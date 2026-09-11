"""Bumble mode: the fly swipes a real Android phone over adb (opt-in).

The phone's screen is streamed into the fly retina with ``adb exec-out screencap
-p``. While the fly is still curious about a profile the feed pages further down
it with an ``up`` swipe; when the fly is done, the feed commits to a decision --
swipe **right** (like) when the interest in the last ~500 ms *spike* window
was high relative to the profiles the fly saw recently, else **left** (pass).

The three adb primitives used here (``get-state``, ``exec-out screencap -p``,
``shell input swipe``), the injectable ``runner`` that makes them testable
without hardware, and the human-like pacing of ``--bumble-247`` follow the
approach taken by BumbleClaw (``bumble_phone_auto.py``). This module is an
independent re-implementation: FlyScroll imports nothing from that project. See
THIRD_PARTY.md.

Safety notes. FlyScroll never signs in and never enters credentials -- the phone
is unlocked and already logged in by the operator. Automating swipes is very
likely against Bumble's terms of service and the account risk is the operator's.
``outputs/flyscroll/bumble/`` ends up holding other people's profile photos, so
it stays local (and is already gitignored).

Requires the Android platform tools on PATH (``adb``), USB debugging enabled and
authorized, the screen unlocked and Bumble open on the swipe deck.
"""

from __future__ import annotations

import csv
import io
import random
import re
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from flyscroll.imaging import center_crop_resize

ROOT = Path(__file__).resolve().parents[1]

_TARGET_W = 360
_TARGET_H = 640

# adb occasionally wedges on a sleeping USB port; a bounded wait keeps the
# neural loop from stalling forever on a single screencap.
ADB_TIMEOUT_SECONDS = 12.0

# Tiny grayscale thumbnails for change detection: small enough that the status
# bar clock ticking over cannot register, large enough that a new profile card
# always does.
_DIFF_W, _DIFF_H = 32, 57

Gesture = tuple[int, int, int, int, int]


# --- adb primitives ------------------------------------------------------------
# Every primitive takes an injectable ``runner`` (default ``subprocess.run``), so
# the decision logic and the feed can be exercised with a scripted fake and no
# phone attached. Stderr is decoded into the RuntimeError message: adb's
# failures ("unauthorized", "device offline") are the useful part.


def adb_base(adb: str = "adb", serial: str = "") -> list[str]:
    command = [adb]
    if serial:
        command.extend(["-s", serial])
    return command


def _decode(stream) -> str:
    if isinstance(stream, (bytes, bytearray)):
        return bytes(stream).decode("utf-8", errors="ignore")
    return "" if stream is None else str(stream)


def _run(args: list[str], runner) -> tuple[int, str, str]:
    """Run one adb command and return ``(returncode, stdout, stderr)`` as text.

    ``text=True`` is deliberately not passed: ``screencap`` needs raw bytes, and
    keeping one call shape for every primitive means one fake in the tests.
    """
    result = runner(
        args,
        capture_output=True,
        check=False,
        timeout=ADB_TIMEOUT_SECONDS,
    )
    return (
        int(getattr(result, "returncode", 0) or 0),
        _decode(getattr(result, "stdout", "")).strip(),
        _decode(getattr(result, "stderr", "")).strip(),
    )


def ensure_device(adb: str = "adb", serial: str = "", runner=subprocess.run) -> None:
    """Fail fast unless exactly one authorized device answers."""
    code, out, err = _run([*adb_base(adb, serial), "get-state"], runner)
    if code != 0 or out != "device":
        detail = err or out or "no device"
        raise RuntimeError(
            f"No authorized adb device ({detail}). Connect the phone over USB, "
            "enable USB debugging in Developer options, and accept the "
            "'Allow USB debugging?' prompt on the phone."
        )


def ensure_awake(adb: str = "adb", serial: str = "", runner=subprocess.run) -> None:
    """Require an awake screen before the session starts.

    A locked or dark phone still screencaps happily -- it just returns the same
    static image forever, which would make every swipe look unconfirmed and the
    fly's novelty collapse on frame two. Better to say so at startup.
    """
    code, out, err = _run(
        [*adb_base(adb, serial), "shell", "dumpsys", "power"], runner
    )
    if code != 0:
        raise RuntimeError(f"adb dumpsys power failed: {err or out or 'unknown error'}")
    if "mWakefulness=Awake" not in out:
        raise RuntimeError(
            "The phone screen is asleep or locked. Unlock the phone and open "
            "Bumble on the swipe deck before starting FlyScroll."
        )


def screencap_png(adb: str = "adb", serial: str = "", runner=subprocess.run) -> bytes:
    result = runner(
        [*adb_base(adb, serial), "exec-out", "screencap", "-p"],
        capture_output=True,
        check=False,
        timeout=ADB_TIMEOUT_SECONDS,
    )
    if int(getattr(result, "returncode", 0) or 0) != 0:
        detail = _decode(getattr(result, "stderr", "")).strip()
        raise RuntimeError(f"adb screencap failed: {detail or 'unknown error'}")
    data = getattr(result, "stdout", b"")
    if not data:
        raise RuntimeError("adb screencap returned no image data")
    if isinstance(data, str):
        data = data.encode("latin-1", errors="ignore")
    return bytes(data)


def swipe(
    adb: str = "adb",
    serial: str = "",
    x1: int = 0,
    y1: int = 0,
    x2: int = 0,
    y2: int = 0,
    duration_ms: int = 250,
    runner=subprocess.run,
) -> None:
    code, out, err = _run(
        [
            *adb_base(adb, serial),
            "shell",
            "input",
            "swipe",
            str(int(x1)),
            str(int(y1)),
            str(int(x2)),
            str(int(y2)),
            str(int(duration_ms)),
        ],
        runner,
    )
    if code != 0:
        raise RuntimeError(f"adb swipe failed: {err or out or 'unknown error'}")


def wm_size(adb: str = "adb", serial: str = "", runner=subprocess.run) -> tuple[int, int]:
    """Screen geometry in the coordinate space ``input swipe`` uses.

    ``Override size`` wins over ``Physical size``: when a display is overridden
    (``wm size 1080x2400`` on a 1440p panel, common on phones set to a lower
    resolution) the input system speaks the overridden coordinates, and physical
    ones would land off-card.
    """
    code, out, err = _run([*adb_base(adb, serial), "shell", "wm", "size"], runner)
    if code != 0:
        raise RuntimeError(f"adb wm size failed: {err or out or 'unknown error'}")
    match = re.search(r"Override size:\s*(\d+)\s*x\s*(\d+)", out) or re.search(
        r"Physical size:\s*(\d+)\s*x\s*(\d+)", out
    )
    if match is None:
        raise RuntimeError(f"Could not parse `adb shell wm size` output: {out!r}")
    return int(match.group(1)), int(match.group(2))


def derive_gestures(width: int, height: int) -> dict[str, Gesture]:
    """Default swipe paths for one screen geometry.

    Horizontal flings cross the middle of the card (85% -> 15% of the width at
    half height) because the photo area, not the buttons, is what a fling acts
    on; ``up`` drags the profile upward from 70% to 30% height to reveal the
    next photo and the bio.
    """
    # round(), not int(): truncating a binary-float product costs a pixel on some
    # geometries, and the coordinates are read back in tests.
    mid_x = round(width * 0.50)
    mid_y = round(height * 0.50)
    x_right = round(width * 0.85)
    x_left = round(width * 0.15)
    return {
        "left": (x_right, mid_y, x_left, mid_y, 250),
        "right": (x_left, mid_y, x_right, mid_y, 250),
        "up": (mid_x, round(height * 0.70), mid_x, round(height * 0.30), 300),
    }


def parse_swipe_spec(spec: str) -> Gesture:
    """Parse a ``"x1,y1,x2,y2,ms"`` CLI override into a gesture tuple."""
    parts = [p.strip() for p in str(spec).split(",")]
    if len(parts) != 5:
        raise ValueError(f"Expected 5 comma-separated values 'x1,y1,x2,y2,ms', got {spec!r}")
    try:
        x1, y1, x2, y2, ms = (int(float(p)) for p in parts)
    except ValueError as exc:
        raise ValueError(f"Non-numeric swipe spec {spec!r}") from exc
    if ms <= 0:
        raise ValueError(f"Swipe duration must be positive in {spec!r}")
    return x1, y1, x2, y2, ms


def _thumb(frame) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return np.asarray(
        img.resize((_DIFF_W, _DIFF_H), Image.Resampling.BILINEAR), dtype=np.float32
    )


def frame_changed(before, after, threshold: float = 8.0) -> bool:
    """Did the phone screen actually move on?

    Mean absolute difference of 32x57 grayscale downsamples. This is the only
    confirmation available: adb reports a swipe as delivered whether or not the
    app reacted, so an out-of-likes dialog or a backgrounded Bumble would
    otherwise be logged as a successful decision.
    """
    if before is None or after is None:
        return True
    a = _thumb(before)
    b = _thumb(after)
    if a.shape != b.shape:
        return True
    return float(np.abs(a - b).mean()) >= float(threshold)


# --- decision policy -----------------------------------------------------------


@dataclass
class RollingRightRate:
    """Self-calibrating like threshold that holds the right-swipe rate near ``target``.

    Absolute interest in Hz depends on the graph scale, the phone's brightness
    and what the photos happen to contain, so a fixed cutoff is guesswork. The
    threshold is instead the percentile of recent per-profile mean interest that
    leaves ``target`` of profiles above it. Before ``min_history`` profiles have
    been scored there is nothing to take a percentile of, so ``fallback``
    (``--bumble-like-threshold``) carries the warm-up.
    """

    target: float = 0.25
    window: int = 40
    min_history: int = 8
    fallback: float = 20.0
    history: deque = field(default_factory=deque)

    def __post_init__(self):
        self.history = deque(maxlen=max(int(self.window), 1))

    @property
    def mode(self) -> str:
        return "rolling" if len(self.history) >= self.min_history else "fallback"

    def threshold(self) -> float:
        if len(self.history) < self.min_history:
            return float(self.fallback)
        pct = 100.0 * (1.0 - float(np.clip(self.target, 0.0, 1.0)))
        return float(np.percentile(np.asarray(self.history, dtype=np.float64), pct))

    def observe(self, mean_interest: float) -> None:
        self.history.append(float(mean_interest))


# --- feed ----------------------------------------------------------------------


class BumbleReel:
    """Duck-types ``Reel``: one Bumble profile, frames pulled live from the phone."""

    def __init__(self, feed: "BumbleFeed", number: int):
        self._feed = feed
        self.number = int(number)
        self.reel_id = f"bumble-{self.number:04d}"
        self.title = f"Bumble #{self.number}"
        self.fps = 20.0
        self.index = 0
        self.frames: list[np.ndarray] = []

    @property
    def frame(self) -> np.ndarray:
        return self._feed.latest_frame()

    def advance(self) -> np.ndarray:
        self.index += 1
        return self._feed.latest_frame()

    @property
    def duration_seconds(self) -> float:
        # A profile has no runtime; boredom or the max-watch timeout ends it.
        return float("inf")


class BumbleFeed:
    """Duck-types ``Feed`` over a real phone: screencaps in, swipes out."""

    CSV_HEADER = (
        "timestamp",
        "profile_n",
        "decision",
        "mean_interest",
        "window_interest",
        "peak_interest",
        "watch_seconds",
        "profile_scrolls",
        "threshold",
        "threshold_mode",
        "reason",
        "dry_run",
        "confirmed",
    )

    def __init__(
        self,
        adb: str = "adb",
        serial: str = "",
        fps: float = 3.0,
        decision_window_seconds: float = 0.5,
        target_right_rate: float = 0.25,
        like_threshold: float | None = None,
        dry_run: bool = False,
        max_swipes: int | None = None,
        max_profile_scrolls: int = 3,
        profile_max_watch: float = 15.0,
        left_swipe: str = "",
        right_swipe: str = "",
        up_swipe: str = "",
        mode_247: bool = False,
        decisions_csv: Path | str | None = None,
        width: int = _TARGET_W,
        height: int = _TARGET_H,
        runner=subprocess.run,
        sleeper=time.sleep,
    ):
        self.adb = str(adb or "adb")
        self.serial = str(serial or "")
        self.fps = float(fps)
        self.width = int(width)
        self.height = int(height)
        self.dry_run = bool(dry_run)
        # Optional ``(rgb) -> (rgb, info)`` censor for frames written to disk.
        # DoomscrollSession sets this to its PrivacyFilter.apply when privacy
        # mode is on; the fly's own frames are never routed through it.
        self.display_frame_filter = None
        self.max_swipes = int(max_swipes) if max_swipes else None
        self.max_profile_scrolls = max(int(max_profile_scrolls), 0)
        self.profile_max_watch = float(profile_max_watch)
        self.mode_247 = bool(mode_247)
        self._runner = runner
        self._sleeper = sleeper
        self._rng = random.Random()

        # Session-facing attributes (the generic ones session.py already reads).
        self.platform_key = "bumble"
        self.platform_title = "Bumble"
        self.capture_backend = "adb_screencap"
        self.last_scroll_error = ""
        self.last_advance = ""
        self.last_decision_label = ""
        self.last_mean_interest = 0.0
        self.likes = 0
        self.passes = 0
        self.swipes = 0
        self.profile_scrolls = 0
        self.device_lost = False

        self._closed = False
        self._thread = None
        self._lock = threading.RLock()
        self._frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._interest_sum = 0.0
        self._interest_seconds = 0.0
        self.peak_interest = 0.0
        # Short rolling window of recent interest -- the "spike" the swipe
        # decision reads, so the fly commits on interest *now* and swipes fast.
        self._window_seconds = max(float(decision_window_seconds), 0.05)
        self._window = deque()  # (clock, hz, dt)
        self._window_clock = 0.0
        self._last_hourly_pause_at = time.monotonic()

        # A 0 (or negative) target right-rate means "use --bumble-like-threshold
        # as a fixed rule" instead of the self-calibrating percentile.
        target = float(target_right_rate or 0.0)
        fallback = float(like_threshold if like_threshold is not None else 20.0)
        self._fixed_threshold = target <= 0.0
        self._rule = RollingRightRate(target=target, fallback=fallback)

        if decisions_csv is None:
            self.decisions_csv = ROOT / "outputs" / "flyscroll" / "bumble" / "decisions.csv"
        else:
            self.decisions_csv = Path(decisions_csv) if decisions_csv else None

        try:
            ensure_device(self.adb, self.serial, self._runner)
            ensure_awake(self.adb, self.serial, self._runner)
            overrides = {
                name: parse_swipe_spec(spec)
                for name, spec in (
                    ("left", left_swipe),
                    ("right", right_swipe),
                    ("up", up_swipe),
                )
                if spec
            }
            if len(overrides) == 3:
                self.screen_size = (0, 0)
                self.gestures = dict(overrides)
            else:
                self.screen_size = wm_size(self.adb, self.serial, self._runner)
                self.gestures = derive_gestures(*self.screen_size)
                self.gestures.update(overrides)
            # Seed the retina synchronously so the first neural tick sees the
            # phone rather than a black frame.
            self._frame = self._grab_frame()
        except BaseException:
            self.close()
            raise

        self.index = 0
        self._reel = BumbleReel(self, 1)
        self.reels = [self._reel]
        if self.fps > 0:
            # A screencap costs ~150-400 ms on a phone, so a few frames a second
            # is what the wire can carry; profiles are static photos, which is
            # plenty for the retina. fps<=0 captures on demand instead (used by
            # the tests, and by anyone who wants one screencap per neural tick).
            self._thread = threading.Thread(
                target=self._capture_loop, name="flyscroll-bumble-capture", daemon=True
            )
            self._thread.start()

    # --- capture ---------------------------------------------------------------

    def _grab_frame(self) -> np.ndarray:
        png = screencap_png(self.adb, self.serial, self._runner)
        return center_crop_resize(Image.open(io.BytesIO(png)), self.width, self.height)

    def _safe_frame(self) -> np.ndarray | None:
        """Capture, recording a device loss instead of raising into the loop."""
        try:
            frame = self._grab_frame()
        except Exception as exc:
            self.device_lost = True
            self.last_scroll_error = f"Lost the phone: {exc}"
            return None
        self._frame = frame
        return frame

    def _capture_loop(self) -> None:
        period = 1.0 / max(self.fps, 0.1)
        while not self._closed:
            began = time.monotonic()
            try:
                frame = self._grab_frame()
            except Exception as exc:
                self.device_lost = True
                self.last_scroll_error = f"Lost the phone: {exc}"
                self._sleeper(1.0)
                continue
            with self._lock:
                self._frame = frame
                if self.device_lost:
                    self.device_lost = False
                    self.last_scroll_error = ""
            elapsed = time.monotonic() - began
            if elapsed < period:
                self._sleeper(period - elapsed)

    @property
    def current(self) -> BumbleReel:
        return self._reel

    def latest_frame(self) -> np.ndarray:
        with self._lock:
            if self._thread is None and not self._closed:
                self._safe_frame()
            return self._frame

    def capture(self, force: bool = False) -> np.ndarray:
        return self.latest_frame()

    # --- interest accumulation -------------------------------------------------

    def on_interest(self, interest_hz: float, wall_seconds: float) -> None:
        """Accumulate the fly's interest over the current profile.

        Time-weighted (sum of interest*dt over sum of dt): neural ticks arrive at
        irregular wall intervals, so a plain mean of samples would over-weight
        whichever ticks happened to be short.
        """
        dt = max(float(wall_seconds), 0.0)
        if dt <= 0:
            return
        value = max(float(interest_hz), 0.0)
        with self._lock:
            self._interest_sum += value * dt
            self._interest_seconds += dt
            self.peak_interest = max(self.peak_interest, value)
            self._window_clock += dt
            self._window.append((self._window_clock, value, dt))
            cutoff = self._window_clock - self._window_seconds
            while len(self._window) > 1 and self._window[0][0] < cutoff:
                self._window.popleft()

    @property
    def dwell_seconds(self) -> float:
        return round(self._interest_seconds, 3)

    @property
    def mean_interest(self) -> float:
        if self._interest_seconds <= 0:
            return 0.0
        return round(self._interest_sum / self._interest_seconds, 3)

    @property
    def window_interest(self) -> float:
        """Time-weighted mean interest over the last ``decision_window_seconds``.

        The signal the swipe decision reads. Falls back to the profile mean
        until the window has any samples."""
        with self._lock:
            total = sum(dt for _c, _hz, dt in self._window)
            if total <= 0:
                return self.mean_interest
            return round(sum(hz * dt for _c, hz, dt in self._window) / total, 3)

    @property
    def window_peak(self) -> float:
        with self._lock:
            return round(max((hz for _c, hz, _dt in self._window), default=0.0), 3)

    @property
    def threshold_mode(self) -> str:
        return "fixed" if self._fixed_threshold else self._rule.mode

    @property
    def current_threshold(self) -> float:
        value = self._rule.fallback if self._fixed_threshold else self._rule.threshold()
        return round(float(value), 3)

    # --- swiping ---------------------------------------------------------------

    def _gesture(self, name: str) -> None:
        x1, y1, x2, y2, ms = self.gestures[name]
        swipe(self.adb, self.serial, x1, y1, x2, y2, ms, runner=self._runner)

    def _settle(self) -> None:
        """Human-like pacing after a decision swipe (BumbleClaw's cadence)."""
        self._sleeper(0.8 + self._rng.uniform(0.0, 0.4))
        if not self.mode_247:
            return
        now = time.monotonic()
        if now - self._last_hourly_pause_at >= 3600.0:
            # This blocks the whole session loop, so the chamber looks frozen for
            # two to three minutes. Documented in the README.
            self._sleeper(self._rng.uniform(120.0, 180.0))
            self._last_hourly_pause_at = now

    def scroll(self) -> BumbleReel:
        """Called when the fly is done with the current *screen*.

        Either pages further into the profile (``last_advance ==
        "profile_scroll"``, same reel returned, session resets boredom) or makes
        the like/pass decision and starts a new profile.

        The whole body holds ``self._lock``: there is one adb channel, and the
        confirmation captures must not interleave with the capture thread's.
        """
        with self._lock:
            if self._closed:
                self.last_scroll_error = "The Bumble feed is closed."
                self.last_advance = "failed"
                return self._reel
            if self.device_lost:
                self.last_advance = "failed"
                return self._reel
            if self.max_swipes is not None and self.swipes >= self.max_swipes:
                self.last_scroll_error = (
                    f"Reached --bumble-max-swipes ({self.max_swipes}); "
                    "the fly keeps watching but no longer swipes."
                )
                self.last_advance = "failed"
                return self._reel

            self.last_scroll_error = ""
            before = self._frame

            # 1. Read further down the profile before judging it: the first photo
            #    is not the profile.
            if (
                self.profile_scrolls < self.max_profile_scrolls
                and self._interest_seconds < self.profile_max_watch
            ):
                if self.dry_run:
                    # Nothing was swiped, so treat the page as turned; a dry run
                    # still walks the same decision path.
                    self.profile_scrolls += 1
                    self.last_advance = "profile_scroll"
                    return self._reel
                self._gesture("up")
                self._sleeper(0.6)
                after = self._safe_frame()
                if after is None:
                    self.last_advance = "failed"
                    return self._reel
                if frame_changed(before, after):
                    self.profile_scrolls += 1
                    self.last_advance = "profile_scroll"
                    return self._reel
                # Unchanged: bottom of the profile. Fall through and decide.

            # 2. Decide from a short "spike" window of interest (the last
            #    decision_window_seconds), not the whole-profile mean: the fly
            #    commits on how interesting the profile is *right now*, so swipes
            #    stay quick. mean/peak are kept for the log and dashboard.
            signal = self.window_interest
            mean = self.mean_interest
            self.last_mean_interest = mean
            threshold = self.current_threshold
            mode = self.threshold_mode
            decision = "like" if signal >= threshold else "pass"
            self._rule.observe(signal)

            # 3. Swipe (unless this is a dry run), then confirm it landed.
            confirmed = False
            if self.dry_run:
                self.swipes += 1
                reason = "dry_run"
            else:
                self._gesture("right" if decision == "like" else "left")
                # Counted even when unconfirmed: the phone did receive a gesture,
                # so --bumble-max-swipes must still spend one.
                self.swipes += 1
                self._settle()
                after = self._safe_frame()
                confirmed = after is not None and frame_changed(before, after)
                if not confirmed:
                    self._sleeper(0.4)
                    after = self._safe_frame()
                    confirmed = after is not None and frame_changed(before, after)
                reason = "swipe_confirmed" if confirmed else "screen_unchanged"

            if not self.dry_run and not confirmed:
                self.last_scroll_error = (
                    "Screen unchanged after swipe — out of likes, a dialog, or "
                    "Bumble isn't on the deck?"
                )
                self.last_advance = "failed"
                # Logged anyway: an unconfirmed decision is the interesting row.
                # The dwell accumulators stay put, so the fly simply retries when
                # it gets bored again.
                self._log_decision(
                    decision=decision,
                    mean=mean,
                    window=signal,
                    threshold=threshold,
                    mode=mode,
                    reason=reason,
                    confirmed=False,
                )
                return self._reel

            # 4. Committed: counters, label, log, next profile.
            if decision == "like":
                self.likes += 1
            else:
                self.passes += 1
            label = "❤ like" if decision == "like" else "✕ pass"
            if self.dry_run:
                label += " (dry run)"
            self.last_decision_label = label
            self.last_advance = decision
            self._log_decision(
                decision=decision,
                mean=mean,
                window=signal,
                threshold=threshold,
                mode=mode,
                reason=reason,
                confirmed=confirmed,
            )
            self.index += 1
            self._reel = BumbleReel(self, self.index + 1)
            self.reels = [self._reel]
            self._interest_sum = 0.0
            self._interest_seconds = 0.0
            self.peak_interest = 0.0
            self._window.clear()
            self._window_clock = 0.0
            self.profile_scrolls = 0
            return self._reel

    # --- logging ---------------------------------------------------------------

    def _log_decision(
        self,
        *,
        decision: str,
        mean: float,
        window: float,
        threshold: float,
        mode: str,
        reason: str,
        confirmed: bool,
    ) -> None:
        """One CSV row plus the profile's last frame, for later calibration."""
        path = self.decisions_csv
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fresh = not path.exists()
            with path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                if fresh:
                    writer.writerow(self.CSV_HEADER)
                writer.writerow(
                    [
                        datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        self.index + 1,
                        decision,
                        f"{mean:.3f}",
                        f"{window:.3f}",
                        f"{self.peak_interest:.3f}",
                        f"{self._interest_seconds:.3f}",
                        self.profile_scrolls,
                        f"{threshold:.3f}",
                        mode,
                        reason,
                        bool(self.dry_run),
                        bool(confirmed),
                    ]
                )
            shot = path.parent / f"profile_{self.index + 1:04d}.jpg"
            saved = self._frame
            if self.display_frame_filter is not None:
                saved = self.display_frame_filter(saved)[0]
            Image.fromarray(saved).save(shot, format="JPEG", quality=60)
        except Exception:
            # Losing a log row must never stop the fly from swiping.
            pass

    def close(self) -> None:
        self._closed = True
        thread = self._thread
        self._thread = None
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=2.0)
