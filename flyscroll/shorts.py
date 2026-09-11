"""Live short-video feeds via Playwright (opt-in, no login). Cross-platform.

Supports public YouTube Shorts, TikTok and Instagram Reels. Playwright opens the
platform in Chromium and advances the feed (wheel, next button, ArrowDown) when
the fly loses interest. Frames reach the fly retina through one of two capture
backends:

* macOS: ScreenCaptureKit streams the Chromium window directly (no CDP
  screenshots, so the video window should not blink from capture).
* Windows / Linux: Chromium's compositor is captured over the DevTools protocol
  (``Page.captureScreenshot`` with ``fromSurface``), which needs no OS-specific
  screen-recording APIs and also works headless.

FlyScroll never signs in, never enters credentials and never works around
CAPTCHAs or login walls. Cookie prompts are answered with the most
privacy-preserving option that still lets public content play; dismissable
"log in" promos are closed like any visitor would. If a platform insists on a
login before showing more content, the feed reports that in ``last_scroll_error``
and simply stays on the current video.

Requires:
  pip install -e ".[shorts]"
  PLAYWRIGHT_BROWSERS_PATH=0 playwright install chromium
  macOS only: System Settings → Privacy & Security → Screen Recording → enable
  the app that launches flyscroll (Terminal, iTerm, Cursor, …).
"""

from __future__ import annotations

import base64
import io
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass

import numpy as np
from PIL import Image

from flyscroll.imaging import center_crop_resize


_TARGET_W = 360
_TARGET_H = 640

_MOBILE_UA = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 "
    "Mobile/15E148 Safari/604.1"
)


@dataclass(frozen=True)
class PlatformSpec:
    """Everything platform-specific about browsing one short-video feed."""

    key: str
    title: str
    url: str
    reel_prefix: str
    # Regex with one capture group; applied to location.pathname and post hrefs.
    id_pattern: str
    # Substring that identifies a post link on the page (``a[href*=...]``).
    href_substr: str
    # Emulate a phone (YouTube's vertical Shorts layout) or a narrow desktop.
    mobile_ua: bool = True
    viewport: tuple[int, int] = (420, 780)
    # Cookie / consent controls, most privacy-preserving first.
    consent_selectors: tuple[str, ...] = ()
    # Dismissable promos and "log in" sheets (a visitor's normal "not now").
    dismiss_selectors: tuple[str, ...] = ()
    # Explicit "next video" controls, tried after the mouse wheel.
    next_selectors: tuple[str, ...] = ()
    # How long a next action may take before the active post is expected to change.
    scroll_settle_seconds: float = 1.0


PLATFORMS: dict[str, PlatformSpec] = {
    "youtube": PlatformSpec(
        key="youtube",
        title="YouTube Shorts",
        url="https://www.youtube.com/shorts",
        reel_prefix="youtube",
        id_pattern=r"/shorts/([^/?#]+)",
        href_substr="/shorts/",
        mobile_ua=True,
        viewport=(420, 780),
        consent_selectors=(
            'button:has-text("Reject all")',
            'button[aria-label="Reject all"]',
            'tp-yt-paper-button:has-text("Reject all")',
            'button:has-text("Accept all")',
            'button:has-text("Accept All")',
            'button[aria-label="Accept all"]',
            'tp-yt-paper-button:has-text("Accept all")',
            'button:has-text("I agree")',
            'button:has-text("Got it")',
            "#introAgreeButton",
            "form[action*='consent'] button",
        ),
        dismiss_selectors=(
            'button[aria-label="No thanks"]',
            'button:has-text("No thanks")',
            'button:has-text("Not now")',
        ),
        next_selectors=(
            "button[aria-label*='Next']",
            "#navigation-button-down button",
            "ytd-shorts #navigation-button-down",
        ),
    ),
    "tiktok": PlatformSpec(
        key="tiktok",
        title="TikTok",
        url="https://www.tiktok.com/foryou",
        reel_prefix="tiktok",
        id_pattern=r"/video/(\d+)",
        href_substr="/video/",
        mobile_ua=False,
        viewport=(520, 900),
        consent_selectors=(
            'button:has-text("Decline optional cookies")',
            'button:has-text("Decline all")',
            'button:has-text("Decline")',
            'button:has-text("Accept all")',
            'button:has-text("Allow all")',
        ),
        dismiss_selectors=(
            '[data-e2e="modal-close-inner-button"]',
            'div[role="dialog"] button[aria-label="Close"]',
            'div[role="dialog"] [aria-label="Close"]',
            'button:has-text("Continue as guest")',
            'button:has-text("Not now")',
        ),
        next_selectors=(
            'button[data-e2e="arrow-right"]',
            '[data-e2e="arrow-right"]',
            'button[aria-label*="next"]',
        ),
        # TikTok animates between posts more slowly than YouTube.
        scroll_settle_seconds=2.0,
    ),
    "instagram": PlatformSpec(
        key="instagram",
        title="Instagram Reels",
        url="https://www.instagram.com/reels/",
        reel_prefix="instagram",
        id_pattern=r"/reels?/([^/?#]+)",
        href_substr="/reel/",
        mobile_ua=False,
        viewport=(520, 900),
        consent_selectors=(
            'button:has-text("Decline optional cookies")',
            'button:has-text("Only allow essential cookies")',
            'button:has-text("Allow all cookies")',
        ),
        dismiss_selectors=(
            'div[role="dialog"] button:has-text("Not now")',
            'div[role="dialog"] button:has-text("Not Now")',
            'button:has-text("Not now")',
            'button:has-text("Not Now")',
            'div[role="dialog"] [aria-label="Close"]',
        ),
        next_selectors=(),
        scroll_settle_seconds=1.5,
    ),
}

DEFAULT_PLATFORM = "youtube"
# Kept for callers that imported the old constant.
DEFAULT_SHORTS_URL = PLATFORMS[DEFAULT_PLATFORM].url


def _ensure_browsers_path() -> None:
    """Use package-local browsers (``PLAYWRIGHT_BROWSERS_PATH=0``) when unset."""
    if "PLAYWRIGHT_BROWSERS_PATH" not in os.environ:
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = "0"


def _require_playwright():
    _ensure_browsers_path()
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            'Live short-video mode needs Playwright. Install with:\n'
            '  pip install -e ".[shorts]"\n'
            "  PLAYWRIGHT_BROWSERS_PATH=0 playwright install chromium"
        ) from exc
    return sync_playwright


# TikTok renders its cookie banner inside a shadow root that CSS selectors
# cannot reach. Prefer declining; accept only if that is the sole way onward.
_SHADOW_COOKIE_JS = """() => {
    const host = document.querySelector('tiktok-cookie-banner');
    const root = host && host.shadowRoot;
    if (!root) return false;
    const buttons = [...root.querySelectorAll('button')];
    const text = (b) => (b.textContent || '').trim().toLowerCase();
    const pick = buttons.find((b) => /decline|reject|only necessary|essential/.test(text(b)))
        || buttons.find((b) => /accept|allow/.test(text(b)));
    if (!pick) return false;
    pick.click();
    return true;
}"""

# Generic last resort: click an affirmative/dismissive control by visible text,
# privacy-preserving choices first.
_TEXT_CONSENT_JS = """() => {
    const words = [
        'decline optional', 'only allow essential', 'reject all', 'decline all',
        'necessary only', 'decline', 'not now', 'no thanks',
        'accept all', 'allow all', 'accept', 'i agree', 'agree', 'continue',
    ];
    const controls = [...document.querySelectorAll(
        'button, input[type="submit"], tp-yt-paper-button, [role="button"]'
    )].filter((el) => {
        const r = el.getBoundingClientRect();
        return r.width > 0 && r.height > 0;
    });
    for (const word of words) {
        const target = controls.find((el) => {
            const text = (el.innerText || el.value || el.getAttribute('aria-label') || '')
                .trim().toLowerCase();
            return text === word || text.includes(word);
        });
        if (target) { target.click(); return word; }
    }
    return '';
}"""

# Identify the active post: the URL first, then the video nearest the viewport
# centre and the post link / id attribute around it, then the media itself.
_ACTIVE_VIDEO_JS = """(args) => {
    const re = new RegExp(args.pattern);
    const fromPath = location.pathname.match(re);
    if (fromPath) return fromPath[1];
    const cy = innerHeight / 2;
    const videos = [...document.querySelectorAll('video')]
        .map((v) => ({ v, r: v.getBoundingClientRect() }))
        .filter((o) => o.r.width > 100 && o.r.height > 100 && o.r.bottom > 0 && o.r.top < innerHeight)
        .sort((a, b) => Math.abs((a.r.top + a.r.bottom) / 2 - cy) - Math.abs((b.r.top + b.r.bottom) / 2 - cy));
    const video = videos.length ? videos[0].v : null;
    if (!video) return '';
    let el = video;
    for (let i = 0; i < 12 && el; i++, el = el.parentElement) {
        const attr = el.getAttribute && (el.getAttribute('data-video-id') || el.getAttribute('video-id'));
        if (attr) return attr;
        let link = null;
        if (el.matches && el.matches('a[href*="' + args.href + '"]')) link = el;
        else if (el.querySelector) link = el.querySelector('a[href*="' + args.href + '"]');
        const m = link && (link.getAttribute('href') || '').match(re);
        if (m) return m[1];
    }
    const src = video.currentSrc || video.src || '';
    return src ? 'src:' + src.slice(-40) : '';
}"""

# A hard login wall: a visible password field and no playable video. We report
# it; we never fill it in or route around it.
_LOGIN_WALL_JS = """() => {
    const visible = (el) => {
        if (!el) return false;
        const r = el.getBoundingClientRect();
        return r.width > 0 && r.height > 0;
    };
    const password = [...document.querySelectorAll('input[type="password"]')].some(visible);
    const video = [...document.querySelectorAll('video')].some((v) => {
        const r = v.getBoundingClientRect();
        return r.width > 100 && r.height > 100;
    });
    return password && !video;
}"""


def _click_first(page, selectors, *, visible_ms: int = 400, click_ms: int = 800) -> bool:
    for sel in selectors:
        try:
            loc = page.locator(sel).first
            if loc.count() and loc.is_visible(timeout=visible_ms):
                loc.click(timeout=click_ms)
                page.wait_for_timeout(400)
                return True
        except Exception:
            continue
    return False


def _dismiss_prompts(page, spec: PlatformSpec) -> None:
    """Best-effort cookie / promo dismissal; headed mode lets the user click if this fails."""
    try:
        if page.evaluate(_SHADOW_COOKIE_JS):
            page.wait_for_timeout(400)
            return
    except Exception:
        pass
    if _click_first(page, spec.consent_selectors):
        return
    if _click_first(page, spec.dismiss_selectors):
        return
    try:
        if page.evaluate(_TEXT_CONSENT_JS):
            page.wait_for_timeout(500)
    except Exception:
        pass


def _wait_through_prompts(page, spec: PlatformSpec, timeout_ms: int = 10_000) -> None:
    deadline = time.monotonic() + timeout_ms / 1000.0
    while time.monotonic() < deadline:
        _dismiss_prompts(page, spec)
        try:
            if page.locator("video").count() > 0:
                return
        except Exception:
            pass
        page.wait_for_timeout(350)


def _active_video_signature(page, spec: PlatformSpec) -> str:
    """Platform post id (or media identity) for verifying that the feed advanced."""
    try:
        return str(
            page.evaluate(
                _ACTIVE_VIDEO_JS, {"pattern": spec.id_pattern, "href": spec.href_substr}
            )
            or ""
        )
    except Exception:
        return ""


def _login_wall_visible(page) -> bool:
    try:
        return bool(page.evaluate(_LOGIN_WALL_JS))
    except Exception:
        return False


def _active_video_metrics(page) -> dict[str, float]:
    try:
        result = page.evaluate(
            """() => {
                const videos = [...document.querySelectorAll('video')]
                    .filter((v) => {
                        const r = v.getBoundingClientRect();
                        return r.width > 100 && r.height > 100;
                    })
                    .sort((a, b) => {
                        const ar = a.getBoundingClientRect();
                        const br = b.getBoundingClientRect();
                        return (br.width * br.height) - (ar.width * ar.height);
                    });
                const v = videos[0];
                return {
                    currentTime: Number(v?.currentTime || 0),
                    duration: Number(v?.duration || 0),
                };
            }"""
        )
        return {
            "watch_seconds": max(0.0, float(result.get("currentTime", 0.0))),
            "duration_seconds": max(0.0, float(result.get("duration", 0.0))),
        }
    except Exception:
        return {"watch_seconds": 0.0, "duration_seconds": 0.0}


def _ensure_unmuted(page) -> None:
    """Keep the active video audible across player replacements."""
    try:
        page.evaluate(
            """() => {
                for (const video of document.querySelectorAll('video')) {
                    video.muted = false;
                    video.defaultMuted = false;
                    video.volume = 1;
                }
            }"""
        )
    except Exception:
        pass


class ShortsReel:
    """Duck-types ``Reel``: live buffer instead of a fixed frame list."""

    def __init__(self, feed: "ShortVideoFeed"):
        self._feed = feed
        self.reel_id = f"{feed.spec.reel_prefix}-{uuid.uuid4().hex[:8]}"
        self.title = feed.spec.title
        self.fps = 20.0
        self.index = 0
        self.frames: list[np.ndarray] = []

    @property
    def frame(self) -> np.ndarray:
        return self._feed.latest_frame()

    def advance(self) -> np.ndarray:
        self.index += 1
        return self._feed.capture(force=False)

    @property
    def duration_seconds(self) -> float:
        return float("inf")


class ShortVideoFeed:
    """Duck-types ``Feed``: Playwright navigates; a capture backend supplies frames."""

    def __init__(
        self,
        url: str = "",
        platform: str = DEFAULT_PLATFORM,
        headless: bool = False,
        width: int = _TARGET_W,
        height: int = _TARGET_H,
    ):
        spec = PLATFORMS.get(str(platform).lower())
        if spec is None:
            raise ValueError(
                f"Unknown platform {platform!r}; choose one of {', '.join(sorted(PLATFORMS))}"
            )
        self.spec = spec
        self.platform_key = spec.key
        self.platform_title = spec.title
        if headless and sys.platform == "darwin":
            # ScreenCaptureKit needs an on-screen window; compositor capture on
            # other platforms works headless, so the flag is honored there.
            print(
                "note: --shorts-headless ignored on macOS; ScreenCaptureKit needs a "
                "visible Chromium window",
                flush=True,
            )
            headless = False
        self.url = url or spec.url
        self.headless = headless
        self.width = width
        self.height = height
        self._lock = threading.RLock()
        self._frame = np.zeros((height, width, 3), dtype=np.uint8)
        self._closed = False
        self._pw = None
        self._browser = None
        self._context = None
        self._page = None
        self._streamer = None
        self._capture_cdp = None
        self._last_capture_at = 0.0
        self._last_audio_check_at = 0.0
        self._last_media_check_at = 0.0
        self.capture_backend = "starting"
        self.last_scroll_error = ""
        self.last_watch_metrics = {
            "watch_seconds": 0.0,
            "duration_seconds": 0.0,
        }
        self._current_watch_metrics = dict(self.last_watch_metrics)
        self._media_last_time = 0.0
        self._media_loops = 0
        self.index = 0
        self._reel = ShortsReel(self)
        self.reels = [self._reel]
        try:
            self._start()
        except BaseException:
            # Close synchronous Playwright objects while its event loop is still
            # alive; deferring this to __del__ can produce un-awaited coroutines.
            self.close()
            raise

    def _start(self) -> None:
        sync_playwright = _require_playwright()
        self._pw = sync_playwright().start()
        try:
            self._browser = self._pw.chromium.launch(
                headless=self.headless,
                args=[
                    "--autoplay-policy=no-user-gesture-required",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-backgrounding-occluded-windows",
                    "--disable-renderer-backgrounding",
                ],
            )
        except Exception as exc:
            msg = str(exc)
            if "Executable doesn't exist" in msg or "playwright install" in msg.lower():
                raise RuntimeError(
                    "Playwright Chromium is missing. From the project venv run:\n"
                    "  PLAYWRIGHT_BROWSERS_PATH=0 playwright install chromium"
                ) from exc
            raise

        spec = self.spec
        context_kwargs = {
            "viewport": {"width": spec.viewport[0], "height": spec.viewport[1]},
            "device_scale_factor": 1,
            "locale": "en-US",
        }
        if spec.mobile_ua:
            context_kwargs["user_agent"] = _MOBILE_UA
        self._context = self._browser.new_context(**context_kwargs)
        self._page = self._context.new_page()
        self._page.goto(self.url, wait_until="domcontentloaded", timeout=60_000)
        _wait_through_prompts(self._page, spec)
        try:
            self._page.wait_for_selector("video", timeout=15_000)
        except Exception:
            pass
        self._page.wait_for_timeout(600)
        _ensure_unmuted(self._page)
        try:
            self._page.bring_to_front()
        except Exception:
            pass
        initial_video_id = _active_video_signature(self._page, spec)
        if initial_video_id:
            self._reel.reel_id = f"{spec.reel_prefix}-{initial_video_id}"
        self._reel.title = f"{spec.title} #1"
        self._reset_media_tracker()

        # Frame capture backend. macOS streams the window via ScreenCaptureKit;
        # every other platform captures Chromium's compositor over the DevTools
        # protocol, which needs no OS screen-recording APIs.
        if sys.platform == "darwin":
            self._start_screencapturekit()
        else:
            self._start_compositor_capture()
            print(
                f"Using Chromium compositor capture ({self.capture_backend}); "
                "ScreenCaptureKit is macOS-only.",
                flush=True,
            )

    def _start_screencapturekit(self) -> None:
        """macOS-only: stream the Chromium window via ScreenCaptureKit.

        Falls back to cross-platform compositor capture if the window cannot be
        found or Screen Recording permission is missing.
        """
        from flyscroll.screencapture import WindowStreamer, find_chrome_window

        # Get the browser's OS pid from Chromium itself. Window title matching is
        # retained only as a fallback because sites can rewrite document.title.
        browser_pid = None
        try:
            cdp = self._browser.new_browser_cdp_session()
            process_info = cdp.send("SystemInfo.getProcessInfo")
            browser_processes = [
                p for p in process_info.get("processInfo", []) if p.get("type") == "browser"
            ]
            if browser_processes:
                browser_pid = int(browser_processes[0]["id"])
            cdp.detach()
        except Exception:
            browser_pid = None

        capture_title = f"FlyScroll Capture {uuid.uuid4().hex[:10]}"
        self._page.evaluate("(title) => { document.title = title; }", capture_title)
        try:
            window = find_chrome_window(
                prefer_title_substr=None if browser_pid is not None else capture_title,
                owner_pid=browser_pid,
            )
            self._streamer = WindowStreamer(
                target_w=self.width, target_h=self.height, fps=20.0
            )
            self._streamer.start(window)
            self._frame = self._streamer.wait_first_frame(timeout=2.5)
            self.capture_backend = "screencapturekit"
        except Exception as exc:
            if self._streamer is not None:
                try:
                    self._streamer.stop()
                except Exception:
                    pass
                self._streamer = None
            self._start_compositor_capture()
            print(
                "ScreenCaptureKit window capture unavailable; using Chromium "
                f"compositor capture ({exc}).",
                flush=True,
            )

    def _start_compositor_capture(self) -> None:
        """Capture Chromium's compositor over CDP (cross-platform, no window access)."""
        if self._context is None or self._page is None:
            raise RuntimeError("Cannot start compositor capture before Chromium is ready")
        self._capture_cdp = self._context.new_cdp_session(self._page)
        self.capture_backend = "chromium_compositor"
        # The video element may still be painting right after navigation; wait a
        # few seconds for the compositor to yield a non-blank frame.
        frame = self._frame
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            frame = self._capture_compositor_frame()
            if frame.size and int(frame.max()) > 2:
                break
            time.sleep(0.1)
        self._frame = frame
        if self._frame.size == 0 or int(self._frame.max()) <= 2:
            raise RuntimeError("Chromium compositor returned only blank frames")

    def _capture_compositor_frame(self) -> np.ndarray:
        if self._capture_cdp is None:
            return self._frame
        shot = self._capture_cdp.send(
            "Page.captureScreenshot",
            {
                "format": "jpeg",
                "quality": 55,
                "fromSurface": True,
                "captureBeyondViewport": False,
            },
        )
        raw = base64.b64decode(shot["data"])
        img = Image.open(io.BytesIO(raw))
        return center_crop_resize(img, self.width, self.height)

    @property
    def current(self) -> ShortsReel:
        return self._reel

    @property
    def current_watch_metrics(self) -> dict[str, float]:
        with self._lock:
            return dict(self._current_watch_metrics)

    def latest_frame(self) -> np.ndarray:
        with self._lock:
            now = time.monotonic()
            if (
                self._page is not None
                and now - self._last_audio_check_at >= 1.0
            ):
                _ensure_unmuted(self._page)
                self._last_audio_check_at = now
            if (
                self._page is not None
                and now - self._last_media_check_at >= 0.25
            ):
                self._update_media_progress()
                self._last_media_check_at = now
            if self._streamer is not None:
                self._frame = self._streamer.latest_frame()
            elif self._capture_cdp is not None:
                # The neural loop is slower than video playback; sample only the
                # newest compositor frame and avoid duplicate captures per tick.
                if now - self._last_capture_at >= 0.075:
                    frame = self._capture_compositor_frame()
                    if frame.size and int(frame.max()) > 2:
                        self._frame = frame
                    self._last_capture_at = now
            return self._frame

    def capture(self, force: bool = False) -> np.ndarray:
        # Stream pushes continuously; just read the latest buffer.
        return self.latest_frame()

    def scroll(self) -> ShortsReel:
        with self._lock:
            if self._closed or self._page is None:
                return self._reel
            spec = self.spec
            _wait_through_prompts(self._page, spec, timeout_ms=2_500)
            before = _active_video_signature(self._page, spec)
            self._update_media_progress()
            watched_metrics = dict(self._current_watch_metrics)
            self.last_scroll_error = ""
            changed = False
            after = ""
            attempts = (
                self._scroll_with_wheel,
                self._scroll_with_next_button,
                self._scroll_with_keyboard,
            )
            for attempt in attempts:
                try:
                    attempt()
                    after = self._wait_for_video_change(
                        before, timeout_seconds=spec.scroll_settle_seconds
                    )
                    if after:
                        changed = True
                        break
                except Exception:
                    continue
            if not changed:
                if _login_wall_visible(self._page):
                    self.last_scroll_error = (
                        f"{spec.title} is asking for a login before showing more videos. "
                        "FlyScroll never signs in, so the feed stays on the current video."
                    )
                else:
                    self.last_scroll_error = (
                        f"{spec.title} remained on video {before or 'unknown'} after all next actions"
                    )
                print(
                    f"warning: {self.last_scroll_error}; skip was not recorded",
                    flush=True,
                )
                return self._reel
            _ensure_unmuted(self._page)
            self.last_watch_metrics = watched_metrics
            self.index += 1
            self._reel.reel_id = f"{spec.reel_prefix}-{after}"
            self._reel.title = f"{spec.title} #{self.index + 1}"
            self._reel.index = 0
            self._reset_media_tracker()
        return self._reel

    def _reset_media_tracker(self) -> None:
        self._media_last_time = 0.0
        self._media_loops = 0
        self._last_media_check_at = 0.0
        self._current_watch_metrics = {
            "watch_seconds": 0.0,
            "duration_seconds": 0.0,
        }

    def _update_media_progress(self) -> None:
        if self._page is None:
            return
        metrics = _active_video_metrics(self._page)
        current = float(metrics["watch_seconds"])
        duration = float(metrics["duration_seconds"])
        if duration <= 0:
            return
        # Players loop short videos by resetting currentTime to zero. Accumulate
        # the completed pass so a full watch cannot later appear as 2–3 seconds.
        if (
            self._media_last_time >= 0.60 * duration
            and current <= 0.25 * duration
            and current + 2.0 < self._media_last_time
        ):
            self._media_loops += 1
        cumulative = self._media_loops * duration + current
        self._media_last_time = current
        self._current_watch_metrics = {
            "watch_seconds": min(cumulative, duration),
            "raw_watch_seconds": cumulative,
            "duration_seconds": duration,
        }

    def _wait_for_video_change(self, before: str, timeout_seconds: float) -> str:
        assert self._page is not None
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            self._page.wait_for_timeout(150)
            after = _active_video_signature(self._page, self.spec)
            if after and after != before:
                return after
        return ""

    def _scroll_with_wheel(self) -> None:
        assert self._page is not None
        w, h = self.spec.viewport
        self._page.mouse.move(w * 0.5, h * 0.72)
        self._page.mouse.wheel(0, h * 0.9)

    def _scroll_with_next_button(self) -> None:
        assert self._page is not None
        for selector in self.spec.next_selectors:
            target = self._page.locator(selector).first
            if target.count() and target.is_visible(timeout=250):
                target.click(timeout=700)
                return
        raise RuntimeError("No visible next-video button")

    def _scroll_with_keyboard(self) -> None:
        assert self._page is not None
        self._page.keyboard.press("ArrowDown")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._streamer is not None:
                try:
                    self._streamer.stop()
                except Exception:
                    pass
                self._streamer = None
            if self._capture_cdp is not None:
                try:
                    self._capture_cdp.detach()
                except Exception:
                    pass
                self._capture_cdp = None
            for obj in (self._context, self._browser):
                try:
                    if obj is not None:
                        obj.close()
                except Exception:
                    pass
            try:
                if self._pw is not None:
                    self._pw.stop()
            except Exception:
                pass
            self._page = None
            self._context = None
            self._browser = None
            self._pw = None

    def __del__(self):  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass


class YouTubeShortsFeed(ShortVideoFeed):
    """Backward-compatible alias: a ``ShortVideoFeed`` fixed to YouTube Shorts."""

    def __init__(
        self,
        url: str = DEFAULT_SHORTS_URL,
        headless: bool = False,
        width: int = _TARGET_W,
        height: int = _TARGET_H,
    ):
        super().__init__(url=url, platform="youtube", headless=headless, width=width, height=height)
