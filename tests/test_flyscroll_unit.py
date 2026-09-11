"""Unit tests that do not require the full MaleCNS graph."""

from __future__ import annotations

import numpy as np

from flyscroll.feed import synthesize_brainrot, Feed
from flyscroll.flyvision import FlyEyeProcessor, PopulationNovelty
from flyscroll.interest import InterestDecoder, HabituationRule
from flyscroll.analytics import WatchAnalytics
from flyscroll.privacy import PrivacyFilter
from flyscroll.retina import luminance_samples, chromatic_samples
from flyscroll.transmitters import transmitter_signs


class _EyeBrain:
    def __init__(self, rows=8, cols=8):
        yy, xx = np.mgrid[0:rows, 0:cols]
        self.hexes = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)
        self.uv = np.column_stack(
            [
                (xx.ravel() + 0.5) / cols,
                (yy.ravel() + 0.5) / rows,
            ]
        ).astype(np.float32)
        self.cell_type = np.asarray(["other"] * (rows * cols), dtype="U16")
        self.chromatic_uv = np.zeros((0, 2), dtype=np.float32)
        self.chromatic_channel = np.zeros(0, dtype="U16")
        self.drive = np.zeros(len(self.cell_type), dtype=np.float32)


def test_transmitter_signs():
    signs, uncertain = transmitter_signs(["acetylcholine", "gaba", "unknown", "acetylcholine,gaba"])
    assert list(signs) == [1, -1, 1, 1]
    assert list(uncertain) == [False, False, True, True]


def test_retina_sampling_shape():
    rgb = np.zeros((64, 36, 3), dtype=np.uint8)
    rgb[:, :, 1] = 255
    uv = np.asarray([[0.2, 0.3], [0.8, 0.7]], dtype=np.float32)
    luma = luminance_samples(rgb, uv)
    assert luma.shape == (2,)
    assert float(luma.mean()) > 0.5
    chroma = chromatic_samples(rgb, uv, np.asarray(["green", "blue"]))
    assert chroma.shape == (2,)
    assert chroma[0] > chroma[1]


def test_fly_eye_separates_on_and_off_transients():
    brain = _EyeBrain()
    eye = FlyEyeProcessor(brain)
    dark = np.zeros((96, 96, 3), dtype=np.uint8)
    bright_patch = dark.copy()
    bright_patch[24:72, 24:72] = 255
    eye.process(dark, brain, 0.05)
    on_frame = eye.process(bright_patch, brain, 0.05)
    off_frame = eye.process(dark, brain, 0.05)
    assert on_frame.features["on"] > on_frame.features["off"]
    assert off_frame.features["off"] > off_frame.features["on"]


def test_fly_eye_distinguishes_opposite_motion():
    brain = _EyeBrain()

    def motion_energy(positions):
        eye = FlyEyeProcessor(brain)
        total = np.zeros(2, dtype=np.float64)
        for x in positions:
            frame = np.zeros((96, 96, 3), dtype=np.uint8)
            frame[:, x : x + 14] = 255
            out = eye.process(frame, brain, 0.05)
            total += [out.features["motion_left"], out.features["motion_right"]]
        return total

    forward = motion_energy([8, 22, 36, 50, 64])
    backward = motion_energy([64, 50, 36, 22, 8])
    assert forward.sum() > 0
    assert backward.sum() > 0
    assert int(np.argmax(forward)) != int(np.argmax(backward))


def test_population_novelty_decays_for_repetition_and_recovers_for_change():
    novelty = PopulationNovelty(n_features=4)
    a = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    b = np.asarray([1.0, 0.0, 0.8, 0.1], dtype=np.float32)
    novelty.update(a, 0.05)
    for _ in range(60):
        repeated = novelty.update(a, 0.05)
    changed = novelty.update(b, 0.05)
    assert repeated["population_novelty"] < 1.0
    assert changed["population_novelty"] > repeated["population_novelty"] + 5.0


def test_full_session_plot_history_is_downsampled_without_losing_span():
    analytics = WatchAnalytics(live_window=120)
    analytics.start_reel("r1", "test")
    for i in range(1000):
        analytics.observe(
            interest=float(i % 17),
            novelty=1000.0 if i == 500 else float(i % 23),
            watch_seconds=i * 0.05,
            surprise=None,
            tick=i,
            dt_seconds=0.05,
        )
    snap = analytics.snapshot()
    interest = snap["live_interest"]
    novelty = snap["live_novelty"]
    assert len(interest) <= 120
    assert len(novelty) <= 120
    assert interest[0]["t"] == 0.05
    assert interest[-1]["t"] == 50.0
    assert max(p["v"] for p in novelty) == 1000.0


def test_watch_outcome_histogram_counts_bored_and_full_watches():
    analytics = WatchAnalytics(max_watch_seconds=5.0)
    analytics.start_reel("a", "bored reel")
    analytics.on_scroll(reason="bored", watch_seconds=2.0, tick=1)
    analytics.start_reel("b", "full reel")
    analytics.on_scroll(reason="timeout", watch_seconds=5.0, tick=2)
    assert analytics.snapshot()["outcome_counts"] == {"bored": 1, "full_watch": 1}


def test_watch_percentage_uses_actual_media_progress_and_duration():
    analytics = WatchAnalytics(max_watch_seconds=5.0)
    analytics.start_reel("a", "long reel")
    entry = analytics.on_scroll(
        reason="timeout",
        watch_seconds=5.0,
        media_watch_seconds=6.0,
        reel_duration_seconds=24.0,
        tick=1,
    )
    assert entry["watch_pct"] == 25.0
    assert entry["watch_seconds"] == 6.0
    assert entry["duration_seconds"] == 24.0
    assert analytics.snapshot()["outcome_counts"] == {"bored": 1, "full_watch": 0}


def test_feed_scrolls():
    reels = synthesize_brainrot(seed=1, n_reels=3, w=72, h=128, fps=10)
    feed = Feed(reels)
    first = feed.current.reel_id
    feed.scroll()
    assert feed.current.reel_id != first


def test_interest_scrolls_when_bored():
    class FakeBrain:
        novelty_mbon = np.asarray([0])
        reverse_dn = np.asarray([0])
        novelty_dan = np.asarray([0])
        kenyon = np.asarray([0])
        counts = np.zeros(1, dtype=np.int32)
        _hz = 80.0

        def mean_rate(self, indices, seconds):
            return self._hz

    dec = InterestDecoder(
        scroll_threshold=10.0,
        peak_fraction=0.4,
        refractory_seconds=0.2,
        boredom_dwell_seconds=0.25,
        max_watch_seconds=5.0,
    )
    brain = FakeBrain()
    # Opening spike to set reel peak.
    for _ in range(5):
        brain._hz = 80.0
        state = dec.observe(brain, 0.1)
    # Then collapse novelty so relative boredom can fire.
    brain._hz = 5.0
    state = None
    for _ in range(40):
        state = dec.observe(brain, 0.1)
        if state["scroll"]:
            break
    assert state["scroll"] is True
    assert state["reason"] == "bored"


def test_dishabituation_recovers_weights():
    class FakeBrain:
        plastic_edges = np.asarray([0], dtype=np.int32)
        ptr = np.asarray([0, 1], dtype=np.int64)
        weight = np.asarray([0.3], dtype=np.float32)
        baseline_weight = np.asarray([1.0], dtype=np.float32)
        counts = np.asarray([1], dtype=np.int32)

    rule = HabituationRule()
    brain = FakeBrain()
    info = rule.dishabituate(brain, surprise=0.8)
    assert brain.weight[0] > 0.3
    assert info["recover_frac"] > 0.3


def test_per_reel_budget_limits_drop():
    class FakeBrain:
        plastic_edges = np.asarray([0], dtype=np.int32)
        ptr = np.asarray([0, 1], dtype=np.int64)
        weight = np.asarray([1.0], dtype=np.float32)
        baseline_weight = np.asarray([1.0], dtype=np.float32)
        counts = np.asarray([200], dtype=np.int32)

    rule = HabituationRule(eta=0.5, per_reel_drop_budget=0.2, min_fraction=0.1, homeostasis_tau_seconds=1e9)
    brain = FakeBrain()
    rule.begin_reel(brain)
    for _ in range(30):
        rule.step(brain, 0.05, learning=True)
    # Cannot fall more than ~20% of (1.0 → 0.1) = 0.18 below start → floor at 0.82
    assert brain.weight[0] >= 0.80


def test_spatial_edge_colors():
    from flyscroll.anatomy import build_spatial_edges, rainbow_rgb
    import numpy as np

    assert rainbow_rgb(0.0) != rainbow_rgb(1.0)
    ptr = np.asarray([0, 1, 2], dtype=np.int64)
    post = np.asarray([1, 0], dtype=np.int32)
    cloud = np.asarray([0, 1], dtype=np.int32)
    sc = np.asarray(["ol_intrinsic", "visual_projection"], dtype="U64")
    xyz = np.asarray([[-1, 0, 0], [1, 0, 0]], dtype=np.float32)
    pairs, colors, node_colors = build_spatial_edges(
        ptr, post, cloud, sc, xyz, max_edges=10
    )
    assert pairs.shape == (2, 2)
    assert colors.shape == (2, 3)
    assert node_colors.shape == (2, 3)
    # Left and right nodes should sit in different hue bands.
    assert not np.array_equal(node_colors[0], node_colors[1])

    from flyscroll.anatomy import normalize_cloud
    import numpy as np

    xyz = np.asarray([[0, 0, 0], [100, 50, 25], [np.nan, 0, 0]], dtype=np.float32)
    mask = np.asarray([True, True, False])
    out = normalize_cloud(xyz, mask)
    assert out.shape == (3, 3)
    assert np.isfinite(out[mask]).all()
    assert (out[~mask] == 0).all()


def test_habituation_depresses_weights():
    class FakeBrain:
        plastic_edges = np.asarray([0], dtype=np.int32)
        ptr = np.asarray([0, 1], dtype=np.int64)
        weight = np.asarray([1.0], dtype=np.float32)
        baseline_weight = np.asarray([1.0], dtype=np.float32)
        counts = np.asarray([50], dtype=np.int32)

    rule = HabituationRule(eta=0.05)
    brain = FakeBrain()
    before = float(brain.weight[0])
    info = rule.step(brain, 0.05, learning=True)
    assert brain.weight[0] < before
    assert info["plastic_edges"] == 1


def test_shorts_portrait_crop():
    from PIL import Image
    import numpy as np

    # Mimic WindowStreamer crop/resize path without ScreenCaptureKit.
    rgb = np.zeros((400, 800, 3), dtype=np.uint8)
    rgb[:, :, 1] = 200
    img = Image.fromarray(rgb, mode="RGB")
    target_w, target_h = 360, 640
    iw, ih = img.size
    target_aspect = target_w / target_h
    src_aspect = iw / ih
    if src_aspect > target_aspect:
        new_w = int(ih * target_aspect)
        left = (iw - new_w) // 2
        img = img.crop((left, 0, left + new_w, ih))
    img = img.resize((target_w, target_h), Image.Resampling.BILINEAR)
    out = np.asarray(img, dtype=np.uint8)
    assert out.shape == (640, 360, 3)


def test_platform_specs_extract_post_ids():
    import re

    from flyscroll.shorts import PLATFORMS, ShortVideoFeed, YouTubeShortsFeed

    assert set(PLATFORMS) == {"youtube", "tiktok", "instagram"}
    samples = {
        "youtube": ("/shorts/InLpslBTylU", "InLpslBTylU"),
        "tiktok": ("/@someone/video/7312345678901234567", "7312345678901234567"),
        "instagram": ("/reel/CxYz_12-Ab/", "CxYz_12-Ab"),
    }
    for key, (path, expected) in samples.items():
        spec = PLATFORMS[key]
        assert spec.key == key
        assert spec.url.startswith("https://")
        assert re.search(spec.id_pattern, path).group(1) == expected
        assert spec.href_substr in path or key == "instagram"
    # The feed root is not a post.
    assert re.search(PLATFORMS["instagram"].id_pattern, "/reels/") is None
    assert re.search(PLATFORMS["youtube"].id_pattern, "/shorts") is None
    assert issubclass(YouTubeShortsFeed, ShortVideoFeed)


def test_short_video_feed_rejects_unknown_platform_before_launching_a_browser():
    import pytest

    from flyscroll.shorts import ShortVideoFeed

    with pytest.raises(ValueError, match="Unknown platform"):
        ShortVideoFeed(platform="myspace")


def test_center_crop_resize_matches_the_shorts_portrait_crop():
    from PIL import Image

    from flyscroll.imaging import center_crop_resize

    rgb = np.zeros((400, 800, 3), dtype=np.uint8)
    rgb[:, :, 1] = 200
    out = center_crop_resize(Image.fromarray(rgb, mode="RGB"), 360, 640)
    assert out.shape == (640, 360, 3)
    assert int(out[:, :, 1].min()) > 150


# --- Bumble mode ---------------------------------------------------------------
# No phone is touched: every adb primitive takes an injectable runner, so a fake
# that records argv and replays scripted results covers the whole decision path.


class _FakeAdb:
    """Records adb argv and answers with scripted results."""

    def __init__(self, screens=None, wm="Physical size: 1080x2400"):
        import io as _io
        from PIL import Image

        self.calls: list[list[str]] = []
        self.wm = wm
        self._screens = list(screens or [])
        self._shot = 0
        self._io = _io
        self._Image = Image

    def _png(self, value: int) -> bytes:
        arr = np.full((240, 108, 3), value, dtype=np.uint8)
        buf = self._io.BytesIO()
        self._Image.fromarray(arr).save(buf, format="PNG")
        return buf.getvalue()

    def __call__(self, args, **kwargs):
        from types import SimpleNamespace

        self.calls.append(list(args))
        joined = " ".join(args)
        if "get-state" in joined:
            return SimpleNamespace(returncode=0, stdout=b"device\n", stderr=b"")
        if "dumpsys power" in joined:
            return SimpleNamespace(
                returncode=0, stdout=b"  mWakefulness=Awake\n", stderr=b""
            )
        if "wm size" in joined:
            return SimpleNamespace(returncode=0, stdout=self.wm.encode(), stderr=b"")
        if "screencap" in joined:
            if self._screens:
                value = self._screens[min(self._shot, len(self._screens) - 1)]
            else:
                value = 10 * self._shot
            self._shot += 1
            return SimpleNamespace(returncode=0, stdout=self._png(value), stderr=b"")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    def swipes(self) -> list[list[str]]:
        return [c for c in self.calls if "swipe" in c]


def _bumble_feed(runner, tmp_path=None, **kwargs):
    from flyscroll.bumble import BumbleFeed

    options = dict(
        fps=0.0,  # capture on demand: no background thread during tests
        runner=runner,
        sleeper=lambda _s: None,
        decisions_csv=(tmp_path / "decisions.csv") if tmp_path is not None else "",
    )
    options.update(kwargs)
    return BumbleFeed(**options)


def test_wm_size_prefers_override_and_derives_gestures():
    from flyscroll.bumble import derive_gestures, wm_size

    runner = _FakeAdb(wm="Physical size: 1440x3200\nOverride size: 1080x2400\n")
    assert wm_size(runner=runner) == (1080, 2400)
    gestures = derive_gestures(1080, 2400)
    assert gestures["left"] == (918, 1200, 162, 1200, 250)
    assert gestures["right"] == (162, 1200, 918, 1200, 250)
    assert gestures["up"] == (540, 1680, 540, 720, 300)


def test_mean_interest_is_time_weighted():
    runner = _FakeAdb()
    feed = _bumble_feed(runner)
    feed.on_interest(10.0, 0.1)
    feed.on_interest(100.0, 2.0)
    assert feed.dwell_seconds == 2.1
    assert abs(feed.mean_interest - 201.0 / 2.1) < 1e-3
    feed.close()


def test_rolling_right_rate_falls_back_then_uses_percentile():
    from flyscroll.bumble import RollingRightRate

    rule = RollingRightRate(target=0.25, window=40, min_history=8, fallback=20.0)
    rule.observe(1.0)
    assert rule.mode == "fallback"
    assert rule.threshold() == 20.0
    for value in range(1, 9):
        rule.observe(float(value * 10))
    assert rule.mode == "rolling"
    # 75th percentile of [1, 10, 20, ... 80], so only the top quarter clears it.
    assert rule.threshold() == 60.0


def test_frame_changed_ignores_identity_and_catches_a_new_screen():
    from flyscroll.bumble import frame_changed

    a = np.zeros((240, 108, 3), dtype=np.uint8)
    b = a.copy()
    b[:, :54] = 255
    assert frame_changed(a, a) is False
    assert frame_changed(a, b) is True


def test_dry_run_scroll_decides_without_touching_the_phone(tmp_path):
    import csv

    runner = _FakeAdb()
    feed = _bumble_feed(runner, tmp_path, dry_run=True, max_profile_scrolls=0)
    first = feed.current.reel_id
    feed.on_interest(90.0, 1.0)
    nxt = feed.scroll()
    assert runner.swipes() == []
    assert nxt.reel_id != first
    assert feed.last_decision_label.endswith("(dry run)")
    rows = list(csv.reader((tmp_path / "decisions.csv").open()))
    assert rows[0][0] == "timestamp" and rows[0][-1] == "confirmed"
    assert rows[1][2] == "like" and rows[1][-2] == "True"
    feed.close()


def test_profile_depth_scrolls_up_before_deciding(tmp_path):
    # Every screencap differs, so each up-swipe reads as a turned page.
    runner = _FakeAdb()
    feed = _bumble_feed(runner, tmp_path, max_profile_scrolls=2)
    first = feed.current.reel_id
    same = feed.scroll()
    assert same.reel_id == first
    assert feed.last_advance == "profile_scroll"
    assert runner.swipes()[0][-5:] == ["540", "1680", "540", "720", "300"]
    feed.scroll()
    assert feed.profile_scrolls == 2
    # Depth exhausted: the next call commits to a horizontal decision swipe.
    nxt = feed.scroll()
    assert nxt.reel_id != first
    assert feed.last_advance in ("like", "pass")
    assert feed.swipes == 1
    feed.close()


def test_unchanged_screen_after_swipe_is_reported_and_not_counted_as_a_profile(tmp_path):
    import csv

    # Constant screen: swipes are delivered but nothing moves.
    runner = _FakeAdb(screens=[40])
    feed = _bumble_feed(runner, tmp_path, max_profile_scrolls=0)
    first = feed.current.reel_id
    same = feed.scroll()
    assert same.reel_id == first
    assert "unchanged" in feed.last_scroll_error
    assert feed.last_advance == "failed"
    assert len(runner.swipes()) == 1
    rows = list(csv.reader((tmp_path / "decisions.csv").open()))
    assert rows[1][-1] == "False"
    feed.close()


def test_max_swipes_stops_after_the_budget():
    runner = _FakeAdb()
    feed = _bumble_feed(runner, max_swipes=1, max_profile_scrolls=0)
    feed.scroll()
    feed.scroll()
    assert len(runner.swipes()) == 1
    assert "max-swipes" in feed.last_scroll_error
    assert feed.last_advance == "failed"
    feed.close()


def test_startup_checks_fail_before_any_other_adb_call():
    from types import SimpleNamespace

    import pytest

    from flyscroll.bumble import ensure_awake, ensure_device

    class _Offline(_FakeAdb):
        def __call__(self, args, **kwargs):
            self.calls.append(list(args))
            return SimpleNamespace(returncode=1, stdout=b"", stderr=b"device offline")

    offline = _Offline()
    with pytest.raises(RuntimeError, match="No authorized adb device"):
        _bumble_feed(offline)
    assert len(offline.calls) == 1

    with pytest.raises(RuntimeError, match="No authorized adb device"):
        ensure_device(runner=offline)

    class _Asleep(_FakeAdb):
        def __call__(self, args, **kwargs):
            if "dumpsys" in " ".join(args):
                self.calls.append(list(args))
                return SimpleNamespace(
                    returncode=0, stdout=b"  mWakefulness=Asleep\n", stderr=b""
                )
            return super().__call__(args, **kwargs)

    with pytest.raises(RuntimeError, match="asleep or locked"):
        ensure_awake(runner=_Asleep())


def test_continue_reel_resets_boredom_but_keeps_watch_seconds():
    dec = InterestDecoder()
    dec.watch_seconds = 3.4
    dec.low_novelty_seconds = 1.2
    dec.low_novelty_samples = 9
    dec.reel_peak_phasic = 55.0
    dec.scrolls = 2
    dec.continue_reel()
    assert dec.watch_seconds == 3.4
    assert dec.scrolls == 2
    assert dec.low_novelty_seconds == 0.0
    assert dec.low_novelty_samples == 0
    assert dec.reel_peak_phasic == 0.0


def test_analytics_stores_an_optional_decision_label():
    analytics = WatchAnalytics(max_watch_seconds=5.0)
    analytics.start_reel("bumble-0001", "Bumble #1")
    entry = analytics.on_scroll(
        reason="bored", watch_seconds=4.0, tick=3, label="❤ like"
    )
    assert entry["label"] == "❤ like"
    assert analytics.snapshot()["scroll_events"][0]["label"] == "❤ like"
    analytics.start_reel("r2", "plain reel")
    assert analytics.on_scroll(reason="bored", watch_seconds=1.0, tick=4)["label"] is None


def test_privacy_filter_returns_a_censored_copy_with_stats():
    rng = np.random.default_rng(3)
    frame = rng.integers(0, 256, (640, 360, 3), dtype=np.uint8)
    before = frame.copy()
    out, info = PrivacyFilter().apply(frame)
    assert out.shape == frame.shape and out.dtype == frame.dtype
    assert out is not frame
    np.testing.assert_array_equal(frame, before)  # the fly still sees the raw frame
    assert set(info) >= {"faces", "text_regions", "enabled"}


def test_privacy_filter_blurs_rendered_text():
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (360, 640), "white")
    draw = ImageDraw.Draw(img)
    # A wide ~22px-tall row of high-contrast strokes on white = a line of text.
    # The band starts MOSTLY WHITE, so a passing assertion means the filter
    # itself painted a censor bar (the earlier version pre-filled it black).
    for x in range(24, 330, 12):
        draw.rectangle([x, 300, x + 6, 322], fill="black")
    frame = np.asarray(img, dtype=np.uint8)
    band = (slice(298, 326), slice(20, 336))
    in_mean = float(frame[band].mean())

    out, info = PrivacyFilter(blur_faces=False).apply(frame)
    assert info["text_regions"] >= 1
    assert in_mean > 120  # sanity: the input band really is mostly white

    # The censor now pixel-blurs (not a black bar), so brightness is roughly
    # preserved but the sharp text edges are gone.
    import cv2

    def edge_energy(a):
        gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        return float(np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)).mean())

    assert not np.array_equal(out[band], frame[band])  # the region was modified
    assert edge_energy(out[band]) < 0.6 * edge_energy(frame[band])  # text edges blurred away


def test_privacy_filter_disabled_is_a_passthrough():
    frame = np.zeros((640, 360, 3), dtype=np.uint8)
    frame[10:20, 10:20] = 255
    out, info = PrivacyFilter(enabled=False).apply(frame)
    np.testing.assert_array_equal(out, frame)
    assert info["enabled"] is False


def test_bumble_decision_reads_the_recent_spike_window_not_the_profile_mean():
    # A long, boring stretch, then a fresh spike in the last <0.5 s. The profile
    # mean stays low, but the 500 ms window is high -- the swipe follows the spike.
    runner = _FakeAdb()
    feed = _bumble_feed(
        runner,
        max_profile_scrolls=0,   # skip paging; go straight to the decision
        target_right_rate=0.0,   # fixed-threshold rule
        like_threshold=50.0,
        decision_window_seconds=0.5,
        dry_run=True,            # decide + log, never swipe the (fake) phone
    )
    for _ in range(100):
        feed.on_interest(10.0, 0.05)   # 5 s of low interest
    for _ in range(6):
        feed.on_interest(100.0, 0.05)  # 0.3 s spike, inside the window

    assert feed.mean_interest < 50.0            # whole-profile mean is low
    assert feed.window_interest >= 50.0         # recent spike is high
    assert feed.window_interest > feed.mean_interest

    feed.scroll()
    assert feed.last_advance == "like"          # decided on the spike, not the mean
