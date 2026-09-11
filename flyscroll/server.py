"""Read-only spectator HTTP server for the doomscroll loop.

Binds 127.0.0.1 only. GET /state and /health. No remote control surface.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from flyscroll.feed import Feed, load_feed
from flyscroll.session import DoomscrollSession, SessionConfig

ROOT = Path(__file__).resolve().parents[1]
UI = Path(__file__).with_name("ui")

# ES modules need a JavaScript MIME type; Windows registries sometimes map .js to
# text/plain, so spell the common types out before consulting mimetypes.
_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".mjs": "application/javascript; charset=utf-8",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".txt": "text/plain; charset=utf-8",
}

latest: dict = {"status": "starting"}
stop = threading.Event()
session: DoomscrollSession | None = None
feed_handle = None


def run_loop(args):
    global latest, session, feed_handle
    if getattr(args, "bumble", False):
        from flyscroll.bumble import BumbleFeed

        feed_handle = BumbleFeed(
            adb=args.adb,
            serial=args.serial,
            fps=args.bumble_fps,
            target_right_rate=args.bumble_target_right_rate,
            like_threshold=args.bumble_like_threshold,
            dry_run=args.bumble_dry_run,
            max_swipes=args.bumble_max_swipes or None,
            max_profile_scrolls=args.bumble_max_profile_scrolls,
            profile_max_watch=args.bumble_max_watch,
            decision_window_seconds=args.bumble_decision_window,
            left_swipe=args.bumble_left_swipe,
            right_swipe=args.bumble_right_swipe,
            up_swipe=args.bumble_up_swipe,
            mode_247=args.bumble_247,
        )
        feed = feed_handle
        n_reels = "live-bumble"
    elif getattr(args, "shorts", False):
        from flyscroll.shorts import ShortVideoFeed

        platform = getattr(args, "platform", "youtube")
        feed_handle = ShortVideoFeed(
            url=args.shorts_url,
            platform=platform,
            headless=args.shorts_headless,
        )
        feed = feed_handle
        n_reels = f"live-{platform}"
    else:
        feed = Feed(
            load_feed(
                Path(args.reels) if args.reels else None,
                seed=args.seed,
                n_reels=args.n_reels,
            )
        )
        feed_handle = feed
        n_reels = len(feed.reels)
    # Bumble profiles are several photos deep, so they get their own watch
    # bounds; the shared --max-watch/--min-watch defaults stay untouched.
    min_watch = args.min_watch
    max_watch = args.max_watch
    if getattr(args, "bumble", False):
        min_watch = args.bumble_min_watch
        max_watch = args.bumble_max_watch
    # A static dating photo goes "low novelty" fast; a short boredom dwell in
    # bumble mode lets the fly commit within about a second, matching the 500 ms
    # decision window.
    boredom = getattr(args, "boredom_dwell", 1.6)
    if getattr(args, "bumble", False):
        boredom = min(boredom, 0.6)
    config = SessionConfig(
        scale=args.scale,
        frame_ms=args.frame_ms,
        learning=not args.no_learning,
        interest_threshold=args.threshold,
        min_watch_seconds=min_watch,
        max_watch_seconds=max_watch,
        seed=args.seed,
        shorts=bool(getattr(args, "shorts", False)),
        privacy=getattr(args, "privacy", False),
        boredom_dwell_seconds=boredom,
        peak_fraction=getattr(args, "peak_fraction", 0.40),
    )
    session = DoomscrollSession(feed, config)
    print(
        json.dumps(
            {
                "status": "running",
                "run_id": session.run_id,
                "port": args.port,
                "scale": args.scale,
                "reels": n_reels,
                "shorts": bool(getattr(args, "shorts", False)),
                "platform": getattr(feed_handle, "platform_key", "local"),
                "dry_run": bool(getattr(feed_handle, "dry_run", False)),
            }
        ),
        flush=True,
    )
    try:
        while not stop.is_set():
            began = time.monotonic()
            state = session.step()
            latest = {k: v for k, v in state.items()}
            target = args.frame_ms / 1000.0
            elapsed = time.monotonic() - began
            if elapsed < target:
                time.sleep(target - elapsed)
    finally:
        closer = getattr(feed_handle, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def _send(self, code: int, body: bytes, content_type: str):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/health":
            body = json.dumps({"ok": True, "status": latest.get("status")}).encode()
            return self._send(200, body, "application/json")
        if path == "/state":
            body = json.dumps(latest).encode()
            return self._send(200, body, "application/json")
        if path == "/anatomy":
            if session is None:
                return self._send(503, b'{"status":"starting"}', "application/json")
            body = json.dumps(session.anatomy).encode()
            return self._send(200, body, "application/json")
        if path in ("/", "/index.html"):
            html = (UI / "index.html").read_bytes()
            return self._send(200, html, "text/html; charset=utf-8")
        return self._static(path)

    def _static(self, path: str):
        """Serve a file from the UI directory (scripts, styles, vendored Three.js).

        Anything that would escape the UI directory is a 404, never a read.
        """
        rel = path.lstrip("/")
        if not rel or ".." in rel.split("/") or "\\" in rel:
            return self._send(404, b"not found", "text/plain")
        root = UI.resolve()
        target = (root / rel).resolve()
        if root not in target.parents or not target.is_file():
            return self._send(404, b"not found", "text/plain")
        ctype = (
            _CONTENT_TYPES.get(target.suffix.lower())
            or mimetypes.guess_type(target.name)[0]
            or "application/octet-stream"
        )
        return self._send(200, target.read_bytes(), ctype)


def _init_macos_graphics() -> None:
    """Initialize AppKit/CGS on the main thread before ScreenCaptureKit touches CG."""
    import sys

    if sys.platform != "darwin":
        return
    from AppKit import NSApplication

    # Required before any CGS/ScreenCaptureKit call or macOS aborts with
    # CGS_REQUIRE_INIT (did_initialize) from a background thread.
    NSApplication.sharedApplication()


def _serve_http(port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Spectator at http://127.0.0.1:{port}/", flush=True)
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--scale", choices=["full", "visual"], default="visual")
    parser.add_argument("--reels", type=str, default="", help="Directory of local videos")
    parser.add_argument("--n-reels", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-ms", type=float, default=50.0)
    parser.add_argument("--threshold", type=float, default=14.0)
    parser.add_argument("--min-watch", type=float, default=0.35, help="Scroll refractory seconds")
    parser.add_argument(
        "--max-watch",
        type=float,
        default=5.0,
        help="Local-feed safety timeout; Shorts use actual video completion",
    )
    parser.add_argument(
        "--boredom-dwell",
        type=float,
        default=1.6,
        help="Seconds phasic novelty must stay low before a bored scroll",
    )
    parser.add_argument(
        "--peak-fraction",
        type=float,
        default=0.40,
        help="Bored when phasic < this fraction of the reel's peak novelty",
    )
    parser.add_argument("--no-learning", action="store_true")
    parser.add_argument(
        "--privacy",
        action="store_true",
        help="Blur faces and text on the shown and logged frames (off by default)",
    )
    parser.add_argument(
        "--shorts",
        action="store_true",
        help="Watch live YouTube Shorts via Playwright (no login)",
    )
    parser.add_argument(
        "--platform",
        choices=["youtube", "tiktok", "instagram"],
        default="youtube",
        help="Public short-video feed to browse with --shorts",
    )
    parser.add_argument(
        "--shorts-url",
        type=str,
        default="",
        help="Override the platform's start URL",
    )
    parser.add_argument(
        "--shorts-headless",
        action="store_true",
        help="Hide the Chromium window (spectator still shows captures)",
    )
    parser.add_argument(
        "--bumble",
        action="store_true",
        help="Let the fly swipe a real Android phone over adb (see README)",
    )
    parser.add_argument("--adb", type=str, default="adb", help="adb executable")
    parser.add_argument("--serial", type=str, default="", help="adb device serial")
    parser.add_argument(
        "--bumble-fps", type=float, default=3.0, help="Screencaps per second"
    )
    parser.add_argument(
        "--bumble-target-right-rate",
        type=float,
        default=0.25,
        help="Rolling fraction of profiles to like; 0 = fixed --bumble-like-threshold",
    )
    parser.add_argument(
        "--bumble-like-threshold",
        type=float,
        default=20.0,
        help="Mean-interest Hz to like: fixed rule, and the warm-up fallback",
    )
    parser.add_argument(
        "--bumble-dry-run",
        action="store_true",
        help="Decide and log but never send a swipe to the phone",
    )
    parser.add_argument(
        "--bumble-max-swipes", type=int, default=0, help="Stop swiping after N (0 = no limit)"
    )
    parser.add_argument(
        "--bumble-max-profile-scrolls",
        type=int,
        default=3,
        help="Up-swipes through a profile's photos before deciding",
    )
    parser.add_argument(
        "--bumble-247",
        action="store_true",
        help="Human-like pacing plus a 2-3 min pause once an hour (blocks the loop)",
    )
    parser.add_argument(
        "--bumble-left-swipe", type=str, default="", help='Override "x1,y1,x2,y2,ms"'
    )
    parser.add_argument(
        "--bumble-right-swipe", type=str, default="", help='Override "x1,y1,x2,y2,ms"'
    )
    parser.add_argument(
        "--bumble-up-swipe", type=str, default="", help='Override "x1,y1,x2,y2,ms"'
    )
    parser.add_argument(
        "--bumble-max-watch",
        type=float,
        default=5.0,
        help="Seconds the fly may spend on one profile (bumble mode only)",
    )
    parser.add_argument(
        "--bumble-min-watch",
        type=float,
        default=0.5,
        help="Swipe refractory seconds (bumble mode only)",
    )
    parser.add_argument(
        "--bumble-decision-window",
        type=float,
        default=0.5,
        help="Seconds of recent interest the swipe decision reads (the spike window)",
    )
    args = parser.parse_args(argv)
    if not args.reels:
        args.reels = ""
    if args.shorts and args.bumble:
        parser.error("--shorts and --bumble are mutually exclusive")

    def handle_sig(*_):
        stop.set()

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    # ScreenCaptureKit + Playwright need the macOS graphics stack initialized
    # on the main thread; run the doomscroll loop there and HTTP in the background.
    if args.shorts:
        _init_macos_graphics()
        server = _serve_http(args.port)

        def http_loop():
            try:
                while not stop.is_set():
                    server.handle_request()
            finally:
                server.server_close()

        http_thread = threading.Thread(target=http_loop, name="flyscroll-http", daemon=True)
        http_thread.start()
        try:
            run_loop(args)
        finally:
            stop.set()
            http_thread.join(timeout=2.0)
        return

    thread = threading.Thread(target=run_loop, args=(args,), daemon=True)
    thread.start()
    server = _serve_http(args.port)
    try:
        while not stop.is_set():
            server.handle_request()
    finally:
        stop.set()
        thread.join(timeout=8.0)
        server.server_close()


if __name__ == "__main__":
    main()
