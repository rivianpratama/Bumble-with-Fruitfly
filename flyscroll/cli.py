"""CLI for FlyScroll: import, prepare, serve, smoke-demo."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="flyscroll",
        description="Fruit-fly connectome doomscroller for short-form video",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_import = sub.add_parser("import", help="Import MaleCNS v1.0 into normalized graph")
    p_import.add_argument("dataset", nargs="?", default="malecns_v1")

    p_prep = sub.add_parser("prepare", help="Build CSR graph + retinal projection")
    p_prep.add_argument("dataset", nargs="?", default="malecns_v1")
    p_prep.add_argument("--scale", choices=["full", "visual"], default="full")

    p_serve = sub.add_parser("serve", help="Run live doomscroll spectator")
    p_serve.add_argument("--port", type=int, default=8767)
    p_serve.add_argument("--scale", choices=["full", "visual"], default="visual")
    p_serve.add_argument("--reels", type=str, default="")
    p_serve.add_argument("--n-reels", type=int, default=8)
    p_serve.add_argument("--seed", type=int, default=0)
    p_serve.add_argument("--frame-ms", type=float, default=50.0)
    p_serve.add_argument("--threshold", type=float, default=14.0)
    p_serve.add_argument("--min-watch", type=float, default=0.35, help="Scroll refractory seconds")
    p_serve.add_argument(
        "--max-watch",
        type=float,
        default=5.0,
        help="Local-feed safety timeout; Shorts use actual video completion",
    )
    p_serve.add_argument(
        "--boredom-dwell",
        type=float,
        default=1.6,
        help="Seconds phasic novelty must stay low before bored-scroll",
    )
    p_serve.add_argument(
        "--peak-fraction",
        type=float,
        default=0.40,
        help="Bored when phasic < this fraction of reel peak novelty",
    )
    p_serve.add_argument("--no-learning", action="store_true")
    p_serve.add_argument(
        "--privacy",
        action="store_true",
        help="Blur faces and text on the shown and logged frames (off by default)",
    )
    p_serve.add_argument(
        "--shorts",
        action="store_true",
        help="Doomscroll a live short-video platform (Playwright; no login). See --platform.",
    )
    p_serve.add_argument(
        "--platform",
        choices=["youtube", "tiktok", "instagram"],
        default="youtube",
        help="Which public feed to browse with --shorts (default: youtube)",
    )
    p_serve.add_argument(
        "--shorts-url",
        type=str,
        default="",
        help="Override the platform's start URL (with --shorts)",
    )
    p_serve.add_argument(
        "--shorts-headless",
        action="store_true",
        help="Hide Chromium window when using --shorts",
    )
    p_serve.add_argument(
        "--bumble",
        action="store_true",
        help="Let the fly swipe a real Android phone over adb (real swipes by default)",
    )
    p_serve.add_argument("--adb", type=str, default="adb", help="adb executable")
    p_serve.add_argument("--serial", type=str, default="", help="adb device serial")
    p_serve.add_argument("--bumble-fps", type=float, default=3.0)
    p_serve.add_argument(
        "--bumble-target-right-rate",
        type=float,
        default=0.25,
        help="Rolling fraction of profiles to like; 0 = fixed --bumble-like-threshold",
    )
    p_serve.add_argument(
        "--bumble-like-threshold",
        type=float,
        default=20.0,
        help="Mean-interest Hz to like: fixed rule, and the warm-up fallback",
    )
    p_serve.add_argument(
        "--bumble-dry-run",
        action="store_true",
        help="Decide and log but never send a swipe to the phone",
    )
    p_serve.add_argument("--bumble-max-swipes", type=int, default=0)
    p_serve.add_argument("--bumble-max-profile-scrolls", type=int, default=3)
    p_serve.add_argument(
        "--bumble-247",
        action="store_true",
        help="Human-like pacing plus a 2-3 min pause once an hour",
    )
    p_serve.add_argument("--bumble-left-swipe", type=str, default="")
    p_serve.add_argument("--bumble-right-swipe", type=str, default="")
    p_serve.add_argument("--bumble-up-swipe", type=str, default="")
    p_serve.add_argument("--bumble-max-watch", type=float, default=5.0)
    p_serve.add_argument("--bumble-min-watch", type=float, default=0.5)
    p_serve.add_argument(
        "--bumble-decision-window",
        type=float,
        default=0.5,
        help="Seconds of recent interest the swipe decision reads (the spike window)",
    )

    p_demo = sub.add_parser("demo", help="Headless few-step smoke run")
    p_demo.add_argument("--scale", choices=["full", "visual"], default="visual")
    p_demo.add_argument("--steps", type=int, default=40)
    p_demo.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(argv)

    if args.cmd == "import":
        from flyscroll.connectome import import_graph

        report = import_graph(args.dataset)
        print(json.dumps({"retained": report["retained_neuron_candidates"], "graph": report["graph"]}, indent=2))
        return 0

    if args.cmd == "prepare":
        from flyscroll.prepare import prepare

        manifest = prepare(args.dataset, scale=args.scale)
        print(json.dumps({k: manifest[k] for k in ("neurons", "edges", "retina_mapped", "scale", "plastic_kc_to_novelty_mbon")}, indent=2))
        return 0

    if args.cmd == "serve":
        from flyscroll.server import main as serve_main

        serve_argv = [
            "--port",
            str(args.port),
            "--scale",
            args.scale,
            "--n-reels",
            str(args.n_reels),
            "--seed",
            str(args.seed),
            "--frame-ms",
            str(args.frame_ms),
            "--threshold",
            str(args.threshold),
            "--min-watch",
            str(args.min_watch),
            "--max-watch",
            str(args.max_watch),
            "--boredom-dwell",
            str(args.boredom_dwell),
            "--peak-fraction",
            str(args.peak_fraction),
        ]
        if args.reels:
            serve_argv.extend(["--reels", args.reels])
        if args.no_learning:
            serve_argv.append("--no-learning")
        if args.privacy:
            serve_argv.append("--privacy")
        if args.shorts:
            serve_argv.append("--shorts")
            serve_argv.extend(["--platform", args.platform])
            if args.shorts_url:
                serve_argv.extend(["--shorts-url", args.shorts_url])
            if args.shorts_headless:
                serve_argv.append("--shorts-headless")
        if args.bumble:
            serve_argv.append("--bumble")
            serve_argv.extend(
                [
                    "--adb",
                    args.adb,
                    "--serial",
                    args.serial,
                    "--bumble-fps",
                    str(args.bumble_fps),
                    "--bumble-target-right-rate",
                    str(args.bumble_target_right_rate),
                    "--bumble-like-threshold",
                    str(args.bumble_like_threshold),
                    "--bumble-max-swipes",
                    str(args.bumble_max_swipes),
                    "--bumble-max-profile-scrolls",
                    str(args.bumble_max_profile_scrolls),
                    "--bumble-max-watch",
                    str(args.bumble_max_watch),
                    "--bumble-min-watch",
                    str(args.bumble_min_watch),
                    "--bumble-decision-window",
                    str(args.bumble_decision_window),
                ]
            )
            if args.bumble_dry_run:
                serve_argv.append("--bumble-dry-run")
            if args.bumble_247:
                serve_argv.append("--bumble-247")
            for flag, value in (
                ("--bumble-left-swipe", args.bumble_left_swipe),
                ("--bumble-right-swipe", args.bumble_right_swipe),
                ("--bumble-up-swipe", args.bumble_up_swipe),
            ):
                if value:
                    serve_argv.extend([flag, value])
        serve_main(serve_argv)
        return 0

    if args.cmd == "demo":
        from flyscroll.feed import Feed, load_feed
        from flyscroll.session import DoomscrollSession, SessionConfig

        feed = Feed(load_feed(seed=args.seed, n_reels=4))
        session = DoomscrollSession(feed, SessionConfig(scale=args.scale, seed=args.seed))
        scrolls = 0
        for _ in range(args.steps):
            state = session.step()
            scrolls += int(state["scrolled"])
        summary = {
            "ticks": args.steps,
            "scrolls": scrolls,
            "final_interest": state["interest"]["interest"],
            "reel": state["reel"]["title"],
            "neural_ms": state["neural_ms"],
            "neurons": session.brain.n,
            "scale": args.scale,
        }
        print(json.dumps(summary, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
