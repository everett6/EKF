"""Command-line entry point: python -m rocket_tracker [options]."""

import argparse
import logging
import sys
import time

import numpy as np

from .config import load_config
from .telemetry import SimulatedFlight, CsvSource
from .tracker import Tracker
from .weather import WeatherService


def build_parser():
    p = argparse.ArgumentParser(
        prog="rocket_tracker",
        description="Track a rocket in flight and keep predicting where it will land. "
                    "Run 'python -m rocket_tracker app' for the full app.")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--sim", action="store_true", help="simulated flight (default)")
    src.add_argument("--replay", metavar="CSV", help="replay a logged flight")
    src.add_argument("--follow", metavar="CSV", help="track live by tailing a CSV a logger is writing")
    p.add_argument("--config", help="JSON file overriding rocket/site/sensor/weather settings")
    p.add_argument("--offline", action="store_true",
                   help="don't fetch live weather; use standard atmosphere + configured surface wind")
    p.add_argument("--weather-refresh", type=float, metavar="S", help="seconds between weather refreshes")
    p.add_argument("--speed", type=float, default=1.0,
                   help="playback speed for --sim/--replay (0 = as fast as possible)")
    p.add_argument("--no-web", action="store_true", help="don't start the dashboard")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--accel-units", choices=["g", "mps2"], default="g")
    p.add_argument("--flip-lon", action="store_true",
                   help="negate logged longitudes (for logs written before the W-sign fix)")
    p.add_argument("--plot", metavar="PNG", help="save a summary plot when the flight ends")
    p.add_argument("--exit", action="store_true", help="exit when the flight ends instead of serving on")
    p.add_argument("-q", "--quiet", action="store_true")
    return p


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "app":
        from .app import main as app_main
        return app_main(argv[1:])
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(message)s")
    cfg = load_config(args.config)
    if args.offline:
        cfg.weather.provider = "standard"
    if args.weather_refresh:
        cfg.weather.refresh_s = args.weather_refresh

    weather = WeatherService(cfg.site, cfg.weather)
    weather.start()
    print(f"Weather: {weather.profile.source}"
          + (f" (live fetch failed: {weather.last_error})" if weather.last_error else ""))
    tracker = Tracker(cfg, weather)

    if args.replay:
        source, paced = CsvSource(args.replay, accel_units=args.accel_units, flip_lon=args.flip_lon), True
    elif args.follow:
        source, paced = CsvSource(args.follow, follow=True, accel_units=args.accel_units,
                                  flip_lon=args.flip_lon), False
    else:
        source, paced = SimulatedFlight(cfg, weather, launch_time=time.time() + 3.0), True
    paced = paced and args.speed > 0

    if not args.no_web:
        from .dashboard import serve
        serve(tracker, args.host, args.port)
        print(f"Dashboard: http://{args.host}:{args.port}")

    wall0 = data0 = None
    last_print = 0.0
    try:
        for m in source:
            if paced:
                if wall0 is None:
                    wall0, data0 = time.time(), m.t
                wait = (m.t - data0) / args.speed - (time.time() - wall0)
                if wait > 0:
                    time.sleep(wait)
            tracker.process(m)
            now = time.time()
            if now - last_print >= 1.0:
                last_print = now
                print(tracker.summary_line())
        print(tracker.summary_line())
        report(tracker, source)
        if args.plot:
            plot(tracker, source, args.plot)
        if not args.no_web and not args.exit:
            print("Flight over; dashboard still running. Ctrl+C to quit.")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        weather.stop()
    return tracker


def report(tracker, source):
    truth = getattr(source, "landing_enu", None)
    if truth is None or not tracker.predictions:
        return
    print("\nLanding prediction error vs simulated truth:")
    for p in tracker.predictions:
        if p.t >= 0 and (p is tracker.predictions[-1] or int(p.t) % 20 == 0):
            err = float(np.linalg.norm(p.landing_enu - truth))
            print(f"  T+{p.t:6.1f}s  error {err:7.1f} m   95% ellipse {p.ellipse[0]:6.0f} x {p.ellipse[1]:.0f} m")


def plot(tracker, source, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tr = np.array([r[:4] for r in tracker.track])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(tr[:, 1], tr[:, 2], label="EKF track")
    lp = np.array([p.landing_enu for p in tracker.predictions])
    ax1.plot(lp[:, 0], lp[:, 1], ".", ms=3, alpha=0.5, label="Predicted landing over time")
    truth = getattr(source, "landing_enu", None)
    if truth is not None:
        ax1.plot(*truth, "k*", ms=12, label="Actual landing")
    ax1.plot(0, 0, "g^", ms=10, label="Pad")
    ax1.set_xlabel("East (m)")
    ax1.set_ylabel("North (m)")
    ax1.axis("equal")
    ax1.legend()
    ax2.plot(tr[:, 0], tr[:, 3], label="EKF altitude")
    if getattr(source, "truth_track", None):
        tt = np.array(source.truth_track)
        ax2.plot(tt[:, 0], tt[:, 3], "--", label="Truth")
    ax2.set_xlabel("Time since launch (s)")
    ax2.set_ylabel("Altitude AGL (m)")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"Saved {path}")
