"""A flight run the app can start, pause, speed up and stop from the UI."""

import copy
import logging
import threading
import time
from pathlib import Path

import numpy as np

from .telemetry import SimulatedFlight, CsvSource
from .tracker import Tracker
from .weather import WeatherService

log = logging.getLogger(__name__)


class FlightSession:
    """Runs one telemetry source through a Tracker on a background thread."""

    def __init__(self, cfg, source="sim", path=None, speed=1.0, offline=False,
                 accel_units="g", flip_lon=False):
        self.cfg = copy.deepcopy(cfg)
        if offline:
            self.cfg.weather.provider = "standard"
        self.kind = source
        self.path = path
        self.speed = float(speed)
        self.accel_units = accel_units
        self.flip_lon = flip_lon
        self.weather = WeatherService(self.cfg.site, self.cfg.weather)
        self.tracker = Tracker(self.cfg, self.weather, async_predictions=True)
        self.state = "starting"   # starting | running | paused | finished | stopped | error
        self.error = None
        self.truth_landing = None
        self._stop = threading.Event()
        self._resume = threading.Event()
        self._resume.set()
        self._anchor = None
        self._thread = threading.Thread(target=self._run, name="flight", daemon=True)
        self.started_at = time.time()

    # --- controls ------------------------------------------------------------------
    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._resume.set()
        if self.state in ("starting", "running", "paused"):
            self.state = "stopped"
        self.weather.stop()
        self.tracker.close()

    def pause(self, paused=True):
        if paused and self.state == "running":
            self._resume.clear()
            self.state = "paused"
        elif not paused and self.state == "paused":
            self._anchor = None
            self._resume.set()
            self.state = "running"

    def set_speed(self, speed):
        self.speed = max(0.0, float(speed))
        self._anchor = None  # re-anchor the clock at the new rate

    # --- run loop ------------------------------------------------------------------
    def _make_source(self):
        if self.kind == "sim":
            return SimulatedFlight(self.cfg, self.weather, launch_time=time.time() + 3.0), True
        if not self.path or not Path(self.path).is_file():
            raise FileNotFoundError(f"No such file: {self.path}")
        follow = self.kind == "follow"
        src = CsvSource(self.path, follow=follow, accel_units=self.accel_units,
                        flip_lon=self.flip_lon, should_stop=self._stop.is_set)
        return src, not follow

    def _run(self):
        try:
            self.weather.start()
            source, paced = self._make_source()
            self.source = source
            self.state = "running"
            for m in source:
                if paced:
                    self._wait_until_due(m.t)
                elif not self._resume.is_set():
                    self._resume.wait()
                if self._stop.is_set():
                    return
                self.tracker.process(m)
            self.truth_landing = getattr(source, "landing_enu", None)
            if self.tracker.ekf is not None:
                self.tracker.run_prediction()
            self.state = "finished"
        except Exception as e:
            log.exception("Flight session failed")
            self.error = f"{type(e).__name__}: {e}"
            self.state = "error"

    def _wait_until_due(self, data_t):
        """Sleep until a sample is due at the current speed, honouring pause/stop."""
        while not self._stop.is_set():
            if not self._resume.is_set():
                self._resume.wait()
                continue
            if self.speed <= 0:
                return
            if self._anchor is None:
                self._anchor = (time.time(), data_t)
            wall0, data0 = self._anchor
            wait = (data_t - data0) / self.speed - (time.time() - wall0)
            if wait <= 0:
                return
            self._stop.wait(min(wait, 0.25))

    def status(self):
        out = {"state": self.state, "source": self.kind, "path": self.path, "speed": self.speed,
               "error": self.error, "started_at": self.started_at}
        if self.truth_landing is not None:
            lat, lon, _ = self.tracker._ll(*self.truth_landing)
            out["truth_landing"] = [lat, lon]
        return out
