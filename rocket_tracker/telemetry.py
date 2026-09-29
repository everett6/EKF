"""Telemetry sources: a simulated flight, and CSV logs (replayed or followed live).

Every source yields Measurement objects in time order. Times are absolute
seconds (Unix time for real logs); the tracker handles launch detection.
"""

import csv
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np

from .atmosphere import G0
from .dynamics import FlightModel, BOOST, LANDED
from .geo import enu_to_geodetic


@dataclass
class Measurement:
    t: float
    kind: str               # "gps" | "baro" | "accel"
    values: tuple           # gps: (lat, lon, alt_msl); baro: (alt_msl,); accel: (ax, ay, az) m/s^2
    truth: dict = field(default=None, repr=False)  # simulator only


class SimulatedFlight:
    """Flies a 'real' rocket that differs from the tracker's assumptions.

    The truth uses the live weather plus a steady wind offset and gusts, and
    scaled drag/thrust/parachute values (SimConfig), then emits noisy GPS,
    barometer and accelerometer readings at the configured rates.
    """

    def __init__(self, cfg, weather_service, launch_time=None, dt=0.02):
        self.cfg = cfg
        self.site = cfg.site
        self.s = cfg.sensors
        self.sim = cfg.sim
        self.rng = np.random.default_rng(cfg.sim.seed)
        self.weather = weather_service
        rocket = cfg.rocket
        self.model = FlightModel(rocket, self.site.altitude_m, lambda: self.weather.profile,
                                 thrust_scale=cfg.sim.thrust_scale, chute_scale=cfg.sim.chute_scale)
        self.t0 = launch_time if launch_time is not None else time.time()
        self.dt = dt
        self.truth_track = []
        self.landing_enu = None

    def __iter__(self):
        X = np.zeros((1, 9))
        X[0, 8] = self.sim.drag_scale
        phase = np.array([BOOST])
        t = 0.0
        gust = np.zeros(2)
        offset = np.array([self.sim.wind_offset_east, self.sim.wind_offset_north])
        periods = {"gps": 1 / self.s.gps_rate_hz, "baro": 1 / self.s.baro_rate_hz,
                   "accel": 1 / self.s.accel_rate_hz}
        next_t = {k: 0.0 for k in periods}
        # A few pad samples before ignition so the tracker sees the rocket at rest.
        for i in range(10):
            yield from self._emit(-1.0 + i * 0.1, X, np.array([0.0, 0.0, G0]))
        while True:
            a = math.exp(-self.dt / self.sim.gust_tau_s)
            gust = a * gust + self.sim.gust_sigma * math.sqrt(1 - a * a) * self.rng.standard_normal(2)
            extra = np.array([[*(offset + gust), 0.0]])
            _, sf = self.model.forces(X, t, phase, wind_extra=extra)
            for kind, per in periods.items():
                if t >= next_t[kind] - 1e-9:
                    next_t[kind] += per
                    yield from self._emit(t, X, sf[0], kinds=(kind,))
            if int(round(t / self.dt)) % max(1, int(round(0.1 / self.dt))) == 0:
                self.truth_track.append((t, *X[0, 0:3]))
            if phase[0] == LANDED:
                self.landing_enu = X[0, 0:2].copy()
                break
            prev = X.copy()
            X = self.model.step(X, t, phase, self.dt, wind_extra=extra)
            t += self.dt
            phase = self.model.next_phase(X, t, phase)
            if phase[0] == LANDED:
                frac = prev[0, 2] / max(prev[0, 2] - X[0, 2], 1e-9)
                X = prev + frac * (X - prev)
                X[0, 2] = 0.0
                X[0, 3:6] = 0.0
        # Keep reporting the landed position briefly, like a recovery beacon.
        for i in range(1, 31):
            yield from self._emit(t + i * 0.1, X, np.array([0.0, 0.0, G0]), kinds=("gps", "baro"))

    def _emit(self, t, X, specific_force, kinds=("gps", "baro", "accel")):
        e, n, u = X[0, 0:3]
        truth = {"enu": (float(e), float(n), float(u))}
        for kind in kinds:
            if kind == "gps":
                ne = e + self.rng.normal(0, self.s.gps_sigma_h_m)
                nn = n + self.rng.normal(0, self.s.gps_sigma_h_m)
                nu = u + self.rng.normal(0, self.s.gps_sigma_v_m)
                lat, lon, alt = enu_to_geodetic(ne, nn, nu, self.site.latitude,
                                                self.site.longitude, self.site.altitude_m)
                yield Measurement(self.t0 + t, "gps", (float(lat), float(lon), float(alt)), truth)
            elif kind == "baro":
                alt = self.site.altitude_m + u + self.rng.normal(0, self.s.baro_sigma_m)
                yield Measurement(self.t0 + t, "baro", (float(alt),), truth)
            elif kind == "accel":
                a = specific_force + self.rng.normal(0, self.s.accel_sigma_mps2, 3)
                yield Measurement(self.t0 + t, "accel", tuple(float(v) for v in a), truth)


# Column names understood by CsvSource; the first match wins. Covers the
# DataLogging*.py formats in this repository and generic names.
_TIME_COLS = ("sys_time", "time", "Time", "timestamp")
_LAT_COLS = ("gps_lat", "lat", "latitude")
_LON_COLS = ("gps_lon", "lon", "longitude")
_ALT_COLS = ("gps_alt", "alt", "altitude")
_BARO_COLS = ("baro_alt", "pressure_alt")
_PRESSURE_COLS = ("pressure_pa", "pressure")
_ACCEL_COLS = (("accel_x", "accel_y", "accel_z"), ("ax", "ay", "az"))


def _pick(row, names):
    for n in names:
        if n in row and row[n] not in ("", None):
            return n
    return None


class CsvSource:
    """Reads telemetry rows from a CSV log.

    follow=True tails a file that a logger (e.g. DataLogging.py) is still
    writing, which gives live tracking without touching the hardware code.
    Accelerations are assumed to be in g (as the MPU9250 logger writes them)
    unless accel_units="mps2".
    """

    def __init__(self, path, follow=False, accel_units="g", flip_lon=False, poll_s=0.05,
                 should_stop=None):
        self.path = path
        self.should_stop = should_stop or (lambda: False)
        self.follow = follow
        self.accel_scale = G0 if accel_units == "g" else 1.0
        self.lon_sign = -1.0 if flip_lon else 1.0
        self.poll_s = poll_s
        self._last_gps = None

    def _rows(self):
        with open(self.path, newline="") as f:
            header = None
            buf = ""
            while True:
                line = f.readline()
                if not line:
                    if not self.follow or self.should_stop():
                        return
                    time.sleep(self.poll_s)
                    continue
                buf += line
                if not buf.endswith("\n") and self.follow:
                    continue  # partial line still being written
                line, buf = buf, ""
                if header is None:
                    header = next(csv.reader([line]))
                    continue
                vals = next(csv.reader([line]), None)
                if vals and len(vals) == len(header):
                    yield dict(zip(header, vals))

    def __iter__(self):
        for row in self._rows():
            tcol = _pick(row, _TIME_COLS)
            if tcol is None:
                continue
            try:
                t = float(row[tcol])
            except ValueError:
                continue
            yield from self._parse(row, t)

    def _parse(self, row, t):
        la, lo, al = _pick(row, _LAT_COLS), _pick(row, _LON_COLS), _pick(row, _ALT_COLS)
        fix = row.get("gps_fix", "True") not in ("False", "0", "false")
        if la and lo and al and fix:
            gps = (float(row[la]), self.lon_sign * float(row[lo]), float(row[al]))
            if gps != self._last_gps and abs(gps[0]) > 1e-6:  # skip repeated/empty fixes
                self._last_gps = gps
                yield Measurement(t, "gps", gps)
        b = _pick(row, _BARO_COLS)
        if b:
            yield Measurement(t, "baro", (float(row[b]),))
        elif _pick(row, _PRESSURE_COLS):
            from .atmosphere import pressure_altitude
            yield Measurement(t, "baro", (float(pressure_altitude(float(row[_pick(row, _PRESSURE_COLS)]))),))
        for cols in _ACCEL_COLS:
            if all(_pick(row, (c,)) for c in cols):
                a = tuple(float(row[c]) * self.accel_scale for c in cols)
                yield Measurement(t, "accel", a)
                break
