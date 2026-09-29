"""Ties telemetry, the EKF, live weather and the landing predictor together."""

import logging
import threading
import time

import numpy as np

from .dynamics import FlightModel, N_STATE, PAD, BOOST, COAST, LANDED, PHASE_NAMES
from .ekf import ExtendedKalmanFilter
from .geo import geodetic_to_enu, enu_to_geodetic
from .predictor import predict_landing

log = logging.getLogger(__name__)

LAUNCH_ACCEL = 3.0 * 9.80665   # specific force that signals ignition
LAUNCH_ALT_AGL = 20.0          # or a climb this high above the pad


class Tracker:
    def __init__(self, cfg, weather_service, rng=None):
        self.cfg = cfg
        self.site = cfg.site
        self.weather = weather_service
        self.model = FlightModel(cfg.rocket, self.site.altitude_m, lambda: self.weather.profile)
        self.rng = rng or np.random.default_rng(0)
        self.ekf = None
        self.launch_time = None
        self.phase = PAD
        self.baro_offset = 0.0
        self._pad_baro = []
        self._pad_gps = []
        self.track = []        # (t, e, n, u, phase)
        self.gps_points = []   # (t, e, n, u)
        self.predictions = []  # Prediction objects
        self.last_prediction = None
        self._next_prediction = 0.0
        self._lock = threading.Lock()
        self.weather.listeners.append(self._on_weather)
        self.events = []

    # --- weather -------------------------------------------------------------------
    def _on_weather(self, profile):
        self._event(f"Weather updated ({profile.source}, valid {profile.valid_time})")
        # Re-run the forecast right away with the new winds.
        self._next_prediction = 0.0

    def _event(self, text):
        t = 0.0 if self.launch_time is None or self.ekf is None else self.ekf.t
        self.events.append((round(t, 1), text))
        log.info("[T+%.1fs] %s", t, text)

    # --- measurement handling ------------------------------------------------------
    def process(self, m):
        if m.kind == "gps":
            enu = np.array(geodetic_to_enu(*m.values, self.site.latitude, self.site.longitude,
                                           self.site.altitude_m), dtype=float)
        if self.ekf is None:
            self._pad(m, enu if m.kind == "gps" else None)
            if self.ekf is None:
                return
        t = m.t - self.launch_time
        if t < self.ekf.t - 0.5:
            return  # stale sample from before launch detection
        self.ekf.predict_to(max(t, self.ekf.t))
        s = self.cfg.sensors
        if m.kind == "gps":
            self.ekf.update_position(enu, s.gps_sigma_h_m, s.gps_sigma_v_m)
            with self._lock:
                self.gps_points.append((t, *enu))
        elif m.kind == "baro":
            self.ekf.update_altitude(m.values[0] - self.site.altitude_m - self.baro_offset,
                                     s.baro_sigma_m)
        elif m.kind == "accel" and self.ekf.phase != LANDED:
            self.ekf.update_specific_force(float(np.linalg.norm(m.values)), s.accel_sigma_mps2)
        if self.ekf.phase != self.phase:
            self._event(f"Phase: {PHASE_NAMES[self.ekf.phase]}")
            self.phase = self.ekf.phase
            self._next_prediction = 0.0
        with self._lock:
            if not self.track or t - self.track[-1][0] >= 0.1:
                self.track.append((t, *self.ekf.x[0:3], self.phase))
        if t >= self._next_prediction and self.phase != LANDED:
            self.run_prediction()
            self._next_prediction = t + self.cfg.predictor.interval_s
        elif self.phase == LANDED and (self.last_prediction is None or self.last_prediction.time_to_landing > 0):
            self.run_prediction()

    def _pad(self, m, enu):
        if m.kind == "baro":
            self._pad_baro.append(m.values[0] - self.site.altitude_m)
            self._pad_baro = self._pad_baro[-200:]
            self.baro_offset = float(np.median(self._pad_baro))
        if m.kind == "gps":
            self._pad_gps.append(enu)
            self._pad_gps = self._pad_gps[-50:]
        launched = False
        if m.kind == "accel" and np.linalg.norm(m.values) > LAUNCH_ACCEL:
            launched = True
        if m.kind == "gps" and enu[2] > LAUNCH_ALT_AGL and len(self._pad_gps) > 5:
            launched = True
        if not launched:
            return
        self.launch_time = m.t
        x0 = np.zeros(N_STATE)
        x0[8] = 1.0
        if self._pad_gps and not (m.kind == "gps" and enu[2] > LAUNCH_ALT_AGL):
            x0[0:2] = np.median(np.array(self._pad_gps)[:, 0:2], axis=0)
        sig_w = self.cfg.weather.forecast_wind_sigma
        P0 = np.diag([3.0, 3.0, 2.0, 1.0, 1.0, 1.0, sig_w, sig_w, 0.15]) ** 2
        phase = BOOST
        if m.kind == "gps" and enu[2] > LAUNCH_ALT_AGL:
            # Detected late from altitude alone: start at the fix with loose velocity.
            x0[0:3] = enu
            x0[5] = 50.0
            P0[3:6, 3:6] = np.diag([30.0, 30.0, 100.0]) ** 2
            phase = COAST
        self.ekf = ExtendedKalmanFilter(self.model, x0, P0, t0=0.0, phase=phase)
        self.phase = phase
        self._event("Launch detected")

    # --- prediction ----------------------------------------------------------------
    def run_prediction(self):
        pc = self.cfg.predictor
        pred = predict_landing(self.model, self.ekf.x, self.ekf.P, self.ekf.t, self.ekf.phase,
                               samples=pc.samples, max_flight_s=pc.max_flight_s, rng=self.rng,
                               gust_sigma=pc.gust_sigma, gust_tau_s=pc.gust_tau_s)
        with self._lock:
            self.last_prediction = pred
            self.predictions.append(pred)
        return pred

    # --- reporting -----------------------------------------------------------------
    def _ll(self, e, n, u=0.0):
        lat, lon, alt = enu_to_geodetic(e, n, u, self.site.latitude, self.site.longitude,
                                        self.site.altitude_m)
        return float(lat), float(lon), float(alt)

    def summary_line(self):
        if self.ekf is None:
            return "On the pad, waiting for launch"
        x, p = self.ekf.x, self.last_prediction
        s = (f"T+{self.ekf.t:6.1f}s {PHASE_NAMES[self.ekf.phase]:>6} alt {x[2]:7.1f} m "
             f"spd {np.linalg.norm(x[3:6]):6.1f} m/s")
        if p is not None:
            e, n = p.landing_enu
            s += (f" | landing {np.hypot(e, n):6.0f} m @ {np.degrees(np.arctan2(e, n)) % 360:5.1f} deg"
                  f" ±{p.ellipse[0]:.0f} m, in {max(p.time_to_landing, 0):5.1f} s")
        return s

    def snapshot(self):
        """JSON-ready state for the dashboard."""
        with self._lock:
            track = list(self.track)
            gps = list(self.gps_points[-400:])
            pred = self.last_prediction
            history = [(p.t, *p.landing_enu) for p in self.predictions]
        out = {
            "site": {"lat": self.site.latitude, "lon": self.site.longitude,
                     "alt": self.site.altitude_m},
            "rocket": self.cfg.rocket.name,
            "weather": self.weather.status(),
            "weather_profile": self.weather.profile.to_dict(),
            "events": self.events[-20:],
            "launched": self.ekf is not None,
        }
        if self.ekf is None:
            return out
        x, P = self.ekf.x, self.ekf.P
        lat, lon, alt = self._ll(*x[0:3])
        spd, wdir = self.weather.profile.wind_at(self.site.altitude_m + x[2])
        out["state"] = {
            "t": round(self.ekf.t, 2), "phase": PHASE_NAMES[self.ekf.phase],
            "lat": lat, "lon": lon, "alt_agl": float(x[2]), "alt_msl": alt,
            "velocity": [float(v) for v in x[3:6]], "speed": float(np.linalg.norm(x[3:6])),
            "pos_sigma": [float(np.sqrt(P[i, i])) for i in range(3)],
            "wind_bias": [float(x[6]), float(x[7])], "drag_scale": float(x[8]),
            "forecast_wind": {"speed": spd, "from_deg": wdir},
            "rejected": self.ekf.rejected,
        }
        out["track"] = [[*self._ll(e, n, u)[0:2], round(float(u), 1), round(t, 1)]
                        for t, e, n, u, _ in track[::max(1, len(track) // 1500)]]
        out["gps"] = [list(self._ll(e, n, u)[0:2]) for t, e, n, u in gps[::2]]
        if pred is not None:
            a, b, ang = pred.ellipse
            th = np.linspace(0, 2 * np.pi, 48)
            c, s_ = np.cos(np.radians(ang)), np.sin(np.radians(ang))
            ex = pred.landing_enu[0] + a * np.cos(th) * c - b * np.sin(th) * s_
            ny = pred.landing_enu[1] + a * np.cos(th) * s_ + b * np.sin(th) * c
            out["prediction"] = {
                "t": round(pred.t, 1),
                "landing": list(self._ll(*pred.landing_enu)[0:2]),
                "landing_enu": [float(v) for v in pred.landing_enu],
                "ellipse_95": {"semi_major_m": a, "semi_minor_m": b, "angle_deg": ang,
                               "polygon": [list(self._ll(e, n)[0:2]) for e, n in zip(ex, ny)]},
                "time_to_landing": pred.time_to_landing,
                "apogee_agl": max(pred.apogee_up, max(r[3] for r in track)),
                "path": [[*self._ll(*p)[0:2], round(float(p[2]), 1)] for p in pred.path_enu[::2]],
                "samples": [list(self._ll(e, n)[0:2]) for e, n in pred.impacts_enu[:80]],
            }
            out["landing_history"] = [[round(t, 1), *self._ll(e, n)[0:2]] for t, e, n in history[-300:]]
        return out
