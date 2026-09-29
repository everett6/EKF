"""Point-mass flight model: thrust, gravity, aerodynamic drag, parachutes and wind.

Everything is vectorised over N states at once so the same code drives the
EKF (N=1, plus Jacobian columns) and the Monte Carlo predictor (N=hundreds).

State vector (9): [east, north, up, v_east, v_north, v_up,
                   wind_bias_east, wind_bias_north, drag_scale]
Positions are metres in the local ENU frame at the launch site.
The wind-bias states absorb the difference between the forecast wind and
the wind actually acting on the rocket; drag_scale absorbs error in the
assumed drag coefficients and parachute sizes.
"""

import numpy as np

from .atmosphere import G0

N_STATE = 9
PAD, BOOST, COAST, DROGUE, MAIN, LANDED = -1, 0, 1, 2, 3, 4
PHASE_NAMES = {PAD: "pad", BOOST: "boost", COAST: "coast", DROGUE: "drogue",
               MAIN: "main", LANDED: "landed"}


class FlightModel:
    def __init__(self, rocket, site_alt, weather_fn, thrust_scale=1.0, chute_scale=1.0):
        """weather_fn() returns the WeatherProfile to use (so live refreshes are picked up)."""
        self.r = rocket
        self.site_alt = site_alt
        self.weather_fn = weather_fn
        self.thrust_scale = thrust_scale
        self.chute_scale = chute_scale
        el, az = np.radians(rocket.launch_elevation_deg), np.radians(rocket.launch_azimuth_deg)
        self.rail_dir = np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])
        if rocket.thrust_curve:
            curve = np.asarray(rocket.thrust_curve, dtype=float)
            self._curve_t, self._curve_f = curve[:, 0], curve[:, 1]
        else:
            self._curve_t = None

    def thrust(self, t):
        """Thrust (N) at time t since ignition (scalar)."""
        if self._curve_t is not None:
            f = float(np.interp(t, self._curve_t, self._curve_f, left=0.0, right=0.0))
        else:
            f = self.r.average_thrust_n if 0.0 <= t < self.r.burn_time_s else 0.0
        return f * self.thrust_scale

    def mass(self, t):
        frac = min(max(t / self.r.burn_time_s, 0.0), 1.0)
        return self.r.dry_mass_kg + self.r.propellant_mass_kg * (1.0 - frac)

    def drag_area(self, phase):
        """Total drag area (Cd*A, m^2) for an array of phases."""
        body, drogue = self.r.body_cda, self.r.drogue_cda * self.chute_scale
        main = self.r.main_cda * self.chute_scale
        # Indexed by phase + 1: pad, boost, coast, drogue, main, landed.
        table = np.array([body, body, body, body + drogue, body + drogue + main,
                          body + drogue + main])
        return table[np.asarray(phase) + 1]

    def forces(self, X, t, phase, wind_extra=None):
        """Return (acceleration (N,3), specific force (N,3)) for states X (N,9)."""
        pos, vel = X[:, 0:3], X[:, 3:6]
        u, v, rho = self.weather_fn().sample(self.site_alt + pos[:, 2])
        wind = np.stack([u + X[:, 6], v + X[:, 7], np.zeros_like(u)], axis=1)
        if wind_extra is not None:
            wind = wind + wind_extra
        v_air = vel - wind
        speed = np.sqrt(np.einsum("ij,ij->i", v_air, v_air))
        m = self.mass(t)
        thrust = self.thrust(t) * (phase == BOOST)
        if thrust.any():
            dist2 = np.einsum("ij,ij->i", pos, pos)
            on_rail = (dist2 < self.r.rail_length_m ** 2) | (speed < 5.0)
            tdir = np.where(on_rail[:, None], self.rail_dir,
                            v_air / np.maximum(speed, 1e-9)[:, None])
        else:
            tdir = 0.0
        k = np.maximum(X[:, 8], 0.05)
        drag = -0.5 * rho[:, None] * speed[:, None] * v_air * (self.drag_area(phase) * k)[:, None]
        specific = (thrust[:, None] * tdir + drag) / m
        accel = specific - np.array([0.0, 0.0, G0])
        landed = phase == LANDED
        accel[landed] = 0.0
        specific[landed] = [0.0, 0.0, G0]
        return accel, specific

    def _deriv(self, X, t, phase, wind_extra):
        a, _ = self.forces(X, t, phase, wind_extra)
        d = np.zeros_like(X)
        d[:, 0:3] = X[:, 3:6]
        d[:, 3:6] = a
        d[phase == LANDED, 0:3] = 0.0
        return d

    def step(self, X, t, phase, dt, wind_extra=None):
        """One RK4 step of length dt with phase held constant."""
        k1 = self._deriv(X, t, phase, wind_extra)
        k2 = self._deriv(X + 0.5 * dt * k1, t + 0.5 * dt, phase, wind_extra)
        k3 = self._deriv(X + 0.5 * dt * k2, t + 0.5 * dt, phase, wind_extra)
        k4 = self._deriv(X + dt * k3, t + dt, phase, wind_extra)
        return X + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    def stable_dt(self, X, t, phase, max_dt=0.5):
        """Largest step that keeps RK4 stable against the stiffest drag (e.g. a chute opening)."""
        speed = np.sqrt(np.einsum("ij,ij->i", X[:, 3:6], X[:, 3:6])) + 1.0
        rate = 1.3 * self.drag_area(phase) * np.maximum(X[:, 8], 0.05) * speed / self.mass(t)
        return float(np.clip(2.0 / max(rate.max(), 1e-6), 0.01, max_dt))

    def next_phase(self, X, t, phase):
        """Apply event logic: burnout, apogee (drogue), main deploy altitude, touchdown."""
        phase = phase.copy()
        up, vup = X[:, 2], X[:, 5]
        burn_end = self._curve_t[-1] if self._curve_t is not None else self.r.burn_time_s
        phase[(phase == BOOST) & (t >= burn_end)] = COAST
        phase[(phase == COAST) & (vup < 0)] = DROGUE
        phase[(phase == DROGUE) & (up < self.r.main_deploy_agl_m)] = MAIN
        phase[(phase >= COAST) & (phase < LANDED) & (up <= 0.0) & (vup < 0)] = LANDED
        return phase
