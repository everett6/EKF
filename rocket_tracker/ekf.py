"""Extended Kalman Filter over the 9-state flight model in dynamics.py.

The filter propagates the full non-linear model with RK4 and linearises it
numerically (finite differences evaluated as one vectorised batch), so the
Jacobian always matches the physics even as phases change.
"""

import numpy as np

from .dynamics import N_STATE, BOOST, COAST, DROGUE, MAIN, LANDED

# Continuous white-noise acceleration intensity per phase (m^2/s^3).
_Q_ACCEL = {BOOST: 25.0, COAST: 4.0, DROGUE: 2.0, MAIN: 1.0, LANDED: 1e-4}
_Q_WIND_BIAS = 0.02   # m^2/s^3 random walk
_Q_DRAG_SCALE = 1e-4  # 1/s random walk
_GATE_CHI2 = {1: 15.0, 3: 22.0}  # ~99.9x% gates for 1 and 3 dof


class ExtendedKalmanFilter:
    def __init__(self, model, x0, P0, t0=0.0, phase=BOOST, max_dt=0.05):
        self.model = model
        self.x = np.asarray(x0, dtype=float).copy()
        self.P = np.asarray(P0, dtype=float).copy()
        self.t = t0
        self.phase = phase
        self.max_dt = max_dt
        self.rejected = 0
        self.rejected_by = {}
        self._settle_until = -1.0

    def _propagate(self, dt):
        X = self.x[None, :]
        eps = np.maximum(1e-4, 1e-6 * np.abs(self.x))
        batch = np.vstack([X, X + np.diag(eps)])
        phases = np.full(len(batch), self.phase)
        out = self.model.step(batch, self.t, phases, dt)
        F = ((out[1:] - out[0]) / eps[:, None]).T
        # Parameter states are constants of motion in the model.
        return out[0], F

    def _process_noise(self, dt):
        q = _Q_ACCEL.get(self.phase, 4.0)
        Q = np.zeros((N_STATE, N_STATE))
        for i in range(3):
            Q[i, i] = q * dt ** 3 / 3
            Q[i, i + 3] = Q[i + 3, i] = q * dt ** 2 / 2
            Q[i + 3, i + 3] = q * dt
        Q[6, 6] = Q[7, 7] = _Q_WIND_BIAS * dt
        Q[8, 8] = _Q_DRAG_SCALE * dt
        return Q

    def predict_to(self, t):
        """Propagate the estimate to time t (seconds since launch)."""
        while self.t < t - 1e-9:
            dt = min(self.max_dt, t - self.t)
            if self.phase == LANDED:
                self.t = t
                return
            x, F = self._propagate(dt)
            self.P = F @ self.P @ F.T + self._process_noise(dt)
            self.x = x
            self.t += dt
            new_phase = self.model.next_phase(self.x[None, :], self.t, np.array([self.phase]))[0]
            if new_phase != self.phase:
                self._on_phase_change(new_phase)

    def _on_phase_change(self, new_phase):
        if new_phase in (DROGUE, MAIN):
            # A parachute just opened; its real drag is uncertain.
            self.P[8, 8] += 0.3 ** 2
            # The estimated deployment moment won't line up exactly with the real
            # one, so ignore the accelerometer through the opening shock.
            self._settle_until = self.t + 2.0
        if new_phase == LANDED:
            self.x[2] = 0.0
            self.x[3:6] = 0.0
        self.phase = new_phase

    def update(self, z, h_fn, R, name="meas"):
        """Generic EKF update with a numerical measurement Jacobian and chi-square gating."""
        z = np.atleast_1d(np.asarray(z, dtype=float))
        R = np.atleast_2d(R)
        hx = np.atleast_1d(h_fn(self.x))
        H = np.zeros((len(z), N_STATE))
        for i in range(N_STATE):
            dx = np.zeros(N_STATE)
            dx[i] = max(1e-4, 1e-6 * abs(self.x[i]))
            H[:, i] = (np.atleast_1d(h_fn(self.x + dx)) - hx) / dx[i]
        y = z - hx
        S = H @ self.P @ H.T + R
        Sinv = np.linalg.inv(S)
        nis = float(y @ Sinv @ y)
        if nis > _GATE_CHI2.get(len(z), 25.0):
            self.rejected += 1
            self.rejected_by[name] = self.rejected_by.get(name, 0) + 1
            return False
        K = self.P @ H.T @ Sinv
        if self.phase in (BOOST, COAST):
            # Schmidt ("consider") update: wind bias and drag scale are barely
            # observable at high speed, so they inform the covariance but are
            # only estimated once the rocket is descending on a parachute.
            K[6:9] = 0.0
        self.x = self.x + K @ y
        I_KH = np.eye(N_STATE) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T  # Joseph form keeps P symmetric PD
        self.x[8] = max(self.x[8], 0.05)
        return True

    # Convenience measurement models ------------------------------------------------

    def update_position(self, enu, sigma_h, sigma_v):
        R = np.diag([sigma_h ** 2, sigma_h ** 2, sigma_v ** 2])
        return self.update(enu, lambda x: x[0:3], R, "gps")

    def update_altitude(self, up, sigma):
        return self.update([up], lambda x: x[2:3], [[sigma ** 2]], "baro")

    def update_specific_force(self, magnitude, sigma):
        """Accelerometer magnitude: orientation-free, so no attitude solution is needed."""
        if self.t < self._settle_until:
            return False
        def h(x):
            _, sf = self.model.forces(x[None, :], self.t, np.array([self.phase]))
            return np.linalg.norm(sf, axis=1)
        # Thrust and drag are only known to a few percent, so widen the noise
        # in proportion to the force the model predicts.
        sigma_model = 0.05 * float(h(self.x)[0])
        return self.update([magnitude], h, [[sigma ** 2 + sigma_model ** 2]], "accel")
