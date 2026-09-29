"""Forward prediction of the remaining trajectory and landing point.

Draws Monte Carlo samples from the EKF's current mean and covariance (which
include the wind-bias and drag-scale states, so forecast-wind and drag
uncertainty are carried through) and flies all of them to the ground at
once with the vectorised flight model using the latest weather profile.
"""

from dataclasses import dataclass, field

import numpy as np

from .dynamics import BOOST, COAST, LANDED

CHI2_95_2DOF = 5.991


@dataclass
class Prediction:
    t: float                      # flight time the prediction was made at
    landing_enu: np.ndarray       # mean landing point (east, north)
    landing_cov: np.ndarray       # 2x2 covariance of landing point
    ellipse: tuple                # (semi-major m, semi-minor m, orientation deg from east) at 95 %
    time_to_landing: float        # mean seconds until touchdown
    apogee_up: float              # predicted apogee (m AGL) of the mean trajectory
    path_enu: np.ndarray          # (K,3) mean-state trajectory, 1 s spacing
    impacts_enu: np.ndarray = field(repr=False, default=None)  # (N,2) per-sample landings


def error_ellipse(cov, chi2=CHI2_95_2DOF):
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0.0)
    major = vecs[:, 1]
    return (float(np.sqrt(chi2 * vals[1])), float(np.sqrt(chi2 * vals[0])),
            float(np.degrees(np.arctan2(major[1], major[0]))))


def predict_landing(model, x, P, t, phase, samples=200, max_flight_s=1800.0, rng=None,
                    gust_sigma=1.0, gust_tau_s=5.0):
    rng = rng or np.random.default_rng()
    P = 0.5 * (P + P.T)
    try:
        L = np.linalg.cholesky(P + 1e-9 * np.eye(len(x)))
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(P)
        L = V @ np.diag(np.sqrt(np.maximum(w, 0)))
    X = x + rng.standard_normal((samples, len(x))) @ L.T
    X[0] = x  # sample 0 is the mean, used for the displayed path
    X[:, 8] = np.maximum(X[:, 8], 0.05)
    ph = np.full(samples, phase)
    if phase == LANDED:
        imp = np.repeat(x[None, 0:2], samples, axis=0)
        return Prediction(t, x[0:2].copy(), np.diag([1e-6, 1e-6]), (0.0, 0.0, 0.0), 0.0,
                          float(x[2]), x[None, 0:3].copy(), imp)
    # Anything on the ground but still climbing is treated as flying.
    impacts = np.full((samples, 2), np.nan)
    impact_t = np.full(samples, np.nan)
    path = [X[0, 0:3].copy()]
    apogee = X[0, 2]
    tc, next_rec = t, t + 1.0
    # Turbulence the forecast can't know about: a random gust process per sample
    # (none on sample 0, the mean path).
    gust = np.zeros((samples, 3))
    gust_scale = np.full((samples, 1), gust_sigma)
    gust_scale[0] = 0.0
    while tc - t < max_flight_s:
        active = np.isnan(impact_t)
        if not active.any():
            break
        if (ph[active] == BOOST).any():
            dt = 0.02
        else:
            dt = model.stable_dt(X[active], tc, ph[active], 0.1 if (ph[active] == COAST).any() else 0.5)
        prev = X.copy()
        if gust_sigma > 0:
            a = np.exp(-dt / gust_tau_s)
            gust[:, 0:2] = a * gust[:, 0:2] + np.sqrt(1 - a * a) * gust_scale * rng.standard_normal((samples, 2))
        X[active] = model.step(X[active], tc, ph[active], dt, wind_extra=gust[active])
        tc += dt
        ph[active] = model.next_phase(X[active], tc, ph[active])
        apogee = max(apogee, X[0, 2])
        hit = active & (ph == LANDED)
        if hit.any():
            # Interpolate to the moment the altitude crossed zero.
            a = prev[hit, 2] / np.maximum(prev[hit, 2] - X[hit, 2], 1e-9)
            a = np.clip(a, 0.0, 1.0)[:, None]
            impacts[hit] = prev[hit, 0:2] + a * (X[hit, 0:2] - prev[hit, 0:2])
            impact_t[hit] = tc - dt + a[:, 0] * dt
        if tc >= next_rec and np.isnan(impact_t[0]):
            path.append(X[0, 0:3].copy())
            next_rec += 1.0
    if not np.isnan(impact_t[0]):
        path.append(np.array([impacts[0, 0], impacts[0, 1], 0.0]))
    ok = ~np.isnan(impact_t)
    pts = impacts[ok] if ok.any() else X[:, 0:2]
    mean = pts.mean(axis=0)
    cov = np.cov(pts.T) if len(pts) > 2 else np.diag([1.0, 1.0])
    ttl = float(np.nanmean(impact_t) - t) if ok.any() else float("nan")
    return Prediction(t, mean, cov, error_ellipse(cov), ttl, float(apogee),
                      np.array(path), pts)
