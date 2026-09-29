import numpy as np

from rocket_tracker.config import Config
from rocket_tracker.dynamics import LANDED
from rocket_tracker.telemetry import SimulatedFlight, CsvSource
from rocket_tracker.tracker import Tracker
from rocket_tracker.weather import WeatherService


def _small_flight_config():
    cfg = Config()
    cfg.weather.provider = "standard"
    # A smaller motor keeps the simulated flight (and the test) short.
    cfg.rocket.average_thrust_n = 900.0
    cfg.rocket.burn_time_s = 2.0
    cfg.rocket.propellant_mass_kg = 1.0
    cfg.rocket.dry_mass_kg = 6.0
    cfg.rocket.main_diameter_m = 1.4
    cfg.predictor.samples = 60
    cfg.predictor.interval_s = 3.0
    cfg.sensors.accel_rate_hz = 20.0
    return cfg


def test_simulated_flight_is_tracked_and_landing_predicted():
    cfg = _small_flight_config()
    ws = WeatherService(cfg.site, cfg.weather)
    sim = SimulatedFlight(cfg, ws, launch_time=1000.0)
    tr = Tracker(cfg, ws, rng=np.random.default_rng(1))
    pos_err = []
    for m in sim:
        tr.process(m)
        if tr.ekf is not None and m.kind == "gps" and tr.ekf.phase != LANDED:
            pos_err.append(np.linalg.norm(tr.ekf.x[0:3] - np.array(m.truth["enu"])))

    assert tr.launch_time is not None and abs(tr.launch_time - 1000.0) < 0.1
    assert tr.ekf.phase == LANDED
    assert np.median(pos_err) < 5.0

    truth = sim.landing_enu
    drogue = [p for p in tr.predictions if 0 < p.time_to_landing < 60]
    assert drogue, "expected predictions during descent"
    # Once the wind has been learned on the way down, predictions close in on the truth...
    late = [np.linalg.norm(p.landing_enu - truth) for p in drogue[len(drogue) // 2:]]
    assert np.median(late) < 60.0
    # ...and the 95 % ellipse is honest about the error most of the time.
    inside = []
    for p in tr.predictions:
        d = truth - p.landing_enu
        inside.append(d @ np.linalg.solve(p.landing_cov + 1e-6 * np.eye(2), d) <= 5.991)
    assert np.mean(inside) > 0.6

    snap = tr.snapshot()
    assert snap["launched"] and "prediction" in snap and snap["state"]["phase"] == "landed"


def test_csv_replay_of_repo_bench_log():
    # The logged bench data never leaves the pad: no launch should be detected.
    cfg = Config()
    cfg.weather.provider = "standard"
    ws = WeatherService(cfg.site, cfg.weather)
    tr = Tracker(cfg, ws)
    kinds = set()
    for m in CsvSource("sensor_data_ui.csv", flip_lon=True):
        kinds.add(m.kind)
        tr.process(m)
    assert kinds == {"gps", "accel"}
    assert tr.ekf is None
    assert len(tr._pad_gps) > 0
    # With the sign fixed the logged fixes sit within a few hundred metres of the pad.
    assert np.hypot(*np.median(np.array(tr._pad_gps), axis=0)[:2]) < 500
