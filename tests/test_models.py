import json
from datetime import datetime, timezone

import numpy as np
import pytest

from rocket_tracker.atmosphere import isa, standard_profile, wind_components
from rocket_tracker.config import Config, load_config
from rocket_tracker.dynamics import FlightModel, BOOST, COAST, DROGUE, MAIN, LANDED
from rocket_tracker.geo import geodetic_to_enu, enu_to_geodetic
from rocket_tracker.predictor import error_ellipse
from rocket_tracker.weather import parse_open_meteo, PRESSURE_LEVELS, WeatherService


def test_isa_reference_values():
    T, p, rho = isa(np.array([0.0, 11000.0, 5000.0]))
    assert T[0] == pytest.approx(288.15)
    assert p[0] == pytest.approx(101325.0)
    assert rho[0] == pytest.approx(1.225, abs=1e-3)
    assert p[1] == pytest.approx(22632, rel=1e-3)
    assert rho[2] == pytest.approx(0.7364, rel=1e-3)


def test_geo_roundtrip():
    lat0, lon0, alt0 = 37.39539, -121.851837, 180.0
    e, n, u = geodetic_to_enu(37.40, -121.84, 1500.0, lat0, lon0, alt0)
    assert e > 0 and n > 0 and u == pytest.approx(1320, abs=5)
    lat, lon, alt = enu_to_geodetic(e, n, u, lat0, lon0, alt0)
    assert lat == pytest.approx(37.40, abs=1e-9)
    assert lon == pytest.approx(-121.84, abs=1e-9)
    assert alt == pytest.approx(1500.0, abs=1e-4)


def test_wind_components_follow_meteorological_convention():
    u, v = wind_components(10.0, 270.0)  # wind FROM the west blows east
    assert u == pytest.approx(10.0) and v == pytest.approx(0.0, abs=1e-9)
    prof = standard_profile(0.0, 10.0, 270.0)
    speed, direction = prof.wind_at(10.0)
    assert speed == pytest.approx(10.0, rel=1e-3) and direction == pytest.approx(270.0)


def _fake_open_meteo(site_alt):
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    hourly = {"time": [now.strftime("%Y-%m-%dT%H:%M")]}
    for i, lvl in enumerate(PRESSURE_LEVELS):
        hourly[f"geopotential_height_{lvl}hPa"] = [44330.77 * (1 - (lvl / 1013.25) ** 0.190263)]
        hourly[f"wind_speed_{lvl}hPa"] = [5.0 + i]
        hourly[f"wind_direction_{lvl}hPa"] = [250.0]
        hourly[f"temperature_{lvl}hPa"] = [15.0 - 0.0065 * hourly[f"geopotential_height_{lvl}hPa"][0]]
    return {"hourly": hourly, "current": {"temperature_2m": 18.0, "surface_pressure": 990.0,
                                          "wind_speed_10m": 3.0, "wind_direction_10m": 250.0}}


def test_parse_open_meteo_drops_levels_below_ground():
    prof = parse_open_meteo(_fake_open_meteo(400.0), site_alt=400.0)
    assert prof.source == "open-meteo"
    assert list(prof.alt[:2]) == pytest.approx([400.0, 410.0])  # ground anchor, 10 m wind
    assert np.all(np.diff(prof.alt) > 0)
    assert prof.alt[2] > 430.0  # 1000/975 hPa levels are underground at 400 m
    u, v, rho = prof.sample(np.array([410.0, 3000.0, 30000.0]))
    assert u[0] > 0 and v[0] > 0  # from 250 deg -> blowing toward ENE
    assert rho[2] < rho[1] < rho[0]


def test_weather_service_falls_back_when_offline():
    cfg = Config()

    def broken(*a, **k):
        raise OSError("no network")

    ws = WeatherService(cfg.site, cfg.weather, fetcher=broken)
    assert ws.refresh() is False
    assert ws.profile.source == "standard-atmosphere"
    assert "no network" in ws.last_error

    ws.fetcher = lambda lat, lon, alt: parse_open_meteo(_fake_open_meteo(alt), alt)
    seen = []
    ws.listeners.append(seen.append)
    assert ws.refresh() is True
    assert ws.profile.source == "open-meteo" and len(seen) == 1


def test_config_override(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"rocket": {"dry_mass_kg": 5.0}, "site": {"altitude_m": 12.0}}))
    cfg = load_config(p)
    assert cfg.rocket.dry_mass_kg == 5.0 and cfg.site.altitude_m == 12.0
    p.write_text(json.dumps({"rocket": {"not_a_field": 1}}))
    with pytest.raises(KeyError):
        load_config(p)


def test_flight_model_phases_in_calm_air():
    cfg = Config()
    prof = standard_profile(cfg.site.altitude_m, 0.0, 0.0)
    m = FlightModel(cfg.rocket, cfg.site.altitude_m, lambda: prof)
    X = np.zeros((1, 9))
    X[0, 8] = 1.0
    ph = np.array([BOOST])
    t, seen, apogee = 0.0, [BOOST], 0.0
    while ph[0] != LANDED and t < 1000:
        X = m.step(X, t, ph, 0.02)
        t += 0.02
        ph = m.next_phase(X, t, ph)
        apogee = max(apogee, X[0, 2])
        if ph[0] != seen[-1]:
            seen.append(int(ph[0]))
    assert seen == [BOOST, COAST, DROGUE, MAIN, LANDED]
    assert 1500 < apogee < 5000
    # With no wind, the rail tilt is the only horizontal push.
    assert np.hypot(X[0, 0], X[0, 1]) < 0.5 * apogee


def test_error_ellipse_axes():
    a, b, ang = error_ellipse(np.diag([100.0, 25.0]))
    assert a == pytest.approx(np.sqrt(5.991 * 100))
    assert b == pytest.approx(np.sqrt(5.991 * 25))
    assert abs(np.cos(np.radians(ang))) == pytest.approx(1.0)
