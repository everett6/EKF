"""Rocket, launch site, sensor and simulation parameters.

Every value has a placeholder default so the tracker runs out of the box;
override any of them with a JSON file (see examples/rocket.json).
"""

import json
import math
from dataclasses import dataclass, field, asdict, fields, is_dataclass


@dataclass
class RocketConfig:
    name: str = "Placeholder L-class"
    dry_mass_kg: float = 12.0
    propellant_mass_kg: float = 2.5
    burn_time_s: float = 3.0
    average_thrust_n: float = 1800.0
    # Optional thrust curve [[t, N], ...]; overrides average_thrust_n when given.
    thrust_curve: list = field(default_factory=list)
    body_diameter_m: float = 0.10
    drag_coefficient: float = 0.50
    drogue_diameter_m: float = 0.6
    drogue_cd: float = 1.5
    main_diameter_m: float = 2.0
    main_cd: float = 2.2
    main_deploy_agl_m: float = 300.0
    launch_elevation_deg: float = 87.0
    launch_azimuth_deg: float = 0.0
    rail_length_m: float = 3.0

    @property
    def body_cda(self):
        return self.drag_coefficient * math.pi * (self.body_diameter_m / 2) ** 2

    @property
    def drogue_cda(self):
        return self.drogue_cd * math.pi * (self.drogue_diameter_m / 2) ** 2

    @property
    def main_cda(self):
        return self.main_cd * math.pi * (self.main_diameter_m / 2) ** 2


@dataclass
class SiteConfig:
    # Default: the origin used in DataLogging.py.
    latitude: float = 37.395390
    longitude: float = -121.851837
    altitude_m: float = 180.0


@dataclass
class WeatherConfig:
    # "open-meteo" fetches live winds aloft; "standard" uses ISA + the surface wind below.
    provider: str = "open-meteo"
    refresh_s: float = 300.0
    surface_wind_speed: float = 4.0
    surface_wind_dir_deg: float = 270.0
    # 1-sigma uncertainty of the forecast wind, used to seed the filter's wind-bias states.
    forecast_wind_sigma: float = 2.0


@dataclass
class SensorConfig:
    gps_rate_hz: float = 10.0
    gps_sigma_h_m: float = 3.0
    gps_sigma_v_m: float = 6.0
    baro_rate_hz: float = 20.0
    baro_sigma_m: float = 2.0
    accel_rate_hz: float = 50.0
    accel_sigma_mps2: float = 0.8


@dataclass
class SimConfig:
    """How the simulated 'real' flight differs from what the tracker assumes."""
    seed: int = 7
    wind_offset_east: float = 2.5
    wind_offset_north: float = -1.5
    gust_sigma: float = 1.0
    gust_tau_s: float = 5.0
    drag_scale: float = 1.10
    thrust_scale: float = 0.97
    chute_scale: float = 0.90


@dataclass
class PredictorConfig:
    samples: int = 200
    interval_s: float = 1.0
    max_flight_s: float = 1800.0
    # Unforecast turbulence flown into every Monte Carlo sample.
    gust_sigma: float = 1.0
    gust_tau_s: float = 5.0


@dataclass
class Config:
    rocket: RocketConfig = field(default_factory=RocketConfig)
    site: SiteConfig = field(default_factory=SiteConfig)
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    sensors: SensorConfig = field(default_factory=SensorConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)

    def to_dict(self):
        return asdict(self)


def _merge(obj, data):
    known = {f.name: f for f in fields(obj)}
    for key, value in data.items():
        if key not in known:
            raise KeyError(f"Unknown config key '{key}' in {type(obj).__name__}")
        current = getattr(obj, key)
        if is_dataclass(current):
            _merge(current, value)
        else:
            setattr(obj, key, value)
    return obj


def load_config(path=None):
    cfg = Config()
    if path:
        with open(path) as f:
            _merge(cfg, json.load(f))
    return cfg
