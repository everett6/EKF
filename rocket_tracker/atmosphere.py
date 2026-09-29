"""Standard atmosphere and altitude-dependent weather profiles."""

import numpy as np

R_AIR = 287.05287  # J/(kg K)
G0 = 9.80665

# ISA layers: base altitude (m), base temperature (K), lapse rate (K/m), base pressure (Pa)
_LAYERS = [
    (0.0, 288.15, -0.0065, 101325.0),
    (11000.0, 216.65, 0.0, 22632.06),
    (20000.0, 216.65, 0.001, 5474.889),
    (32000.0, 228.65, 0.0028, 868.0187),
    (47000.0, 270.65, 0.0, 110.9063),
]


def isa(alt_msl):
    """International Standard Atmosphere. Returns (temperature K, pressure Pa, density kg/m^3)."""
    h = np.clip(np.asarray(alt_msl, dtype=float), -500.0, 50000.0)
    T = np.empty_like(h)
    p = np.empty_like(h)
    for i, (hb, Tb, L, pb) in enumerate(_LAYERS):
        top = _LAYERS[i + 1][0] if i + 1 < len(_LAYERS) else np.inf
        m = (h >= hb) & (h < top) if i else (h < top)
        dh = h[m] - hb
        if L == 0.0:
            T[m] = Tb
            p[m] = pb * np.exp(-G0 * dh / (R_AIR * Tb))
        else:
            T[m] = Tb + L * dh
            p[m] = pb * (T[m] / Tb) ** (-G0 / (L * R_AIR))
    return T, p, p / (R_AIR * T)


def pressure_altitude(pressure_pa, sea_level_pa=101325.0):
    """Barometric altitude (m) from static pressure, troposphere formula."""
    return 44330.77 * (1.0 - (np.asarray(pressure_pa) / sea_level_pa) ** 0.190263)


class WeatherProfile:
    """Wind, temperature and pressure tabulated by altitude above mean sea level.

    Between levels values are linearly interpolated. Above the top level the
    wind is held constant and density follows the ISA shape; below the lowest
    level the lowest values are held.
    """

    def __init__(self, alt_msl, wind_east, wind_north, temperature_k, pressure_pa,
                 source="custom", valid_time=None, fetched_at=None):
        order = np.argsort(alt_msl)
        self.alt = np.asarray(alt_msl, dtype=float)[order]
        self.u = np.asarray(wind_east, dtype=float)[order]
        self.v = np.asarray(wind_north, dtype=float)[order]
        self.T = np.asarray(temperature_k, dtype=float)[order]
        self.p = np.asarray(pressure_pa, dtype=float)[order]
        self.rho = self.p / (R_AIR * self.T)
        self.source = source
        self.valid_time = valid_time
        self.fetched_at = fetched_at
        _, _, rho_isa_top = isa(self.alt[-1:])
        self._rho_scale_top = self.rho[-1] / rho_isa_top[0]

    def sample(self, alt_msl):
        """Vectorised lookup. Returns (wind_east, wind_north, density)."""
        h = np.asarray(alt_msl, dtype=float)
        u = np.interp(h, self.alt, self.u)
        v = np.interp(h, self.alt, self.v)
        rho = np.interp(h, self.alt, self.rho)
        above = h > self.alt[-1]
        if np.any(above):
            _, _, rho_isa = isa(h[above])
            rho = np.array(rho, dtype=float)
            rho[above] = rho_isa * self._rho_scale_top
        return u, v, rho

    def wind_at(self, alt_msl):
        u, v, _ = self.sample(np.atleast_1d(alt_msl))
        speed = float(np.hypot(u[0], v[0]))
        # Meteorological convention: direction the wind blows FROM, degrees clockwise from north.
        direction = float(np.degrees(np.arctan2(-u[0], -v[0])) % 360.0)
        return speed, direction

    def to_dict(self):
        return {
            "source": self.source,
            "valid_time": self.valid_time,
            "fetched_at": self.fetched_at,
            "levels": [
                {"alt_msl": round(float(a), 1), "wind_east": round(float(u), 2),
                 "wind_north": round(float(v), 2), "temperature_c": round(float(t) - 273.15, 1),
                 "pressure_hpa": round(float(p) / 100, 1)}
                for a, u, v, t, p in zip(self.alt, self.u, self.v, self.T, self.p)
            ],
        }


def wind_components(speed, direction_from_deg):
    """Convert speed and meteorological 'from' direction to (east, north) components."""
    d = np.radians(direction_from_deg)
    return -speed * np.sin(d), -speed * np.cos(d)


def standard_profile(site_alt, surface_wind_speed=0.0, surface_wind_dir=0.0, top=30000.0,
                     source="standard-atmosphere"):
    """ISA temperature/pressure with a 1/7 power-law wind profile from a surface wind."""
    alt = np.concatenate([[site_alt, site_alt + 10.0], np.linspace(site_alt + 50.0, top, 60)])
    T, p, _ = isa(alt)
    agl = np.maximum(alt - site_alt, 1.0)
    speed = surface_wind_speed * np.minimum((agl / 10.0) ** (1 / 7), 2.5)
    u, v = wind_components(speed, surface_wind_dir)
    return WeatherProfile(alt, u, v, T, p, source=source)
