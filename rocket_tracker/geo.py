"""WGS84 geodetic <-> local East/North/Up conversions (no external deps)."""

import numpy as np

A = 6378137.0
F = 1 / 298.257223563
E2 = F * (2 - F)


def geodetic_to_ecef(lat, lon, alt):
    lat, lon = np.radians(lat), np.radians(lon)
    n = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
    x = (n + alt) * np.cos(lat) * np.cos(lon)
    y = (n + alt) * np.cos(lat) * np.sin(lon)
    z = (n * (1 - E2) + alt) * np.sin(lat)
    return x, y, z


def ecef_to_geodetic(x, y, z):
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - E2))
    for _ in range(6):
        n = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
        alt = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1 - E2 * n / (n + alt)))
    n = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - n
    return np.degrees(lat), np.degrees(lon), alt


def _rotation(lat0, lon0):
    la, lo = np.radians(lat0), np.radians(lon0)
    return np.array([
        [-np.sin(lo), np.cos(lo), 0.0],
        [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
        [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)],
    ])


def geodetic_to_enu(lat, lon, alt, lat0, lon0, alt0):
    """Return (east, north, up) in metres relative to the reference point."""
    x, y, z = geodetic_to_ecef(lat, lon, alt)
    x0, y0, z0 = geodetic_to_ecef(lat0, lon0, alt0)
    d = np.array([np.subtract(x, x0), np.subtract(y, y0), np.subtract(z, z0)])
    e, n, u = np.tensordot(_rotation(lat0, lon0), d, axes=1)
    return e, n, u


def enu_to_geodetic(e, n, u, lat0, lon0, alt0):
    """Return (lat, lon, alt) for a local ENU offset from the reference point."""
    d = np.tensordot(_rotation(lat0, lon0).T, np.array([e, n, u], dtype=float), axes=1)
    x0, y0, z0 = geodetic_to_ecef(lat0, lon0, alt0)
    return ecef_to_geodetic(d[0] + x0, d[1] + y0, d[2] + z0)
