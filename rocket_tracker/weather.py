"""Live weather: winds, temperature and pressure aloft from Open-Meteo.

Open-Meteo (https://open-meteo.com) is free and needs no API key. The
service refreshes the profile in a background thread; whenever a fetch
fails it keeps the last good profile, or falls back to a standard
atmosphere with the configured surface wind.
"""

import json
import logging
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import numpy as np

from .atmosphere import WeatherProfile, standard_profile, wind_components

log = logging.getLogger(__name__)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
PRESSURE_LEVELS = [1000, 975, 950, 925, 900, 850, 800, 700, 600, 500, 400, 300, 250, 200]


def build_open_meteo_url(lat, lon):
    hourly = []
    for lvl in PRESSURE_LEVELS:
        for var in ("temperature", "wind_speed", "wind_direction", "geopotential_height"):
            hourly.append(f"{var}_{lvl}hPa")
    params = {
        "latitude": f"{lat:.5f}",
        "longitude": f"{lon:.5f}",
        "hourly": ",".join(hourly),
        "current": "temperature_2m,surface_pressure,wind_speed_10m,wind_direction_10m",
        "wind_speed_unit": "ms",
        "forecast_days": 2,
        "past_days": 1,
        "timezone": "GMT",
    }
    return OPEN_METEO_URL + "?" + urllib.parse.urlencode(params)


def parse_open_meteo(data, site_alt, when=None):
    """Turn an Open-Meteo JSON response into a WeatherProfile for the hour nearest `when`."""
    when = when or datetime.now(timezone.utc)
    hourly = data["hourly"]
    times = [datetime.fromisoformat(s).replace(tzinfo=timezone.utc) for s in hourly["time"]]
    idx = int(np.argmin([abs((tt - when).total_seconds()) for tt in times]))
    cur = data.get("current", {})
    sfc_alt = site_alt + 10.0
    alts, us, vs, Ts, ps = [], [], [], [], []
    if cur.get("wind_speed_10m") is not None:
        u, v = wind_components(cur["wind_speed_10m"], cur["wind_direction_10m"])
        alts.append(sfc_alt)
        us.append(u)
        vs.append(v)
        Ts.append(cur["temperature_2m"] + 273.15)
        ps.append(cur["surface_pressure"] * 100.0)
    for lvl in PRESSURE_LEVELS:
        try:
            h = hourly[f"geopotential_height_{lvl}hPa"][idx]
            spd = hourly[f"wind_speed_{lvl}hPa"][idx]
            wd = hourly[f"wind_direction_{lvl}hPa"][idx]
            tc = hourly[f"temperature_{lvl}hPa"][idx]
        except (KeyError, IndexError):
            continue
        if None in (h, spd, wd, tc) or h <= sfc_alt + 20.0:
            continue  # level missing or below ground at this site
        u, v = wind_components(spd, wd)
        alts.append(h)
        us.append(u)
        vs.append(v)
        Ts.append(tc + 273.15)
        ps.append(lvl * 100.0)
    if len(alts) < 3:
        raise ValueError("Open-Meteo response had too few usable levels")
    if alts[0] > site_alt:
        # Anchor the ground so low altitudes don't extrapolate oddly.
        alts.insert(0, site_alt)
        us.insert(0, us[0] * 0.8)
        vs.insert(0, vs[0] * 0.8)
        Ts.insert(0, Ts[0])
        ps.insert(0, ps[0] * np.exp(9.80665 * (alts[1] - site_alt) / (287.05 * Ts[0])))
    return WeatherProfile(alts, us, vs, Ts, ps, source="open-meteo",
                          valid_time=times[idx].isoformat(),
                          fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))


def fetch_open_meteo(lat, lon, site_alt, timeout=10.0):
    req = urllib.request.Request(build_open_meteo_url(lat, lon),
                                 headers={"User-Agent": "rocket-tracker/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    return parse_open_meteo(data, site_alt)


class WeatherService:
    """Holds the current weather profile and keeps it fresh in the background."""

    def __init__(self, site, cfg, fetcher=fetch_open_meteo):
        self.site = site
        self.cfg = cfg
        self.fetcher = fetcher
        self._lock = threading.Lock()
        self._profile = self._fallback()
        self.last_error = None
        self.updates = 0
        self._stop = threading.Event()
        self._thread = None
        self.listeners = []

    def _fallback(self):
        return standard_profile(self.site.altitude_m, self.cfg.surface_wind_speed,
                                self.cfg.surface_wind_dir_deg)

    @property
    def profile(self):
        with self._lock:
            return self._profile

    def set_profile(self, profile):
        with self._lock:
            self._profile = profile
            self.updates += 1
        for fn in self.listeners:
            fn(profile)

    def refresh(self):
        if self.cfg.provider != "open-meteo":
            return False
        try:
            prof = self.fetcher(self.site.latitude, self.site.longitude, self.site.altitude_m)
        except Exception as e:  # network down, API change, etc.
            self.last_error = f"{type(e).__name__}: {e}"
            log.warning("Weather fetch failed, keeping %s profile (%s)",
                        self.profile.source, self.last_error)
            return False
        self.last_error = None
        self.set_profile(prof)
        log.info("Weather updated from Open-Meteo (valid %s)", prof.valid_time)
        return True

    def start(self):
        if self.cfg.provider != "open-meteo":
            return
        self.refresh()
        self._thread = threading.Thread(target=self._run, name="weather", daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.wait(self.cfg.refresh_s):
            self.refresh()

    def stop(self):
        self._stop.set()

    def status(self):
        p = self.profile
        return {"source": p.source, "valid_time": p.valid_time, "fetched_at": p.fetched_at,
                "updates": self.updates, "error": self.last_error,
                "refresh_s": self.cfg.refresh_s}
