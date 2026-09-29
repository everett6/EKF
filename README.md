Matthew, here is some files, I still have a lot of work to do like combining and simpyfing the code. Change whatever you feel neccesary

# Rocket tracker and landing predictor

`rocket_tracker/` follows a rocket after launch and keeps predicting where it
is going and where it will land, re-forecasting as the weather updates.

* **Extended Kalman Filter** fuses GPS fixes, barometric altitude and the
  accelerometer (magnitude only, so no attitude solution is needed) with a
  physics model: thrust, gravity, drag, drogue and main parachutes, and wind.
  Besides position and velocity it estimates a **wind correction** (how far
  the real wind differs from the forecast) and a **drag scale** (how far the
  real drag and parachutes differ from the configured values).
* **Live weather** – winds, temperature and pressure by altitude from
  [Open-Meteo](https://open-meteo.com) (free, no API key), refreshed in the
  background during the flight (default every 5 min). If the fetch fails it
  keeps the last good profile, or falls back to the standard atmosphere with
  a configured surface wind.
* **Landing prediction** – every second (and immediately after a phase change
  or a weather update) 200 Monte Carlo copies of the current estimate, spread
  by the filter's uncertainty, are flown to the ground. The result is the
  predicted path, apogee, time to landing and landing point with a 95 %
  uncertainty ellipse.
* **Live map** – a small built-in web dashboard (satellite/OSM map, track,
  GPS fixes, predicted path, landing ellipse, altitude chart, weather panel).

## Quick start

```bash
pip install -r requirements.txt
python -m rocket_tracker                 # simulated flight + dashboard at http://127.0.0.1:8000
python -m rocket_tracker --speed 5       # same, 5x faster
python -m rocket_tracker --offline --speed 0 --no-web --plot flight.png   # headless, as fast as possible
```

The simulator flies a "real" rocket that deliberately differs from what the
tracker assumes (extra wind plus gusts, 10 % more drag, 3 % less thrust,
10 % smaller parachutes) so you can watch the filter learn the difference.
At the end it prints the landing-prediction error over time.

### With real telemetry

The tracker reads CSV logs, including the format `DataLoggingGraphics.py`
writes (`sys_time`, `gps_lat`, `gps_lon`, `gps_alt`, `accel_x/y/z` in g).
Optional columns `baro_alt` (m MSL) or `pressure_pa` add the barometer.

```bash
python -m rocket_tracker --follow /home/everett/Programs/Data/datalogger.csv   # live, while the logger runs
python -m rocket_tracker --replay flight.csv --speed 2                           # replay a recorded flight
```

Launch is detected from a >3 g acceleration (or a 20 m climb if there is no
accelerometer). Before launch the barometer is zeroed against the pad.

> **Longitude sign fix.** `DataLogging.py` and `DataLoggingGraphics.py`
> negated longitudes that `pynmea2` had already signed, so west longitudes
> were logged as positive (e.g. `121.85` instead of `-121.85` in
> `sensor_data_ui.csv`). That is fixed; use `--flip-lon` to read logs
> recorded before the fix.

## Configuration

All rocket, site, sensor, weather, simulator and predictor settings have
placeholder defaults in `rocket_tracker/config.py` (default site: the origin
used in `DataLogging.py`). Override any of them with a JSON file:

```bash
python -m rocket_tracker --config examples/rocket.json
```

The values that matter most for prediction are the masses, thrust/burn time
(or a `thrust_curve` of `[t, N]` pairs), body diameter and drag coefficient,
parachute diameters and drag coefficients, main deployment altitude and the
launch site.

## Layout

| File | What it does |
| --- | --- |
| `rocket_tracker/dynamics.py` | Vectorised point-mass flight model and flight phases |
| `rocket_tracker/ekf.py` | Extended Kalman Filter (numerical Jacobians, outlier gating) |
| `rocket_tracker/predictor.py` | Monte Carlo trajectory and landing prediction |
| `rocket_tracker/weather.py` | Open-Meteo client and background refresh |
| `rocket_tracker/atmosphere.py` | Standard atmosphere and altitude-interpolated weather profiles |
| `rocket_tracker/telemetry.py` | Simulated flight and CSV replay/follow sources |
| `rocket_tracker/tracker.py` | Wires everything together, launch detection, dashboard state |
| `rocket_tracker/dashboard.py`, `static/index.html` | Web dashboard |

Run the tests with `python -m pytest`.
