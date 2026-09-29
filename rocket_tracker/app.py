"""Rocket Tracker app: a local web application with setup, run controls and live views.

    python -m rocket_tracker app            # opens the app in a window or your browser

Everything runs on this computer. The page talks to a small JSON API:

    GET  /api/state                 live flight state, run status, weather, pad forecast
    GET  /api/config                current configuration
    POST /api/config                update configuration (partial dict), 409 while a run is active
    POST /api/config/reset          restore defaults
    POST /api/run                   {"source": "sim"|"replay"|"follow", "path", "speed", "offline", "flip_lon"}
    POST /api/control               {"action": "pause"|"resume"|"stop"} or {"speed": 5}
    POST /api/weather/refresh       fetch fresh weather now
    GET  /api/files                 CSV logs available for replay
    POST /api/upload?name=f.csv     upload a CSV log (request body is the file)
    GET  /api/export.csv            the filtered track and predictions of the current run
    GET  /tiles/<layer>/<z>/<x>/<y> map tiles, cached on disk so the map keeps working offline
"""

import argparse
import csv
import io
import json
import logging
import os
import re
import threading
import time
import urllib.request
import webbrowser
from dataclasses import asdict
from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np

from .config import Config, load_config, merge_config
from .dynamics import PHASE_NAMES
from .session import FlightSession
from .dynamics import BOOST
from .tracker import Tracker, _predict_in_process
from .weather import WeatherService

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
CACHE = Path(os.environ.get("ROCKET_TRACKER_CACHE", Path.home() / ".cache" / "rocket_tracker"))
TILE_SOURCES = {
    "osm": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    "satellite": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
}
MAX_UPLOAD = 200 * 1024 * 1024
_TILE_MISSES = {}  # url -> time of last failed fetch, so offline maps don't keep retrying


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


class AppState:
    def __init__(self, cfg, data_dir):
        self.cfg = cfg
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.session = None
        self.lock = threading.Lock()
        # Idle weather + tracker give the pad forecast before any run starts.
        self._idle = None
        self._pad = None
        self._pad_key = None
        self._pad_future = None
        from concurrent.futures import ProcessPoolExecutor
        self._pad_pool = ProcessPoolExecutor(max_workers=1)
        self._rebuild_idle()

    # --- idle / pad forecast -------------------------------------------------------
    def _rebuild_idle(self):
        if self._idle:
            self._idle[0].stop()
        ws = WeatherService(self.cfg.site, self.cfg.weather)
        threading.Thread(target=ws.start, daemon=True).start()
        self._idle = (ws, Tracker(self.cfg, ws))
        self._pad = None
        self._pad_key = None

    def pad_forecast(self):
        """Latest 'if launched now' forecast; recomputed in a worker process when inputs change."""
        ws, tr = self._active_weather_tracker()
        key = (id(tr), ws.updates)
        fut = self._pad_future
        if fut is not None and fut.done():
            self._pad_future = None
            try:
                pred_key, result = fut.result()
                if pred_key == key:
                    self._pad = tr.prediction_dict(result)
            except Exception:
                log.exception("Pad forecast failed")
        if self._pad_key != key and self._pad_future is None:
            self._pad_key = key
            x0, P0 = tr.pad_state()
            job = self._pad_pool.submit(_predict_in_process, tr.cfg.rocket, tr.site.altitude_m,
                                        ws.profile, x0, P0, 0.0, BOOST, tr.cfg.predictor, 3)
            self._pad_future = _Keyed(job, key)
        return self._pad

    def _active_weather_tracker(self):
        if self.session is not None:
            return self.session.weather, self.session.tracker
        return self._idle

    @property
    def running(self):
        return self.session is not None and self.session.state in ("starting", "running", "paused")

    # --- actions -------------------------------------------------------------------
    def update_config(self, data):
        if self.running:
            raise PermissionError("Stop the current run before changing the setup")
        self.cfg = merge_config(self.cfg, data)
        self._rebuild_idle()

    def start_run(self, req):
        source = req.get("source", "sim")
        if source not in ("sim", "replay", "follow"):
            raise ValueError("source must be sim, replay or follow")
        path = req.get("path")
        if source == "replay" and path and not os.path.isabs(path):
            path = str(self.data_dir / Path(path).name)
        with self.lock:
            if self.session is not None:
                self.session.stop()
            self.session = FlightSession(
                self.cfg, source=source, path=path, speed=float(req.get("speed", 1.0)),
                offline=bool(req.get("offline", False)), flip_lon=bool(req.get("flip_lon", False)),
                accel_units=req.get("accel_units", "g")).start()
            self._pad = None
        return self.session.status()

    def control(self, req):
        s = self.session
        if s is None:
            raise ValueError("Nothing is running")
        if "speed" in req:
            s.set_speed(req["speed"])
        action = req.get("action")
        if action == "pause":
            s.pause(True)
        elif action == "resume":
            s.pause(False)
        elif action == "stop":
            s.stop()
        return s.status()

    def refresh_weather(self):
        ws, _ = self._active_weather_tracker()
        if ws.cfg.provider != "open-meteo":
            return {"ok": False, "error": "Weather is set to offline (standard atmosphere)"}
        ok = ws.refresh()
        return {"ok": ok, "error": ws.last_error}

    def files(self):
        out = []
        for p in sorted(self.data_dir.glob("*.csv")):
            st = p.stat()
            out.append({"name": p.name, "size": st.st_size, "modified": st.st_mtime})
        return out

    def save_upload(self, name, body):
        name = Path(name or "upload.csv").name
        name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
        if not name.lower().endswith(".csv"):
            name += ".csv"
        (self.data_dir / name).write_bytes(body)
        return {"name": name, "size": len(body)}

    # --- state ---------------------------------------------------------------------
    def snapshot(self):
        ws, tr = self._active_weather_tracker()
        out = tr.snapshot()
        out["run"] = self.session.status() if self.session else {"state": "idle"}
        out["phases"] = [PHASE_NAMES[k] for k in sorted(PHASE_NAMES)]
        if not out.get("launched"):
            out["pad_forecast"] = self.pad_forecast()
        return out

    def export_csv(self):
        _, tr = self._active_weather_tracker()
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["t_s", "lat", "lon", "alt_agl_m", "speed_mps", "phase",
                    "pred_landing_lat", "pred_landing_lon", "pred_ellipse_major_m"])
        preds = list(tr.predictions)
        j = 0
        for t, e, n, u, ph, spd in list(tr.track):
            while j + 1 < len(preds) and preds[j + 1].t <= t:
                j += 1
            lat, lon, _ = tr._ll(e, n, u)
            row = [round(t, 2), f"{lat:.7f}", f"{lon:.7f}", round(u, 2), round(spd, 2), PHASE_NAMES[ph]]
            if preds and preds[j].t <= t:
                plat, plon, _ = tr._ll(*preds[j].landing_enu)
                row += [f"{plat:.7f}", f"{plon:.7f}", round(preds[j].ellipse[0], 1)]
            w.writerow(row)
        return buf.getvalue().encode()


class _Keyed:
    """A future whose result is tagged with the inputs it was computed for."""

    def __init__(self, fut, key):
        self.fut, self.key = fut, key

    def done(self):
        return self.fut.done()

    def result(self):
        return self.key, self.fut.result()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, app=None, **kw):
        self.app = app
        super().__init__(*a, directory=str(STATIC), **kw)

    def log_message(self, *args):
        pass

    def _send(self, code, body, ctype="application/json", extra=None):
        if not isinstance(body, (bytes, bytearray)):
            body = json.dumps(body, default=_json_default).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _body(self, limit=1024 * 1024):
        n = int(self.headers.get("Content-Length") or 0)
        if n > limit:
            raise ValueError("Request too large")
        return self.rfile.read(n) if n else b""

    def _json_body(self):
        raw = self._body()
        return json.loads(raw) if raw else {}

    def do_GET(self):
        url = urlparse(self.path)
        p = url.path
        try:
            if p == "/api/state":
                return self._send(200, self.app.snapshot())
            if p == "/api/config":
                return self._send(200, asdict(self.app.cfg))
            if p == "/api/files":
                return self._send(200, {"dir": str(self.app.data_dir), "files": self.app.files()})
            if p == "/api/export.csv":
                return self._send(200, self.app.export_csv(), "text/csv",
                                  {"Content-Disposition": 'attachment; filename="flight_track.csv"'})
            if p.startswith("/tiles/"):
                return self._tile(p)
        except (BrokenPipeError, ConnectionResetError):
            return None
        except Exception as e:
            log.exception("GET %s failed", p)
            return self._send(500, {"error": str(e)})
        return super().do_GET()

    def do_POST(self):
        url = urlparse(self.path)
        p = url.path
        try:
            if p == "/api/config":
                self.app.update_config(self._json_body())
                return self._send(200, asdict(self.app.cfg))
            if p == "/api/config/reset":
                self.app.update_config(asdict(Config()))
                return self._send(200, asdict(self.app.cfg))
            if p == "/api/run":
                return self._send(200, self.app.start_run(self._json_body()))
            if p == "/api/control":
                return self._send(200, self.app.control(self._json_body()))
            if p == "/api/weather/refresh":
                return self._send(200, self.app.refresh_weather())
            if p == "/api/upload":
                name = parse_qs(url.query).get("name", ["upload.csv"])[0]
                return self._send(200, self.app.save_upload(name, self._body(MAX_UPLOAD)))
        except PermissionError as e:
            return self._send(409, {"error": str(e)})
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
            return self._send(400, {"error": str(e).strip("'\"")})
        except Exception as e:
            log.exception("POST %s failed", p)
            return self._send(500, {"error": str(e)})
        return self._send(404, {"error": "not found"})

    def _tile(self, path):
        m = re.fullmatch(r"/tiles/(\w+)/(\d+)/(\d+)/(\d+)\.?\w*", path)
        if not m or m.group(1) not in TILE_SOURCES:
            return self._send(404, {"error": "unknown tile"})
        layer, z, x, y = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        f = CACHE / "tiles" / layer / str(z) / str(x) / f"{y}.img"
        if not f.exists():
            url = TILE_SOURCES[layer].format(z=z, x=x, y=y)
            if time.time() - _TILE_MISSES.get(url, 0) < 60:
                return self._send(404, {"error": "tile unavailable offline"})
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "rocket-tracker/0.2 (local app)"})
                with urllib.request.urlopen(req, timeout=8) as r:
                    data = r.read()
            except Exception:
                _TILE_MISSES[url] = time.time()
                return self._send(404, {"error": "tile unavailable offline"})
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(data)
        data = f.read_bytes()
        ctype = "image/png" if data[:4] == b"\x89PNG" else "image/jpeg"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "max-age=86400")
        self.end_headers()
        self.wfile.write(data)


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        # A browser closing a connection mid-response is routine, not an error.
        import sys
        if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


def serve(app, host="127.0.0.1", port=8000):
    server = _Server((host, port), partial(Handler, app=app))
    threading.Thread(target=server.serve_forever, name="app-http", daemon=True).start()
    return server


def open_window(url):
    """Open the app in its own window if pywebview is installed, else in the browser."""
    try:
        import webview  # optional: pip install pywebview
    except ImportError:
        webbrowser.open(url)
        return False
    webview.create_window("Rocket Tracker", url, width=1400, height=900)
    webview.start()
    return True


def main(argv=None):
    p = argparse.ArgumentParser(prog="rocket_tracker app", description="Rocket Tracker app")
    p.add_argument("--config", help="JSON configuration to start from")
    p.add_argument("--data-dir", default="flights", help="folder for uploaded/replayable CSV logs")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-open", action="store_true", help="don't open a window/browser")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    app = AppState(load_config(args.config), args.data_dir)
    serve(app, args.host, args.port)
    url = f"http://{args.host}:{args.port}/"
    print(f"Rocket Tracker running at {url}  (Ctrl+C to quit)")
    try:
        if args.no_open or not open_window(url):
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        if app.session:
            app.session.stop()
