import json
import threading
import time
import urllib.request
from dataclasses import asdict

import pytest

from rocket_tracker.app import AppState, serve
from rocket_tracker.config import Config


@pytest.fixture
def app_url(tmp_path):
    cfg = Config()
    cfg.weather.provider = "standard"
    cfg.predictor.samples = 40
    app = AppState(cfg, tmp_path / "flights")
    server = serve(app, "127.0.0.1", 0)
    yield app, f"http://127.0.0.1:{server.server_address[1]}"
    if app.session:
        app.session.stop()
    server.shutdown()


def call(url, body=None, raw=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_pages_and_config(app_url):
    app, url = app_url
    code, body = call(url + "/")
    assert code == 200 and b"Rocket Tracker" in body
    for asset in ("/app.js", "/app.css", "/vendor/leaflet/leaflet.js"):
        assert call(url + asset)[0] == 200
    code, body = call(url + "/api/config")
    assert json.loads(body)["rocket"]["dry_mass_kg"] == 12.0
    code, body = call(url + "/api/config", {"rocket": {"dry_mass_kg": 8}})
    assert code == 200 and json.loads(body)["rocket"]["dry_mass_kg"] == 8.0
    code, body = call(url + "/api/config", {"rocket": {"dry_mass_kg": -1}})
    assert code == 400 and b"positive" in body
    code, body = call(url + "/api/config", {"rocket": {"wings": 2}})
    assert code == 400


def test_pad_forecast_appears_before_launch(app_url):
    app, url = app_url
    for _ in range(100):
        state = json.loads(call(url + "/api/state")[1])
        if state.get("pad_forecast"):
            break
        time.sleep(0.2)
    pf = state["pad_forecast"]
    assert state["run"]["state"] == "idle" and not state["launched"]
    assert pf["apogee_agl"] > 1000 and len(pf["ellipse_95"]["polygon"]) > 10


def test_sim_run_upload_replay_and_export(app_url):
    app, url = app_url
    code, body = call(url + "/api/run", {"source": "sim", "speed": 0, "offline": True})
    assert code == 200
    # Setup is locked while a run is active.
    assert call(url + "/api/config", {"rocket": {"dry_mass_kg": 9}})[0] == 409
    for _ in range(600):
        state = json.loads(call(url + "/api/state")[1])
        if state["run"]["state"] in ("finished", "error"):
            break
        time.sleep(0.2)
    assert state["run"]["state"] == "finished", state["run"]
    assert state["state"]["phase"] == "landed" and "truth_landing" in state["run"]
    code, csv_body = call(url + "/api/export.csv")
    lines = csv_body.decode().splitlines()
    assert code == 200 and lines[0].startswith("t_s,lat,lon") and len(lines) > 100

    # Upload the exported bench log from the repo and replay it.
    with open("sensor_data_ui.csv", "rb") as f:
        code, body = call(url + "/api/upload?name=..%2Fbench%20log.csv", raw=f.read())
    name = json.loads(body)["name"]
    assert code == 200 and name == "bench_log.csv"
    files = json.loads(call(url + "/api/files")[1])["files"]
    assert [f["name"] for f in files] == ["bench_log.csv"]
    code, body = call(url + "/api/run", {"source": "replay", "path": name, "speed": 0, "flip_lon": True})
    assert code == 200
    for _ in range(100):
        state = json.loads(call(url + "/api/state")[1])
        if state["run"]["state"] == "finished":
            break
        time.sleep(0.1)
    assert state["run"]["state"] == "finished" and not state["launched"]

    code, body = call(url + "/api/run", {"source": "replay", "path": "missing.csv"})
    assert code == 200
    for _ in range(50):
        state = json.loads(call(url + "/api/state")[1])
        if state["run"]["state"] == "error":
            break
        time.sleep(0.1)
    assert "No such file" in state["run"]["error"]


def test_pause_resume_speed(app_url):
    app, url = app_url
    call(url + "/api/run", {"source": "sim", "speed": 1, "offline": True})
    time.sleep(0.5)
    assert json.loads(call(url + "/api/control", {"action": "pause"})[1])["state"] == "paused"
    assert json.loads(call(url + "/api/control", {"action": "resume"})[1])["state"] == "running"
    assert json.loads(call(url + "/api/control", {"speed": 5})[1])["speed"] == 5.0
    assert json.loads(call(url + "/api/control", {"action": "stop"})[1])["state"] == "stopped"
