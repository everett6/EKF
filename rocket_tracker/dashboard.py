"""Tiny built-in web dashboard: live map, altitude and weather (stdlib only)."""

import json
import logging
import threading
from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import numpy as np

STATIC = Path(__file__).parent / "static"
log = logging.getLogger(__name__)


def _default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


class _Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, tracker=None, **kw):
        self.tracker = tracker
        super().__init__(*a, directory=str(STATIC), **kw)

    def do_GET(self):
        if self.path.startswith("/api/state"):
            body = json.dumps(self.tracker.snapshot(), default=_default).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def log_message(self, *args):
        pass


def serve(tracker, host="127.0.0.1", port=8000):
    server = ThreadingHTTPServer((host, port), partial(_Handler, tracker=tracker))
    threading.Thread(target=server.serve_forever, name="dashboard", daemon=True).start()
    log.info("Dashboard at http://%s:%d", host, port)
    return server
