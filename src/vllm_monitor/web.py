"""Web dashboard: a background scraper plus a tiny HTTP server serving one HTML page and a JSON API."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import math
from typing import Any, Deque, List, Optional, Union
from urllib.parse import parse_qs, urlparse

from .analysis import Analysis, Setup, analyze
from .metrics import GENERATION_TOKENS, PROMPT_TOKENS, KV_USAGE, RUNNING, WAITING, Histogram, Snapshot, VLLMClient
from .predict import Prediction
from .recorder import Recorder
from .stats import compute_window


def _clean(o: Any) -> Any:
    """Replace NaN/inf (invalid JSON) with None."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    return o


def _hist(h: Optional[Histogram]) -> Optional[dict]:
    if h is None or h.count <= 0:
        return None
    return {"p50": h.quantile(0.5), "p95": h.quantile(0.95), "mean": h.mean, "count": h.count}


def _pred(p: Optional[Prediction]) -> Optional[dict]:
    return None if p is None else {**asdict(p), "capped": p.capped}


def analysis_json(an: Analysis) -> dict:
    s = an.stats
    stats = {k: getattr(s, k) for k in (
        "duration", "gen_tps", "prompt_tps", "requests_per_s", "running_avg", "running_max", "running_busy_avg",
        "busy_frac", "waiting_avg", "waiting_max", "kv_usage_avg", "kv_usage_max", "preemptions",
        "prefix_hit_rate", "spec_acceptance", "finished", "avg_prompt_len", "avg_gen_len")}
    stats.update({k: _hist(getattr(s, k)) for k in ("ttft", "itl", "e2e", "queue", "prompt_len", "gen_len")})
    return {
        "stats": stats,
        "expected": _pred(an.expected),
        "best": _pred(an.best),
        "knee": _pred(an.knee),
        "expected_now": an.expected.output_tps * s.busy_frac if an.expected else None,
        "efficiency": an.efficiency,
        "verdict": an.verdict,
        "advice": [asdict(a) for a in an.advice],
    }


def setup_json(setup: Setup) -> dict:
    return {
        "url": setup.url, "model_id": setup.model_id, "max_model_len": setup.max_model_len,
        "gpu": setup.gpu.name if setup.gpu else None, "tp": setup.tp, "notes": setup.notes,
        "gpu_memory_utilization": setup.gpu_memory_utilization, "kv_capacity_tokens": setup.kv_capacity_tokens,
        "kv_cache_dtype": setup.kv_cache_dtype, "prefix_caching": setup.prefix_caching,
        "params_b": setup.model.total_params / 1e9 if setup.model else None,
    }


def chart_point(prev: Optional[Snapshot], snap: Snapshot, analysis: Optional[dict]) -> Optional[dict]:
    """One sample of the page's time series: rates since `prev` and gauges at `snap`."""
    if prev is None or snap.t <= prev.t:
        return None
    dt = snap.t - prev.t
    rate = lambda names: max(0.0, (snap.get(names, 0.0) - prev.get(names, 0.0)) / dt)  # noqa: E731
    return {
        "t": snap.t,
        "gen": rate(GENERATION_TOKENS),
        "prompt": rate(PROMPT_TOKENS),
        "running": snap.get(RUNNING, 0.0),
        "waiting": snap.get(WAITING, 0.0),
        "kv": snap.get(KV_USAGE),
        "expected": analysis["expected_now"] if analysis else None,
    }


class MonitorService:
    """Scrapes the server in a background thread and keeps a time series for the page."""

    def __init__(self, client: VLLMClient, setup: Setup, interval: float = 2.0, window: float = 60.0,
                 history: float = 3600.0, recorder: Optional[Recorder] = None):
        self.client, self.setup, self.interval, self.window = client, setup, interval, window
        self.recorder = recorder
        self.snaps: Deque[Snapshot] = deque()
        self.points: Deque[dict] = deque(maxlen=max(10, int(history / interval)))
        self.analysis: Optional[dict] = None
        self.error: Optional[str] = None
        self.last_ok: Optional[float] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def tick(self) -> None:
        try:
            snap = self.client.scrape()
        except Exception as e:
            with self._lock:
                self.error = f"{type(e).__name__}: {e}"
            return
        if self.recorder:
            self.recorder.write(snap)
        with self._lock:
            prev = self.snaps[-1] if self.snaps else None
            self.snaps.append(snap)
            while len(self.snaps) > 2 and self.snaps[-1].t - self.snaps[0].t > self.window:
                self.snaps.popleft()
            stats = compute_window(list(self.snaps))
            an = analyze(stats, self.setup) if stats else None
            self.analysis = analysis_json(an) if an else None
            self.error, self.last_ok = None, snap.t
            p = chart_point(prev, snap, self.analysis)
            if p:
                self.points.append(p)

    def _loop(self) -> None:
        while not self._stop.is_set():
            t0 = time.time()
            self.tick()
            self._stop.wait(max(0.0, self.interval - (time.time() - t0)))

    def start(self) -> "MonitorService":
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def state(self, since: float = 0.0) -> dict:
        with self._lock:
            return {
                "now": time.time(),
                "interval": self.interval,
                "window": self.window,
                "setup": setup_json(self.setup),
                "analysis": self.analysis,
                "error": self.error,
                "last_ok": self.last_ok,
                "points": [p for p in self.points if p["t"] > since],
            }


class ReplayService:
    """Serves a finished recording: the whole run as a time series, plus analysis of any time range in it."""

    def __init__(self, snaps: List[Snapshot], setup: Setup, window: float = 60.0, source: Optional[str] = None):
        self.snaps, self.setup, self.window, self.source = snaps, setup, window, source
        gaps = sorted(b.t - a.t for a, b in zip(snaps, snaps[1:]) if b.t > a.t)
        self.interval = gaps[len(gaps) // 2] if gaps else 1.0
        # the "expected" line uses the same trailing window as the live dashboard
        self.points: List[dict] = []
        lo = 0
        for i in range(1, len(snaps)):
            while i - lo > 1 and snaps[i].t - snaps[lo].t > window:
                lo += 1
            stats = compute_window(snaps[lo:i + 1])
            an = analysis_json(analyze(stats, setup)) if stats else None
            p = chart_point(snaps[i - 1], snaps[i], an)
            if p:
                self.points.append(p)
        self.overall = self.analyze_range(snaps[0].t, snaps[-1].t)

    def analyze_range(self, t0: float, t1: float) -> Optional[dict]:
        stats = compute_window([s for s in self.snaps if t0 <= s.t <= t1])
        return analysis_json(analyze(stats, self.setup)) if stats else None

    def start(self) -> "ReplayService":
        return self

    def stop(self) -> None:
        pass

    def state(self, since: float = 0.0) -> dict:
        start, end = self.snaps[0].t, self.snaps[-1].t
        return {
            "mode": "replay", "source": self.source, "start": start, "end": end,
            "now": end,
            "interval": self.interval,
            "window": self.window,
            "setup": setup_json(self.setup),
            "analysis": self.overall,
            "error": None,
            "last_ok": end,
            "points": [p for p in self.points if p["t"] > since],
        }


def _page() -> bytes:
    return files("vllm_monitor").joinpath("static/index.html").read_bytes()


def make_server(service: Union[MonitorService, ReplayService], host: str = "127.0.0.1",
                port: int = 8080) -> ThreadingHTTPServer:
    page = _page()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: bytes, ctype: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            u = urlparse(self.path)
            if u.path in ("/", "/index.html"):
                self._send(page, "text/html; charset=utf-8")
            elif u.path == "/api/state":
                since = float(parse_qs(u.query).get("since", ["0"])[0] or 0)
                self._send(json.dumps(_clean(service.state(since))).encode(), "application/json")
            elif u.path == "/api/range" and isinstance(service, ReplayService):
                q = parse_qs(u.query)
                try:
                    t0, t1 = float(q["t0"][0]), float(q["t1"][0])
                except (KeyError, ValueError):
                    self.send_error(400, "need numeric t0 and t1")
                    return
                body = {"t0": t0, "t1": t1, "analysis": service.analyze_range(t0, t1)}
                self._send(json.dumps(_clean(body)).encode(), "application/json")
            else:
                self.send_error(404)

        def log_message(self, *args):
            pass

    return ThreadingHTTPServer((host, port), Handler)
