"""Scrape and parse vLLM's Prometheus ``/metrics`` endpoint.

vLLM has renamed several metrics across releases, so every logical metric below
lists all the names it has been exported under (newest first).
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from prometheus_client.parser import text_string_to_metric_families

# Gauges
RUNNING = ("vllm:num_requests_running",)
WAITING = ("vllm:num_requests_waiting",)
KV_USAGE = ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")

# Counters
PROMPT_TOKENS = ("vllm:prompt_tokens_total",)
GENERATION_TOKENS = ("vllm:generation_tokens_total",)
PREEMPTIONS = ("vllm:num_preemptions_total",)
PREFIX_QUERIES = ("vllm:prefix_cache_queries_total", "vllm:gpu_prefix_cache_queries_total")
PREFIX_HITS = ("vllm:prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits_total")
REQUEST_SUCCESS = ("vllm:request_success_total",)
SPEC_ACCEPTED = ("vllm:spec_decode_num_accepted_tokens_total",)
SPEC_DRAFT = ("vllm:spec_decode_num_draft_tokens_total",)

# Histograms (base names, without _bucket/_sum/_count)
TTFT = ("vllm:time_to_first_token_seconds",)
ITL = ("vllm:inter_token_latency_seconds", "vllm:time_per_output_token_seconds")
E2E = ("vllm:e2e_request_latency_seconds",)
QUEUE_TIME = ("vllm:request_queue_time_seconds",)
PREFILL_TIME = ("vllm:request_prefill_time_seconds",)
DECODE_TIME = ("vllm:request_decode_time_seconds",)
REQ_PROMPT_TOKENS = ("vllm:request_prompt_tokens",)
REQ_GEN_TOKENS = ("vllm:request_generation_tokens",)

# Info gauges: value is always 1, the payload is in the labels.
CACHE_CONFIG = ("vllm:cache_config_info",)

_SCALARS = set(
    RUNNING + WAITING + KV_USAGE + PROMPT_TOKENS + GENERATION_TOKENS + PREEMPTIONS
    + PREFIX_QUERIES + PREFIX_HITS + REQUEST_SUCCESS + SPEC_ACCEPTED + SPEC_DRAFT
)
_HISTOGRAMS = set(
    TTFT + ITL + E2E + QUEUE_TIME + PREFILL_TIME + DECODE_TIME + REQ_PROMPT_TOKENS + REQ_GEN_TOKENS
)
_INFOS = set(CACHE_CONFIG)
# Gauges averaged (not summed) across engines / data-parallel ranks.
_MEAN_GAUGES = set(KV_USAGE)
# Labels kept when aggregating; all others (model_name, engine, ...) are summed over.
_KEPT_LABELS = ("le", "finished_reason")


def _key(name: str, labels: Dict[str, str]) -> str:
    kept = [f"{k}={labels[k]}" for k in _KEPT_LABELS if k in labels]
    return f"{name}{{{','.join(kept)}}}" if kept else name


@dataclass
class Histogram:
    buckets: List[Tuple[float, float]]  # (upper bound, cumulative count), sorted by bound
    sum: float
    count: float

    def __sub__(self, older: "Histogram") -> "Histogram":
        prev = dict(older.buckets)
        buckets = [(le, max(0.0, c - prev.get(le, 0.0))) for le, c in self.buckets]
        return Histogram(buckets, max(0.0, self.sum - older.sum), max(0.0, self.count - older.count))

    @property
    def mean(self) -> Optional[float]:
        return self.sum / self.count if self.count > 0 else None

    def quantile(self, q: float) -> Optional[float]:
        """Prometheus-style quantile estimate (linear interpolation inside a bucket)."""
        if self.count <= 0 or not self.buckets:
            return None
        rank = q * self.count
        lower_le, lower_c = 0.0, 0.0
        for le, c in self.buckets:
            if c >= rank:
                if math.isinf(le):
                    return lower_le
                if c == lower_c:
                    return le
                return lower_le + (le - lower_le) * (rank - lower_c) / (c - lower_c)
            lower_le, lower_c = le, c
        return lower_le


@dataclass
class Snapshot:
    """Metrics at one point in time, aggregated over engines/models."""

    t: float
    values: Dict[str, float]
    info: Dict[str, Dict[str, str]] = field(default_factory=dict)

    def get(self, names: Sequence[str], default: Optional[float] = None) -> Optional[float]:
        for n in names:
            if n in self.values:
                return self.values[n]
        return default

    def by_label(self, names: Sequence[str], label: str) -> Dict[str, float]:
        for n in names:
            prefix = f"{n}{{{label}="
            out = {k[len(prefix):-1]: v for k, v in self.values.items() if k.startswith(prefix)}
            if out:
                return out
        return {}

    def histogram(self, names: Sequence[str]) -> Optional[Histogram]:
        for n in names:
            if f"{n}_count" not in self.values:
                continue
            prefix = f"{n}_bucket{{le="
            buckets = sorted(
                (float(k[len(prefix):-1]), v) for k, v in self.values.items() if k.startswith(prefix)
            )
            return Histogram(buckets, self.values.get(f"{n}_sum", 0.0), self.values[f"{n}_count"])
        return None

    def to_json(self) -> dict:
        return {"t": self.t, "v": self.values}

    @classmethod
    def from_json(cls, d: dict, info: Optional[dict] = None) -> "Snapshot":
        return cls(d["t"], d["v"], info or {})


def parse_metrics(text: str, t: Optional[float] = None) -> Snapshot:
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    info: Dict[str, Dict[str, str]] = {}
    for family in text_string_to_metric_families(text):
        for s in family.samples:
            name = s.name
            if name in _INFOS:
                info[name] = dict(s.labels)
                continue
            base = name.rsplit("_", 1)[0] if name.endswith(("_bucket", "_sum", "_count")) else name
            if name not in _SCALARS and base not in _HISTOGRAMS:
                continue
            k = _key(name, s.labels)
            sums[k] = sums.get(k, 0.0) + s.value
            counts[k] = counts.get(k, 0) + 1
    values = {k: (v / counts[k] if k in _MEAN_GAUGES else v) for k, v in sums.items()}
    return Snapshot(time.time() if t is None else t, values, info)


class VLLMClient:
    """Minimal HTTP client for a vLLM OpenAI-compatible server."""

    def __init__(self, url: str = "http://localhost:8000", timeout: float = 5.0, api_key: Optional[str] = None):
        url = url.rstrip("/")
        for suffix in ("/metrics", "/v1"):
            if url.endswith(suffix):
                url = url[: -len(suffix)]
        if "://" not in url:
            url = "http://" + url
        self.base = url
        self.timeout = timeout
        self.api_key = api_key or os.environ.get("VLLM_API_KEY")

    def _get(self, path: str, auth: bool = False) -> bytes:
        req = urllib.request.Request(self.base + path)
        if auth and self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return r.read()

    def scrape(self) -> Snapshot:
        t = time.time()
        return parse_metrics(self._get("/metrics").decode(), t)

    def models(self) -> List[dict]:
        return json.loads(self._get("/v1/models", auth=True)).get("data", [])
