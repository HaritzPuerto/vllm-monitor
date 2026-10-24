"""Turn a sequence of snapshots into rates and distributions over a time window."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

from . import metrics as m
from .metrics import Histogram, Snapshot


@dataclass
class WindowStats:
    duration: float
    n_samples: int
    # throughput
    gen_tps: float
    prompt_tps: float
    requests_per_s: float
    # scheduler state (averaged over the snapshots in the window)
    running_avg: float
    running_min: float
    running_max: float
    running_busy_avg: float  # mean running while at least one request was running
    busy_frac: float  # fraction of snapshots with >= 1 running request
    waiting_avg: float
    waiting_max: float
    kv_usage_avg: Optional[float]
    kv_usage_max: Optional[float]
    preemptions: float
    prefix_hit_rate: Optional[float]
    spec_acceptance: Optional[float]
    finished: Dict[str, float] = field(default_factory=dict)
    # per-request distributions over the window
    ttft: Optional[Histogram] = None
    itl: Optional[Histogram] = None
    e2e: Optional[Histogram] = None
    queue: Optional[Histogram] = None
    prompt_len: Optional[Histogram] = None
    gen_len: Optional[Histogram] = None

    @property
    def finished_total(self) -> float:
        return sum(self.finished.values())

    @property
    def avg_prompt_len(self) -> Optional[float]:
        if self.prompt_len is not None and self.prompt_len.mean is not None:
            return self.prompt_len.mean
        return self.prompt_tps / self.requests_per_s if self.requests_per_s > 0 else None

    @property
    def avg_gen_len(self) -> Optional[float]:
        if self.gen_len is not None and self.gen_len.mean is not None:
            return self.gen_len.mean
        return self.gen_tps / self.requests_per_s if self.requests_per_s > 0 else None


def _delta(a: Snapshot, b: Snapshot, names) -> float:
    va, vb = a.get(names, 0.0), b.get(names, 0.0)
    return vb if vb < va else vb - va  # a drop means the server restarted


def _hist_delta(a: Snapshot, b: Snapshot, names) -> Optional[Histogram]:
    hb = b.histogram(names)
    if hb is None:
        return None
    ha = a.histogram(names)
    if ha is None or hb.count < ha.count:
        return hb
    return hb - ha


def _ratio(num: float, den: float) -> Optional[float]:
    return num / den if den > 0 else None


def compute_window(snaps: Sequence[Snapshot]) -> Optional[WindowStats]:
    if len(snaps) < 2:
        return None
    a, b = snaps[0], snaps[-1]
    dt = b.t - a.t
    if dt <= 0:
        return None

    running = [s.get(m.RUNNING, 0.0) for s in snaps]
    waiting = [s.get(m.WAITING, 0.0) for s in snaps]
    kv = [v for v in (s.get(m.KV_USAGE) for s in snaps) if v is not None]
    busy = [r for r in running if r > 0]

    fa, fb = a.by_label(m.REQUEST_SUCCESS, "finished_reason"), b.by_label(m.REQUEST_SUCCESS, "finished_reason")
    finished = {k: max(0.0, v - fa.get(k, 0.0)) for k, v in fb.items()}

    return WindowStats(
        duration=dt,
        n_samples=len(snaps),
        gen_tps=_delta(a, b, m.GENERATION_TOKENS) / dt,
        prompt_tps=_delta(a, b, m.PROMPT_TOKENS) / dt,
        requests_per_s=sum(finished.values()) / dt,
        running_avg=sum(running) / len(running),
        running_min=min(running),
        running_max=max(running),
        running_busy_avg=sum(busy) / len(busy) if busy else 0.0,
        busy_frac=len(busy) / len(running),
        waiting_avg=sum(waiting) / len(waiting),
        waiting_max=max(waiting),
        kv_usage_avg=sum(kv) / len(kv) if kv else None,
        kv_usage_max=max(kv) if kv else None,
        preemptions=_delta(a, b, m.PREEMPTIONS),
        prefix_hit_rate=_ratio(_delta(a, b, m.PREFIX_HITS), _delta(a, b, m.PREFIX_QUERIES)),
        spec_acceptance=_ratio(_delta(a, b, m.SPEC_ACCEPTED), _delta(a, b, m.SPEC_DRAFT)),
        finished=finished,
        ttft=_hist_delta(a, b, m.TTFT),
        itl=_hist_delta(a, b, m.ITL),
        e2e=_hist_delta(a, b, m.E2E),
        queue=_hist_delta(a, b, m.QUEUE_TIME),
        prompt_len=_hist_delta(a, b, m.REQ_PROMPT_TOKENS),
        gen_len=_hist_delta(a, b, m.REQ_GEN_TOKENS),
    )
