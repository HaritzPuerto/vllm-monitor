"""Compare observed throughput with the estimate and turn symptoms into advice."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

from .hardware import GPUSpec
from .model import ModelSpec
from .predict import Assumptions, Estimator, Prediction
from .stats import WindowStats

GOOD_EFF, SLOW_EFF = 0.6, 0.35


@dataclass
class Setup:
    """What we know about the deployment being monitored."""

    url: Optional[str] = None
    model_id: Optional[str] = None
    max_model_len: Optional[int] = None
    cache_info: Dict[str, str] = field(default_factory=dict)
    gpu: Optional[GPUSpec] = None
    tp: int = 1
    model: Optional[ModelSpec] = None
    max_num_seqs: Optional[int] = None
    assumptions: Optional[Assumptions] = None
    notes: List[str] = field(default_factory=list)  # why some features are unavailable

    def _cache(self, key: str) -> Optional[str]:
        v = self.cache_info.get(key)
        return None if v in (None, "", "None") else v

    @property
    def gpu_memory_utilization(self) -> Optional[float]:
        v = self._cache("gpu_memory_utilization")
        return float(v) if v else None

    @property
    def kv_capacity_tokens(self) -> Optional[float]:
        blocks, size = self._cache("num_gpu_blocks"), self._cache("block_size")
        return float(blocks) * float(size) if blocks and size else None

    @property
    def kv_cache_dtype(self) -> Optional[str]:
        return self._cache("cache_dtype")

    @property
    def prefix_caching(self) -> Optional[bool]:
        v = self._cache("enable_prefix_caching")
        return None if v is None else v.lower() == "true"

    def estimator(self, assumptions: Optional[Assumptions] = None) -> Optional[Estimator]:
        if self.model is None or self.gpu is None:
            return None
        a = replace(assumptions or self.assumptions or Assumptions())
        if self.gpu_memory_utilization:
            a.gpu_memory_utilization = self.gpu_memory_utilization
        if self.kv_cache_dtype and "fp8" in self.kv_cache_dtype:
            a.kv_dtype_bytes = 1.0
        if self.max_num_seqs:
            a.max_num_seqs = self.max_num_seqs
        return Estimator(self.model, self.gpu, self.tp, a, kv_capacity_tokens=self.kv_capacity_tokens)


@dataclass
class Advice:
    level: str  # "ok" | "info" | "warn" | "crit"
    title: str
    detail: str = ""
    actions: List[str] = field(default_factory=list)


@dataclass
class Analysis:
    stats: WindowStats
    expected: Optional[Prediction] = None  # estimate at the observed load
    best: Optional[Prediction] = None  # estimate at full concurrency
    knee: Optional[Prediction] = None  # smallest concurrency close to best
    efficiency: Optional[float] = None  # observed / expected output tok/s
    verdict: str = "unknown"  # good | fair | slow | idle | unknown
    advice: List[Advice] = field(default_factory=list)


def analyze(stats: WindowStats, setup: Setup, estimator: Optional[Estimator] = None) -> Analysis:
    est = estimator or setup.estimator()
    an = Analysis(stats)
    busy = stats.busy_frac > 0.05 and stats.gen_tps > 0
    I, O = stats.avg_prompt_len, stats.avg_gen_len
    if est is not None and busy and I and O:
        an.expected = est.predict(max(stats.running_busy_avg, 1.0), I, O)
        an.best = est.best(I, O)
        an.knee = est.knee(I, O)
        # The estimate assumes the GPU is always busy; scale by the observed busy fraction.
        an.efficiency = stats.gen_tps / (an.expected.output_tps * stats.busy_frac)
    if not busy:
        an.verdict = "idle"
    elif an.efficiency is not None:
        an.verdict = "good" if an.efficiency >= GOOD_EFF else "fair" if an.efficiency >= SLOW_EFF else "slow"
    an.advice = advise(an, setup)
    return an


def _kv_actions(setup: Setup, stats: WindowStats) -> List[str]:
    acts = []
    gmu = setup.gpu_memory_utilization
    if gmu is None or gmu < 0.95:
        cur = f" (currently {gmu:g})" if gmu else ""
        acts.append(f"Raise --gpu-memory-utilization{cur} towards 0.95 if nothing else shares the GPU.")
    if not (setup.kv_cache_dtype and "fp8" in setup.kv_cache_dtype):
        acts.append("Use --kv-cache-dtype fp8 to roughly double KV capacity (small quality impact).")
    p99 = None
    if stats.prompt_len and stats.gen_len:
        q1, q2 = stats.prompt_len.quantile(0.99), stats.gen_len.quantile(0.99)
        if q1 is not None and q2 is not None:
            p99 = q1 + q2
    if setup.max_model_len and p99 and setup.max_model_len > 2 * p99:
        acts.append(
            f"Lower --max-model-len (now {setup.max_model_len}); p99 request length is ~{p99:.0f} tokens."
        )
    acts.append("Add GPUs: more --tensor-parallel-size (more KV room) or another replica / --data-parallel-size.")
    return acts


def advise(an: Analysis, setup: Setup) -> List[Advice]:
    s = an.stats
    out: List[Advice] = []

    if an.verdict == "idle":
        out.append(Advice("info", "Server is idle", "No requests were running in this window."))
        return out

    kv = s.kv_usage_avg
    queueing = s.waiting_avg >= 1

    # -- the GPU is waiting on the client ------------------------------------------------
    if s.busy_frac < 0.8:
        out.append(Advice(
            "warn", f"GPU idle {100 * (1 - s.busy_frac):.0f}% of the time",
            "There are gaps with no running requests, so the client is not keeping the server fed.",
            ["Submit requests continuously (async client / worker pool) instead of in synchronous batches.",
             "If you preprocess between batches, overlap it with generation."],
        ))
    low_conc = s.waiting_avg < 0.5 and (kv is None or kv < 0.5)
    if an.knee is not None:
        low_conc = low_conc and s.running_busy_avg < 0.5 * an.knee.concurrency
    else:
        low_conc = low_conc and s.running_busy_avg < 16
    if low_conc:
        detail = f"Only ~{s.running_busy_avg:.0f} requests in flight on average and nothing is queued."
        acts = []
        if an.knee is not None and an.expected is not None and an.best is not None:
            gain = an.knee.output_tps / an.expected.output_tps
            detail += (
                f" At ~{an.knee.concurrency:.0f} concurrent requests the estimate is "
                f"{an.knee.output_tps:,.0f} tok/s ({gain:.1f}x now); max ~{an.best.output_tps:,.0f} tok/s "
                f"at {an.best.concurrency:.0f}."
            )
            acts.append(f"Send ~{an.knee.concurrency:.0f}+ requests concurrently "
                        f"(e.g. asyncio.Semaphore({an.knee.concurrency:.0f}), or pass all prompts to one call).")
        else:
            acts.append("Send more requests concurrently: vLLM batches them, throughput grows almost linearly "
                        "with concurrency until the GPU saturates.")
        out.append(Advice("warn", "Client-bound: concurrency too low", detail, acts))

    # -- server-side limits --------------------------------------------------------------
    if queueing and kv is not None and kv > 0.85:
        out.append(Advice(
            "warn", "KV cache full: requests are queueing",
            f"KV usage {100 * kv:.0f}%, {s.waiting_avg:.1f} requests waiting on average.",
            _kv_actions(setup, s),
        ))
    elif queueing and (kv is None or kv < 0.6) and s.running_max - s.running_min <= 1 and s.running_avg >= 4:
        limit = f"--max-num-seqs (now {setup.max_num_seqs})" if setup.max_num_seqs else "--max-num-seqs"
        out.append(Advice(
            "warn", f"Scheduler limit: running plateaus at {s.running_max:.0f}",
            "Requests wait although the KV cache has room, so a scheduler cap is binding.",
            [f"Raise {limit}.",
             "With long prompts, also raise --max-num-batched-tokens (e.g. 8192-16384)."],
        ))
    elif queueing and s.queue and (s.queue.quantile(0.5) or 0) > 1:
        out.append(Advice(
            "info", "Server saturated",
            f"Median queue time {s.queue.quantile(0.5):.1f}s. Throughput is fine, latency suffers.",
            ["If latency matters, add replicas; if only throughput matters, nothing to do."],
        ))

    if s.preemptions > 0:
        kv_advice_given = any(a.title.startswith("KV cache full") for a in out)
        out.append(Advice(
            "crit" if s.preemptions > 10 else "warn", f"{s.preemptions:.0f} preemptions",
            "Requests were evicted from the KV cache and must be recomputed (wasted GPU work).",
            ["Lower --max-num-seqs so admitted requests fit."] + ([] if kv_advice_given else _kv_actions(setup, s)[:2]),
        ))

    # -- efficiency vs estimate ----------------------------------------------------------
    if an.efficiency is not None and an.efficiency < SLOW_EFF and s.busy_frac >= 0.5:
        out.append(Advice(
            "warn", f"Slower than expected: {100 * an.efficiency:.0f}% of estimate",
            f"Observed {s.gen_tps:,.0f} tok/s vs ~{an.expected.output_tps * s.busy_frac:,.0f} tok/s "
            f"expected for {setup.model_id or 'this model'} on {setup.tp}x {setup.gpu.name}.",
            ["Check the server was not started with --enforce-eager (disables CUDA graphs).",
             "CPU-side bottlenecks: guided/structured decoding, logprobs, n>1, very large batches of "
             "tiny requests, slow detokenization.",
             "Check nvidia-smi: other processes on the GPU, thermal/power throttling.",
             "Make sure --gpu/--tp passed to vllm-monitor match the deployment."],
        ))

    # -- workload shape ------------------------------------------------------------------
    if s.gen_tps > 0 and s.prompt_tps > 5 * s.gen_tps:
        acts = ["Raise --max-num-batched-tokens (e.g. 8192-16384) for bigger prefill chunks."]
        hit = s.prefix_hit_rate
        if setup.prefix_caching is False:
            acts.insert(0, "Enable --enable-prefix-caching if prompts share a prefix (system prompt, few-shot).")
        elif hit is not None and hit < 0.2:
            acts.insert(0, f"Prefix-cache hit rate is {100 * hit:.0f}%: put shared text (system prompt, "
                           "few-shot examples) first and keep it byte-identical across requests.")
        out.append(Advice(
            "info", f"Prefill-heavy workload ({s.prompt_tps / s.gen_tps:.0f}:1 prompt:output tokens)",
            "Judge throughput by total tokens/s; prompt processing dominates GPU time.", acts,
        ))

    if s.finished_total >= 5:
        trunc = s.finished.get("length", 0) / s.finished_total
        if trunc > 0.3:
            out.append(Advice(
                "info", f"{100 * trunc:.0f}% of requests hit max_tokens",
                "finish_reason=length: outputs may be truncated.",
                ["Raise max_tokens if this is not intended."],
            ))

    if s.spec_acceptance is not None and s.spec_acceptance < 0.3:
        out.append(Advice(
            "warn", f"Speculative decoding acceptance only {100 * s.spec_acceptance:.0f}%",
            "Drafts are mostly rejected; speculation is likely costing throughput.",
            ["Try fewer speculative tokens, another draft method, or disable speculative decoding."],
        ))

    if setup.gpu is None or setup.model is None:
        out.append(Advice("info", "No throughput estimate", "; ".join(setup.notes) or
                          "Pass --gpu and --tp so observed throughput can be compared with an estimate."))
    if not any(a.level in ("warn", "crit") for a in out):
        out.insert(0, Advice("ok", "Looks healthy", "No bottleneck detected in this window."))
    return out
