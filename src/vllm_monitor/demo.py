"""A fake vLLM metrics server for trying the dashboard without a GPU (and for tests)."""

from __future__ import annotations

import json
import random
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Tuple

from .hardware import GPUS
from .model import spec_from_config
from .predict import Assumptions, Estimator

DEMO_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DEMO_GPU = "H100-SXM"
DEMO_CONFIG = {
    "hidden_size": 3584, "num_hidden_layers": 28, "num_attention_heads": 28, "num_key_value_heads": 4,
    "intermediate_size": 18944, "vocab_size": 152064, "tie_word_embeddings": False, "torch_dtype": "bfloat16",
}
DEMO_SPEC = spec_from_config(DEMO_CONFIG, DEMO_MODEL_ID, weight_bytes=15.23e9)


@dataclass
class Scenario:
    description: str
    running: float
    waiting: float = 0.0
    prompt_len: float = 800
    gen_len: float = 300
    efficiency: float = 0.75  # fraction of the roofline estimate the fake server achieves
    idle_frac: float = 0.0
    gpu_memory_utilization: float = 0.9
    preemptions_per_s: float = 0.0
    prefix_hit_rate: float = 0.3


SCENARIOS: Dict[str, Scenario] = {
    "healthy": Scenario("well-tuned batch job", running=128, waiting=1),
    "low-concurrency": Scenario("client sends few requests at a time", running=4, idle_frac=0.3),
    "kv-bound": Scenario("long requests, KV cache full", running=100, waiting=40, prompt_len=3000,
                         gen_len=1500, gpu_memory_utilization=0.5, preemptions_per_s=0.2),
    "slow": Scenario("server much slower than it should be", running=64, efficiency=0.25),
    "prefill-heavy": Scenario("long prompts, short answers", running=64, prompt_len=6000, gen_len=40,
                              prefix_hit_rate=0.02),
}

_LAT_BUCKETS = [0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5,
                10.0, 20.0, 40.0, 80.0, 160.0, 640.0, 2560.0]
_TOK_BUCKETS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]


class _Hist:
    def __init__(self, bounds: List[float]):
        self.bounds, self.counts, self.sum, self.count = bounds, [0] * len(bounds), 0.0, 0

    def observe(self, v: float) -> None:
        for i, b in enumerate(self.bounds):
            if v <= b:
                self.counts[i] += 1
        self.sum += v
        self.count += 1

    def lines(self, name: str, labels: str) -> List[str]:
        out = [f"# TYPE {name} histogram"]
        out += [f'{name}_bucket{{{labels},le="{b}"}} {c}' for b, c in zip(self.bounds, self.counts)]
        out += [f'{name}_bucket{{{labels},le="+Inf"}} {self.count}',
                f"{name}_sum{{{labels}}} {self.sum}", f"{name}_count{{{labels}}} {self.count}"]
        return out


class FakeVLLM:
    def __init__(self, scenario: str = "healthy", seed: int = 0):
        self.sc = SCENARIOS[scenario]
        self.rng = random.Random(seed)
        a = Assumptions(gpu_memory_utilization=self.sc.gpu_memory_utilization)
        self.est = Estimator(DEMO_SPEC, GPUS[DEMO_GPU], 1, a)
        self.capacity = self.est.kv_capacity_tokens()
        pred = self.est.predict(self.sc.running, self.sc.prompt_len, self.sc.gen_len)
        self.gen_tps = pred.output_tps * self.sc.efficiency
        self.c: Dict[str, float] = dict(prompt=0.0, gen=0.0, preempt=0.0, pq=0.0, ph=0.0, stop=0.0, length=0.0)
        self._pending = 0.0
        self.h = {k: _Hist(_LAT_BUCKETS) for k in ("ttft", "itl", "e2e", "queue")}
        self.h.update(prompt=_Hist(_TOK_BUCKETS), gen=_Hist(_TOK_BUCKETS))
        self.state: Tuple[float, float, float] = (0, 0, 0)
        self.last = time.time()
        self.lock = threading.Lock()

    def _jitter(self, x: float, rel: float = 0.1) -> float:
        return max(0.0, x * (1 + self.rng.gauss(0, rel)))

    def advance(self) -> None:
        now = time.time()
        dt, self.last = now - self.last, now
        sc = self.sc
        if self.rng.random() < sc.idle_frac:
            self.state = (0, 0, 0)
            return
        running = round(self._jitter(sc.running, 0.03))
        waiting = round(self._jitter(sc.waiting, 0.2))
        kv = min(1.0, running * (sc.prompt_len + sc.gen_len / 2) / self.capacity)
        self.state = (running, waiting, kv)
        gen = self._jitter(self.gen_tps * dt, 0.05)
        reqs = gen / sc.gen_len
        prompt = reqs * sc.prompt_len
        self.c["gen"] += gen
        self.c["prompt"] += prompt
        self.c["pq"] += prompt
        self.c["ph"] += prompt * sc.prefix_hit_rate
        self.c["preempt"] += self.rng.random() < sc.preemptions_per_s * dt
        self._pending += reqs
        tpot = running / self.gen_tps if self.gen_tps else 0.05
        while self._pending >= 1:
            self._pending -= 1
            p, g = self._jitter(sc.prompt_len, 0.3), max(1.0, self._jitter(sc.gen_len, 0.3))
            q = self._jitter(sc.waiting / max(reqs / dt, 1e-6), 0.3) if sc.waiting else 0.0
            ttft = q + self._jitter(p / 20000, 0.3)
            self.c["length" if self.rng.random() < 0.1 else "stop"] += 1
            for k, v in (("ttft", ttft), ("itl", self._jitter(tpot, 0.2)), ("queue", q),
                         ("e2e", ttft + g * tpot), ("prompt", p), ("gen", g)):
                self.h[k].observe(v)

    def metrics_text(self) -> str:
        with self.lock:
            self.advance()
            lb = f'engine="0",model_name="{DEMO_MODEL_ID}"'
            running, waiting, kv = self.state
            c = self.c
            blocks = int(self.capacity // 16)
            lines = [
                "# TYPE vllm:num_requests_running gauge", f"vllm:num_requests_running{{{lb}}} {running}",
                "# TYPE vllm:num_requests_waiting gauge", f"vllm:num_requests_waiting{{{lb}}} {waiting}",
                "# TYPE vllm:kv_cache_usage_perc gauge", f"vllm:kv_cache_usage_perc{{{lb}}} {kv}",
                "# TYPE vllm:prompt_tokens_total counter", f"vllm:prompt_tokens_total{{{lb}}} {c['prompt']}",
                "# TYPE vllm:generation_tokens_total counter", f"vllm:generation_tokens_total{{{lb}}} {c['gen']}",
                "# TYPE vllm:num_preemptions_total counter", f"vllm:num_preemptions_total{{{lb}}} {c['preempt']}",
                "# TYPE vllm:prefix_cache_queries_total counter", f"vllm:prefix_cache_queries_total{{{lb}}} {c['pq']}",
                "# TYPE vllm:prefix_cache_hits_total counter", f"vllm:prefix_cache_hits_total{{{lb}}} {c['ph']}",
                "# TYPE vllm:request_success_total counter",
                f'vllm:request_success_total{{{lb},finished_reason="stop"}} {c["stop"]}',
                f'vllm:request_success_total{{{lb},finished_reason="length"}} {c["length"]}',
                "# TYPE vllm:cache_config_info gauge",
                f'vllm:cache_config_info{{block_size="16",cache_dtype="auto",enable_prefix_caching="True",'
                f'gpu_memory_utilization="{self.sc.gpu_memory_utilization}",num_gpu_blocks="{blocks}"}} 1.0',
            ]
            for key, name in (("ttft", "vllm:time_to_first_token_seconds"), ("itl", "vllm:inter_token_latency_seconds"),
                              ("e2e", "vllm:e2e_request_latency_seconds"), ("queue", "vllm:request_queue_time_seconds"),
                              ("prompt", "vllm:request_prompt_tokens"), ("gen", "vllm:request_generation_tokens")):
                lines += self.h[key].lines(name, lb)
            return "\n".join(lines) + "\n"

    def models_json(self) -> str:
        return json.dumps({"object": "list", "data": [
            {"id": DEMO_MODEL_ID, "object": "model", "root": DEMO_MODEL_ID, "max_model_len": 32768}]})


def serve(fake: FakeVLLM, port: int = 0) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path.startswith("/metrics"):
                body, ctype = fake.metrics_text(), "text/plain; version=0.0.4"
            elif self.path.startswith("/v1/models"):
                body, ctype = fake.models_json(), "application/json"
            else:
                self.send_error(404)
                return
            data = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
