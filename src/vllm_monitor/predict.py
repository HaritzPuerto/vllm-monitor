"""Roofline estimate of vLLM throughput for a model / GPU / workload.

Decoding is modelled as one forward pass per step over the ``B`` running requests:

    step_time = max(bytes_read / bandwidth, flops / peak_flops) + fixed overhead

where bytes_read = weights (only the experts actually hit, for MoE) + the KV cache of
the batch.  Prefill is compute-bound.  Over the lifetime of a batch of ``B`` requests
with ``I`` prompt and ``O`` output tokens, prefill and decode share the GPU, so the
cycle time is the roofline of their combined bytes and flops.

Real deployments reach a fraction of the datasheet peaks; ``Assumptions`` holds those
efficiency factors.  Treat results as a ballpark (roughly +/-30%), good enough to tell
"healthy" apart from "something is off" and to pick a sensible concurrency.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Optional

from .hardware import GPUSpec
from .model import ModelSpec

GB = 1e9


@dataclass
class Assumptions:
    bandwidth_eff: float = 0.75  # achieved / peak HBM bandwidth
    flops_eff: float = 0.5  # achieved / peak tensor-core FLOPs
    step_overhead_s: float = 0.002  # scheduling, sampling, launch overhead per step
    allreduce_s: float = 15e-6  # per all-reduce (2 per layer) when tp > 1
    gpu_memory_utilization: float = 0.9
    reserved_gb_per_gpu: float = 2.5  # activations, CUDA graphs, NCCL buffers
    kv_dtype_bytes: float = 2.0
    max_num_seqs: int = 256  # vLLM default (some versions/GPUs use 512 or 1024)


@dataclass
class Prediction:
    concurrency: float  # requests actually in flight (after capping by KV capacity)
    requested_concurrency: float
    prompt_len: float
    output_len: float
    output_tps: float
    prompt_tps: float
    requests_per_s: float
    step_time_s: float  # one decode step at this batch size, no prefill
    ttft_s: float  # unloaded prefill time for one prompt
    tpot_s: float  # per-request time per output token, prefill interference included
    bound: str  # "memory" or "compute"
    kv_capacity_tokens: Optional[float]
    max_concurrency: Optional[float]

    @property
    def capped(self) -> bool:
        return self.concurrency < self.requested_concurrency


class Estimator:
    def __init__(
        self,
        model: ModelSpec,
        gpu: GPUSpec,
        tp: int = 1,
        assumptions: Optional[Assumptions] = None,
        kv_capacity_tokens: Optional[float] = None,
    ):
        self.model, self.gpu, self.tp = model, gpu, tp
        self.a = assumptions or Assumptions()
        self._kv_capacity = kv_capacity_tokens

    # ---------------------------------------------------------------- hardware
    @property
    def bandwidth(self) -> float:
        return self.tp * self.gpu.bandwidth_gbs * GB * self.a.bandwidth_eff

    @property
    def flops(self) -> float:
        peak = self.gpu.fp8_tflops if (self.model.fp8_weights and self.gpu.fp8_tflops) else self.gpu.bf16_tflops
        return self.tp * peak * 1e12 * self.a.flops_eff

    @property
    def kv_bytes_per_token(self) -> float:
        return self.model.kv_bytes_per_token(self.a.kv_dtype_bytes, self.tp)

    def kv_capacity_tokens(self) -> float:
        """Tokens that fit in the KV cache (exact if the server reported its block count)."""
        if self._kv_capacity is not None:
            return self._kv_capacity
        per_gpu = self.gpu.memory_gb * GB * self.a.gpu_memory_utilization - self.a.reserved_gb_per_gpu * GB
        free = self.tp * per_gpu - self.model.weight_bytes
        return max(0.0, free / self.kv_bytes_per_token)

    def fits(self) -> bool:
        return self.kv_capacity_tokens() > 0

    # ---------------------------------------------------------------- per-step costs
    def _weight_bytes_read(self, batch: float) -> float:
        m = self.model
        if not m.num_experts:
            return m.weight_bytes
        # Fraction of experts hit by at least one of `batch` tokens (uniform routing).
        hit = 1 - (1 - m.experts_per_token / m.num_experts) ** max(batch, 1)
        return (m.total_params - m.expert_params * (1 - hit)) * m.bytes_per_param

    def _attn_flops_per_token(self, context: float) -> float:
        m = self.model
        return 4 * m.num_layers * m.num_heads * m.head_dim * context

    def _overhead(self) -> float:
        comm = 2 * self.model.num_layers * self.a.allreduce_s if self.tp > 1 else 0.0
        return self.a.step_overhead_s + comm

    def decode_step_time(self, batch: float, context: float) -> float:
        bytes_ = self._weight_bytes_read(batch) + batch * context * self.kv_bytes_per_token
        flops = batch * (2 * self.model.active_params + self._attn_flops_per_token(context))
        return max(bytes_ / self.bandwidth, flops / self.flops) + self._overhead()

    def prefill_flops(self, prompt_len: float) -> float:
        # causal attention: on average each token attends to half the prompt
        return prompt_len * (2 * self.model.active_params + self._attn_flops_per_token(prompt_len / 2))

    # ---------------------------------------------------------------- workload
    def predict(self, concurrency: float, prompt_len: float, output_len: float) -> Prediction:
        cap = self.kv_capacity_tokens()
        max_conc = cap / max(prompt_len + output_len, 1)
        B = max(1.0, min(concurrency, max_conc, self.a.max_num_seqs)) if max_conc >= 1 else 1.0
        I, O = max(prompt_len, 1.0), max(output_len, 1.0)
        ctx = I + O / 2

        # One "cycle": B requests are prefilled and decoded for O steps.
        dec_bytes = O * (self._weight_bytes_read(B) + B * ctx * self.kv_bytes_per_token)
        dec_flops = O * B * (2 * self.model.active_params + self._attn_flops_per_token(ctx))
        pre_flops = B * self.prefill_flops(I)
        t_mem = dec_bytes / self.bandwidth
        t_cmp = (dec_flops + pre_flops) / self.flops
        cycle = max(t_mem, t_cmp) + O * self._overhead()

        return Prediction(
            concurrency=B,
            requested_concurrency=concurrency,
            prompt_len=I,
            output_len=O,
            output_tps=B * O / cycle,
            prompt_tps=B * I / cycle,
            requests_per_s=B / cycle,
            step_time_s=self.decode_step_time(B, ctx),
            ttft_s=self.prefill_flops(I) / self.flops + self._overhead(),
            tpot_s=cycle / O,
            bound="memory" if t_mem >= t_cmp else "compute",
            kv_capacity_tokens=cap,
            max_concurrency=max_conc,
        )

    def sweep(self, prompt_len: float, output_len: float, max_concurrency: Optional[int] = None) -> List[Prediction]:
        top = max_concurrency or self.a.max_num_seqs
        levels, b = [], 1
        while b < top:
            levels.append(b)
            b *= 2
        levels.append(top)
        return [self.predict(c, prompt_len, output_len) for c in levels]

    def best(self, prompt_len: float, output_len: float) -> Prediction:
        """Highest-throughput operating point (all slots the scheduler/KV cache allow)."""
        return self.predict(self.a.max_num_seqs, prompt_len, output_len)

    def knee(self, prompt_len: float, output_len: float, frac: float = 0.85) -> Prediction:
        """Smallest power-of-two concurrency reaching `frac` of the best throughput."""
        target = frac * self.best(prompt_len, output_len).output_tps
        for p in self.sweep(prompt_len, output_len):
            if p.output_tps >= target:
                return p
        return self.best(prompt_len, output_len)

    def with_assumptions(self, **kw) -> "Estimator":
        return Estimator(self.model, self.gpu, self.tp, replace(self.a, **kw), self._kv_capacity)
