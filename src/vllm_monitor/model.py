"""Model architecture info needed for performance estimates, read from a HF ``config.json``."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_DTYPE_BYTES = {
    "float32": 2, "float16": 2, "bfloat16": 2,  # vLLM serves fp32 checkpoints in 16-bit
    "F32": 2, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "I32": 4, "I64": 8,
}
CACHE_DIR = Path(os.environ.get("VLLM_MONITOR_CACHE", Path.home() / ".cache" / "vllm-monitor"))


@dataclass
class ModelSpec:
    name: str
    num_layers: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    total_params: float
    active_params: float  # params touched per token (differs from total for MoE)
    expert_params: float  # params living in routed experts (0 for dense models)
    num_experts: int
    experts_per_token: int
    weight_bytes: float  # bytes of weights in GPU memory (all TP ranks combined)
    kv_elems_per_token: float  # KV cache elements per token summed over layers
    mla: bool = False  # multi-head latent attention: one latent KV shared by all heads
    fp8_weights: bool = False

    @property
    def bytes_per_param(self) -> float:
        return self.weight_bytes / self.total_params

    def kv_bytes_per_token(self, kv_dtype_bytes: float = 2.0, tp: int = 1) -> float:
        """KV bytes per token across all GPUs, including head replication when tp > kv heads."""
        replicas = tp if self.mla else max(1.0, tp / self.num_kv_heads)
        return self.kv_elems_per_token * kv_dtype_bytes * replicas


def _bytes_per_param(cfg: dict) -> tuple:
    """Return (bytes per weight, fp8 weights?) from dtype + quantization_config."""
    q = cfg.get("quantization_config") or {}
    method = str(q.get("quant_method", "")).lower()
    bits = q.get("bits") or q.get("w_bit")
    if method == "compressed-tensors":
        for group in (q.get("config_groups") or {}).values():
            w = group.get("weights") or {}
            bits = w.get("num_bits")
            if w.get("type") == "float" and bits == 8:
                return 1.0, True
            break
    if method in ("fp8", "fbgemm_fp8", "modelopt") or "fp8" in method:
        return 1.0, True
    if method == "mxfp4":
        return 0.55, False
    if bits:
        return bits / 8 * 1.06, False  # + scales / zero points
    dtype = cfg.get("torch_dtype") or cfg.get("dtype") or "bfloat16"
    return float(_DTYPE_BYTES.get(str(dtype), 2)), False


def spec_from_config(cfg: dict, name: str = "model", weight_bytes: Optional[float] = None) -> ModelSpec:
    c = {**cfg, **cfg["text_config"]} if isinstance(cfg.get("text_config"), dict) else cfg
    h = c["hidden_size"]
    L = c["num_hidden_layers"]
    nh = c["num_attention_heads"]
    kvh = c.get("num_key_value_heads") or nh
    hd = c.get("head_dim") or h // nh
    vocab = c.get("vocab_size", 32000)
    inter = c.get("intermediate_size") or 4 * h

    if c.get("kv_lora_rank"):  # DeepSeek-style MLA
        rope, nope = c.get("qk_rope_head_dim", 64), c.get("qk_nope_head_dim", 128)
        v_hd, q_lora, kv_lora = c.get("v_head_dim", 128), c.get("q_lora_rank"), c["kv_lora_rank"]
        q_dim = nh * (rope + nope)
        attn = (h * q_lora + q_lora * q_dim if q_lora else h * q_dim)
        attn += h * (kv_lora + rope) + kv_lora * nh * (nope + v_hd) + nh * v_hd * h
        kv_elems = L * (kv_lora + rope)
        mla = True
    else:
        attn = 2 * h * nh * hd + 2 * h * kvh * hd
        kv_elems = 2 * L * kvh * hd
        mla = False

    n_exp = c.get("num_local_experts") or c.get("num_experts") or c.get("n_routed_experts") or 0
    k = c.get("num_experts_per_tok") or c.get("num_experts_per_token") or (2 if n_exp else 0)
    moe_inter = c.get("moe_intermediate_size") or inter
    expert = 3 * h * moe_inter
    shared = c.get("shared_expert_intermediate_size")
    shared = 3 * h * shared if shared else (c.get("n_shared_experts") or 0) * expert
    moe_layers = (L - (c.get("first_k_dense_replace") or 0)) if n_exp else 0
    dense_layers = L - moe_layers

    embed = vocab * h * (1 if c.get("tie_word_embeddings") else 2)
    base = embed + L * attn + dense_layers * 3 * h * inter + moe_layers * (shared + h * n_exp)
    expert_params = moe_layers * n_exp * expert
    total = base + expert_params
    active = base + moe_layers * k * expert

    bpp, fp8 = _bytes_per_param(c)
    return ModelSpec(
        name=name, num_layers=L, hidden_size=h, num_heads=nh, num_kv_heads=kvh, head_dim=hd,
        total_params=total, active_params=active, expert_params=expert_params,
        num_experts=n_exp, experts_per_token=k,
        weight_bytes=weight_bytes or total * bpp, kv_elems_per_token=kv_elems, mla=mla, fp8_weights=fp8,
    )


# --------------------------------------------------------------------------- loading


def _hf_get(repo: str, filename: str, revision: str = "main") -> Optional[dict]:
    """Fetch a JSON file from the HF hub (local HF cache first, then our cache, then network)."""
    try:
        from huggingface_hub import try_to_load_from_cache

        p = try_to_load_from_cache(repo, filename, revision=revision)
        if isinstance(p, str):
            return json.loads(Path(p).read_text())
    except Exception:
        pass
    cached = CACHE_DIR / "hf" / repo / revision / filename
    if cached.exists():
        return json.loads(cached.read_text())
    if filename.startswith("api:"):
        url = f"https://huggingface.co/api/models/{repo}"
    else:
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
    req = urllib.request.Request(url)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(data))
    return data


def load_model_spec(ref: str, revision: str = "main") -> ModelSpec:
    """Load a ModelSpec from a local model dir, a config.json path, or a HF repo id."""
    path = Path(ref).expanduser()
    if path.is_file():
        path = path.parent
    if path.is_dir():
        cfg = json.loads((path / "config.json").read_text())
        index = path / "model.safetensors.index.json"
        if index.exists():
            weight_bytes = json.loads(index.read_text()).get("metadata", {}).get("total_size")
        else:
            weight_bytes = sum(f.stat().st_size for f in path.glob("*.safetensors")) or None
        return spec_from_config(cfg, name=path.name, weight_bytes=weight_bytes)

    cfg = _hf_get(ref, "config.json", revision)
    if cfg is None:
        raise FileNotFoundError(f"no config.json for {ref!r} (local path or HF repo id)")
    weight_bytes = None
    index = _hf_get(ref, "model.safetensors.index.json", revision)
    if index:
        weight_bytes = index.get("metadata", {}).get("total_size")
    else:
        info = _hf_get(ref, "api:model_info", revision) or {}
        params = (info.get("safetensors") or {}).get("parameters") or {}
        weight_bytes = sum(n * _DTYPE_BYTES.get(dt, 2) for dt, n in params.items()) or None
    return spec_from_config(cfg, name=ref, weight_bytes=weight_bytes)
