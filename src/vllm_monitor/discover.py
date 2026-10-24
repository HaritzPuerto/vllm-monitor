"""Work out what is being served (model, GPU, cache config) from the server and CLI flags."""

from __future__ import annotations

from typing import Optional

from .analysis import Setup
from .hardware import GPUS, detect_gpus, resolve_gpu
from .metrics import CACHE_CONFIG, Snapshot, VLLMClient
from .model import ModelSpec, load_model_spec


def resolve_model(setup: Setup, model_ref: Optional[str] = None, root: Optional[str] = None) -> None:
    for ref in [r for r in (model_ref, root, setup.model_id) if r]:
        try:
            setup.model = load_model_spec(ref)
            return
        except Exception as e:  # network, gated repo, unknown architecture ...
            last = f"{type(e).__name__}: {e}"
    if model_ref or setup.model_id:
        setup.notes.append(f"could not load model config ({last}); pass --model-config <dir|config.json|repo>")


def resolve_hardware(setup: Setup, gpu: Optional[str]) -> None:
    if gpu:
        setup.gpu = resolve_gpu(gpu)
        if setup.gpu is None:
            setup.notes.append(f"unknown GPU {gpu!r}; known: {', '.join(GPUS)}")
        return
    local = detect_gpus()
    if local:
        setup.gpu = resolve_gpu(*local[0])
    if setup.gpu is None:
        setup.notes.append("GPU unknown; pass --gpu (see `vllm-monitor gpus`)")


def discover(
    client: VLLMClient,
    snapshot: Optional[Snapshot] = None,
    gpu: Optional[str] = None,
    tp: int = 1,
    model_ref: Optional[str] = None,
    max_num_seqs: Optional[int] = None,
    model: Optional[ModelSpec] = None,
) -> Setup:
    setup = Setup(url=client.base, tp=tp, max_num_seqs=max_num_seqs)
    root = None
    try:
        models = client.models()
        if models:
            setup.model_id = models[0].get("id")
            setup.max_model_len = models[0].get("max_model_len")
            root = models[0].get("root")
    except Exception as e:
        setup.notes.append(f"/v1/models failed ({e}); set VLLM_API_KEY if the server uses --api-key")
    if snapshot is not None:
        setup.cache_info = next((snapshot.info[n] for n in CACHE_CONFIG if n in snapshot.info), {})
    resolve_hardware(setup, gpu)
    if model is not None:
        setup.model = model
    else:
        resolve_model(setup, model_ref, root)
    return setup


def setup_header(setup: Setup, model_ref: Optional[str] = None) -> dict:
    """Serializable description of a Setup, stored at the top of recordings."""
    return {
        "url": setup.url, "model_id": setup.model_id, "max_model_len": setup.max_model_len,
        "cache_info": setup.cache_info, "gpu": setup.gpu.name if setup.gpu else None, "tp": setup.tp,
        "model_ref": model_ref, "max_num_seqs": setup.max_num_seqs,
    }


def setup_from_header(h: dict, gpu: Optional[str] = None, tp: Optional[int] = None,
                      model_ref: Optional[str] = None, max_num_seqs: Optional[int] = None) -> Setup:
    setup = Setup(url=h.get("url"), model_id=h.get("model_id"), max_model_len=h.get("max_model_len"),
                  cache_info=h.get("cache_info") or {}, tp=tp or h.get("tp") or 1,
                  max_num_seqs=max_num_seqs or h.get("max_num_seqs"))
    g = gpu or h.get("gpu")
    if g:
        resolve_hardware(setup, g)
    else:
        setup.notes.append("GPU unknown; pass --gpu")
    resolve_model(setup, model_ref or h.get("model_ref"))
    return setup
