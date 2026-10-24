"""GPU specifications and detection."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class GPUSpec:
    name: str
    memory_gb: float
    bandwidth_gbs: float  # HBM bandwidth, GB/s
    bf16_tflops: float  # dense tensor-core peak (FP16 on GPUs without BF16)
    fp8_tflops: Optional[float] = None  # dense; None if FP8 tensor cores are unsupported


# Datasheet numbers (dense, no sparsity).
_SPECS = [
    GPUSpec("B200", 180, 8000, 2250, 4500),
    GPUSpec("H200", 141, 4800, 989, 1979),
    GPUSpec("GH200", 96, 4000, 989, 1979),
    GPUSpec("H100-SXM", 80, 3350, 989, 1979),
    GPUSpec("H100-NVL", 94, 3900, 835, 1671),
    GPUSpec("H100-PCIe", 80, 2000, 756, 1513),
    GPUSpec("A100-SXM-80GB", 80, 2039, 312),
    GPUSpec("A100-PCIe-80GB", 80, 1935, 312),
    GPUSpec("A100-40GB", 40, 1555, 312),
    GPUSpec("L40S", 48, 864, 362, 733),
    GPUSpec("L40", 48, 864, 181, 362),
    GPUSpec("L4", 24, 300, 121, 242),
    GPUSpec("A40", 48, 696, 150),
    GPUSpec("RTX-A6000", 48, 768, 155),
    GPUSpec("A10", 24, 600, 125),
    GPUSpec("RTX-4090", 24, 1008, 165, 330),
    GPUSpec("RTX-3090", 24, 936, 71),
    GPUSpec("V100-32GB", 32, 900, 125),
    GPUSpec("V100-16GB", 16, 900, 125),
    GPUSpec("MI300X", 192, 5300, 1307, 2615),
]
GPUS = {g.name: g for g in _SPECS}


def _norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def resolve_gpu(name: str, memory_gb: Optional[float] = None) -> Optional[GPUSpec]:
    """Map a user string or an ``nvidia-smi`` name (e.g. 'NVIDIA A100-SXM4-80GB') to a spec."""
    n = _norm(name)
    for g in _SPECS:
        if _norm(g.name) == n:
            return g
    big = memory_gb is None or memory_gb > 50
    rules = [
        ("GH200", "GH200"), ("B200", "B200"), ("H200", "H200"),
        ("H100", "H100-PCIe" if "PCIE" in n else "H100-NVL" if "NVL" in n else "H100-SXM"),
        ("A100", ("A100-PCIe-80GB" if "PCIE" in n else "A100-SXM-80GB")
         if "80" in n or (big and "40" not in n) else "A100-40GB"),
        ("L40S", "L40S"), ("L40", "L40"), ("L4", "L4"),
        ("A6000", "RTX-A6000"), ("A40", "A40"), ("A10", "A10"),
        ("4090", "RTX-4090"), ("3090", "RTX-3090"),
        ("V100", "V100-32GB" if "32" in n or (memory_gb or 0) > 20 else "V100-16GB"),
        ("MI300X", "MI300X"),
    ]
    for pattern, key in rules:
        if pattern in n:
            return GPUS[key]
    return None


def detect_gpus() -> List[Tuple[str, float]]:
    """Return [(name, memory_gb)] for local GPUs via nvidia-smi, or [] if unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        name, mem = (p.strip() for p in line.rsplit(",", 1))
        gpus.append((name, float(mem) / 1024))
    return gpus


def detect_gpu() -> Optional[GPUSpec]:
    gpus = detect_gpus()
    return resolve_gpu(*gpus[0]) if gpus else None
