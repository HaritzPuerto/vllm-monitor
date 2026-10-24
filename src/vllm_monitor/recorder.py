"""Record snapshots to JSON Lines (optionally gzipped) and read them back."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import IO, List, Tuple

from . import __version__
from .metrics import Snapshot


def _open(path: Path, mode: str) -> IO[str]:
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(path, mode, encoding="utf-8")


class Recorder:
    def __init__(self, path: str, header: dict):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = _open(self.path, "a")
        self._f.write(json.dumps({"type": "header", "version": __version__, **header}) + "\n")
        self._f.flush()

    def write(self, snap: Snapshot) -> None:
        self._f.write(json.dumps(snap.to_json(), separators=(",", ":")) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def read_recording(path: str) -> Tuple[dict, List[Snapshot]]:
    """Return (last header, snapshots). Appended sessions are merged into one timeline."""
    header: dict = {}
    snaps: List[Snapshot] = []
    with _open(Path(path), "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:  # truncated last line of a killed job
                continue
            if d.get("type") == "header":
                header = d
            else:
                snaps.append(Snapshot.from_json(d, {"vllm:cache_config_info": header.get("cache_info") or {}}))
    return header, snaps
