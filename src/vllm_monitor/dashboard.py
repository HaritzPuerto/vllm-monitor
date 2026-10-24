"""Rich terminal rendering: the live dashboard and the static report."""

from __future__ import annotations

import time
from pathlib import Path
from collections import deque
from typing import Deque, List, Optional, Sequence

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .analysis import Analysis, Setup, analyze
from .metrics import GENERATION_TOKENS, Histogram, Snapshot, VLLMClient
from .recorder import Recorder
from .stats import compute_window

_LEVEL_STYLE = {"ok": ("green", "✓"), "info": ("cyan", "i"), "warn": ("yellow", "!"), "crit": ("red", "✗")}
_VERDICT_STYLE = {
    "good": ("bold green", "as expected"), "fair": ("bold yellow", "somewhat slow"),
    "slow": ("bold red", "SLOW"), "idle": ("dim", "idle"), "unknown": ("dim", "no estimate"),
}
_BARS = "▁▂▃▄▅▆▇█"


def sparkline(values: Sequence[float], width: int = 40) -> str:
    vals = list(values)[-width:]
    if not vals:
        return ""
    top = max(vals) or 1.0
    return "".join(_BARS[min(len(_BARS) - 1, int(v / top * (len(_BARS) - 1)))] for v in vals)


def _fmt(v: Optional[float], unit: str = "", digits: int = 0) -> str:
    return "-" if v is None else f"{v:,.{digits}f}{unit}"


def _secs(v: Optional[float]) -> str:
    if v is None:
        return "-"
    return f"{v * 1000:.0f} ms" if v < 1 else f"{v:.2f} s"


def _bar(frac: Optional[float], width: int = 12) -> Text:
    if frac is None:
        return Text("-")
    n = int(round(min(max(frac, 0), 1) * width))
    style = "green" if frac < 0.7 else "yellow" if frac < 0.9 else "red"
    return Text("█" * n, style=style) + Text("░" * (width - n), style="dim") + Text(f" {100 * frac:.0f}%")


def _kv_table(rows) -> Table:
    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim")
    t.add_column(justify="right")
    for r in rows:
        t.add_row(*r)
    return t


def _q(h: Optional[Histogram], q: float) -> Optional[float]:
    return h.quantile(q) if h else None


def render(setup: Setup, an: Optional[Analysis], history: Sequence[float] = (), title: str = "",
           error: Optional[str] = None, width: int = 120) -> RenderableType:
    hw = f"{setup.tp}x {setup.gpu.name}" if setup.gpu else "GPU ?"
    head = Text.assemble(("vllm-monitor ", "bold"), (setup.model_id or "?", "bold cyan"),
                         f"  ·  {hw}  ·  {setup.url or ''}  ", (title, "dim"))
    if error:
        return Group(head, Panel(Text(error, style="red"), title="error"))
    if an is None:
        return Group(head, Text("collecting samples…", style="dim"))
    s = an.stats

    thr = _kv_table([
        ("output", Text(f"{s.gen_tps:,.0f} tok/s", style="bold")),
        ("prompt", f"{s.prompt_tps:,.0f} tok/s"),
        ("total", f"{s.gen_tps + s.prompt_tps:,.0f} tok/s"),
        ("requests", f"{s.requests_per_s:,.2f} /s"),
    ])
    thr_panel = Panel(Group(thr, Text(sparkline(history), style="cyan")), title="throughput", border_style="cyan")

    style, label = _VERDICT_STYLE[an.verdict]
    if an.expected is not None:
        e, b = an.expected, an.best
        headroom = b.output_tps / max(s.gen_tps, 1e-9)
        rows = [
            ("verdict", Text(label, style=style)),
            ("efficiency", Text(f"{100 * an.efficiency:.0f}%", style=style)),
            ("expected", f"{e.output_tps * s.busy_frac:,.0f} tok/s"),
            ("at load", f"{e.concurrency:.0f} reqs, {e.bound}-bound"),
            ("best", f"{b.output_tps:,.0f} tok/s @ {b.concurrency:.0f}"),
            ("headroom", Text(f"{headroom:.1f}x possible", style="bold yellow" if headroom >= 1.5 else "green")),
            ("KV fits", f"~{b.max_concurrency:,.0f} reqs"),
        ]
        exp_panel = Panel(_kv_table(rows), title="vs. expected", border_style=style.split()[-1])
    else:
        exp_panel = Panel(Text(label, style=style), title="vs. expected", border_style="dim")

    sched = _kv_table([
        ("running", f"{s.running_avg:.1f} (max {s.running_max:.0f})"),
        ("waiting", f"{s.waiting_avg:.1f} (max {s.waiting_max:.0f})"),
        ("KV cache", _bar(s.kv_usage_avg)),
        ("GPU busy", _bar(s.busy_frac)),
        ("preempted", f"{s.preemptions:.0f}"),
        ("prefix hit", _fmt(None if s.prefix_hit_rate is None else 100 * s.prefix_hit_rate, "%")),
    ])
    sched_panel = Panel(sched, title="scheduler", border_style="magenta")

    lat = Table(box=None, padding=(0, 2), show_edge=False)
    for col in ("", "p50", "p95"):
        lat.add_column(col, justify="right" if col else "left", style=None if col else "dim")
    for name, h in (("TTFT", s.ttft), ("inter-token", s.itl), ("end-to-end", s.e2e), ("queue", s.queue)):
        lat.add_row(name, _secs(_q(h, 0.5)), _secs(_q(h, 0.95)))
    work = _kv_table([
        ("prompt len", f"{_fmt(s.avg_prompt_len)} (p95 {_fmt(_q(s.prompt_len, 0.95))})"),
        ("output len", f"{_fmt(s.avg_gen_len)} (p95 {_fmt(_q(s.gen_len, 0.95))})"),
        ("finished", ", ".join(f"{k} {v:.0f}" for k, v in sorted(s.finished.items()) if v) or "-"),
        ("max_model_len", _fmt(setup.max_model_len)),
    ])
    lat_panel = Panel(lat, title="latency", border_style="blue")
    work_panel = Panel(work, title="workload", border_style="blue")

    adv = Table.grid(padding=(0, 1))
    adv.add_column(width=1)
    adv.add_column()
    for a in an.advice:
        color, icon = _LEVEL_STYLE[a.level]
        acts = Table.grid(padding=(0, 1))
        acts.add_column(width=1)
        acts.add_column()
        for act in a.actions:
            acts.add_row("→", Text(act, style="dim"))
        body = [Text(a.title, style=f"bold {color}")] + ([Text(a.detail)] if a.detail else []) + [acts]
        adv.add_row(Text(icon, style=color), Group(*body))
    adv_panel = Panel(adv, title="advice", border_style="yellow")

    cols = 3 if width >= 120 else 2
    panels = [thr_panel, exp_panel, sched_panel, lat_panel, work_panel]
    grid = Table.grid(expand=True)
    for _ in range(cols):
        grid.add_column(ratio=1)
    for i in range(0, len(panels), cols):
        row = panels[i:i + cols]
        grid.add_row(*row, *[""] * (cols - len(row)))
    return Group(head, grid, adv_panel)


def watch(client: VLLMClient, setup: Setup, interval: float = 2.0, window: float = 60.0,
          record: Optional[Recorder] = None, console: Optional[Console] = None,
          duration: Optional[float] = None) -> None:
    console = console or Console()
    snaps: Deque[Snapshot] = deque()
    history: Deque[float] = deque(maxlen=120)
    start = time.time()
    with Live(console=console, refresh_per_second=4, screen=False) as live:
        while duration is None or time.time() - start < duration:
            t0 = time.time()
            error = None
            an = None
            try:
                snap = client.scrape()
                if record:
                    record.write(snap)
                if snaps:
                    prev = snaps[-1]
                    dt = snap.t - prev.t
                    if dt > 0:
                        history.append(max(0.0, (snap.get(GENERATION_TOKENS, 0) - prev.get(GENERATION_TOKENS, 0)) / dt))
                snaps.append(snap)
                while len(snaps) > 2 and snaps[-1].t - snaps[0].t > window:
                    snaps.popleft()
                stats = compute_window(list(snaps))
                an = analyze(stats, setup) if stats else None
            except Exception as e:  # server not up yet / restarting
                error = f"{client.base}: {type(e).__name__}: {e} (retrying)"
            live.update(render(setup, an, history, title=f"window {window:.0f}s · every {interval:g}s · ctrl-c to quit",
                               error=error, width=console.width))
            time.sleep(max(0.0, interval - (time.time() - t0)))


def print_report(console: Console, setup: Setup, overall: Analysis, timeline: List[Analysis],
                 history: Sequence[float], source: str) -> None:
    console.print(render(setup, overall, history, title=f"report · {Path(source).name} · {overall.stats.duration / 60:.1f} min",
                         width=console.width))
    if len(timeline) < 2:
        return
    t = Table(title="timeline", box=None, header_style="bold")
    for col in ("t (min)", "out tok/s", "prompt tok/s", "running", "waiting", "KV", "eff.", "verdict"):
        t.add_column(col, justify="right")
    elapsed = 0.0
    for a in timeline:
        s = a.stats
        style, label = _VERDICT_STYLE[a.verdict]
        t.add_row(
            f"{elapsed / 60:.1f}", f"{s.gen_tps:,.0f}", f"{s.prompt_tps:,.0f}", f"{s.running_avg:.1f}",
            f"{s.waiting_avg:.1f}", _fmt(None if s.kv_usage_avg is None else 100 * s.kv_usage_avg, "%"),
            _fmt(None if a.efficiency is None else 100 * a.efficiency, "%"), Text(label, style=style),
        )
        elapsed += s.duration
    console.print(t)
