"""Command line interface: ``vllm-monitor {serve,watch,record,report,replay,estimate,gpus,demo}``."""

from __future__ import annotations

import argparse
import signal
import socket
import sys
import time
from typing import List, Optional

from rich.console import Console
from rich.table import Table

from . import __version__
from .analysis import analyze
from .dashboard import print_report, render, watch
from .discover import discover, setup_from_header, setup_header
from .hardware import GPUS, resolve_gpu
from .metrics import GENERATION_TOKENS, VLLMClient
from .model import load_model_spec
from .predict import Assumptions, Estimator
from .recorder import Recorder, read_recording
from .stats import compute_window

DEFAULT_URL = "http://localhost:8000"


def _add_setup_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("deployment (for throughput estimates)")
    g.add_argument("--gpu", help="GPU type, e.g. H100-SXM, A100-80GB (auto-detected via nvidia-smi if local)")
    g.add_argument("--tp", type=int, default=None, help="tensor-parallel size of the server (default 1)")
    g.add_argument("--model-config", help="model dir, config.json or HF repo id (default: served model name)")
    g.add_argument("--max-num-seqs", type=int, help="the server's --max-num-seqs, if not the default")
    _add_eff_args(g)


def _add_eff_args(g) -> None:
    g.add_argument("--bandwidth-eff", type=float, default=Assumptions.bandwidth_eff,
                   help="achieved fraction of peak HBM bandwidth (default %(default)s)")
    g.add_argument("--flops-eff", type=float, default=Assumptions.flops_eff,
                   help="achieved fraction of peak FLOPs (default %(default)s)")


def _assumptions(args) -> Assumptions:
    return Assumptions(bandwidth_eff=args.bandwidth_eff, flops_eff=args.flops_eff)


def _exit_on_sigterm() -> None:
    """Turn SIGTERM (job end, condor_rm, scancel, kill) into KeyboardInterrupt so cleanup runs.

    Without this a gzip recording is left without its trailer and cannot be read back.
    """
    def handler(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, handler)


def _connect(args, console: Console):
    client = VLLMClient(args.url, api_key=args.api_key)
    snap = None
    with console.status(f"connecting to {client.base} …"):
        deadline = time.time() + args.wait
        while True:
            try:
                snap = client.scrape()
                break
            except Exception as e:
                if time.time() > deadline:
                    console.print(f"[yellow]cannot reach {client.base}/metrics yet ({e}); will keep retrying[/]")
                    break
                time.sleep(2)
    setup = discover(client, snap, gpu=args.gpu, tp=args.tp or 1, model_ref=args.model_config,
                     max_num_seqs=args.max_num_seqs)
    setup.assumptions = _assumptions(args)
    for n in setup.notes:
        console.print(f"[dim]note: {n}[/]")
    return client, setup


def cmd_watch(args) -> int:
    console = Console()
    client, setup = _connect(args, console)
    rec = Recorder(args.record, setup_header(setup, args.model_config)) if args.record else None
    _exit_on_sigterm()
    try:
        watch(client, setup, interval=args.interval, window=args.window, record=rec, console=console)
    except KeyboardInterrupt:
        pass
    finally:
        if rec:
            rec.close()
            console.print(f"recorded to {args.record}; analyse with: vllm-monitor report {args.record}")
    return 0


def _serve(service, host: str, port: int, console: Console) -> None:
    from .web import make_server

    server = make_server(service.start(), host, port)
    port = server.server_address[1]
    console.print(f"[bold]dashboard:[/] http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}")
    if host == "127.0.0.1":
        node = socket.gethostname()
        console.print(f"[dim]remote? on your laptop run:  ssh -N -L {port}:localhost:{port} {node}"
                      f"   (add -J <login-node> if {node} is behind one), then open http://localhost:{port}[/]")
    console.print("[dim]ctrl-c to stop[/]")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        service.stop()
        server.server_close()


def cmd_serve(args) -> int:
    from .web import MonitorService

    console = Console()
    client, setup = _connect(args, console)
    rec = Recorder(args.record, setup_header(setup, args.model_config)) if args.record else None
    service = MonitorService(client, setup, interval=args.interval, window=args.window,
                             history=args.history, recorder=rec)
    _exit_on_sigterm()
    try:
        _serve(service, args.host, args.port, console)
    finally:
        if rec:
            rec.close()
    return 0


def cmd_record(args) -> int:
    console = Console(force_terminal=False, soft_wrap=True)
    client, setup = _connect(args, console)
    rec = Recorder(args.output, setup_header(setup, args.model_config))
    console.print(f"recording {client.base} every {args.interval:g}s to {args.output}")
    _exit_on_sigterm()
    start, last_log, prev = time.time(), 0.0, None
    try:
        while args.duration is None or time.time() - start < args.duration:
            t0 = time.time()
            try:
                snap = client.scrape()
                rec.write(snap)
                if prev is not None and t0 - last_log >= args.log_every:
                    st = compute_window([prev, snap])
                    if st:
                        kv = "-" if st.kv_usage_avg is None else f"{100 * st.kv_usage_avg:.0f}%"
                        console.print(
                            f"{time.strftime('%H:%M:%S')} out {st.gen_tps:8,.0f} tok/s | prompt {st.prompt_tps:8,.0f} tok/s"
                            f" | running {snap.get(('vllm:num_requests_running',), 0):4.0f}"
                            f" waiting {snap.get(('vllm:num_requests_waiting',), 0):4.0f} | KV {kv}")
                    prev, last_log = snap, t0
                elif prev is None:
                    prev = snap
            except Exception as e:
                console.print(f"{time.strftime('%H:%M:%S')} scrape failed: {e}")
            time.sleep(max(0.0, args.interval - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        rec.close()
    console.print(f"stopped; analyse with: vllm-monitor report {args.output}")
    return 0


def _load_recording(args, console: Console):
    """Snapshots and Setup of a recording; CLI flags override what the header says."""
    header, snaps = read_recording(args.file)
    if args.skip:
        snaps = [s for s in snaps if s.t - snaps[0].t >= args.skip]
    if len(snaps) < 2:
        console.print("[red]recording has fewer than 2 samples[/]")
        return None, None
    setup = setup_from_header(header, gpu=args.gpu, tp=args.tp, model_ref=args.model_config,
                              max_num_seqs=args.max_num_seqs)
    setup.assumptions = _assumptions(args)
    for n in setup.notes:
        console.print(f"[dim]note: {n}[/]")
    return snaps, setup


def cmd_report(args) -> int:
    console = Console()
    snaps, setup = _load_recording(args, console)
    if snaps is None:
        return 1
    overall = analyze(compute_window(snaps), setup)

    total = snaps[-1].t - snaps[0].t
    seg = args.segment * 60 if args.segment else max(60.0, total / 12)
    timeline, cur = [], [snaps[0]]
    for s in snaps[1:]:
        cur.append(s)
        if s.t - cur[0].t >= seg:
            timeline.append(analyze(compute_window(cur), setup))
            cur = [s]
    if len(cur) >= 2:
        timeline.append(analyze(compute_window(cur), setup))

    history = []
    for a, b in zip(snaps, snaps[1:]):
        if b.t > a.t:
            history.append(max(0.0, (b.get(GENERATION_TOKENS, 0) - a.get(GENERATION_TOKENS, 0)) / (b.t - a.t)))
    step = max(1, len(history) // 80)
    history = [sum(history[i:i + step]) / len(history[i:i + step]) for i in range(0, len(history), step)]
    print_report(console, setup, overall, timeline, history, args.file)
    console.print(f"[dim]explore it in the browser: vllm-monitor replay {args.file}[/]")
    return 0


def cmd_replay(args) -> int:
    from .web import ReplayService

    console = Console()
    snaps, setup = _load_recording(args, console)
    if snaps is None:
        return 1
    with console.status(f"analysing {len(snaps):,} samples …"):
        service = ReplayService(snaps, setup, window=args.window, source=args.file)
    _serve(service, args.host, args.port, console)
    return 0


def cmd_estimate(args) -> int:
    console = Console()
    gpu = resolve_gpu(args.gpu)
    if gpu is None:
        console.print(f"[red]unknown GPU {args.gpu!r}[/]; known: {', '.join(GPUS)}")
        return 1
    with console.status(f"loading config for {args.model} …"):
        model = load_model_spec(args.model)
    a = _assumptions(args)
    a.gpu_memory_utilization = args.gpu_memory_utilization
    a.kv_dtype_bytes = 1.0 if args.kv_cache_dtype.startswith("fp8") else 2.0
    a.max_num_seqs = args.max_num_seqs or a.max_num_seqs
    est = Estimator(model, gpu, args.tp, a)
    I, O = args.input_len, args.output_len

    cap = est.kv_capacity_tokens()
    console.print(f"[bold]{model.name}[/] on [bold]{args.tp}x {gpu.name}[/]  "
                  f"(input {I}, output {O} tokens)")
    console.print(f"  params {model.total_params / 1e9:.1f}B"
                  + (f" ({model.active_params / 1e9:.1f}B active)" if model.num_experts else "")
                  + f", weights {model.weight_bytes / 1e9:.1f} GB, KV {est.kv_bytes_per_token / 1024:.0f} KiB/token")
    if cap <= 0:
        console.print(f"[red]  does not fit: weights need {model.weight_bytes / 1e9:.1f} GB; "
                      f"increase --tp or use a quantized model[/]")
        return 1
    console.print(f"  KV cache ~{cap:,.0f} tokens → at most ~{cap / (I + O):,.0f} concurrent requests of this length")

    t = Table(box=None, header_style="bold")
    for col in ("concurrency", "output tok/s", "total tok/s", "req/s", "TPOT", "TTFT (idle)", "bound"):
        t.add_column(col, justify="right")
    levels = est.sweep(I, O)
    if args.concurrency:
        levels.append(est.predict(args.concurrency, I, O))
        levels.sort(key=lambda p: p.requested_concurrency)
    knee = est.knee(I, O)
    for p in levels:
        conc = f"{p.requested_concurrency:.0f}" + (f" (→{p.concurrency:.0f})" if p.capped else "")
        style = "bold green" if p.requested_concurrency == knee.requested_concurrency else None
        t.add_row(conc, f"{p.output_tps:,.0f}", f"{p.output_tps + p.prompt_tps:,.0f}", f"{p.requests_per_s:,.2f}",
                  f"{p.tpot_s * 1000:.1f} ms", f"{p.ttft_s * 1000:.0f} ms", p.bound, style=style)
    console.print(t)
    best = est.best(I, O)
    console.print(f"  recommended client concurrency: ≥{knee.concurrency:.0f} "
                  f"(~{100 * knee.output_tps / best.output_tps:.0f}% of max {best.output_tps:,.0f} tok/s)")
    console.print(f"  [dim]roofline estimate with bandwidth_eff={a.bandwidth_eff}, flops_eff={a.flops_eff}; "
                  f"expect ±30%[/]")
    if args.prompts:
        p = est.predict(args.concurrency or knee.concurrency, I, O)
        hours = args.prompts / p.requests_per_s / 3600
        console.print(f"  {args.prompts:,} prompts at concurrency {p.concurrency:.0f}: ~{hours:.1f} h")
    return 0


def cmd_gpus(args) -> int:
    t = Table(box=None, header_style="bold")
    for col in ("name", "memory GB", "bandwidth GB/s", "BF16 TFLOPs", "FP8 TFLOPs"):
        t.add_column(col, justify="right" if col != "name" else "left")
    for g in GPUS.values():
        t.add_row(g.name, f"{g.memory_gb:g}", f"{g.bandwidth_gbs:g}", f"{g.bf16_tflops:g}",
                  f"{g.fp8_tflops:g}" if g.fp8_tflops else "-")
    Console().print(t)
    return 0


def cmd_demo(args) -> int:
    from .demo import DEMO_GPU, DEMO_SPEC, FakeVLLM, serve

    console = Console()
    fake = FakeVLLM(args.scenario)
    server = serve(fake)
    client = VLLMClient(f"http://127.0.0.1:{server.server_address[1]}")
    setup = discover(client, client.scrape(), gpu=DEMO_GPU, tp=1, model=DEMO_SPEC)
    console.print(f"[dim]demo scenario '{args.scenario}': {fake.sc.description}[/]")
    try:
        if args.web:
            from .web import MonitorService

            _serve(MonitorService(client, setup, interval=args.interval, window=args.window), "127.0.0.1",
                   args.port, console)
        elif args.once:
            snaps = []
            for _ in range(5):
                snaps.append(client.scrape())
                time.sleep(args.interval)
            console.print(render(setup, analyze(compute_window(snaps), setup), title="demo", width=console.width))
        else:
            watch(client, setup, interval=args.interval, window=args.window, console=console)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
    return 0


def build_parser() -> argparse.ArgumentParser:
    from .demo import SCENARIOS

    p = argparse.ArgumentParser(prog="vllm-monitor", description=__doc__)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def server_cmd(name, help_):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("url", nargs="?", default=DEFAULT_URL, help="vLLM server (default %(default)s)")
        sp.add_argument("--interval", type=float, default=2.0, help="seconds between scrapes")
        sp.add_argument("--api-key", help="API key for /v1/models (or VLLM_API_KEY)")
        sp.add_argument("--wait", type=float, default=10, help="seconds to wait for the server at startup")
        _add_setup_args(sp)
        return sp

    w = server_cmd("watch", "live dashboard")
    w.add_argument("--window", type=float, default=60, help="seconds of history used for rates")
    w.add_argument("--record", metavar="FILE", help="also record to FILE (.jsonl or .jsonl.gz)")
    w.set_defaults(func=cmd_watch)

    r = server_cmd("record", "headless recording for batch jobs (one log line per --log-every)")
    r.add_argument("-o", "--output", required=True, help=".jsonl or .jsonl.gz")
    r.add_argument("--log-every", type=float, default=60, help="seconds between status lines")
    r.add_argument("--duration", type=float, help="stop after this many seconds")
    r.set_defaults(func=cmd_record, interval=5.0)

    rp = sub.add_parser("report", help="analyse a recording")
    rp.add_argument("file")
    rp.add_argument("--segment", type=float, help="timeline segment length in minutes")
    rp.add_argument("--skip", type=float, default=0, help="ignore the first N seconds (warm-up)")
    _add_setup_args(rp)
    rp.set_defaults(func=cmd_report)

    rpl = sub.add_parser("replay", help="open a recording in the web dashboard; drag over any part to analyse it")
    rpl.add_argument("file")
    rpl.add_argument("--port", type=int, default=8080)
    rpl.add_argument("--host", default="127.0.0.1", help="interface to bind (default %(default)s)")
    rpl.add_argument("--window", type=float, default=60, help="seconds of history behind the 'expected' line")
    rpl.add_argument("--skip", type=float, default=0, help="ignore the first N seconds (warm-up)")
    _add_setup_args(rpl)
    rpl.set_defaults(func=cmd_replay)

    e = sub.add_parser("estimate", help="predict throughput for a model / GPU / workload (no server needed)")
    e.add_argument("--model", required=True, help="HF repo id, model dir or config.json")
    e.add_argument("--gpu", required=True)
    e.add_argument("--tp", type=int, default=1)
    e.add_argument("--input-len", type=int, default=1000)
    e.add_argument("--output-len", type=int, default=300)
    e.add_argument("--concurrency", type=int, help="also show this concurrency")
    e.add_argument("--prompts", type=int, help="number of prompts in your job, to estimate wall time")
    e.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    e.add_argument("--kv-cache-dtype", default="auto")
    e.add_argument("--max-num-seqs", type=int)
    _add_eff_args(e)
    e.set_defaults(func=cmd_estimate)

    sv = server_cmd("serve", "web dashboard (open it in a browser, or through an SSH tunnel)")
    sv.add_argument("--port", type=int, default=8080)
    sv.add_argument("--host", default="127.0.0.1",
                    help="interface to bind; keep 127.0.0.1 and use an SSH tunnel on shared machines")
    sv.add_argument("--window", type=float, default=60, help="seconds of history used for rates and advice")
    sv.add_argument("--history", type=float, default=3600, help="seconds of chart history kept in memory")
    sv.add_argument("--record", metavar="FILE", help="also record to FILE (.jsonl or .jsonl.gz)")
    sv.set_defaults(func=cmd_serve)

    sub.add_parser("gpus", help="list known GPU specs").set_defaults(func=cmd_gpus)

    d = sub.add_parser("demo", help="dashboard against a simulated server (no GPU needed)")
    d.add_argument("--scenario", choices=list(SCENARIOS), default="low-concurrency")
    d.add_argument("--interval", type=float, default=1.0)
    d.add_argument("--window", type=float, default=30)
    d.add_argument("--once", action="store_true", help="print one frame and exit")
    d.add_argument("--web", action="store_true", help="web dashboard instead of the terminal one")
    d.add_argument("--port", type=int, default=8080)
    d.set_defaults(func=cmd_demo)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
