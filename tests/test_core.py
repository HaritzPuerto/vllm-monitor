import json

import pytest

from vllm_monitor import metrics as m
from vllm_monitor.analysis import Setup, analyze
from vllm_monitor.cli import main
from vllm_monitor.demo import DEMO_CONFIG, DEMO_SPEC, FakeVLLM, serve
from vllm_monitor.hardware import GPUS, resolve_gpu
from vllm_monitor.metrics import parse_metrics
from vllm_monitor.model import spec_from_config
from vllm_monitor.predict import Estimator
from vllm_monitor.recorder import read_recording
from vllm_monitor.stats import compute_window

TWO_ENGINES = """
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="x"} 3.0
vllm:num_requests_running{engine="1",model_name="x"} 5.0
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{engine="0",model_name="x"} 0.2
vllm:gpu_cache_usage_perc{engine="1",model_name="x"} 0.4
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{engine="0",model_name="x"} 100.0
vllm:generation_tokens_total{engine="1",model_name="x"} 50.0
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{engine="0",le="0.1",model_name="x"} 2.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="1.0",model_name="x"} 4.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="+Inf",model_name="x"} 4.0
vllm:time_to_first_token_seconds_sum{engine="0",model_name="x"} 1.2
vllm:time_to_first_token_seconds_count{engine="0",model_name="x"} 4.0
# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{block_size="16",num_gpu_blocks="1000",gpu_memory_utilization="0.9"} 1.0
# TYPE some_other_metric gauge
some_other_metric 7.0
"""


def test_parse_aggregates_engines_and_old_names():
    s = parse_metrics(TWO_ENGINES, t=0)
    assert s.get(m.RUNNING) == 8  # summed
    assert s.get(m.KV_USAGE) == pytest.approx(0.3)  # averaged, found under the old name
    assert s.get(m.GENERATION_TOKENS) == 150
    assert "some_other_metric" not in s.values
    h = s.histogram(m.TTFT)
    assert h.count == 4 and h.mean == pytest.approx(0.3)
    assert h.quantile(0.5) == pytest.approx(0.1)
    assert 0.1 < h.quantile(0.75) < 1.0
    assert Setup(cache_info=s.info["vllm:cache_config_info"]).kv_capacity_tokens == 16000


def test_param_count_dense_and_moe():
    assert DEMO_SPEC.total_params == pytest.approx(7.6e9, rel=0.02)  # Qwen2.5-7B
    moe = spec_from_config({
        "hidden_size": 2048, "num_hidden_layers": 48, "num_attention_heads": 32, "num_key_value_heads": 4,
        "head_dim": 128, "vocab_size": 151936, "intermediate_size": 6144, "moe_intermediate_size": 768,
        "num_experts": 128, "num_experts_per_tok": 8, "torch_dtype": "bfloat16",
    })  # Qwen3-30B-A3B
    assert moe.total_params == pytest.approx(30.5e9, rel=0.03)
    assert moe.active_params == pytest.approx(3.3e9, rel=0.1)


def test_resolve_gpu():
    assert resolve_gpu("NVIDIA A100-SXM4-80GB").name == "A100-SXM-80GB"
    assert resolve_gpu("NVIDIA A100-SXM4-40GB").name == "A100-40GB"
    assert resolve_gpu("NVIDIA H100 80GB HBM3").name == "H100-SXM"
    assert resolve_gpu("NVIDIA H100 PCIe").name == "H100-PCIe"
    assert resolve_gpu("NVIDIA L40S").name == "L40S"
    assert resolve_gpu("NVIDIA L4").name == "L4"
    assert resolve_gpu("h200").name == "H200"
    assert resolve_gpu("TPU v5") is None


def test_prediction_monotonic_and_capped():
    est = Estimator(DEMO_SPEC, GPUS["H100-SXM"], kv_capacity_tokens=50_000)
    preds = est.sweep(1000, 300)
    tps = [p.output_tps for p in preds]
    assert tps == sorted(tps)
    big = est.predict(256, 1000, 300)
    assert big.capped and big.concurrency == pytest.approx(50_000 / 1300)
    # a 70B bf16 model does not fit on one 80GB GPU
    big_model = spec_from_config({**DEMO_CONFIG, "hidden_size": 8192, "num_hidden_layers": 80,
                                  "num_attention_heads": 64, "num_key_value_heads": 8, "intermediate_size": 28672})
    assert not Estimator(big_model, GPUS["H100-SXM"]).fits()
    assert Estimator(big_model, GPUS["H100-SXM"], tp=4).fits()


def _simulate(scenario, n=10, step=2.0):
    fake = FakeVLLM(scenario, seed=1)
    snaps = []
    for i in range(n):
        fake.last -= step  # pretend `step` seconds passed
        snaps.append(parse_metrics(fake.metrics_text(), t=i * step))
    setup = Setup(model_id="demo", gpu=GPUS["H100-SXM"], model=DEMO_SPEC, max_model_len=32768,
                  cache_info=snaps[-1].info["vllm:cache_config_info"])
    return analyze(compute_window(snaps), setup)


@pytest.mark.parametrize("scenario,expected_title,verdicts", [
    ("healthy", "Looks healthy", {"good"}),
    ("low-concurrency", "Client-bound", {"good", "fair"}),
    ("kv-bound", "KV cache full", {"good", "fair"}),
    ("slow", "Slower than expected", {"slow"}),
    ("prefill-heavy", "Prefill-heavy", {"good", "fair"}),
])
def test_scenarios_diagnosed(scenario, expected_title, verdicts):
    an = _simulate(scenario)
    titles = [a.title for a in an.advice]
    assert any(t.startswith(expected_title) for t in titles), titles
    assert an.verdict in verdicts


def test_record_and_report_roundtrip(tmp_path, capsys):
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "config.json").write_text(json.dumps(DEMO_CONFIG))
    server = serve(FakeVLLM("healthy"))
    url = f"http://127.0.0.1:{server.server_address[1]}"
    out = tmp_path / "run.jsonl.gz"
    try:
        assert main(["record", url, "-o", str(out), "--interval", "0.3", "--duration", "2",
                     "--log-every", "0.5", "--gpu", "H100-SXM", "--model-config", str(tmp_path / "model")]) == 0
    finally:
        server.shutdown()
    header, snaps = read_recording(str(out))
    assert header["gpu"] == "H100-SXM" and header["model_id"] == "Qwen/Qwen2.5-7B-Instruct"
    assert len(snaps) >= 4
    assert main(["report", str(out)]) == 0
    assert "vs. expected" in capsys.readouterr().out


def test_web_api_serves_page_and_state():
    import threading
    import urllib.request

    from vllm_monitor.metrics import VLLMClient
    from vllm_monitor.web import MonitorService, make_server

    fake = FakeVLLM("healthy")
    fake_server = serve(fake)
    client = VLLMClient(f"http://127.0.0.1:{fake_server.server_address[1]}")
    setup = Setup(model_id="demo", gpu=GPUS["H100-SXM"], model=DEMO_SPEC)
    service = MonitorService(client, setup, interval=0.1)
    for _ in range(3):
        fake.last -= 2  # pretend 2 s passed between scrapes
        service.tick()
    web = make_server(service, port=0)
    threading.Thread(target=web.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{web.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "<title>vLLM monitor</title>" in page
        state = json.loads(urllib.request.urlopen(base + "/api/state?since=0").read())
        assert state["setup"]["gpu"] == "H100-SXM"
        assert len(state["points"]) == 2 and state["analysis"]["expected_now"] > 0
        later = json.loads(urllib.request.urlopen(f"{base}/api/state?since={state['points'][-1]['t']}").read())
        assert later["points"] == []
    finally:
        web.shutdown()
        fake_server.shutdown()


def test_replay_analyses_any_range_of_a_recording():
    import threading
    import urllib.request

    from vllm_monitor.web import ReplayService, make_server

    # a finished job: 10 min with few requests in flight, then 10 min well tuned
    snaps, t, counters = [], 0.0, None
    for scenario in ("low-concurrency", "healthy"):
        fake = FakeVLLM(scenario)
        if counters:
            fake.c = counters
        for _ in range(120):
            fake.last -= 5
            snaps.append(parse_metrics(fake.metrics_text(), t=t))
            t += 5
        counters = fake.c
    setup = Setup(model_id="demo", gpu=GPUS["H100-SXM"], model=DEMO_SPEC)
    service = ReplayService(snaps, setup, source="run.jsonl.gz")
    web = make_server(service, port=0)
    threading.Thread(target=web.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{web.server_address[1]}"
    get = lambda path: json.loads(urllib.request.urlopen(base + path).read())  # noqa: E731
    try:
        state = get("/api/state")
        assert state["mode"] == "replay" and len(state["points"]) == len(snaps) - 1
        assert all(p["expected"] for p in state["points"][20:])
        first = get(f"/api/range?t0=0&t1={600 - 5}")["analysis"]
        assert any(a["title"].startswith("Client-bound") for a in first["advice"])
        second = get(f"/api/range?t0=600&t1={t}")["analysis"]
        assert not any(a["title"].startswith("Client-bound") for a in second["advice"])
        assert get("/api/range?t0=10&t1=12")["analysis"] is None  # fewer than 2 samples
    finally:
        web.shutdown()


@pytest.mark.parametrize("cmd", ["serve", "record"])
def test_sigterm_leaves_readable_gzip(tmp_path, cmd):
    """Jobs stop the monitor with SIGTERM; the .gz recording must still be complete."""
    import signal
    import subprocess
    import sys
    import time

    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "config.json").write_text(json.dumps(DEMO_CONFIG))
    server = serve(FakeVLLM("healthy"))
    out = tmp_path / "run.jsonl.gz"
    args = [sys.executable, "-m", "vllm_monitor.cli", cmd, f"http://127.0.0.1:{server.server_address[1]}",
            "--interval", "0.2", "--gpu", "H100-SXM", "--model-config", str(tmp_path / "model")]
    args += ["--record", str(out), "--port", "0"] if cmd == "serve" else ["-o", str(out)]
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        deadline = time.time() + 20
        while time.time() < deadline and (not out.exists() or out.stat().st_size < 200):
            time.sleep(0.2)
        time.sleep(1)
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=10) == 0, proc.stderr.read().decode()
    finally:
        proc.kill()
        server.shutdown()
    import gzip

    gzip.decompress(out.read_bytes())  # raises EOFError on a truncated stream
    header, snaps = read_recording(str(out))
    assert header["gpu"] == "H100-SXM" and len(snaps) >= 3


def test_read_truncated_gzip(tmp_path):
    """A SIGKILLed job can't close the file; report must still read what was flushed."""
    import gzip

    lines = [json.dumps({"type": "header", "gpu": "H100-SXM"})] + [
        json.dumps({"t": float(i), "v": {"vllm:generation_tokens_total": 100.0 * i}}) for i in range(50)]
    data = gzip.compress(("\n".join(lines) + "\n").encode())
    out = tmp_path / "cut.jsonl.gz"
    out.write_bytes(data[: len(data) - 12])  # drop the trailer and some data
    header, snaps = read_recording(str(out))
    assert header["gpu"] == "H100-SXM" and len(snaps) >= 40
