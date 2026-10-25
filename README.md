# vllm-monitor

Live throughput monitor and tuning advisor for [vLLM](https://github.com/vllm-project/vllm) servers.

It answers three questions while your inference job runs:

1. **How fast is it going?** Output/prompt tokens per second, requests/s, latency percentiles,
   queue, KV-cache usage and preemptions, all read from vLLM's Prometheus `/metrics` endpoint.
2. **Is that slower than it should be?** A roofline model of your model × GPU × TP ×
   workload predicts the throughput you should get at the observed concurrency and
   prompt/output lengths, and how much more is possible.
3. **What should I change?** Rule-based advice, e.g. "client-bound: send ≥128 requests
   concurrently", "KV cache full: raise `--gpu-memory-utilization` / use `--kv-cache-dtype fp8`
   / lower `--max-model-len`", "`--max-num-seqs` is the limit", "preemptions", "prefill-heavy:
   fix prefix caching", "slower than expected: check `--enforce-eager`".

```
pip install vllm-monitor            # or: pip install -e . from this repo
vllm-monitor demo --web             # try the web dashboard against a simulated server
```

## Usage

```bash
# web dashboard: open http://localhost:8080
vllm-monitor serve http://localhost:8000 --port 8080

# terminal dashboard (server on the same machine: GPU is auto-detected via nvidia-smi)
vllm-monitor watch http://localhost:8000

# monitoring a server on another node: say which GPU / TP the server uses
vllm-monitor watch http://gpu-node-12:8000 --gpu A100-80GB --tp 2

# headless recording for batch jobs (one log line per minute, Ctrl-C/SIGTERM safe)
vllm-monitor record http://localhost:8000 -o logs/run.jsonl.gz

# analyse a recording afterwards: summary, timeline, advice
vllm-monitor report logs/run.jsonl.gz --skip 120   # ignore 2 min of warm-up

# replay a finished job in the web dashboard; drag across a chart to analyse that stretch
vllm-monitor replay logs/run.jsonl.gz --port 8080

# plan before you run: throughput vs. concurrency, KV capacity, wall-time
vllm-monitor estimate --model Qwen/Qwen2.5-7B-Instruct --gpu H100 \
    --input-len 1000 --output-len 300 --prompts 100000

vllm-monitor gpus                   # known GPU specs
```

The served model's `config.json` is read from the local path (if the server was started from
a directory), the local HF cache, or the Hugging Face Hub (set `HF_TOKEN` for gated models).
Override with `--model-config <dir|config.json|repo-id>`. If the server uses `--api-key`, set
`VLLM_API_KEY`.

## On a cluster (HTCondor / Slurm)

Record from inside the job, next to the server, and analyse later:

```bash
#!/bin/bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 &
until curl -sf localhost:8000/health; do sleep 5; done

vllm-monitor record localhost:8000 -o logs/$JOB_ID.jsonl.gz &   # GPU auto-detected
MON=$!

python my_inference_client.py        # your workload

kill $MON
```

For the web dashboard, start `vllm-monitor serve localhost:8000 --record logs/$JOB_ID.jsonl.gz &`
in the job instead of `record`. It binds to `127.0.0.1` only, so reach it with an SSH tunnel
from your laptop:

```bash
# compute node reachable via the login node
ssh -N -L 8080:localhost:8080 -J <user>@<login-node> <user>@<compute-node>
# then open http://localhost:8080
```

If compute nodes don't accept SSH, run `serve` on the login node instead, pointing it at
`http://<compute-node>:8000` with `--gpu` and `--tp`, and tunnel to the login node:
`ssh -N -L 8080:localhost:8080 <user>@<login-node>`. `serve` prints the exact tunnel command
for the host it runs on.

With `record`, the job log shows a status line per minute, and `vllm-monitor report logs/<id>.jsonl.gz`
gives the full analysis.

## Replaying a finished job

`vllm-monitor replay logs/<id>.jsonl.gz` opens the recording in the web dashboard. The charts span the
whole run, and the speedometer, advice and stats describe it. Drag across any chart to select part of
the run, e.g. a slow phase after warm-up, and the verdict, "what to change next run" and latency
stats are recomputed for just that stretch. The recording keeps the GPU, TP and model from the job, so
replay works on your laptop. To ask "what if", re-run it with other flags: `--gpu`, `--tp`,
`--max-num-seqs`, `--bandwidth-eff`, `--flops-eff`. For a live view, either `condor_ssh_to_job` / `srun --jobid ... --pty`
into the job and run `vllm-monitor watch`, or run it from the login node against
`http://<node>:8000` with `--gpu` and `--tp`.

## How the estimate works

Each decode step is a forward pass over the `B` running requests:

    step_time = max(bytes_read / HBM_bandwidth, FLOPs / peak_FLOPs) + overhead
    bytes_read = weights (only experts hit, for MoE) + KV cache of the batch

Prefill is compute-bound and shares the GPU with decode. Datasheet peaks are scaled by
efficiency factors (`--bandwidth-eff 0.75`, `--flops-eff 0.5` by default). KV capacity comes
from the server's `cache_config_info` metric when available, else from GPU memory.

The verdict compares observed output tok/s with the estimate at the *observed* load:

| efficiency | verdict       |
|-----------:|---------------|
|     ≥ 60 % | as expected   |
|   35–60 %  | somewhat slow |
|     < 35 % | SLOW          |

"headroom" compares with the best operating point (all scheduler slots filled). A server can be
"as expected" yet have 40× headroom: then the bottleneck is your client, not vLLM.

The estimate is a ballpark (expect ±30%). If your setup is consistently off, calibrate
`--bandwidth-eff` / `--flops-eff` once from a run you know is well tuned.

## Supported vLLM versions

Metric names from vLLM ≥ 0.6 (V0) and the V1 engine are handled, including renamed metrics
(`gpu_cache_usage_perc` → `kv_cache_usage_perc`, `time_per_output_token_seconds` →
`inter_token_latency_seconds`) and data-parallel engines.

Offline `LLM.generate()` scripts expose no HTTP endpoint and are not supported yet; serve the
model with `vllm serve` and send requests to it instead.

## Development

```bash
uv venv && uv pip install -e '.[dev]' && pytest
```
