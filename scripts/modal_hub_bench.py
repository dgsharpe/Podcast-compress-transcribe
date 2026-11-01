"""mimi-hub's current pipeline on Modal GPUs, through scripts/throughput_server.py: the hub's own code and settings
(fp16 streaming Mimi, native fp16 Parakeet encoder with length-bucketed batches, batched pyannote at a 2 s step),
a backlog of 8 x 5 min clips, then variants of how it's run.

    uvx --from modal modal run scripts/modal_hub_bench.py [--gpus L40S,H100] [--plan baseline]

Each plan entry is one throughput_server.py run (fresh process, so env settings apply) in the same container.
Needs locally: data/ref/ref_24k.wav, data/modal_pyannote/ (the gated community-1 snapshot, cp -rL out of the HF
cache). Results: outputs/modal_hub_bench.jsonl (one line per run).
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
# $/s, modal.com/pricing (2026-10-31)
PRICE = {"T4": 0.000164, "L4": 0.000222, "A10": 0.000306, "L40S": 0.000542, "A100-40GB": 0.000583,
         "A100-80GB": 0.000694, "RTX-PRO-6000": 0.000842, "H100": 0.001097, "H200": 0.001261, "B200": 0.001736}
CPU_PRICE, MEM_PRICE = 0.0000131, 0.00000222  # per core / GiB, $/s
MEM_GIB = 64
NA = "/root/na"  # the repository's layout inside the container
ENV = {"MIMI_HUB_MODELS": "/models/hub", "MIMI_HUB_DATA": "/tmp/hub_data", "HF_HUB_OFFLINE": "1",
       "TPUT_PYANNOTE_PATH": "/models/pyannote"}


def _download():
    from huggingface_hub import hf_hub_download, snapshot_download
    snapshot_download("kyutai/mimi")
    hf_hub_download("nvidia/parakeet-tdt-0.6b-v2", "parakeet-tdt-0.6b-v2.nemo")  # for hub/parakeet_native.py
    # onnxruntime rejects external weights that resolve outside the model's directory, as the HF cache's symlinks
    # into blobs/ do: make the snapshot's files real ones
    snap = Path(snapshot_download("istupakov/parakeet-tdt-0.6b-v2-onnx"))
    for f in snap.iterdir():
        if f.is_symlink():
            blob = f.resolve()
            f.unlink()
            blob.rename(f)
    import onnx_asr
    onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v2")  # preprocessor, decoder (and the unused ONNX encoder)
    onnx_asr.load_vad("silero")


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install("torch==2.14.1", "torchaudio", index_url="https://download.pytorch.org/whl/cu130")
    # as hub/setup_cuda.sh: onnxruntime on the CPU for Parakeet's preprocessing, VAD and decoder
    .pip_install("transformers==5.22.1", "onnx-asr[cpu,hub]==0.12.0", "onnxruntime==1.31.1", "pyannote-audio==4.0.7",
                 "librosa", "soundfile", "scipy", "fastapi>=0.162", "python-multipart", "httpx", "uvicorn",
                 "huggingface_hub", "safetensors", "nvidia-ml-py")
    .run_function(_download)
    .env(ENV)
    .add_local_dir(ROOT / "data/modal_pyannote", "/models/pyannote")
    .add_local_file(ROOT / "data/ref/ref_24k.wav", f"{NA}/data/ref/ref_24k.wav")
    .add_local_dir(ROOT / "hub", f"{NA}/hub", ignore=["__pycache__", "*.pyc"])
    .add_local_file(ROOT / "scripts/mimi_file.py", f"{NA}/scripts/mimi_file.py")
    .add_local_file(ROOT / "scripts/throughput_server.py", f"{NA}/scripts/throughput_server.py")
    .add_local_file(ROOT / "scripts/parakeet_decoder_check.py", f"{NA}/scripts/parakeet_decoder_check.py")
    .add_local_file(ROOT / "scripts/diar_precision_check.py", f"{NA}/scripts/diar_precision_check.py")
)
app = modal.App("mimi-hub-bench", image=image)

# name -> runs: (args, extra env); args are throughput_server.py's unless they start with another script's name
PLANS = {
    "baseline": [
        (["--workers", "1,2,4,8"], {}),
        (["--workers", "2", "--procs", "2"], {}),
        (["--workers", "2", "--procs", "4"], {}),
        (["--workers", "4", "--tf32"], {}),
    ],
    # CPU-bound at 16 cores: where ASR's CPU time goes, then more processes with fewer threads each
    "procs": [
        (["--workers", "1", "--profile"], {}),
        (["--workers", "2", "--procs", "2"], {}),
        (["--workers", "2", "--procs", "4"], {"MIMI_HUB_THREADS": "8"}),
        (["--workers", "2", "--procs", "4"], {"MIMI_HUB_THREADS": "4"}),
        (["--workers", "1", "--procs", "8"], {"MIMI_HUB_THREADS": "4"}),
    ],
    # the native TDT decoder (now the default) and pyannote's ResNet frames in reduced precision
    "native": [
        (["parakeet_decoder_check.py"], {}),
        (["diar_precision_check.py", "fp32:32", "tf32:32", "fp16:32", "bf16:32", "fp16:128"], {}),
        (["--workers", "4"], {}),
        (["--workers", "2", "--procs", "2"], {}),
        (["--workers", "2", "--procs", "2"], {"MIMI_HUB_DIAR_DTYPE": "fp16"}),
        (["--workers", "2", "--procs", "4"], {"MIMI_HUB_DIAR_DTYPE": "fp16", "MIMI_HUB_THREADS": "8"}),
    ],
    # the hub's NVIDIA defaults now (native decoder, fp16 pyannote frames, single-threaded VAD): processes x threads
    "tuned": [
        (["--workers", "2", "--procs", "2"], {}),
        (["--workers", "2", "--procs", "3"], {"MIMI_HUB_THREADS": "8"}),
        (["--workers", "2", "--procs", "4"], {"MIMI_HUB_THREADS": "8"}),
        (["--workers", "2", "--procs", "4"], {"MIMI_HUB_THREADS": "4"}),
        (["--workers", "2", "--procs", "6"], {"MIMI_HUB_THREADS": "4"}),
    ],
}


@app.function(gpu="L40S", cpu=16.0, memory=MEM_GIB * 1024, timeout=3600)
def bench(gpu: str, runs: list) -> list:
    import torch
    sys.path.insert(0, NA)
    info = {"gpu": gpu, "device": torch.cuda.get_device_name(0), "vcpus": os.cpu_count(),
            "cpu_model": next((ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo")
                               if ln.startswith("model name")), "?")}
    print("INFO " + json.dumps(info), flush=True)
    from hub import parakeet_native  # convert the .nemo once, before the runs (and their processes) need it
    model_dir = Path(ENV["MIMI_HUB_MODELS"]) / "nemo-parakeet-tdt-0.6b-v2-hf"
    parakeet_native.convert("nemo-parakeet-tdt-0.6b-v2", model_dir)
    parakeet_native.convert_decoder("nemo-parakeet-tdt-0.6b-v2", model_dir)
    out = []
    for args, env in runs:
        t = time.time()
        cmd = ([f"{NA}/scripts/{args[0]}", *args[1:]] if args and args[0].endswith(".py")
               else [f"{NA}/scripts/throughput_server.py", "--label", gpu, *args])
        p = subprocess.run([sys.executable, *cmd], cwd=NA, env={**os.environ, **env}, capture_output=True, text=True)
        rows = [{"tag": ln.split(" ", 1)[0], **json.loads(ln.split(" ", 1)[1])} for ln in p.stdout.splitlines()
                if re.match(r"^[A-Z]+ \{", ln)]
        if p.returncode or not rows:
            print(f"RUN FAILED {args} {env}:\n{p.stderr[-3000:]}", flush=True)
            out.append({**info, "args": args, "env": env, "error": p.stderr[-1500:]})
            continue
        for r in rows:
            r = {**info, **r, "args": args, "env": env, "run_s": round(time.time() - t)}
            print("RESULT " + json.dumps(r), flush=True)
            out.append(r)
    return out


@app.local_entrypoint()
def main(gpus: str = "L40S,H100", plan: str = "baseline", cpu: float = 16.0, wait_min: float = 45):
    runs = PLANS[plan]
    calls = {g: bench.with_options(gpu=g, cpu=cpu).spawn(g, runs) for g in (x.strip() for x in gpus.split(",")) if g}
    out = ROOT / "outputs/modal_hub_bench.jsonl"
    out.parent.mkdir(exist_ok=True)
    deadline = time.time() + wait_min * 60
    while calls and time.time() < deadline:  # as they finish: one GPU type without capacity mustn't block the rest
        for g, call in list(calls.items()):
            try:
                rows = call.get(timeout=0)
            except (TimeoutError, modal.exception.TimeoutError):
                continue
            except Exception as e:
                rows = [{"gpu": g, "error": f"{type(e).__name__}: {e}"}]
            del calls[g]
            with open(out, "a") as f:
                for r in rows:
                    r["plan"], r["cpu"] = plan, cpu
                    f.write(json.dumps(r) + "\n")
                    if "error" in r:
                        print(f"{g:6s} ERROR {r.get('args')} {r['error'][-300:]}", flush=True)
                        continue
                    if r.get("tag") != "RESULT":
                        keep = {k: v for k, v in r.items() if k not in ("gpu", "device", "vcpus", "cpu_model", "args",
                                                                         "env", "plan", "cpu")}
                        print(f"{g:6s} {json.dumps(keep)}", flush=True)
                        continue
                    usd_h = (PRICE.get(g, 0) + cpu * CPU_PRICE + MEM_GIB * MEM_PRICE) * 3600 / r["x_rt"]
                    print(f"{g:6s} procs={r['procs']} workers={r['workers']} env={r['env']} "
                          f"{r['x_rt']:6.1f}x  gpu {r['gpu_busy_pct']:5.1f}%  cpu {r['cpu_cores_used']:4.1f} cores  "
                          f"${usd_h:.4f}/audio h  stages {r['clip_stage_s']}", flush=True)
        time.sleep(5)
    for g, call in calls.items():
        call.cancel()
        print(f"{g}: no result within {wait_min:g} min; cancelled", flush=True)
