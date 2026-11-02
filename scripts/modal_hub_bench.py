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
import signal
import subprocess
import sys
import time
import types
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
    # a free-threaded Python (3.14t, no GIL) with the same packages, for runs with PY=FT_PYTHON
    .run_commands(
        "pip install -q uv==0.13.6",  # its interpreter list must include 3.14.10
        # pinned: torch's cp314t wheel needs a 3.14 C API (PyCriticalSection_BeginMutex) "3.14t" didn't resolve to
        "uv python install cpython-3.14.10+freethreaded",
        "uv venv -q /ft --python cpython-3.14.10+freethreaded",
        "uv pip install -q --python /ft/bin/python torch==2.14.1 torchaudio --index-url "
        "https://download.pytorch.org/whl/cu130",
        "uv pip install -q --python /ft/bin/python torch==2.14.1 transformers==5.22.1 'onnx-asr[cpu,hub]==0.12.0' "
        "onnxruntime==1.31.1 pyannote-audio==4.0.7 librosa soundfile scipy 'fastapi>=0.162' python-multipart httpx "
        "uvicorn huggingface_hub safetensors nvidia-ml-py threadpoolctl soxr",
        # protobuf's C extensions (imported via pyannote's telemetry) segfault without the GIL, even when only probed
        # (before PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION is read): removed, and the build-time constant protobuf reads
        # first (api_version 0 = pure Python) set, so it doesn't probe at all
        "rm -rf /ft/lib/python3.14t/site-packages/google/_upb /ft/lib/python3.14t/site-packages/google/protobuf/pyext",
        "echo 'api_version = 0' > /ft/lib/python3.14t/site-packages/google/protobuf/internal/_api_implementation.py",
    )
    # CUDA's compiler and headers for AOTInductor (torch.export + a C++ library built around the kernels)
    .run_commands(
        "apt-get update -q && apt-get install -y -q wget build-essential",
        "wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian$(. /etc/os-release; echo $VERSION_ID)"
        "/x86_64/cuda-keyring_1.1-1_all.deb && dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb",
        "apt-get update -q && apt-get install -y -q cuda-nvcc-13-0 cuda-cudart-dev-13-0 cuda-driver-dev-13-0",
        # TensorRT's C++ headers, for hub/trt_seanet.py's shim (the pip package's libnvinfer, same version)
        "apt-get install -y -q libnvinfer-headers-dev=11.4.0.106-1+cuda13.4",
    )
    # TensorRT and the ONNX exporter (the standard Python only), for scripts/trt_mimi_check.py
    .pip_install("tensorrt-cu13", "onnx", "onnxscript")
    .env({**ENV, "CUDA_HOME": "/usr/local/cuda-13.0"})
    .add_local_dir(ROOT / "data/modal_pyannote", "/models/pyannote")
    .add_local_file(ROOT / "data/ref/ref_24k.wav", f"{NA}/data/ref/ref_24k.wav")
    .add_local_dir(ROOT / "hub", f"{NA}/hub", ignore=["__pycache__", "*.pyc"])
    .add_local_file(ROOT / "scripts/mimi_file.py", f"{NA}/scripts/mimi_file.py")
    .add_local_file(ROOT / "scripts/throughput_server.py", f"{NA}/scripts/throughput_server.py")
    .add_local_file(ROOT / "scripts/parakeet_decoder_check.py", f"{NA}/scripts/parakeet_decoder_check.py")
    .add_local_file(ROOT / "scripts/diar_precision_check.py", f"{NA}/scripts/diar_precision_check.py")
    .add_local_file(ROOT / "scripts/roofline_server.py", f"{NA}/scripts/roofline_server.py")
    .add_local_file(ROOT / "scripts/mimi_pieces_check.py", f"{NA}/scripts/mimi_pieces_check.py")
    .add_local_file(ROOT / "scripts/fp16_weights_check.py", f"{NA}/scripts/fp16_weights_check.py")
    .add_local_file(ROOT / "scripts/bulk_bench.py", f"{NA}/scripts/bulk_bench.py")
    .add_local_file(ROOT / "scripts/mimi_static_check.py", f"{NA}/scripts/mimi_static_check.py")
    .add_local_file(ROOT / "scripts/model_ceiling.py", f"{NA}/scripts/model_ceiling.py")
    .add_local_file(ROOT / "scripts/seg_profile.py", f"{NA}/scripts/seg_profile.py")
    .add_local_file(ROOT / "scripts/compile_check.py", f"{NA}/scripts/compile_check.py")
    .add_local_file(ROOT / "scripts/aoti_check.py", f"{NA}/scripts/aoti_check.py")
    .add_local_file(ROOT / "scripts/trt_mimi_check.py", f"{NA}/scripts/trt_mimi_check.py")
    .add_local_file(ROOT / "scripts/episode_timeline.py", f"{NA}/scripts/episode_timeline.py")
)
app = modal.App("mimi-hub-bench", image=image)

FT_PYTHON = "/ft/bin/python"
# PYTHON_GIL=0: keep the GIL off even if an extension module asks for it; protobuf's C extension (imported by
# pyannote's telemetry) segfaults on import without the GIL, so its pure-Python implementation
FT_ENV = {"PY": FT_PYTHON, "PYTHON_GIL": "0", "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python"}
MPS_ENV = {"MIMI_HUB_THREADS": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "MPS": "1"}
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
    # 8 processes ran an L40S out of memory (5.5 GB each): fp16 encoder weights and a less fragmenting allocator
    "push2": [
        (["roofline_server.py"], {}),
        (["--clips", "32", "--workers", "2", "--procs", "6"], {"MIMI_HUB_THREADS": "4"}),
        (["--clips", "32", "--workers", "2", "--procs", "6"], {"MIMI_HUB_THREADS": "4", "MIMI_HUB_ASR_HALF": "1",
                                                               "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}),
        (["--clips", "32", "--workers", "2", "--procs", "8"], {"MIMI_HUB_THREADS": "4", "MIMI_HUB_ASR_HALF": "1",
                                                               "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}),
        (["--clips", "48", "--workers", "1", "--procs", "10"], {"MIMI_HUB_THREADS": "3", "MIMI_HUB_ASR_HALF": "1",
                                                                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}),
    ],
    # CUDA graphs for pyannote (no gain: compute-bound, removed); now Mimi as lockstep pieces in one batch, and the
    # ResNet in channels-last layout
    "push4": [
        (["mimi_pieces_check.py", "4:250", "8:250", "16:250"], {}),
        (["diar_precision_check.py", "fp16:32", "fp16:32:cl1", "fp16:64:cl1"], {}),
    ],
    # channels-last ResNet now the NVIDIA default; where the CPU goes (each stage alone), and thread counts
    "push5": [
        (["--stage-cpu", "--clips", "8"], {"MIMI_HUB_THREADS": "3"}),
        (["--stage-cpu", "--clips", "8"], {"MIMI_HUB_THREADS": "1"}),
        (["--clips", "48", "--workers", "1", "--procs", "10"], {"MIMI_HUB_THREADS": "3", "MIMI_HUB_ASR_HALF": "1",
                                                                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}),
        (["--clips", "48", "--workers", "1", "--procs", "10"], {"MIMI_HUB_THREADS": "1", "MIMI_HUB_ASR_HALF": "1",
                                                                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}),
    ],
    # one OpenBLAS thread (now the hub's default): ASR's CPU use, then more processes on the freed cores
    "push6": [
        (["--stage-cpu", "--clips", "8"], {"MIMI_HUB_THREADS": "1"}),
        (["--clips", "48", "--workers", "1", "--procs", "10"], {"MIMI_HUB_THREADS": "1", "MIMI_HUB_ASR_HALF": "1",
                                                                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}),
        (["--clips", "64", "--workers", "1", "--procs", "14"], {"MIMI_HUB_THREADS": "1", "MIMI_HUB_ASR_HALF": "1",
                                                                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}),
        (["--clips", "64", "--workers", "2", "--procs", "10"], {"MIMI_HUB_THREADS": "1", "MIMI_HUB_ASR_HALF": "1",
                                                                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}),
    ],
    # NVIDIA MPS: several processes' kernels at once on the GPU instead of time-slicing between them
    "push7": [
        (["--clips", "48", "--workers", "1", "--procs", "10"], {"MIMI_HUB_THREADS": "1", "MIMI_HUB_ASR_HALF": "1",
                                                                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}),
        (["--clips", "48", "--workers", "1", "--procs", "10"], {"MIMI_HUB_THREADS": "1", "MIMI_HUB_ASR_HALF": "1",
                                                                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                                                                "MPS": "1"}),
        (["--clips", "64", "--workers", "1", "--procs", "16"], {"MIMI_HUB_THREADS": "1", "MIMI_HUB_ASR_HALF": "1",
                                                                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                                                                "MPS": "1"}),
    ],
    # MPS on an H100: more processes (3.2 GB each)
    "push8": [
        (["--clips", "80", "--workers", "1", "--procs", "20"], {"MIMI_HUB_THREADS": "1", "MIMI_HUB_ASR_HALF": "1",
                                                                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                                                                "MPS": "1"}),
        (["--clips", "96", "--workers", "1", "--procs", "24"], {"MIMI_HUB_THREADS": "1", "MIMI_HUB_ASR_HALF": "1",
                                                                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                                                                "MPS": "1"}),
    ],
    # the segmentation model's LSTM in fp16 (DER 0.0002 on the server), with MPS
    "push9": [
        (["diar_precision_check.py", "fp16:32:cl1", "fp16:32:cl1:seg16"], {}),
        (["--clips", "64", "--workers", "1", "--procs", "16"], {"MIMI_HUB_THREADS": "1", "MIMI_HUB_ASR_HALF": "1",
                                                                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                                                                "MIMI_HUB_SEG_DTYPE": "fp16", "MPS": "1"}),
    ],
    # MPS on an H100: 20 / 24 processes (fp16 weights converted before the GPU, 4 loading at a time), and a profile
    # of one process's kernels under full load
    "push10": [
        (["--clips", "64", "--workers", "1", "--procs", "16", "--torch-profile"], MPS_ENV),
        (["--clips", "80", "--workers", "1", "--procs", "20"], MPS_ENV),
        (["--clips", "96", "--workers", "1", "--procs", "24"], MPS_ENV),
    ],
    # new defaults: fp16-stored autocast weights (identical), folded batch norms, fp16 segmentation LSTM
    "push11": [
        (["fp16_weights_check.py"], {}),
        (["diar_precision_check.py", "fp16:32:cl1", "fp16:32:cl1:seg16:fold"], {}),
        (["parakeet_decoder_check.py"], {}),
        (["--clips", "64", "--workers", "1", "--procs", "16"], MPS_ENV),
        (["--clips", "80", "--workers", "1", "--procs", "20"], MPS_ENV),
        (["--clips", "96", "--workers", "1", "--procs", "24"], MPS_ENV),
    ],
    # cross-episode batching (hub/bulk.py, one process) vs 16 processes with MPS, same container
    "bulk1": [
        (["bulk_bench.py", "--clips", "96", "--check", "2", "--prep", "24"], {}),
        (["bulk_bench.py", "--clips", "96", "--check", "0", "--prep", "24"],
         {"MIMI_BULK_MIMI_BATCH": "32", "MIMI_BULK_ASR_BATCH": "128", "MIMI_BULK_DIAR_ROWS": "512",
          "MIMI_BULK_DIAR_THREADS": "24", "MIMI_BULK_IN_FLIGHT": "48"}),
        (["fp16_weights_check.py"], {}),
        (["--clips", "64", "--workers", "1", "--procs", "16"], MPS_ENV),
    ],
    # bulk after keeping Parakeet's encoder output on the GPU (2 ASR workers): worker counts and batch sizes
    "bulk2": [
        (["bulk_bench.py", "--clips", "96", "--check", "1", "--prep", "24"], {}),
        (["bulk_bench.py", "--clips", "96", "--check", "0", "--prep", "24"], {"MIMI_BULK_MIMI_WORKERS": "2"}),
        (["bulk_bench.py", "--clips", "128", "--check", "0", "--prep", "24"],
         {"MIMI_BULK_MIMI_WORKERS": "2", "MIMI_BULK_ASR_WORKERS": "3", "MIMI_BULK_MIMI_BATCH": "32",
          "MIMI_BULK_ASR_BATCH": "128", "MIMI_BULK_DIAR_ROWS": "512", "MIMI_BULK_DIAR_THREADS": "24",
          "MIMI_BULK_IN_FLIGHT": "64"}),
    ],
    # bulk is GIL-bound (main process ~1.6 cores): longer Mimi chunks, bigger batches, several bulk processes
    "bulk3": [
        (["bulk_bench.py", "--clips", "96", "--check", "0", "--prep", "24"],
         {"MIMI_BULK_CHUNK_FRAMES": "1000", "MIMI_BULK_ASR_BATCH": "256", "MIMI_BULK_MIMI_BATCH": "32"}),
        (["bulk_bench.py", "--clips", "128", "--check", "0", "--prep", "24", "--procs", "3"],
         {"MIMI_BULK_CHUNK_FRAMES": "1000", "MIMI_BULK_ASR_BATCH": "256", "MIMI_BULK_MIMI_BATCH": "32",
          "MPS": "1"}),
        (["bulk_bench.py", "--clips", "192", "--check", "0", "--prep", "24", "--procs", "6"],
         {"MIMI_BULK_CHUNK_FRAMES": "1000", "MIMI_BULK_ASR_BATCH": "128", "MIMI_BULK_MIMI_BATCH": "16",
          "MPS": "1"}),
    ],
    # several bulk processes under MPS (audio now handed over in spool files: /dev/shm ran out with 3)
    "bulk4": [
        (["bulk_bench.py", "--clips", "128", "--check", "0", "--prep", "24", "--procs", "3"],
         {"MIMI_BULK_ASR_BATCH": "128", "MIMI_BULK_MIMI_BATCH": "32", "MPS": "1"}),
        (["bulk_bench.py", "--clips", "192", "--check", "0", "--prep", "30", "--procs", "6"],
         {"MIMI_BULK_ASR_BATCH": "128", "MIMI_BULK_MIMI_BATCH": "16", "MPS": "1"}),
    ],
    # GPU memory per bulk process (3 with big batches ran an H100 out of memory), then 4 with smaller batches
    "bulk5": [
        (["bulk_bench.py", "--clips", "96", "--check", "0", "--prep", "24"], {}),
        (["bulk_bench.py", "--clips", "128", "--check", "0", "--prep", "32", "--procs", "4"],
         {"MIMI_BULK_MIMI_BATCH": "8", "MIMI_BULK_ASR_BATCH": "64", "MIMI_BULK_DIAR_ROWS": "128",
          "MIMI_BULK_DIAR_THREADS": "8", "MIMI_BULK_IN_FLIGHT": "16",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "MPS": "1"}),
    ],
    # bulk mode on free-threaded Python 3.14t (no GIL) vs 3.12, same container; then more threads
    "ft1": [
        (["bulk_bench.py", "--clips", "96", "--check", "1", "--prep", "24"], {}),
        (["bulk_bench.py", "--clips", "96", "--check", "1", "--prep", "24"], FT_ENV),
        (["bulk_bench.py", "--clips", "128", "--check", "0", "--prep", "24"],
         {**FT_ENV, "MIMI_BULK_MIMI_WORKERS": "2", "MIMI_BULK_ASR_WORKERS": "4", "MIMI_BULK_DIAR_THREADS": "24"}),
    ],
    "ft2": [
        (["bulk_bench.py", "--clips", "96", "--check", "1", "--prep", "24"], FT_ENV),
        (["bulk_bench.py", "--clips", "128", "--check", "0", "--prep", "24"],
         {**FT_ENV, "MIMI_BULK_MIMI_WORKERS": "2", "MIMI_BULK_ASR_WORKERS": "4", "MIMI_BULK_DIAR_THREADS": "24"}),
    ],
    # free-threaded crash hunt: a stack on a segfault (faulthandler), and once without forcing the GIL off (Python
    # then names any extension module that needs it)
    "ft3": [
        (["bulk_bench.py", "--clips", "16", "--check", "0", "--prep", "8"], {**FT_ENV, "PYTHONFAULTHANDLER": "1"}),
        (["bulk_bench.py", "--clips", "16", "--check", "0", "--prep", "8"],
         {"PY": FT_PYTHON, "PYTHONFAULTHANDLER": "1", "PYTHONWARNINGS": "always"}),
    ],
    "ft4": [
        (["bulk_bench.py", "--clips", "16", "--check", "0", "--prep", "8"], {**FT_ENV, "PYTHONFAULTHANDLER": "1"}),
    ],
    "ft5": [
        (["bulk_bench.py", "--clips", "96", "--check", "1", "--prep", "24"], {**FT_ENV, "PYTHONFAULTHANDLER": "1"}),
        (["bulk_bench.py", "--clips", "128", "--check", "0", "--prep", "24"],
         {**FT_ENV, "PYTHONFAULTHANDLER": "1", "MIMI_BULK_MIMI_WORKERS": "2", "MIMI_BULK_ASR_WORKERS": "4",
          "MIMI_BULK_DIAR_THREADS": "24"}),
    ],
    # bulk mode with Mimi as CUDA graphs, on Python 3.12 and free-threaded 3.14t (protobuf's C extension removed)
    "ft6": [
        (["bulk_bench.py", "--clips", "96", "--check", "1", "--prep", "24"], {}),
        (["bulk_bench.py", "--clips", "96", "--check", "1", "--prep", "24"], {**FT_ENV, "PYTHONFAULTHANDLER": "1"}),
        (["bulk_bench.py", "--clips", "128", "--check", "0", "--prep", "24"],
         {**FT_ENV, "PYTHONFAULTHANDLER": "1", "MIMI_BULK_MIMI_WORKERS": "2", "MIMI_BULK_ASR_WORKERS": "4",
          "MIMI_BULK_DIAR_THREADS": "24"}),
    ],
    "ft7": [
        (["bulk_bench.py", "--clips", "96", "--check", "1", "--prep", "24"], {**FT_ENV, "PYTHONFAULTHANDLER": "1"}),
        (["bulk_bench.py", "--clips", "128", "--check", "0", "--prep", "24"],
         {**FT_ENV, "PYTHONFAULTHANDLER": "1", "MIMI_BULK_ASR_WORKERS": "4", "MIMI_BULK_DIAR_THREADS": "24"}),
    ],
    # both CUDA graphs (Mimi, TDT decoder) with free-threaded 3.14t, and on 3.12 for comparison
    "ft8": [
        (["bulk_bench.py", "--clips", "96", "--check", "1", "--prep", "24"], {**FT_ENV, "PYTHONFAULTHANDLER": "1"}),
        (["bulk_bench.py", "--clips", "128", "--check", "0", "--prep", "24"],
         {**FT_ENV, "PYTHONFAULTHANDLER": "1", "MIMI_BULK_ASR_WORKERS": "3", "MIMI_BULK_DIAR_THREADS": "24"}),
        (["bulk_bench.py", "--clips", "96", "--check", "0", "--prep", "24"], {}),
    ],
    # decoder graphs captured at load (mid-run capture stalled the device); deeper pipelines (IN_FLIGHT)
    "ft9": [
        (["bulk_bench.py", "--clips", "96", "--check", "1", "--prep", "24"], {**FT_ENV, "PYTHONFAULTHANDLER": "1"}),
        (["bulk_bench.py", "--clips", "128", "--check", "0", "--prep", "32"],
         {**FT_ENV, "PYTHONFAULTHANDLER": "1", "MIMI_BULK_IN_FLIGHT": "96"}),
        (["bulk_bench.py", "--clips", "160", "--check", "0", "--prep", "32"],
         {**FT_ENV, "PYTHONFAULTHANDLER": "1", "MIMI_BULK_IN_FLIGHT": "128", "MIMI_BULK_DIAR_THREADS": "24"}),
    ],
    # the hang with load-time decoder graphs: free-threaded / 3.12, under a watchdog that prints every thread's stack
    "ft10": [
        (["bulk_bench.py", "--clips", "32", "--check", "0", "--prep", "16"], {**FT_ENV, "BULK_WATCHDOG_S": "240"}),
        (["bulk_bench.py", "--clips", "32", "--check", "0", "--prep", "16"], {"BULK_WATCHDOG_S": "240"}),
        (["bulk_bench.py", "--clips", "96", "--check", "0", "--prep", "24"],
         {**FT_ENV, "BULK_WATCHDOG_S": "400", "MIMI_BULK_ASR_GRAPH": "0"}),
    ],
    # free-threaded with the fbank (vmap) lock: both graphs, then a deeper pipeline
    "ft11": [
        (["bulk_bench.py", "--clips", "96", "--check", "1", "--prep", "24"], {**FT_ENV, "BULK_WATCHDOG_S": "500"}),
        (["bulk_bench.py", "--clips", "128", "--check", "0", "--prep", "32"],
         {**FT_ENV, "BULK_WATCHDOG_S": "500", "MIMI_BULK_IN_FLIGHT": "96"}),
    ],
    # Python 3.12 (stable) with both CUDA graphs captured at load, and a deeper pipeline
    "g1": [
        (["bulk_bench.py", "--clips", "96", "--check", "1", "--prep", "24"], {"BULK_WATCHDOG_S": "500"}),
        (["bulk_bench.py", "--clips", "128", "--check", "0", "--prep", "32"],
         {"BULK_WATCHDOG_S": "500", "MIMI_BULK_IN_FLIGHT": "96"}),
    ],
    # both graphs captured at load, each decoder graph in its own pool (a shared one corrupted out-of-order replays)
    "g2": [
        (["bulk_bench.py", "--clips", "96", "--check", "1", "--prep", "24"], {"BULK_WATCHDOG_S": "400"}),
        (["bulk_bench.py", "--clips", "96", "--check", "1", "--prep", "24"], {**FT_ENV, "BULK_WATCHDOG_S": "400"}),
        (["bulk_bench.py", "--clips", "128", "--check", "0", "--prep", "32"],
         {**FT_ENV, "BULK_WATCHDOG_S": "400", "MIMI_BULK_IN_FLIGHT": "96"}),
    ],
    "g3": [
        (["bulk_bench.py", "--clips", "32", "--check", "0", "--prep", "16"], {"BULK_WATCHDOG_S": "240"}),
    ],
    # the candidate: free-threaded 3.14t + Mimi graphs, eager TDT search; twice, and with a deeper pipeline
    "g4": [
        (["bulk_bench.py", "--clips", "96", "--check", "1", "--prep", "24"], {**FT_ENV, "BULK_WATCHDOG_S": "300"}),
        (["bulk_bench.py", "--clips", "128", "--check", "0", "--prep", "32"],
         {**FT_ENV, "BULK_WATCHDOG_S": "300", "MIMI_BULK_IN_FLIGHT": "96"}),
        (["bulk_bench.py", "--clips", "96", "--check", "0", "--prep", "24"], {**FT_ENV, "BULK_WATCHDOG_S": "300"}),
    ],
    # each model's ceiling at large batches (eager, compiled), and the 16-process MPS setup at the 2 s / 5 s step
    "ceil1": [
        (["model_ceiling.py"], {}),
        (["model_ceiling.py", "--compile"], {}),
        (["--clips", "64", "--workers", "1", "--procs", "16"], MPS_ENV),
        (["--clips", "64", "--workers", "1", "--procs", "16"], {**MPS_ENV, "MIMI_HUB_DIAR_STEP": "0.5"}),
    ],
    # the segmentation model in parts (SincNet fp32 / TF32, LSTM), compiled ceilings, TF32 SincNet's DER, and bulk
    # mode (free-threaded, Mimi graphs, 96 in flight) at the 5 s step
    "seg1": [
        (["seg_profile.py"], {}),
        (["diar_precision_check.py", "fp16:32:cl1:seg16:fold", "fp16:32:cl1:seg16:fold:segtf32"], {}),
        (["model_ceiling.py", "--compile"], {}),
        (["bulk_bench.py", "--clips", "128", "--check", "0", "--prep", "32"],
         {**FT_ENV, "BULK_WATCHDOG_S": "300", "MIMI_BULK_IN_FLIGHT": "96", "MIMI_HUB_DIAR_STEP": "0.5"}),
    ],
    # compiled ceilings (models loaded first), and bulk mode on a 256-episode backlog at the 5 s step (128 episodes
    # with 96 in flight was mostly ramp-up and tail: no stage above 60% busy)
    "seg2": [
        (["model_ceiling.py", "--compile"], {}),
        (["bulk_bench.py", "--clips", "256", "--check", "0", "--prep", "32"],
         {**FT_ENV, "BULK_WATCHDOG_S": "400", "MIMI_BULK_IN_FLIGHT": "96", "MIMI_HUB_DIAR_STEP": "0.5"}),
    ],
    # bulk mode's steady state: a 1,024-episode backlog (85 h) at the 5 s step; each model through torch.compile
    "seg3": [
        (["bulk_bench.py", "--clips", "1024", "--check", "0", "--prep", "32", "--subtype", "PCM_16"],
         {**FT_ENV, "BULK_WATCHDOG_S": "850", "MIMI_BULK_IN_FLIGHT": "96", "MIMI_HUB_DIAR_STEP": "0.5"}),
        (["model_ceiling.py", "--compile", "parakeet"], {}),
        (["model_ceiling.py", "--compile", "mimi_seanet"], {}),
        (["model_ceiling.py", "--compile", "resnet"], {}),
        (["model_ceiling.py", "--compile", "segmentation"], {}),
    ],
    # the same models with max-autotune (slow compiles: a run of its own, within the function's 25 min)
    "seg4": [
        (["model_ceiling.py", "--compile", "parakeet", "--mode", "max-autotune-no-cudagraphs"], {}),
        (["model_ceiling.py", "--compile", "mimi_seanet", "--mode", "max-autotune-no-cudagraphs"], {}),
        (["model_ceiling.py", "--compile", "resnet", "--mode", "max-autotune-no-cudagraphs"], {}),
        (["model_ceiling.py", "--compile", "segmentation", "--mode", "max-autotune-no-cudagraphs"], {}),
    ],
    # the compiled Parakeet encoder and ResNet in the pipeline (MIMI_HUB_COMPILE, on by default on NVIDIA): checked
    # against eager first (which also fills the container's inductor cache for the processes after it), then the
    # 16-process MPS setup and bulk mode (expandable segments: 10 of 1,024 episodes failed in ASR with GPU memory full)
    "comp1": [
        (["compile_check.py"], {}),
        (["--clips", "64", "--workers", "1", "--procs", "16"], {**MPS_ENV, "MIMI_HUB_DIAR_STEP": "0.5"}),
        (["bulk_bench.py", "--clips", "1024", "--check", "0", "--prep", "32", "--subtype", "PCM_16"],
         {**FT_ENV, "BULK_WATCHDOG_S": "850", "MIMI_BULK_IN_FLIGHT": "96", "MIMI_HUB_DIAR_STEP": "0.5",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}),
    ],
    # AOTInductor in the pipeline (MIMI_HUB_COMPILE auto = aot): checked against eager (which builds the packages),
    # then the 16-process MPS setup and free-threaded bulk mode, both at the 5 s step
    "aot2": [
        (["compile_check.py", "aot"], {}),
        (["--clips", "64", "--workers", "1", "--procs", "16"], {**MPS_ENV, "MIMI_HUB_DIAR_STEP": "0.5"}),
        (["bulk_bench.py", "--clips", "1024", "--check", "0", "--prep", "32", "--subtype", "PCM_16"],
         {**FT_ENV, "BULK_WATCHDOG_S": "880", "MIMI_BULK_IN_FLIGHT": "96", "MIMI_HUB_DIAR_STEP": "0.5",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}),
    ],
    # bulk mode again with the encoder calls serialized across its ASR workers (40 of 1,024 failed in cuDNN)
    "aot3": [
        (["bulk_bench.py", "--clips", "1024", "--check", "0", "--prep", "32", "--subtype", "PCM_16"],
         {**FT_ENV, "BULK_WATCHDOG_S": "880", "MIMI_BULK_IN_FLIGHT": "96", "MIMI_HUB_DIAR_STEP": "0.5",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}),
    ],
    # where bulk mode's GPU time goes now (AOT models, 5 s step): 5 s of kernels per stage, 40 s into the run
    "prof1": [
        (["bulk_bench.py", "--clips", "512", "--check", "0", "--prep", "32", "--subtype", "PCM_16", "--profile",
          "40:5"],
         {**FT_ENV, "BULK_WATCHDOG_S": "880", "MIMI_BULK_IN_FLIGHT": "96", "MIMI_HUB_DIAR_STEP": "0.5",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}),
    ],
    # fbank in the ResNet batcher, 128-window calls and a 10 ms merge wait; then Mimi at batch 32; then its profile
    "opt1": [
        (["bulk_bench.py", "--clips", "1024", "--check", "0", "--prep", "32", "--subtype", "PCM_16"],
         {**FT_ENV, "BULK_WATCHDOG_S": "880", "MIMI_BULK_IN_FLIGHT": "96", "MIMI_HUB_DIAR_STEP": "0.5",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}),
        (["bulk_bench.py", "--clips", "1024", "--check", "0", "--prep", "32", "--subtype", "PCM_16"],
         {**FT_ENV, "BULK_WATCHDOG_S": "880", "MIMI_BULK_IN_FLIGHT": "96", "MIMI_HUB_DIAR_STEP": "0.5",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "MIMI_BULK_MIMI_BATCH": "32"}),
        (["bulk_bench.py", "--clips", "512", "--check", "0", "--prep", "32", "--subtype", "PCM_16", "--profile",
          "40:5"],
         {**FT_ENV, "BULK_WATCHDOG_S": "880", "MIMI_BULK_IN_FLIGHT": "96", "MIMI_HUB_DIAR_STEP": "0.5",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "MIMI_BULK_MIMI_BATCH": "32"}),
    ],
    # the ASR loop is the limit: its streams at high priority (the new default), then with 3 and 4 ASR workers
    "asr1": [
        (["bulk_bench.py", "--clips", "1024", "--check", "0", "--prep", "32", "--subtype", "PCM_16"],
         {**FT_ENV, "BULK_WATCHDOG_S": "880", "MIMI_BULK_IN_FLIGHT": "96", "MIMI_HUB_DIAR_STEP": "0.5",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}),
        (["bulk_bench.py", "--clips", "1024", "--check", "0", "--prep", "32", "--subtype", "PCM_16"],
         {**FT_ENV, "BULK_WATCHDOG_S": "880", "MIMI_BULK_IN_FLIGHT": "96", "MIMI_HUB_DIAR_STEP": "0.5",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "MIMI_BULK_ASR_WORKERS": "3"}),
        (["bulk_bench.py", "--clips", "1024", "--check", "0", "--prep", "32", "--subtype", "PCM_16"],
         {**FT_ENV, "BULK_WATCHDOG_S": "880", "MIMI_BULK_IN_FLIGHT": "96", "MIMI_HUB_DIAR_STEP": "0.5",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "MIMI_BULK_ASR_WORKERS": "4"}),
    ],
    # one bulk process stops scaling (bigger batches, more ASR workers: slower): 2 and 4 under MPS, the episodes in
    # flight split between them
    "mp1": [
        (["bulk_bench.py", "--clips", "1024", "--check", "0", "--prep", "32", "--subtype", "PCM_16", "--procs", "2"],
         {**FT_ENV, "MPS": "1", "BULK_WATCHDOG_S": "880", "MIMI_HUB_DIAR_STEP": "0.5",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "MIMI_BULK_IN_FLIGHT": "48"}),
        (["bulk_bench.py", "--clips", "1024", "--check", "0", "--prep", "32", "--subtype", "PCM_16", "--procs", "4"],
         {**FT_ENV, "MPS": "1", "BULK_WATCHDOG_S": "880", "MIMI_HUB_DIAR_STEP": "0.5",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "MIMI_BULK_IN_FLIGHT": "24"}),
    ],
    # every setup tops out at 1,900-1,970x, and more parallel work inside a process makes it slower: the CPU? (run
    # with --cpu 64): one bulk process with more prep workers, two under MPS, and the 16 hub processes
    "cpu1": [
        (["bulk_bench.py", "--clips", "1024", "--check", "0", "--prep", "48", "--subtype", "PCM_16"],
         {**FT_ENV, "BULK_WATCHDOG_S": "880", "MIMI_BULK_IN_FLIGHT": "96", "MIMI_HUB_DIAR_STEP": "0.5",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}),
        (["bulk_bench.py", "--clips", "1024", "--check", "0", "--prep", "48", "--subtype", "PCM_16", "--procs", "2"],
         {**FT_ENV, "BULK_WATCHDOG_S": "880", "MIMI_BULK_IN_FLIGHT": "96", "MIMI_HUB_DIAR_STEP": "0.5",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "MPS": "1", "MIMI_BULK_IN_FLIGHT": "48"}),
        (["--clips", "64", "--workers", "1", "--procs", "16"], {**MPS_ENV, "MIMI_HUB_DIAR_STEP": "0.5"}),
    ],
    # Mimi's SEANet as a TensorRT engine vs eager: 32 x 20 s (the ceiling test's shape) and 16 x 20 s (bulk's batch)
    "trt1": [
        (["trt_mimi_check.py", "32", "20"], {}),
        (["trt_mimi_check.py", "16", "20"], {}),
    ],
    # one episode at a time, warm (the on-demand case): a real podcast URL and the 43.8 min reference as an upload,
    # stage times from the request; no ad detection (no LLM here); diarization at the 2 s and 5 s step
    "lat1": [
        # each its own data dir: the hub would serve the episodes the run before converted
        (["episode_timeline.py"], {"MIMI_HUB_SEGMENTER": "off", "MIMI_HUB_DATA": "/tmp/hub_data_lat_a"}),
        (["episode_timeline.py"], {"MIMI_HUB_SEGMENTER": "off", "MIMI_HUB_DIAR_STEP": "0.5",
                                   "MIMI_HUB_DATA": "/tmp/hub_data_lat_b"}),
    ],
    # the TDT search: eager, as CUDA graphs captured on demand, and captured up front then replayed from a worker
    # thread (bulk mode's way, which got stuck before): tokens vs onnx_asr's loop and decode time
    "dec1": [
        (["parakeet_decoder_check.py"], {}),
        (["parakeet_decoder_check.py"], {"DECODER_GRAPH": "1"}),
        (["parakeet_decoder_check.py"], {"DECODER_GRAPH": "1", "DECODER_PRECAPTURE": "1"}),
    ],
    # the TDT search as CUDA graphs in bulk mode (captured at load; replays matched onnx_asr 548/548 and decoded
    # 2.7x faster on their own), against the same run without
    "dec2": [
        (["bulk_bench.py", "--clips", "1024", "--check", "0", "--prep", "32", "--subtype", "PCM_16"],
         {**FT_ENV, "BULK_WATCHDOG_S": "880", "MIMI_BULK_IN_FLIGHT": "96", "MIMI_HUB_DIAR_STEP": "0.5",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "MIMI_BULK_ASR_GRAPH": "1"}),
        (["bulk_bench.py", "--clips", "1024", "--check", "0", "--prep", "32", "--subtype", "PCM_16"],
         {**FT_ENV, "BULK_WATCHDOG_S": "880", "MIMI_BULK_IN_FLIGHT": "96", "MIMI_HUB_DIAR_STEP": "0.5",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}),
    ],
    # where bulk mode's decoder graphs hang (both ASR threads stuck in a GPU sync; alone they matched and ran 2.7x
    # faster): one ASR worker / no expandable segments (free-threaded), and the GIL Python
    "dec3a": [
        (["bulk_bench.py", "--clips", "128", "--check", "0", "--prep", "32", "--subtype", "PCM_16"],
         {**FT_ENV, "BULK_WATCHDOG_S": "600", "MIMI_BULK_IN_FLIGHT": "96", "MIMI_HUB_DIAR_STEP": "0.5",
          "MIMI_BULK_ASR_GRAPH": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
          "MIMI_BULK_ASR_WORKERS": "1"}),
        (["bulk_bench.py", "--clips", "128", "--check", "0", "--prep", "32", "--subtype", "PCM_16"],
         {**FT_ENV, "BULK_WATCHDOG_S": "600", "MIMI_BULK_IN_FLIGHT": "96", "MIMI_HUB_DIAR_STEP": "0.5",
          "MIMI_BULK_ASR_GRAPH": "1"}),
    ],
    "dec3b": [
        (["bulk_bench.py", "--clips", "128", "--check", "0", "--prep", "32", "--subtype", "PCM_16"],
         {"BULK_WATCHDOG_S": "600", "MIMI_BULK_IN_FLIGHT": "96", "MIMI_HUB_DIAR_STEP": "0.5",
          "MIMI_BULK_ASR_GRAPH": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}),
    ],
    # one episode, warm, with the hub's order and in latency mode (MIMI_HUB_LATENCY), at the 2 s and 5 s step
    "lat2": [
        (["episode_timeline.py"], {"MIMI_HUB_SEGMENTER": "off", "MIMI_HUB_DATA": "/tmp/hub_data_l0"}),
        (["episode_timeline.py"], {"MIMI_HUB_SEGMENTER": "off", "MIMI_HUB_LATENCY": "1",
                                   "MIMI_HUB_DATA": "/tmp/hub_data_l1"}),
        (["episode_timeline.py"], {"MIMI_HUB_SEGMENTER": "off", "MIMI_HUB_LATENCY": "1", "MIMI_HUB_DIAR_STEP": "0.5",
                                   "MIMI_HUB_DATA": "/tmp/hub_data_l2"}),
    ],
    # decoder graphs with their replays serialized across ASR workers (bulk, 1,024 episodes), and one episode in
    # latency mode with sharded VAD (MIMI_HUB_VAD_SHARDS 16, latency mode's default)
    "dec4": [
        (["bulk_bench.py", "--clips", "1024", "--check", "0", "--prep", "32", "--subtype", "PCM_16"],
         {**FT_ENV, "BULK_WATCHDOG_S": "600", "MIMI_BULK_IN_FLIGHT": "96", "MIMI_HUB_DIAR_STEP": "0.5",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "MIMI_BULK_ASR_GRAPH": "1"}),
        (["episode_timeline.py"], {"MIMI_HUB_SEGMENTER": "off", "MIMI_HUB_LATENCY": "1", "MIMI_HUB_DIAR_STEP": "0.5",
                                   "MIMI_HUB_DATA": "/tmp/hub_data_l3"}),
    ],
    # decoder graphs, one set shared by the ASR workers (a second slot's set hung): 128 then 1,024 episodes; then one
    # episode in latency mode with sharded VAD
    "dec5": [
        (["bulk_bench.py", "--clips", "128", "--check", "0", "--prep", "32", "--subtype", "PCM_16"],
         {**FT_ENV, "BULK_WATCHDOG_S": "600", "MIMI_BULK_IN_FLIGHT": "96", "MIMI_HUB_DIAR_STEP": "0.5",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "MIMI_BULK_ASR_GRAPH": "1"}),
        (["bulk_bench.py", "--clips", "1024", "--check", "0", "--prep", "32", "--subtype", "PCM_16"],
         {**FT_ENV, "BULK_WATCHDOG_S": "600", "MIMI_BULK_IN_FLIGHT": "96", "MIMI_HUB_DIAR_STEP": "0.5",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "MIMI_BULK_ASR_GRAPH": "1"}),
        (["episode_timeline.py"], {"MIMI_HUB_SEGMENTER": "off", "MIMI_HUB_LATENCY": "1", "MIMI_HUB_DIAR_STEP": "0.5",
                                   "MIMI_HUB_DATA": "/tmp/hub_data_l4"}),
    ],
    # AOTInductor prototype: Parakeet (one bucket) and the ResNet as packages, with the GIL and free-threaded
    "aot1": [
        (["aoti_check.py"], {"MIMI_HUB_COMPILE": "0"}),
        (["aoti_check.py"], {**FT_ENV, "MIMI_HUB_COMPILE": "0"}),
    ],
    # bulk mode alone (its free-threaded Python compiles its own kernels; after the others it ran out of time)
    "comp3": [
        (["bulk_bench.py", "--clips", "1024", "--check", "0", "--prep", "32", "--subtype", "PCM_16"],
         {**FT_ENV, "BULK_WATCHDOG_S": "880", "MIMI_BULK_IN_FLIGHT": "96", "MIMI_HUB_DIAR_STEP": "0.5",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}),
    ],
    # Parakeet's encoder compiled with dynamic shapes was slower than eager on the H100 (0.84x; static 1.66x): which
    # dynamic dim costs it
    "comp2": [
        (["model_ceiling.py", "--compile", "parakeet", "--dynamic", d], {}) for d in ("batch", "time", "all")
    ],
    # pushing one GPU: where its time goes, then a 32-clip backlog (160 min) at more processes
    "push1": [
        (["roofline_server.py"], {}),
        (["--workers", "1", "--profile"], {}),
        (["--clips", "32", "--workers", "2", "--procs", "6"], {"MIMI_HUB_THREADS": "4"}),
        (["--clips", "32", "--workers", "2", "--procs", "8"], {"MIMI_HUB_THREADS": "4"}),
        (["--clips", "32", "--workers", "1", "--procs", "8"], {"MIMI_HUB_THREADS": "4"}),
        (["--clips", "32", "--workers", "1", "--procs", "12"], {"MIMI_HUB_THREADS": "2"}),
    ],
}


# Guards against a hung run billing a GPU: each run is killed (with the worker processes it spawned) after
# RUN_TIMEOUT_S, the container after 25 min, and the local entrypoint cancels its calls when it gives up or is stopped.
RUN_TIMEOUT_S = 900
# torch.compile's kernels (hub MIMI_HUB_COMPILE) kept between containers: compiling Parakeet's buckets took ~6 min
CACHE = modal.Volume.from_name("mimi-hub-inductor-cache", create_if_missing=True)


@app.function(gpu="L40S", cpu=16.0, memory=MEM_GIB * 1024, timeout=25 * 60, volumes={"/cache": CACHE})
def bench(gpu: str, runs: list) -> list:
    import torch
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/cache/inductor"
    os.environ["MIMI_HUB_AOTI"] = "/cache/aoti"  # hub/aoti.py's packages, shared by both Pythons and later runs
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
    mps_env = {"CUDA_MPS_PIPE_DIRECTORY": "/tmp/mps-pipe", "CUDA_MPS_LOG_DIRECTORY": "/tmp/mps-log"}

    def mps(on):
        """NVIDIA's Multi-Process Service: kernels of several processes run on the GPU at once, not in turns."""
        e = {**os.environ, **mps_env}
        if on:
            for d in mps_env.values():
                os.makedirs(d, exist_ok=True)
            r = subprocess.run(["nvidia-cuda-mps-control", "-d"], env=e, capture_output=True, text=True)
            return f"exit {r.returncode} {r.stderr.strip()[-200:]}"
        subprocess.run(["nvidia-cuda-mps-control"], input="quit\n", env=e, capture_output=True, text=True)

    for args, env in runs:
        t = time.time()
        env = dict(env)
        use_mps = env.pop("MPS", None) == "1"
        python = env.pop("PY", sys.executable)
        if use_mps:
            print(f"MPS start: {mps(True)}", flush=True)
            env.update(mps_env)
        cmd = ([f"{NA}/scripts/{args[0]}", *args[1:]] if args and args[0].endswith(".py")
               else [f"{NA}/scripts/throughput_server.py", "--label", gpu, *args])
        proc = subprocess.Popen([python, *cmd], cwd=NA, env={**os.environ, **env}, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, start_new_session=True)
        try:
            stdout, stderr = proc.communicate(timeout=RUN_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)  # the run and every process it started
            stdout, stderr = proc.communicate()
            stderr += f"\nkilled after {RUN_TIMEOUT_S} s"
        p = types.SimpleNamespace(returncode=proc.returncode, stdout=stdout, stderr=stderr)
        CACHE.commit()  # what this run compiled (even if it then failed), for the next containers
        rows = [{"tag": ln.split(" ", 1)[0], **json.loads(ln.split(" ", 1)[1])} for ln in p.stdout.splitlines()
                if re.match(r"^[A-Z]+ \{", ln)]
        if p.returncode or not rows:
            fatal = p.stderr.find("Fatal Python error")  # faulthandler: the crash and every thread's stack
            if fatal < 0:
                fatal = p.stderr.find("--- thread ")  # scripts/bulk_bench.py's watchdog: where every thread was
            shown = p.stderr[fatal:fatal + 20000] if fatal >= 0 else p.stderr[-3000:]
            print(f"RUN FAILED {args} {env} (exit {p.returncode}):\n{shown}", flush=True)
            out.append({**info, "args": args, "env": env, "error": p.stderr[-1500:]})
            continue
        if use_mps:
            mps(False)
        if any(r.get("error") for r in rows):  # e.g. a worker process that died: why
            print(f"RUN STDERR {args}:\n{p.stderr[-4000:]}", flush=True)
        for r in rows:
            r = {**info, **r, "args": args, "env": env, "mps": use_mps, "run_s": round(time.time() - t)}
            print("RESULT " + json.dumps(r), flush=True)
            out.append(r)
    return out


@app.local_entrypoint()
def main(gpus: str = "L40S,H100", plan: str = "baseline", cpu: float = 16.0, mem: int = MEM_GIB, wait_min: float = 25):
    runs = PLANS[plan]
    names = [g for g in (x.strip() for x in gpus.split(",")) if g]
    calls = {g: bench.with_options(gpu=g, cpu=cpu, memory=mem * 1024).spawn(g, runs) for g in names}

    def stop(*_):  # stopped from outside (SIGTERM / SIGHUP): don't leave the containers running
        for call in calls.values():
            call.cancel()
        raise SystemExit("stopped; remote calls cancelled")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGHUP, stop)
    out = ROOT / "outputs/modal_hub_bench.jsonl"
    out.parent.mkdir(exist_ok=True)
    try:
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
                        r["plan"], r["cpu"], r["mem_gib"] = plan, cpu, mem
                        f.write(json.dumps(r) + "\n")
                        if r.get("error"):  # (TIMELINE rows carry "error": null)
                            print(f"{g:6s} ERROR {r.get('args')} {r['error'][-300:]}", flush=True)
                            continue
                        if r.get("tag") != "RESULT":
                            skip = ("gpu", "device", "vcpus", "cpu_model", "args", "env", "plan", "cpu", "mem_gib")
                            keep = {k: v for k, v in r.items() if k not in skip}
                            print(f"{g:6s} {json.dumps(keep)}", flush=True)
                            continue
                        usd_h = (PRICE.get(g, 0) + cpu * CPU_PRICE + mem * MEM_PRICE) * 3600 / r["x_rt"]
                        print(f"{g:6s} procs={r['procs']} workers={r['workers']} env={r['env']} "
                              f"{r['x_rt']:6.1f}x  gpu {r['gpu_busy_pct']:5.1f}%  "
                              f"cpu {r['cpu_cores_used']:4.1f} cores  "
                              f"${usd_h:.4f}/audio h  stages {r['clip_stage_s']}  "
                              f"peak {r.get('gpu_peak_gb_per_proc')} GB/proc",
                              flush=True)
            time.sleep(5)
    finally:  # however this ends (deadline, error, Ctrl-C): nothing left running in the cloud
        for g, call in calls.items():
            call.cancel()
            print(f"{g}: no result within {wait_min:g} min (or stopped); cancelled", flush=True)
