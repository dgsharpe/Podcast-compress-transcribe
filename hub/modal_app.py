"""mimi-hub as your own server on Modal: a GPU only while it works (it scales to zero), ad detection through an
OpenRouter key; no machine or subscription of your own.

    modal secret create mimi-hub MIMI_HUB_TOKEN=<a long random string> [OPENROUTER_API_KEY=<key>] [HF_TOKEN=<token>]
    modal deploy hub/modal_app.py              -> https://<workspace>--mimi-hub-serve.<region>.modal.run

then in the app, Settings > Advanced > Mimi server: that URL followed by /t/<the token> (the app sends no header;
hub/mimi_hub.py takes the token from the path). The server refuses to start without one: its URL is public.

The first start converts the models and compiles their kernels (several minutes, kept on the mimi-hub-models
Volume). Then Modal snapshots the container with its models loaded (2-3 snapshots per GPU type, each made on a cold
start of ~1 min), and a cold start restores one: models ready ~8 s after the request, against ~45 s loading them
(2026-11-02, L40S; transcripts after a restore agreed with a fresh load's as well as two fresh loads did). Episodes, transcripts and .mimi files live on the mimi-hub-data Volume. Without OPENROUTER_API_KEY in
the secret, ad detection uses the key the app sends with each conversion (X-OpenRouter-Key; held in memory only).

Diarization (speaker labels): pyannote's community-1 on the GPU if it's available: it's gated, so accept its
conditions on huggingface.co and put HF_TOKEN in the secret (downloaded once to the models Volume), or keep a copy
in data/modal_pyannote (cp -rL of its HF cache snapshot), which is baked into the image. Otherwise the CPU
diarizer (sherpa-onnx).

Settings, from the environment at deploy time: MIMI_MODAL_GPU (L40S,H100: Modal GPU types in order of preference,
the next when one has no capacity), MIMI_MODAL_CPU
(16 cores: latency mode's sharded VAD), MIMI_MODAL_MEMORY_GIB (32), MIMI_MODAL_IDLE_S (20: how long the container
stays up after the last request or job; after a warm-up, 30 s more), MIMI_MODAL_MAX_BUSY_S (7200: a hung job is let go after this),
MIMI_MODAL_APP (mimi-hub; another name deploys a second server with its own data Volume). Modal bills while the
container is up: an L40S with 16 cores is ~$3 an hour; a launch restored from its snapshot ~$0.02-0.05, plus ~$0.02
per hour of audio. Hub settings (MIMI_HUB_*, see
hub/mimi_hub.py) can go in the secret.

Background work outlives the request that started it: while the hub is busy, the container keeps a request to its
own GET /v1/hold in flight, so Modal doesn't scale it down mid-conversion.
"""

import os
import shutil
import threading
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
NA = "/root/na"  # the repository's layout inside the container (hub/, scripts/mimi_file.py)
APP = os.environ.get("MIMI_MODAL_APP", "mimi-hub")  # (another name: a second server, with its own data Volume)
# an L40S matched the H100 (cold start, warm-up, then a 43 min episode: first audio 1.8 vs 2.7 s, ready 13 s) at half
# the price; L4s were often unavailable (scripts/remote_episode.py, 2026-11-02)
# a list, tried in order: one type alone can be out of capacity for many minutes (L40S, 2026-11-02: 9 min); each
# type builds its own kernel packages on its first start (H100's are made, L40S's too)
GPU = [g.strip() for g in os.environ.get("MIMI_MODAL_GPU", "L40S,H100").split(",") if g.strip()]
CPU = float(os.environ.get("MIMI_MODAL_CPU", "16"))
MEMORY_GIB = int(os.environ.get("MIMI_MODAL_MEMORY_GIB", "32"))
# short: an episode's work holds the container (GET /v1/hold), the app checks on ads every 15 s and its transcript
# every 10 s until they're ready, and after POST /v1/warm the hub holds it MIMI_HUB_WARM_LINGER_S (30)
IDLE_S = int(os.environ.get("MIMI_MODAL_IDLE_S", "20"))
# a hung job mustn't keep a GPU billing: after this long busy without a break, the container is let go
MAX_BUSY_S = int(os.environ.get("MIMI_MODAL_MAX_BUSY_S", "7200"))
PYANNOTE_LOCAL = ROOT / "data/modal_pyannote"
PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"
SHERPA = "https://github.com/k2-fsa/sherpa-onnx/releases/download"


def _download():
    """The public models, into the image's HF cache (hub/prefetch.py, and the .nemo for hub/parakeet_native.py)."""
    import sys
    sys.path.insert(0, "/opt/prefetch")
    import prefetch
    from huggingface_hub import hf_hub_download
    prefetch.main()
    hf_hub_download("nvidia/parakeet-tdt-0.6b-v2", "parakeet-tdt-0.6b-v2.nemo")


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libsndfile1", "ffmpeg", "curl", "bzip2", "wget", "build-essential")
    # CUDA's compiler and headers, for hub/aoti.py's ahead-of-time packages (built once, on the first start)
    .run_commands(
        "wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian$(. /etc/os-release; echo $VERSION_ID)"
        "/x86_64/cuda-keyring_1.1-1_all.deb && dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb",
        "apt-get update -q && apt-get install -y -q cuda-nvcc-13-0 cuda-cudart-dev-13-0 cuda-driver-dev-13-0",
    )
    .pip_install("torch==2.14.1", "torchaudio", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("transformers==5.22.1", "onnx-asr[cpu,hub]==0.12.0", "onnxruntime==1.31.1", "pyannote-audio==4.0.7",
                 "sherpa-onnx>=2.0.1", "librosa", "soundfile", "soxr", "scipy", "threadpoolctl", "fastapi>=0.162",
                 "python-multipart", "httpx", "huggingface_hub", "safetensors")
    # the CPU diarizer's models (hub/fetch_models.sh), copied to the models Volume on start
    .run_commands(
        "mkdir -p /opt/sherpa && "
        f"curl -sL {SHERPA}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
        " | tar -xj -C /opt/sherpa && "
        f"curl -sL -o /opt/sherpa/wespeaker_en_voxceleb_resnet34_LM.onnx "
        f"{SHERPA}/speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx")
    .add_local_file(ROOT / "hub/prefetch.py", "/opt/prefetch/prefetch.py", copy=True)
    .run_function(_download)
    .env({"CUDA_HOME": "/usr/local/cuda-13.0", "HF_HUB_OFFLINE": "1"})
)
if PYANNOTE_LOCAL.exists():
    image = image.add_local_dir(PYANNOTE_LOCAL, "/opt/pyannote")
image = (image
         .add_local_dir(ROOT / "hub", f"{NA}/hub", ignore=["__pycache__", "*.pyc", "test_*.py"])
         .add_local_file(ROOT / "scripts/mimi_file.py", f"{NA}/scripts/mimi_file.py"))

app = modal.App(APP, image=image)
data = modal.Volume.from_name(f"{APP}-data", create_if_missing=True, version=2)
models = modal.Volume.from_name("mimi-hub-models", create_if_missing=True, version=2)


def _prepare():
    """Settings and models before the hub is imported (it reads its settings then)."""
    if not os.environ.get("MIMI_HUB_TOKEN"):
        raise RuntimeError("set MIMI_HUB_TOKEN in the mimi-hub secret: the server's URL is public")
    defaults = {
        "MIMI_HUB_DATA": "/data", "MIMI_HUB_MODELS": "/models/hub",
        # inductor's build directory on the local disk: building in it on the Volume failed (CppCompileError) after
        # two packages; the packages themselves (MIMI_HUB_MODELS/aoti) are what's kept
        "TORCHINDUCTOR_CACHE_DIR": "/tmp/inductor-cache",
        "MIMI_HUB_AOTI_MARCH": "x86-64-v3",  # every Modal host has it: packages built once serve all of them
        "MIMI_HUB_LATENCY": "1",  # one episode at a time, as fast as possible (sharded VAD over the cores)
        "MIMI_HUB_DIAR_STEP": "0.5",  # pyannote's window step, 5 s (the hub's default is 2 s)
        "MIMI_HUB_PRELOAD": "1",
        "MIMI_HUB_SEGMENTER": "openrouter:anthropic/claude-haiku-5.5",
    }
    for k, v in defaults.items():
        os.environ.setdefault(k, v)
    hub_models = Path(os.environ["MIMI_HUB_MODELS"])
    hub_models.mkdir(parents=True, exist_ok=True)
    for f in Path("/opt/sherpa").iterdir():
        if not (hub_models / f.name).exists():
            (shutil.copytree if f.is_dir() else shutil.copy)(f, hub_models / f.name)
    if Path("/opt/pyannote").exists():
        os.environ.setdefault("MIMI_HUB_PYANNOTE_PATH", "/opt/pyannote")
    elif os.environ.get("HF_TOKEN"):
        # once, to the models Volume (gated: the token's account must have accepted its conditions); in a child
        # process, as the image's HF_HUB_OFFLINE=1 is read on import
        import subprocess
        import sys
        r = subprocess.run([sys.executable, "-c", "import sys; from huggingface_hub import snapshot_download; "
                            f"print(snapshot_download({PYANNOTE_MODEL!r}, cache_dir='/models/hf'))"],
                           env={**os.environ, "HF_HUB_OFFLINE": "0"}, capture_output=True, text=True, timeout=600)
        if r.returncode == 0:
            os.environ.setdefault("MIMI_HUB_PYANNOTE_PATH", r.stdout.strip().splitlines()[-1])
            models.commit()
        else:
            print(f"pyannote not downloaded, the CPU diarizer instead: {r.stderr[-500:]}", flush=True)


def _keepalive(hub, base):
    """While the hub is busy, a GET /v1/hold of its own in flight (Modal keeps a container with a request in
    flight); after each busy spell, the Volumes committed."""
    import httpx
    headers = {"Authorization": f"Bearer {os.environ['MIMI_HUB_TOKEN']}"}
    was_busy, since = False, None
    while True:
        if hub.busy() and base[0]:
            was_busy, since = True, since or time.time()
            if time.time() - since > MAX_BUSY_S:
                print(f"keepalive: busy for over {MAX_BUSY_S} s; no longer holding the container", flush=True)
                time.sleep(60)
                continue
            try:
                # short: a request of 300 s was cut off (and its cancellation killed the container)
                httpx.get(f"{base[0]}/v1/hold", params={"s": 60}, headers=headers, timeout=90)
            except Exception as e:
                print(f"keepalive: {type(e).__name__}: {e}", flush=True)
                time.sleep(5)
            continue
        if was_busy and not hub.busy():
            was_busy, since = False, None
            for v in (data, models):
                try:
                    v.commit()
                except Exception as e:
                    print(f"commit: {type(e).__name__}: {e}", flush=True)
        time.sleep(2)


# MIMI_MODAL_SNAPSHOT=gpu (default): the models are loaded onto the GPU once and the container's memory, GPU memory
# included, is snapshotted (Modal's GPU memory snapshots, alpha); later cold starts restore it instead of loading.
# off: every cold start loads them.
SNAPSHOT = os.environ.get("MIMI_MODAL_SNAPSHOT", "gpu")


@app.cls(gpu=GPU, cpu=CPU, memory=MEMORY_GIB * 1024, volumes={"/data": data, "/models": models},
         secrets=[modal.Secret.from_name("mimi-hub")], timeout=3600, scaledown_window=IDLE_S, max_containers=1,
         enable_memory_snapshot=SNAPSHOT == "gpu",
         experimental_options={"enable_gpu_snapshot": True} if SNAPSHOT == "gpu" else None)
@modal.concurrent(max_inputs=200)
class Hub:
    @modal.enter(snap=True)
    def load(self):
        """Settings, the hub and (with a snapshot) its models: what a snapshot keeps."""
        import sys
        t0 = time.time()
        _prepare()
        t1 = time.time()
        sys.path.insert(0, NA)
        from hub import mimi_hub
        print(f"startup: settings {t1 - t0:.1f} s, hub imported {time.time() - t1:.1f} s", flush=True)
        if SNAPSHOT == "gpu":
            mimi_hub.warm(vad_workers=False)  # (worker processes and threads: after the restore)
        self.hub = mimi_hub

    @modal.enter(snap=False)
    def started(self):
        """After a restore (or the first start): what a snapshot can't hold."""
        t0 = time.time()
        if SNAPSHOT == "gpu":
            self.hub.models()  # the VAD worker processes
        self.base = [None]  # the public URL, from the first request (a self-request reaches this container)
        threading.Thread(target=_keepalive, args=(self.hub, self.base), daemon=True, name="modal-keepalive").start()
        print(f"startup: after restore {time.time() - t0:.1f} s", flush=True)

    @modal.asgi_app(label=f"{APP}-serve")
    def serve(self):
        base, app = self.base, self.hub.app

        @app.middleware("http")
        async def own_url(request, call_next):
            if base[0] is None and request.headers.get("host"):
                base[0] = f"https://{request.headers['host']}"
            return await call_next(request)
        return app
