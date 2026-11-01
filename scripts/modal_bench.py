"""mimi-hub's pipeline on Modal GPUs: the same 43.8 min episode, each stage alone and then overlapped as the hub runs it.

    uvx --from modal modal run scripts/modal_bench.py [--gpus T4,L4,A10,L40S,A100-40GB,H100]

Stages (as on the server): Mimi encode (transformers MimiModel, fp32), Parakeet TDT 0.6B v2 (onnx_asr; here on
onnxruntime's CUDA provider), pyannote community-1 with hub/pyannote_gpu.py's batched embedding step (stock pyannote
timed too). "pipeline" = Mimi encode || (ASR -> diarization), the hub's order on one GPU. Needs locally:
data/ref/ref_24k.wav and data/modal_pyannote/ (the community-1 snapshot, cp -rL out of the HF cache: it's gated,
so the container gets the files instead of a token). Results: outputs/modal_bench.json.
"""

import json
import os
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
# $/s, modal.com/pricing (2026-10-31)
PRICE = {"T4": 0.000164, "L4": 0.000222, "A10": 0.000306, "L40S": 0.000542, "A100-40GB": 0.000583,
         "A100-80GB": 0.000694, "RTX-PRO-6000": 0.000842, "H100": 0.001097, "H200": 0.001261, "B200": 0.001736}
CPU_CORES, MEM_GIB = 8.0, 24
CPU_MEM_PRICE = CPU_CORES * 0.0000131 + MEM_GIB * 0.00000222  # $/s for the container's CPU + memory


def _download():
    from huggingface_hub import snapshot_download
    snapshot_download("kyutai/mimi")
    # a plain directory: onnxruntime rejects external weights that resolve outside the model dir (HF cache symlinks)
    snapshot_download("istupakov/parakeet-tdt-0.6b-v2-onnx", local_dir="/models/parakeet")
    import onnx_asr
    onnx_asr.load_vad("silero")


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libsndfile1")
    # all CUDA 13: onnxruntime-gpu 1.31's CUDA provider needs libcublasLt.so.13 (with cu128 torch it silently fell
    # back to the CPU)
    .pip_install("torch==2.14.1", "torchaudio", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("onnxruntime-gpu[cuda,cudnn]==1.31.1", "transformers>=4.50", "onnx-asr[hub]>=0.12", "librosa",
                 "soundfile", "scipy", "pyannote-audio>=4", "huggingface_hub")
    .run_function(_download)
    # CUDA 13's pip wheels install under nvidia/cu13/lib, which onnxruntime's preload_dlls() doesn't search
    .env({"LD_LIBRARY_PATH": "/usr/local/lib/python3.12/site-packages/nvidia/cu13/lib:"
                             "/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib"})
    .add_local_dir(ROOT / "data/modal_pyannote", "/models/pyannote")
    .add_local_file(ROOT / "data/ref/ref_24k.wav", "/data/ref_24k.wav")
    .add_local_file(ROOT / "scripts/mimi_file.py", "/root/mimi_file.py")
    .add_local_file(ROOT / "hub/pyannote_gpu.py", "/root/pyannote_gpu.py")
)
app = modal.App("mimi-pipeline-bench", image=image)


@app.function(gpu="T4", cpu=CPU_CORES, memory=MEM_GIB * 1024, timeout=3600)
def bench(gpu: str) -> dict:
    import threading
    import time
    import types

    import librosa
    import onnxruntime as ort
    import soundfile as sf
    import torch

    ort.preload_dlls()  # CUDA / cuDNN from the torch wheels' nvidia-* packages
    import mimi_file
    import onnx_asr
    import pyannote_gpu
    from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization

    res = {"gpu": gpu, "device": torch.cuda.get_device_name(0), "torch": torch.__version__, "ort": ort.__version__}
    wav, sr = sf.read("/data/ref_24k.wav", dtype="float32")
    res["audio_s"] = len(wav) / sr
    w16 = librosa.resample(wav, orig_sr=sr, target_sr=16000)

    t0 = time.perf_counter()
    mimi = mimi_file.load_model("cuda")
    base = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v2", "/models/parakeet", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    res["asr_providers"] = base.asr._encoder.get_providers()
    if "CUDAExecutionProvider" not in res["asr_providers"]:  # don't pay for a GPU benchmarking a CPU fallback
        raise RuntimeError(f"Parakeet encoder not on CUDA: {res['asr_providers']}")
    asr = base.with_vad(onnx_asr.load_vad("silero", providers=["CPUExecutionProvider"])).with_timestamps()
    pipe = pyannote_gpu.load("cuda", path="/models/pyannote")
    res["load_s"] = round(time.perf_counter() - t0, 1)

    def timed(f):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = f()
        torch.cuda.synchronize()
        return out, round(time.perf_counter() - t, 1)

    encode = lambda: mimi_file.encode_array(mimi, wav)  # noqa: E731
    transcribe = lambda: list(asr.recognize(w16, sample_rate=16000))  # noqa: E731
    diarize = lambda: pipe({"waveform": torch.from_numpy(w16)[None], "sample_rate": 16000})  # noqa: E731

    # warm-up: kernel selection / graph setup on 2 min, outside the timings
    mimi_file.encode_array(mimi, wav[: 120 * sr])
    list(asr.recognize(w16[: 120 * 16000], sample_rate=16000))
    pipe({"waveform": torch.from_numpy(w16[: 120 * 16000])[None], "sample_rate": 16000})

    _, res["mimi_s"] = timed(encode)
    segs, res["asr_s"] = timed(transcribe)
    res["asr_words"] = sum(len(s.text.split()) for s in segs)
    out, res["diar_s"] = timed(diarize)
    ann = getattr(out, "speaker_diarization", out)
    res["speakers_min"] = sorted((round(ann.label_duration(k) / 60, 1) for k in ann.labels()), reverse=True)[:4]

    batched = pipe.get_embeddings
    pipe.get_embeddings = types.MethodType(SpeakerDiarization.get_embeddings, pipe)
    _, res["diar_stock_s"] = timed(diarize)
    pipe.get_embeddings = batched

    times = {}

    def run(name, f):
        _, times[name] = timed(f)

    def asr_then_diar():
        run("asr", transcribe)
        run("diar", diarize)

    t = time.perf_counter()
    threads = [threading.Thread(target=run, args=("mimi", encode)), threading.Thread(target=asr_then_diar)]
    [th.start() for th in threads]
    [th.join() for th in threads]
    res["pipeline_s"] = round(time.perf_counter() - t, 1)
    res["pipeline_stages"] = times
    res = json.loads(json.dumps(res, default=str))  # plain types: torch.__version__ is a torch str subclass
    print("RESULT " + json.dumps(res), flush=True)  # also in `modal app logs`, should the caller be gone
    return res


@app.function(gpu="T4", cpu=CPU_CORES, memory=MEM_GIB * 1024, timeout=1800)
def asr_profile(gpu: str) -> dict:
    """Where Parakeet's time goes on 43.8 min: VAD, preprocessing, encoder, decoder (on CUDA, then on 1 CPU thread)."""
    import time

    import librosa
    import onnxruntime as ort
    import soundfile as sf

    ort.preload_dlls()
    import onnx_asr

    wav, sr = sf.read("/data/ref_24k.wav", dtype="float32")
    w16 = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    vad = onnx_asr.load_vad("silero", providers=["CPUExecutionProvider"])
    totals, counts = {}, {}

    def wrap(obj, attr, key, method="run"):
        target = getattr(obj, attr)
        f = getattr(target, method) if method else target

        def timed(*a, **k):
            t = time.perf_counter()
            try:
                return f(*a, **k)
            finally:
                totals[key] = totals.get(key, 0.0) + time.perf_counter() - t
                counts[key] = counts.get(key, 0) + 1
        if method:
            setattr(target, method, timed)
        else:
            setattr(obj, attr, timed)

    wrap(vad, "_model", "vad")
    res = {"gpu": gpu, "vcpus": os.cpu_count()}
    # ORT's CUDA provider benchmarks every cuDNN conv algorithm for each new input shape by default (EXHAUSTIVE);
    # VAD batches all differ in length, so it searches on every encoder call
    for search in ("EXHAUSTIVE", "HEURISTIC"):
        base = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v2", "/models/parakeet",
                                   providers=[("CUDAExecutionProvider", {"cudnn_conv_algo_search": search}),
                                              "CPUExecutionProvider"])
        wrap(base.asr, "_encoder", "encoder")
        wrap(base.asr, "_decoder_joint", "decoder")
        wrap(base.asr, "_preprocessor", "preprocess", method=None)
        asr = base.with_vad(vad).with_timestamps()
        list(asr.recognize(w16[: 60 * 16000], sample_rate=16000))  # warm-up
        totals.clear()
        counts.clear()
        t = time.perf_counter()
        segs = list(asr.recognize(w16, sample_rate=16000))
        res[search.lower()] = {"total_s": round(time.perf_counter() - t, 1),
                               **{k: round(v, 1) for k, v in totals.items()},
                               "words": sum(len(x.text.split()) for x in segs)}
    return res


@app.function(gpu="H100", cpu=CPU_CORES, memory=MEM_GIB * 1024, timeout=3600)
def throughput(gpu: str, ns: list, clip_min: float = 5.0, search: str = "HEURISTIC") -> dict:
    """Aggregate speed with n episodes in flight at once (different clip_min-long slices of the episode), each run
    as the hub runs one: Mimi encode || (ASR -> diarization). Mimi and Parakeet are shared; pyannote is per episode."""
    import threading
    import time

    import librosa
    import onnxruntime as ort
    import soundfile as sf
    import torch

    ort.preload_dlls()
    import mimi_file
    import onnx_asr
    import pyannote_gpu

    wav, sr = sf.read("/data/ref_24k.wav", dtype="float32")
    n_max, step = max(ns), int(clip_min * 60 * sr)
    clips = [wav[k * step:(k + 1) * step] for k in range(n_max)]
    assert all(len(c) == step for c in clips), "episode too short for that many clips"
    clips16 = [librosa.resample(c, orig_sr=sr, target_sr=16000) for c in clips]
    mimi = mimi_file.load_model("cuda")
    base = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v2", "/models/parakeet",
                               providers=[("CUDAExecutionProvider", {"cudnn_conv_algo_search": search}),
                                          "CPUExecutionProvider"])
    asr = base.with_vad(onnx_asr.load_vad("silero", providers=["CPUExecutionProvider"])).with_timestamps()
    pipes = [pyannote_gpu.load("cuda", path="/models/pyannote") for _ in range(n_max)]

    def episode(k):
        enc = threading.Thread(target=mimi_file.encode_array, args=(mimi, clips[k]))
        enc.start()
        list(asr.recognize(clips16[k], sample_rate=16000))
        pipes[k]({"waveform": torch.from_numpy(clips16[k])[None], "sample_rate": 16000})
        enc.join()

    for k in range(n_max):  # warm-up: every pyannote copy, and the shapes of a full clip
        pipes[k]({"waveform": torch.from_numpy(clips16[k][: 30 * 16000])[None], "sample_rate": 16000})
    episode(0)
    res = {"gpu": gpu, "device": torch.cuda.get_device_name(0), "clip_s": clip_min * 60, "search": search, "runs": {}}
    for n in ns:
        torch.cuda.synchronize()
        t = time.perf_counter()
        threads = [threading.Thread(target=episode, args=(k,)) for k in range(n)]
        [th.start() for th in threads]
        [th.join() for th in threads]
        torch.cuda.synchronize()
        wall = time.perf_counter() - t
        res["runs"][str(n)] = {"wall_s": round(wall, 1), "x_rt": round(n * clip_min * 60 / wall, 1),
                               "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
        print(f"THROUGHPUT {gpu} n={n}: {res['runs'][str(n)]}", flush=True)
    return res


@app.function(gpu="T4", cpu=2.0, timeout=600)
def check() -> dict:
    """Cheap preflight: driver, torch CUDA and onnxruntime's CUDA provider on a real Parakeet session."""
    import subprocess

    import onnxruntime as ort
    import torch
    print(subprocess.run("find / \\( -name 'libcublasLt.so*' -o -name 'libcudnn.so.*' -o -name 'libcudart.so*' \\) "
                         "-not -path '/proc/*' 2>/dev/null; pip list 2>/dev/null | grep -iE 'nvidia|torch|onnxruntime'",
                         shell=True, capture_output=True, text=True).stdout, flush=True)
    ort.preload_dlls()
    import onnx_asr
    base = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v2", "/models/parakeet",
                               providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    return {"nvidia-smi": smi, "torch": str(torch.__version__), "torch_cuda": torch.cuda.is_available(),
            "ort": ort.__version__, "encoder_providers": base.asr._encoder.get_providers()}


@app.local_entrypoint()
def main(gpus: str = "T4,L4,A10,L40S,A100-40GB,H100", preflight: bool = True, wait_min: float = 30):
    if preflight:
        info = check.remote()
        print("preflight:", info)
        if "CUDAExecutionProvider" not in info["encoder_providers"]:
            raise SystemExit("onnxruntime has no CUDA provider; not starting the GPU runs")
    names = [g.strip() for g in gpus.split(",") if g.strip()]
    calls = {g: bench.with_options(gpu=g).spawn(g) for g in names}
    out = ROOT / "outputs/modal_bench.json"
    out.parent.mkdir(exist_ok=True)
    results = []

    def row(r):
        if "error" in r:
            print(f"{r['gpu']:10s} {r['error'][:100]}", flush=True)
            return
        rate = r["audio_s"] / r["pipeline_s"]
        cost = (PRICE.get(r["gpu"], 0) + CPU_MEM_PRICE) * r["pipeline_s"] * 3600 / r["audio_s"]
        print(f"{r['gpu']:10s} {r['mimi_s']:6.1f} {r['asr_s']:6.1f} {r['diar_s']:6.1f} {r['diar_stock_s']:6.1f} "
              f"{r['pipeline_s']:6.1f} {rate:6.0f} {cost:9.4f}   {r['device']}", flush=True)

    print(f"\n{'gpu':10s} {'mimi':>6s} {'asr':>6s} {'diar':>6s} {'stock':>6s} {'pipe':>6s} {'x RT':>6s} {'$/audio h':>9s}")
    # rows as they finish: one GPU type without capacity mustn't hold up (or lose) the others
    deadline = time.time() + wait_min * 60
    while calls and time.time() < deadline:
        for g, call in list(calls.items()):
            try:
                r = call.get(timeout=0)
            except (TimeoutError, modal.exception.TimeoutError):
                continue
            except Exception as e:
                r = {"gpu": g, "error": f"{type(e).__name__}: {e}"}
            del calls[g]
            results.append(r)
            row(r)
            out.write_text(json.dumps(results, indent=1))
        time.sleep(5)
    for g, call in calls.items():  # never scheduled (no capacity) or still running at the deadline
        call.cancel()
        results.append({"gpu": g, "error": f"no result within {wait_min:g} min (no capacity?); cancelled"})
        row(results[-1])
    out.write_text(json.dumps(results, indent=1))
    print(f"\nresults: {out}")


@app.local_entrypoint()
def profile(gpus: str = "L40S,H100"):
    """modal run scripts/modal_bench.py::profile [--gpus L40S,H100]: Parakeet's time split, per GPU host."""
    calls = {g.strip(): asr_profile.with_options(gpu=g.strip()).spawn(g.strip()) for g in gpus.split(",") if g.strip()}
    for g, call in calls.items():
        try:
            print("PROFILE " + json.dumps(call.get(timeout=1500)), flush=True)
        except Exception as e:
            print(f"PROFILE {g} failed: {type(e).__name__}: {e}", flush=True)


@app.local_entrypoint()
def batch(gpus: str = "H100,L40S,A10", ns: str = "1,2,4,8", clip_min: float = 5.0, search: str = "HEURISTIC"):
    """modal run scripts/modal_bench.py::batch: throughput with 1..n episodes in flight, and $ per audio hour."""
    counts = [int(x) for x in ns.split(",")]
    calls = {g.strip(): throughput.with_options(gpu=g.strip()).spawn(g.strip(), counts, clip_min, search)
             for g in gpus.split(",") if g.strip()}
    results = []
    for g, call in calls.items():
        try:
            r = call.get(timeout=2400)
        except Exception as e:
            print(f"{g}: failed: {type(e).__name__}: {e}", flush=True)
            continue
        results.append(r)
        for n, run in r["runs"].items():
            cost = (PRICE.get(g, 0) + CPU_MEM_PRICE) * 3600 / run["x_rt"]
            print(f"BATCH {g:10s} n={n:>2s}  wall {run['wall_s']:6.1f} s  {run['x_rt']:6.1f}x real time  "
                  f"${cost:.4f}/audio h  peak {run['gpu_mem_gb']} GB", flush=True)
    (ROOT / "outputs/modal_batch.json").write_text(json.dumps(results, indent=1))
