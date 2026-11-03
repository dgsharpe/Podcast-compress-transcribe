"""Throughput experiments on a GPU: a backlog of clips processed by concurrent workers.

    .venv-rocm/bin/python scripts/throughput_server.py [--workers 1,2,4] [--procs 1] [--clips 8] [--clip-min 5]
                                                       [--label E0] [--tf32]

Each clip runs exactly as mimi-hub runs an episode, with the hub's own models and settings: Mimi encode ||
(Parakeet ASR -> pyannote diarization, which the hub serialises with a lock). [--procs] P > 1 splits the clips over P
processes, each with its own models and [workers] threads (does one Python process / GIL limit a fast GPU?).
Reports aggregate x real time (clip audio / wall time), the mean GPU busy % (amdgpu's gpu_busy_percent or NVML,
sampled every 100 ms), CPU cores in use, and per-clip stage times. Results are appended to
outputs/throughput_server.jsonl. scripts/modal_hub_bench.py runs this on Modal's GPUs.
"""

import argparse
import json
import os
import queue
import resource
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MIMI_HUB_DATA", tempfile.mkdtemp(prefix="mimi-tput-"))  # never touch hub_data
os.environ.setdefault("HF_HUB_OFFLINE", "1")

BUSY = Path("/sys/class/drm/card1/device/gpu_busy_percent")


def _busy_reader():
    """A function returning the GPU's busy % now, or None if neither amdgpu's sysfs file nor NVML is there."""
    if BUSY.exists():
        return lambda: int(BUSY.read_text())
    if sys.platform == "darwin":  # Apple silicon: the GPU's own utilization counter
        import re
        import subprocess

        def ioreg():
            out = subprocess.run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"], capture_output=True,
                                 text=True).stdout
            return int(re.search(r'"Device Utilization %"=(\d+)', out).group(1))
        return ioreg
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return lambda: pynvml.nvmlDeviceGetUtilizationRates(h).gpu
    except Exception:
        return None


class GpuBusy:
    """Mean GPU busy % while running."""

    def __enter__(self):
        self.samples, self.stop, self.read = [], threading.Event(), _busy_reader()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def _run(self):
        while self.read and not self.stop.wait(0.1):
            try:
                self.samples.append(self.read())
            except (OSError, ValueError):
                pass

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        self.mean = round(sum(self.samples) / max(1, len(self.samples)), 1)


def hub_encode(model, wav):
    """Mimi codes the way the hub's encode_live makes them: one stream over its chunks (MIMI_STREAMING), else the
    old windows with 20 s of left context."""
    import torch
    from hub import mimi_hub
    import mimi_file
    if not mimi_hub.MIMI_STREAMING:
        return mimi_file.encode_array(model, wav)
    kv, pad, out = None, None, []
    with torch.inference_mode():
        for f0, f1 in mimi_hub.chunk_bounds(len(wav) // mimi_file.SPF):
            x = torch.from_numpy(wav[f0 * mimi_file.SPF:f1 * mimi_file.SPF].copy())[None, None].to(model.device)
            o = mimi_hub.mimi_encode(model, x, encoder_past_key_values=kv, padding_cache=pad, use_streaming=True)
            kv, pad = o.encoder_past_key_values, o.padding_cache
            out.append(o.audio_codes[0].T.cpu().numpy())
    return out


def baseline_episode(m, wav24, wav16):
    """One clip the way hub.mimi_hub.run() handles an episode (minus file I/O): (cues, speakers, stage seconds)."""
    from hub import mimi_hub
    t0 = time.perf_counter()
    times = {}

    def encode():
        hub_encode(m["mimi"], wav24)
        times["mimi"] = time.perf_counter() - t0

    enc = threading.Thread(target=encode)
    enc.start()
    cues = [c for s in m["asr"].recognize(wav16, sample_rate=16000) for c in mimi_hub.cues(s)]
    times["asr"] = time.perf_counter() - t0
    turns, _ = mimi_hub.diarize(wav16)
    times["diar"] = time.perf_counter() - t0 - times["asr"]
    enc.join()
    return len(cues), len({t[2] for t in turns}), times


def run_backlog(episode_fn, m, clips, workers):
    """Process every clip with [workers] threads pulling from one queue: wall time, mean GPU busy %, results,
    errors, CPU seconds used."""
    q = queue.Queue()
    for k in range(len(clips)):
        q.put(k)
    out, errors = {}, []

    def worker():
        while True:
            try:
                k = q.get_nowait()
            except queue.Empty:
                return
            try:
                out[k] = episode_fn(m, *clips[k])
            except Exception as e:  # count it, keep going
                errors.append(f"clip {k}: {type(e).__name__}: {e}")

    r0 = resource.getrusage(resource.RUSAGE_SELF)
    with GpuBusy() as busy:
        t = time.perf_counter()
        threads = [threading.Thread(target=worker) for _ in range(workers)]
        [th.start() for th in threads]
        [th.join() for th in threads]
        wall = time.perf_counter() - t
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    cpu_s = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    return wall, busy.mean, out, errors, cpu_s


def load_clips(path, n, clip_min, ks=None):
    import librosa
    import soundfile as sf
    wav, sr = sf.read(path, dtype="float32")
    step = int(clip_min * 60 * sr)
    ks = range(n) if ks is None else ks
    avail = len(wav) // step
    assert avail > 0, "the audio is shorter than one clip"
    clips = [wav[k % avail * step:(k % avail + 1) * step] for k in ks]  # a long backlog reuses the audio
    return [(w, librosa.resample(w, orig_sr=sr, target_sr=16000)) for w in clips]


def setup(tf32):
    from hub import mimi_hub
    if tf32:  # fp32 matmuls / convs on NVIDIA tensor cores, and cuDNN's per-shape algorithm search
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    m = mimi_hub.models()
    if os.environ.get("TPUT_PYANNOTE_PATH"):  # a local copy of the gated model (Modal), not the HF cache
        from hub import pyannote_gpu
        mimi_hub._models["diar_pyannote"] = pyannote_gpu.load(m["device"], path=os.environ["TPUT_PYANNOTE_PATH"])
    return m


PROFILE = {}  # --profile: seconds and calls per component, summed over every clip


def profile_hooks(m):
    """Time Parakeet's VAD / preprocessing / encoder / decoder calls and pyannote's segmentation / embedding steps."""
    from hub import mimi_hub

    def wrap(obj, attr, key):
        f = getattr(obj, attr)

        def timed(*a, **k):
            t = time.perf_counter()
            try:
                return f(*a, **k)
            finally:
                s, n = PROFILE.get(key, (0.0, 0))
                PROFILE[key] = (s + time.perf_counter() - t, n + 1)
        setattr(obj, attr, timed)

    asr = m["asr"].asr
    wrap(m["asr"].vad._model, "run", "vad")
    wrap(asr, "_preprocessor", "preprocess")
    wrap(asr._encoder, "run", "encoder")
    wrap(asr._decoder_joint, "run", "decoder")
    pipe = mimi_hub._models.get("diar_pyannote")
    if pipe is not None:
        wrap(pipe, "get_segmentations", "segmentation")
        wrap(pipe, "get_embeddings", "embeddings")


def _proc(a, ks, workers, barrier, results, index=0, loading=None):
    """One of --procs processes: its own models, its share [ks] of the clips."""
    clips = load_clips(a.wav, None, a.clip_min, ks)
    if loading is not None:  # a few at a time: loading peaks at several GB of RAM per process
        loading.acquire()
    try:
        m = setup(a.tf32)
        baseline_episode(m, *clips[0])  # warm-up
    finally:
        if loading is not None:
            loading.release()
    barrier.wait(timeout=900)
    t = time.time()
    if a.torch_profile and index == 0:
        wall, _, out, errors, cpu_s = _profiled(lambda: run_backlog(baseline_episode, m, clips, workers))
    else:
        wall, _, out, errors, cpu_s = run_backlog(baseline_episode, m, clips, workers)
    results.put((t, t + wall, list(out.values()), errors, cpu_s, _gpu_peak_gb()))


def _profiled(f):
    """f() under torch.profiler: prints the GPU kernels taking the most time."""
    import torch
    from torch.profiler import ProfilerActivity, profile
    try:  # the work runs in worker threads
        cfg = {"experimental_config": torch._C._profiler._ExperimentalConfig(profile_all_threads=True)}
    except (AttributeError, TypeError):
        cfg = {}
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], **cfg) as prof:
        r = f()
    ev = prof.key_averages()

    def selfdev(e):
        return getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)
    kernels = sorted((e for e in ev if selfdev(e) > 0), key=selfdev, reverse=True)
    total = sum(selfdev(e) for e in kernels)
    top = [{"kernel": e.key[:90], "ms": round(selfdev(e) / 1000, 1), "calls": e.count,
            "pct": round(100 * selfdev(e) / max(1, total), 1)} for e in kernels[:25]]
    print("TORCHPROF " + json.dumps({"wall_s": round(r[0], 1), "kernel_ms_total": round(total / 1000, 1),
                                     "top": top}), flush=True)
    return r


def _gpu_peak_gb():
    import torch
    return round(torch.cuda.max_memory_allocated() / 2**30, 1) if torch.cuda.is_available() else None


def summarize(a, workers, wall, busy, outs, errors, cpu_s, gpu_peak_gb=None):
    stages = {k: round(sum(o[2].get(k, 0.0) for o in outs) / max(1, len(outs)), 2) for k in ("mimi", "asr", "diar")}
    return {"label": a.label, "procs": a.procs, "workers": workers, "clips": a.clips, "clip_min": a.clip_min,
            "tf32": a.tf32, "wall_s": round(wall, 1), "x_rt": round(a.clips * a.clip_min * 60 / wall, 1),
            "gpu_busy_pct": busy, "cpu_cores_used": round(cpu_s / wall, 1), "clip_stage_s": stages,
            "gpu_peak_gb_per_proc": gpu_peak_gb, "errors": errors, "cues": sum(o[0] for o in outs)}


def stage_cpu(a):
    """Mimi, ASR and diarization each run alone over every clip: where the CPU (and the wall time) goes."""
    import torch
    from hub import mimi_hub
    clips = load_clips(a.wav, a.clips, a.clip_min)
    m = setup(a.tf32)
    baseline_episode(m, *clips[0])
    stages = {"mimi": lambda w24, w16: hub_encode(m["mimi"], w24),
              "asr": lambda w24, w16: [c for s in m["asr"].recognize(w16, sample_rate=16000)
                                       for c in mimi_hub.cues(s)],
              "diar": lambda w24, w16: mimi_hub.diarize(w16)}
    audio_s = a.clips * a.clip_min * 60
    res = {"label": a.label, "clips": a.clips, "threads": mimi_hub.THREADS}
    for name, f in stages.items():
        torch.cuda.synchronize()
        r0, t0 = resource.getrusage(resource.RUSAGE_SELF), time.perf_counter()
        for w24, w16 in clips:
            f(w24, w16)
        torch.cuda.synchronize()
        r1, t1 = resource.getrusage(resource.RUSAGE_SELF), time.perf_counter()
        cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
        res[name] = {"cpu_ms_per_audio_s": round(1000 * cpu / audio_s, 2), "wall_ms_per_audio_s":
                     round(1000 * (t1 - t0) / audio_s, 3), "cores": round(cpu / (t1 - t0), 1)}
    print("STAGECPU " + json.dumps(res), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", default="1,2,4", help="threads per process")
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--clips", type=int, default=8)
    ap.add_argument("--clip-min", type=float, default=5.0)
    ap.add_argument("--label", default="E0")
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--profile", action="store_true", help="component times (one process; per clip)")
    ap.add_argument("--torch-profile", action="store_true", help="profile process 0's share of the backlog")
    ap.add_argument("--stage-cpu", action="store_true",
                    help="each stage alone over all clips: CPU-seconds and wall seconds per audio-second")
    ap.add_argument("--wav", default=str(ROOT / "data/ref/ref_24k.wav"))
    a = ap.parse_args()

    if a.stage_cpu:
        stage_cpu(a)
        return
    results = []
    for workers in [int(x) for x in a.workers.split(",")]:
        if a.procs == 1:
            if not results:
                clips = load_clips(a.wav, a.clips, a.clip_min)
                m = setup(a.tf32)
                baseline_episode(m, *clips[0])  # warm-up: models, kernel selection for a full clip
                if a.profile:
                    profile_hooks(m)
            PROFILE.clear()
            wall, busy, out, errors, cpu_s = run_backlog(baseline_episode, m, clips, workers)
            res = summarize(a, workers, wall, busy, list(out.values()), errors, cpu_s, _gpu_peak_gb())
            if a.profile:
                res["profile_per_clip"] = {k: [round(v[0] / a.clips, 3), round(v[1] / a.clips)]
                                           for k, v in PROFILE.items()}
        else:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            barrier, q, loading = ctx.Barrier(a.procs + 1), ctx.Queue(), ctx.Semaphore(4)
            shares = [list(range(a.clips))[p::a.procs] for p in range(a.procs)]
            ps = [ctx.Process(target=_proc, args=(a, ks, workers, barrier, q, k, loading))
                  for k, ks in enumerate(shares)]
            [p.start() for p in ps]
            got, dead = [], []
            try:
                barrier.wait(timeout=900)  # every process loaded and warmed up
            except threading.BrokenBarrierError:
                dead = [f"process {k} died loading (exit {p.exitcode})" for k, p in enumerate(ps) if not p.is_alive()]
            if not dead:
                with GpuBusy() as busy:
                    while len(got) < len(ps):  # a process that dies (e.g. out of GPU memory) never reports
                        try:
                            got.append(q.get(timeout=5))
                        except queue.Empty:
                            if all(not p.is_alive() for p in ps) and q.empty():
                                break
                dead = [f"process {k} exited {p.exitcode}" for k, p in enumerate(ps) if p.exitcode not in (0, None)]
            [p.kill() for p in ps if p.is_alive()]
            [p.join() for p in ps]
            if dead or not got:
                res = {"label": a.label, "procs": a.procs, "workers": workers, "clips": a.clips, "error": dead}
                print("RESULT " + json.dumps(res), flush=True)
                continue
            wall = max(g[1] for g in got) - min(g[0] for g in got)
            res = summarize(a, workers, wall, busy.mean, [o for g in got for o in g[2]],
                            [e for g in got for e in g[3]], sum(g[4] for g in got),
                            max((g[5] for g in got if g[5] is not None), default=None))
        results.append(res)
        print("RESULT " + json.dumps(res), flush=True)
        (ROOT / "outputs").mkdir(exist_ok=True)
        with open(ROOT / "outputs/throughput_server.jsonl", "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
