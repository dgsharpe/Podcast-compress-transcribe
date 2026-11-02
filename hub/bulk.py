"""Batch conversion: many episodes on one GPU at once, each model's work batched across episodes.

    python -m hub.bulk OUT_DIR AUDIO [AUDIO ...] [--prep N]      (files or URLs; ffmpeg on PATH)

The hub (hub/mimi_hub.py) runs one episode per job, so its GPU batches are one episode's: Mimi one stream (batch
1), Parakeet up to 16 segments, pyannote 32 windows; filling a fast GPU took many processes, each with its own copy
of the models (~4 GB each: an H100 holds 16). Here one process holds one copy, and every GPU call takes its batch
from all the episodes in flight:
  - Mimi: each episode is cut into pieces of PIECE_FRAMES (5 min), each streamed from WARMUP_FRAMES (20 s, Mimi's
    attention window) before it (the first from 0, as the hub's stream); MIMI_BATCH pieces from any episodes run
    in lockstep. Codes differ from the hub's single stream as fp16 batching rounds differently (~96% identical,
    nearly all in the fine codebooks).
  - Parakeet: the VAD segments of all episodes in flight, sorted by length, ASR_BATCH at a time through the encoder
    and the native TDT decoder.
  - pyannote: each episode's pipeline runs in its own thread (its CPU steps unchanged); its segmentation batches and
    ResNet frames go through shared batchers that run requests from all of them together.
Each model runs on its own CUDA stream. Decoding, resampling, VAD and Parakeet's features run in a pool of
processes (one Python process couldn't keep up), the audio handed over as memory-mapped spool files (local disk,
in the page cache: /dev/shm is small in containers). Output per episode: the
hub's .mimi and .vtt (speaker-voiced cues).
"""

import argparse
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

PIECE_FRAMES = int(os.environ.get("MIMI_BULK_PIECE_FRAMES", "3750"))  # 5 min of 12.5 Hz frames
WARMUP_FRAMES = 250  # Mimi's transformer context (20 s)
# streamed 20 s at a time: B pieces x 20 s of 24 kHz activations stay a few GB; longer chunks mean fewer kernel
# launches (from Python, which one process can only do one thread at a time) per second of audio
CHUNK_FRAMES = int(os.environ.get("MIMI_BULK_CHUNK_FRAMES", "250"))
MIMI_BATCH = int(os.environ.get("MIMI_BULK_MIMI_BATCH", "16"))
MIMI_WORKERS = int(os.environ.get("MIMI_BULK_MIMI_WORKERS", "1"))  # threads, each on its own CUDA stream
# Mimi through hub/mimi_static.py: fixed-size streaming state, each chunk of a batch one CUDA graph replay instead of
# hundreds of kernel launches (codes as transformers' streaming to ~98%: attention rounds a little differently).
# auto: on NVIDIA (launch-bound there); the server's ROCm GPU is compute-bound and gained nothing.
MIMI_GRAPH = os.environ.get("MIMI_BULK_MIMI_GRAPH", "auto")
ASR_BATCH = int(os.environ.get("MIMI_BULK_ASR_BATCH", "64"))
ASR_WORKERS = int(os.environ.get("MIMI_BULK_ASR_WORKERS", "2"))  # threads, each on its own CUDA stream
# the TDT search as CUDA graphs (hub/parakeet_native.NativeTdtDecoder.decode_graphed): 16 steps per launch, the same
# tokens and timestamps (548/548 segments). Off by default: on an H100 it was no faster than the eager search (590x
# vs 789x end to end), and graphs captured at load hung on replay from the worker threads (on demand they didn't).
ASR_GRAPH = os.environ.get("MIMI_BULK_ASR_GRAPH", "0")
DIAR_ROWS = int(os.environ.get("MIMI_BULK_DIAR_ROWS", "256"))  # segmentation / ResNet windows per GPU call
DIAR_THREADS = int(os.environ.get("MIMI_BULK_DIAR_THREADS", "16"))
IN_FLIGHT = int(os.environ.get("MIMI_BULK_IN_FLIGHT", "32"))  # episodes prepared and not yet written


# ---------- CPU side: a pool of processes ----------

_prep_models = {}


def _prep_init():
    import onnxruntime as ort
    import onnx_asr
    import threadpoolctl
    from onnx_asr.preprocessors.numpy_preprocessor import NemoPreprocessorNumpy
    threadpoolctl.threadpool_limits(1, user_api="blas")
    so = ort.SessionOptions()
    so.intra_op_num_threads = so.inter_op_num_threads = 1
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    _prep_models["vad"] = onnx_asr.load_vad("silero", sess_options=so)
    _prep_models["features"] = NemoPreprocessorNumpy("nemo128")  # Parakeet TDT 0.6B v2's


def _prep(src, spool):
    """Decode [src] to 24 kHz, resample to 16 kHz, find its speech (Silero VAD) and Parakeet's features of each
    segment, as the hub does. The audio goes back in a spool file in [spool], the rest pickled."""
    import uuid
    import numpy as np
    import soxr
    import mimi_file
    t = time.perf_counter()
    http = ["-user_agent", "Mozilla/5.0 (mimi-hub)"] if str(src).startswith("http") else []
    raw = subprocess.run(["ffmpeg", "-v", "error", *http, "-i", str(src), "-map", "0:a:0", "-ac", "1", "-ar",
                          str(mimi_file.SR), "-f", "f32le", "-"], capture_output=True, check=True).stdout
    w24 = np.frombuffer(raw, np.float32)
    n_frames = math.ceil(len(w24) / mimi_file.SPF)
    w16 = soxr.resample(w24, mimi_file.SR, 16000, quality="HQ").astype(np.float32)  # = librosa's soxr_hq
    path = Path(spool) / f"{uuid.uuid4().hex}.f32"  # the main process deletes it when done
    buf = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(n_frames * mimi_file.SPF + len(w16),))
    buf[:len(w24)] = w24
    buf[len(w24):n_frames * mimi_file.SPF] = 0.0  # the last frame's tail
    buf[n_frames * mimi_file.SPF:] = w16
    buf.flush()
    del buf
    vad = _prep_models["vad"]
    segments = [(int(s), int(e)) for s, e in next(iter(vad.segment_batch(w16[None], np.array([len(w16)]), 16000)))]
    feats = []
    for s, e in segments:
        f, n = _prep_models["features"](w16[None, s:e].copy(), np.array([e - s]))
        feats.append((f[0], int(n[0])))  # (128, frames), valid frames
    return {"spool": str(path), "n24": n_frames * mimi_file.SPF, "n16": len(w16), "frames": n_frames,
            "duration_s": len(w24) / mimi_file.SR, "segments": segments,
            "feats": feats, "prep_s": time.perf_counter() - t}


# ---------- GPU side: one process, a batcher per model ----------

BUSY = {}  # seconds each GPU worker / batcher spent running batches (for the summary)


def _busy(name, t0):
    BUSY[name] = BUSY.get(name, 0.0) + time.perf_counter() - t0

class Episode:
    def __init__(self, src, out_dir, prep):
        import numpy as np
        self.src, self.name = src, Path(str(src)).stem
        self.out_dir, self.p = Path(out_dir), prep
        buf = np.load(prep["spool"], mmap_mode="r")
        self.w24, self.w16 = buf[:prep["n24"]], buf[prep["n24"]:]
        self.codes = np.zeros((prep["frames"], 32), np.int64)
        self.n_pieces = math.ceil(prep["frames"] / PIECE_FRAMES)
        self.asr = [None] * len(prep["segments"])
        self.lock = threading.Lock()
        self.left = {"mimi": self.n_pieces, "asr": len(self.asr)}
        self.done = {k: threading.Event() for k in ("mimi", "asr", "diar")}
        self.t = {"start": time.perf_counter()}
        self.turns, self.errors = [], []
        if not self.asr:
            self.done["asr"].set()

    def fail(self, stage, e):
        """[stage] failed for this episode: it won't be written, and nothing waits on it."""
        self.errors.append(f"{stage}: {type(e).__name__}: {e}")
        self.t[stage] = time.perf_counter()
        self.done[stage].set()

    def finish(self, stage, n=1):
        with self.lock:
            self.left[stage] -= n
            last = self.left[stage] == 0
        if last:
            self.t[stage] = time.perf_counter()
            self.done[stage].set()

    def release(self):
        self.w24 = self.w16 = None
        Path(self.p["spool"]).unlink(missing_ok=True)


class Batcher:
    """fn(x) for calls from many threads, merged along dim 0 into batches of about [max_rows] (a call waits up to
    [wait_s] for others to join), on the batcher's own CUDA stream."""

    def __init__(self, fn, max_rows, wait_s=0.003, name="batcher"):
        import torch
        self.fn, self.max_rows, self.wait_s, self.name = fn, max_rows, wait_s, name
        self.q, self.stream = queue.Queue(), torch.cuda.Stream()
        self.calls = self.rows = 0
        threading.Thread(target=self._run, daemon=True, name=name).start()

    def __call__(self, x):
        import torch
        item = {"x": x, "done": threading.Event(), "ready": torch.cuda.current_stream().record_event()}
        self.q.put(item)
        item["done"].wait()
        if "error" in item:
            raise item["error"]
        item["y"].record_stream(torch.cuda.current_stream())  # allocated on our stream, used on the caller's
        return item["y"]

    def _run(self):
        import torch
        while True:
            items = [self.q.get()]
            rows, deadline = len(items[0]["x"]), time.perf_counter() + self.wait_s
            while rows < self.max_rows:
                try:
                    items.append(self.q.get(timeout=max(0.0, deadline - time.perf_counter())))
                except queue.Empty:
                    break
                rows += len(items[-1]["x"])
            groups = {}
            for it in items:  # only same-shaped inputs can be concatenated
                groups.setdefault((tuple(it["x"].shape[1:]), it["x"].dtype), []).append(it)
            for group in groups.values():
                t0 = time.perf_counter()
                try:
                    with torch.cuda.stream(self.stream), torch.inference_mode():
                        for it in group:
                            self.stream.wait_event(it["ready"])
                        y = self.fn(torch.cat([it["x"] for it in group]))
                        self.stream.synchronize()
                    k = 0
                    for it in group:
                        it["y"], k = y[k:k + len(it["x"])], k + len(it["x"])
                except Exception as e:  # noqa: BLE001 - handed to the callers
                    for it in group:
                        it["error"] = e
                self.calls, self.rows = self.calls + 1, self.rows + sum(len(it["x"]) for it in group)
                _busy(self.name, t0)
                for it in group:
                    it["done"].set()


class Models:
    def __init__(self):
        import torch
        from hub import mimi_hub, pyannote_gpu
        self.torch, self.hub = torch, mimi_hub
        m = mimi_hub.models()  # the hub's models and settings (fp16 weights, native encoder / decoder, ...)
        self.device = m["device"]
        assert self.device == "cuda", "hub/bulk.py needs a CUDA (or ROCm) GPU"
        self.mimi, self.asr = m["mimi"], m["asr"].asr
        from hub import parakeet_native
        if not isinstance(self.asr._decoding, parakeet_native.NativeTdtDecoder):
            # on ROCm the hub decodes on the CPU (it's GPU-bound); here that would all fall on this one process
            path = parakeet_native.convert_decoder(mimi_hub.ASR_MODEL, mimi_hub.MODELS_DIR / f"{mimi_hub.ASR_MODEL}-hf")
            self.asr._decoding = parakeet_native.NativeTdtDecoder(path, self.device, self.asr._blank_idx,
                                                                  self.asr._vocab_size,
                                                                  max_symbols=self.asr._max_tokens_per_step)
        pipe = pyannote_gpu.load(self.device, path=os.environ.get("TPUT_PYANNOTE_PATH"))
        mode = pyannote_gpu._frames_mode(torch.device(self.device))
        emb = pipe._embedding.model_
        frames = Batcher(lambda fb: pyannote_gpu.resnet_frames(emb, fb, mode), DIAR_ROWS, name="bulk-resnet")
        pyannote_gpu.FRAMES_FN = frames
        seg = pipe._segmentation.model
        seg.forward = Batcher(seg.forward, DIAR_ROWS, name="bulk-segmentation")
        self.pipe, self.batchers = pipe, {"resnet": frames, "segmentation": seg.forward}
        if ASR_GRAPH == "1" or (ASR_GRAPH == "auto" and not torch.version.hip):  # captured now, not mid-run
            for k in range(ASR_WORKERS):
                self.asr._decoding.capture_graphs(k, max_batch=ASR_BATCH)
        self.mimi_static = []
        # one set per Mimi worker, captured now, before any worker runs on the GPU
        if MIMI_GRAPH == "1" or (MIMI_GRAPH == "auto" and not torch.version.hip):
            from hub.mimi_static import StaticMimiEncoder
            L = PIECE_FRAMES + WARMUP_FRAMES
            assert L % CHUNK_FRAMES == 0, f"a piece ({L} frames) must be whole chunks of {CHUNK_FRAMES} for the graph"
            sizes = sorted({max(1, MIMI_BATCH // d) for d in (1, 2, 4)})  # a partial batch pads to the next size
            for _ in range(MIMI_WORKERS):
                encs = {}
                for b in sizes:
                    encs[b] = StaticMimiEncoder(self.mimi, b, CHUNK_FRAMES)
                    encs[b].capture()
                self.mimi_static.append(encs)


def mimi_worker(models, pieces, name="mimi", index=0):
    """Lockstep batches of up to MIMI_BATCH pieces (episode, k) from any episodes."""
    import numpy as np
    import torch
    import mimi_file
    from hub import mimi_hub
    spf, L = mimi_file.SPF, PIECE_FRAMES + WARMUP_FRAMES
    stream = torch.cuda.Stream()
    # this worker's captured encoders by batch size (reused by the next convert(): the previous one's workers sit idle
    # on their queue)
    statics = models.mimi_static[index] if index < len(models.mimi_static) else None
    while True:
        batch = [pieces.get()]
        while len(batch) < MIMI_BATCH:
            try:
                batch.append(pieces.get(timeout=0.01))
            except queue.Empty:
                break
        t0 = time.perf_counter()
        try:
            static = statics[min(b for b in statics if b >= len(batch))] if statics else None
            x = np.zeros((static.B if static is not None else len(batch), L * spf), np.float32)
            for b, (ep, k) in enumerate(batch):
                a = 0 if k == 0 else k * PIECE_FRAMES - WARMUP_FRAMES
                seg = ep.w24[a * spf:(a + L) * spf]
                x[b, :len(seg)] = seg
            with torch.cuda.stream(stream):
                xt = torch.from_numpy(x).pin_memory().to("cuda", non_blocking=True)[:, None]
                out = []
                if static is not None:
                    static.reset()
                    for c0 in range(0, L, CHUNK_FRAMES):
                        out.append(static.encode(xt[..., c0 * spf:(c0 + CHUNK_FRAMES) * spf]).transpose(1, 2))
                else:
                    kv, pad = None, None
                    for c0 in range(0, L, CHUNK_FRAMES):
                        o = mimi_hub.mimi_encode(models.mimi, xt[..., c0 * spf:(c0 + CHUNK_FRAMES) * spf],
                                                 encoder_past_key_values=kv, padding_cache=pad, use_streaming=True)
                        kv, pad = o.encoder_past_key_values, o.padding_cache
                        out.append(o.audio_codes.transpose(1, 2))  # (B, frames, 32)
                codes = torch.cat(out, dim=1).cpu().numpy()
        except Exception as e:  # noqa: BLE001
            for ep, _ in batch:
                ep.fail("mimi", e)
            continue
        for b, (ep, k) in enumerate(batch):
            lo = 0 if k == 0 else WARMUP_FRAMES  # piece 0 has no warm-up
            f0 = k * PIECE_FRAMES
            n = min(PIECE_FRAMES, ep.p["frames"] - f0)
            ep.codes[f0:f0 + n] = codes[b, lo:lo + n]
            ep.finish("mimi")
        _busy(name, t0)


def asr_worker(models, segments, name="asr", slot=0):
    """Segments (episode, j) of any episodes, pooled and sorted by length, ASR_BATCH at a time. The features go to
    the GPU in one pinned copy and the encoder's output stays there for the decoder."""
    import numpy as np
    import torch
    from onnx_asr.vad import TimestampedSegmentResult
    asr, stream, pending = models.asr, torch.cuda.Stream(), []
    graphed = ASR_GRAPH == "1" or (ASR_GRAPH == "auto" and not torch.version.hip)
    encoder = asr._encoder  # hub/parakeet_native.NativeEncoder
    while True:
        if not pending:
            pending.append(segments.get())
        while True:  # take everything queued; wait briefly for more while fewer than 4 batches
            try:
                pending.append(segments.get(timeout=0.02 if len(pending) < 4 * ASR_BATCH else 0))
            except queue.Empty:
                break
        pending.sort(key=lambda it: it[0].p["feats"][it[1]][1])
        t0 = time.perf_counter()
        for i in range(0, len(pending), ASR_BATCH):
            batch = pending[i:i + ASR_BATCH]
            feats = [ep.p["feats"][j] for ep, j in batch]
            T = max(f.shape[1] for f, _ in feats)
            x = torch.zeros((len(batch), T, feats[0][0].shape[0]), dtype=torch.float32).pin_memory()  # (B, T, F)
            xn = x.numpy()
            for b, (f, _) in enumerate(feats):
                xn[b, :f.shape[1]] = f.T
            lens = torch.tensor([n for _, n in feats], dtype=torch.int64)
            try:
                with torch.cuda.stream(stream):
                    enc, enc_lens = encoder.encode(x, lens)
                    found = (asr._decoding.decode_graphed(enc, enc_lens, slot) if graphed
                             else asr._decoding(enc, enc_lens, need_logprobs=True))
                    results = list(map(asr._decode_tokens, *zip(*found)))
            except Exception as e:  # noqa: BLE001
                for ep, _ in batch:
                    ep.fail("asr", e)
                continue
            for (ep, j), res in zip(batch, results, strict=True):
                s, e = ep.p["segments"][j]
                ep.asr[j] = TimestampedSegmentResult(s / 16000, e / 16000, res.text, res.timestamps, res.tokens,
                                                     res.logprobs)
                ep.finish("asr")
        _busy(name, t0)
        pending = []


def diarize_one(models, ep):
    import torch
    t0 = time.perf_counter()
    try:
        out = models.pipe({"waveform": torch.from_numpy(ep.w16)[None], "sample_rate": 16000})
        ann = getattr(out, "speaker_diarization", out)
        ep.turns = models.hub.number_turns([(t.start, t.end, spk) for t, _, spk in ann.itertracks(yield_label=True)])
    except Exception as e:  # noqa: BLE001 - a transcript without speakers is still useful, as in the hub
        ep.errors.append(f"diar (written without speakers): {type(e).__name__}: {e}")
    ep.t["diar"] = time.perf_counter()
    _busy("diar_threads", t0)
    ep.done["diar"].set()


def write_outputs(models, ep):
    import mimi_file
    hub = models.hub
    for e in ep.done.values():
        e.wait()
    if any(not err.startswith("diar") for err in ep.errors):
        ep.cues = 0
        ep.release()
        raise RuntimeError("; ".join(ep.errors))
    meta = dict(title=ep.name, source=str(ep.src), id=ep.name, encoder="transformers MimiModel (hub/bulk.py)",
                duration_s=ep.p["duration_s"])
    mimi_file.write(ep.out_dir / f"{ep.name}.mimi", ep.codes, 32, meta)
    cues = [c for seg in ep.asr for c in hub.cues(seg)]
    hub.write_vtt(ep.out_dir / f"{ep.name}.vtt", cues, ep.turns)
    ep.t["written"] = time.perf_counter()
    ep.cues = len(cues)
    ep.release()


def convert(sources, out_dir, prep_workers=None, models=None, log=print):
    """Convert [sources] into [out_dir]; returns per-episode stats. [models]: a loaded Models (reused)."""
    import multiprocessing
    import tempfile
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spool = tempfile.mkdtemp(prefix="mimi-bulk-spool-", dir=os.environ.get("MIMI_BULK_SPOOL"))
    models = models or Models()
    pieces, segments = queue.Queue(), queue.Queue()
    for k in range(MIMI_WORKERS):
        threading.Thread(target=mimi_worker, args=(models, pieces, f"mimi{k}", k), daemon=True,
                         name=f"bulk-mimi{k}").start()
    for k in range(ASR_WORKERS):
        threading.Thread(target=asr_worker, args=(models, segments, f"asr{k}", k), daemon=True,
                         name=f"bulk-asr{k}").start()
    diar = ThreadPoolExecutor(DIAR_THREADS, thread_name_prefix="bulk-diar")
    writers = ThreadPoolExecutor(4, thread_name_prefix="bulk-write")
    slots = threading.Semaphore(IN_FLIGHT)
    import resource
    BUSY.clear()
    stats, t0, r0 = [], time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
    with ProcessPoolExecutor(prep_workers or max(1, (os.cpu_count() or 2) // 2),
                             mp_context=multiprocessing.get_context("spawn"), initializer=_prep_init) as pool:
        def submit(src):
            slots.acquire()  # bounds the episodes held in memory
            return pool.submit(_prep, src, spool)

        futures = {}
        feeder = threading.Thread(target=lambda: [futures.setdefault(s, submit(s)) for s in sources], daemon=True)
        feeder.start()
        done = []
        for src in sources:
            while src not in futures:
                time.sleep(0.01)
            try:
                ep = Episode(src, out_dir, futures[src].result())
            except Exception as e:  # noqa: BLE001 - e.g. a file ffmpeg can't read: skip it
                slots.release()
                log(f"{src}: {type(e).__name__}: {e}")
                stats.append({"name": Path(str(src)).stem, "error": f"{type(e).__name__}: {e}"})
                continue
            for k in range(ep.n_pieces):
                pieces.put((ep, k))
            for j in range(len(ep.asr)):
                segments.put((ep, j))
            diar.submit(diarize_one, models, ep)

            def written(f, ep=ep):
                slots.release()
                if f.exception():
                    log(f"{ep.name}: {f.exception()!r}")
            w = writers.submit(write_outputs, models, ep)
            w.add_done_callback(written)
            done.append((ep, w))
        for ep, w in done:
            if w.exception():
                stats.append({"name": ep.name, "error": "; ".join(ep.errors) or repr(w.exception())})
                continue
            stats.append({"name": ep.name, "duration_s": round(ep.p["duration_s"], 1), "errors": ep.errors,
                          "prep_s": round(ep.p["prep_s"], 2), "cues": ep.cues,
                          "speakers": len({t[2] for t in ep.turns}), "turns": ep.turns,
                          **{k: round(v - ep.t["start"], 2) for k, v in ep.t.items() if k != "start"}})
    wall = time.perf_counter() - t0
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    audio = sum(s.get("duration_s", 0) for s in stats)
    ok = [s for s in stats if "prep_s" in s]
    n_prep = prep_workers or max(1, (os.cpu_count() or 2) // 2)

    def mean(k):
        return round(sum(s.get(k, 0) for s in ok) / max(1, len(ok)), 2)
    summary = {"episodes": len(stats), "audio_min": round(audio / 60, 1), "wall_s": round(wall, 1),
               "x_rt": round(audio / wall, 1),
               "batchers": {k: {"calls": b.calls, "rows_per_call": round(b.rows / max(1, b.calls), 1)}
                            for k, b in models.batchers.items()},
               # where the time goes: busy fraction of the wall time per GPU worker (diar_threads: summed over the
               # threads), the main process's CPU in cores (~1 = held by the GIL), the prep pool's use
               "busy_frac": {k: round(v / wall, 2) for k, v in BUSY.items()},
               "main_cpu_cores": round((r1.ru_utime - r0.ru_utime + r1.ru_stime - r0.ru_stime) / wall, 2),
               "prep_pool_use": round(sum(s["prep_s"] for s in ok) / (n_prep * wall), 2),
               "mean_prep_s": mean("prep_s"), "mean_done_after_prep_s": {k: mean(k) for k in ("mimi", "asr", "diar")}}
    diar.shutdown()
    writers.shutdown()
    import shutil
    shutil.rmtree(spool, ignore_errors=True)
    failed = [s for s in stats if s.get("error")]
    summary["failed"] = len(failed)
    summary["gpu_reserved_peak_gb"] = round(models.torch.cuda.max_memory_reserved() / 2**30, 1)
    import sysconfig
    summary["python"] = sys.version.split()[0] + ("t" if sysconfig.get_config_var("Py_GIL_DISABLED") else "")
    summary["gil"] = sys._is_gil_enabled() if hasattr(sys, "_is_gil_enabled") else True
    if failed:
        summary["first_error"] = failed[0]["error"][:300]
    return summary, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir")
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--prep", type=int, default=None, help="CPU worker processes (default: half the cores)")
    a = ap.parse_args()
    summary, stats = convert(a.sources, a.out_dir, a.prep)
    for s in stats:
        print("EPISODE " + json.dumps({k: v for k, v in s.items() if k != "turns"}), flush=True)
    print("BULK " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
