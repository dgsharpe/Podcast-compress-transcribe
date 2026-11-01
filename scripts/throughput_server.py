"""Throughput experiments on the server's GPU: a backlog of clips processed by c concurrent workers.

    .venv-rocm/bin/python scripts/throughput_server.py [--workers 1,2,4] [--clips 8] [--clip-min 5] [--label E0]

E0 (baseline) runs each clip exactly as mimi-hub runs an episode, with the hub's own models and settings:
Mimi encode || (Parakeet ASR -> pyannote diarization, which the hub serialises with a lock). Reports aggregate
x real time (clip audio / wall time) and the mean GPU busy % (amdgpu's gpu_busy_percent, sampled every 100 ms).
Results are appended to outputs/halo_throughput.jsonl.
"""

import argparse
import json
import os
import queue
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


class GpuBusy:
    """Mean of amdgpu's busy % while running."""

    def __enter__(self):
        self.samples, self.stop = [], threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def _run(self):
        while not self.stop.wait(0.1):
            try:
                self.samples.append(int(BUSY.read_text()))
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
            x = torch.from_numpy(wav[f0 * mimi_file.SPF:f1 * mimi_file.SPF].copy())[None, None].cuda()
            o = model.encode(x, num_quantizers=32, encoder_past_key_values=kv, padding_cache=pad, use_streaming=True)
            kv, pad = o.encoder_past_key_values, o.padding_cache
            out.append(o.audio_codes[0].T.cpu().numpy())
    return out


def baseline_episode(m, wav24, wav16):
    """One clip the way hub.mimi_hub.run() handles an episode (minus file I/O)."""
    from hub import mimi_hub
    enc = threading.Thread(target=hub_encode, args=(m["mimi"], wav24))
    enc.start()
    cues = [c for s in m["asr"].recognize(wav16, sample_rate=16000) for c in mimi_hub.cues(s)]
    turns, _ = mimi_hub.diarize(wav16)
    enc.join()
    return len(cues), len({t[2] for t in turns})


def run_backlog(episode_fn, m, clips, workers):
    """Process every clip with [workers] threads pulling from one queue; wall time and mean GPU busy %."""
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

    with GpuBusy() as busy:
        t = time.perf_counter()
        threads = [threading.Thread(target=worker) for _ in range(workers)]
        [th.start() for th in threads]
        [th.join() for th in threads]
        wall = time.perf_counter() - t
    return wall, busy.mean, out, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", default="1,2,4")
    ap.add_argument("--clips", type=int, default=8)
    ap.add_argument("--clip-min", type=float, default=5.0)
    ap.add_argument("--label", default="E0")
    ap.add_argument("--wav", default=str(ROOT / "data/ref/ref_24k.wav"))
    a = ap.parse_args()

    import librosa
    import soundfile as sf
    from hub import mimi_hub

    wav, sr = sf.read(a.wav, dtype="float32")
    step = int(a.clip_min * 60 * sr)
    clips = [(wav[k * step:(k + 1) * step],) for k in range(a.clips)]
    assert all(len(c[0]) == step for c in clips), "not enough audio for that many clips"
    clips = [(w, librosa.resample(w, orig_sr=sr, target_sr=16000)) for (w,) in clips]
    m = mimi_hub.models()
    baseline_episode(m, *clips[0])  # warm-up: models, kernel selection for a full clip
    audio_s = a.clips * a.clip_min * 60
    for workers in [int(x) for x in a.workers.split(",")]:
        wall, busy, out, errors = run_backlog(baseline_episode, m, clips, workers)
        res = {"label": a.label, "workers": workers, "clips": a.clips, "clip_min": a.clip_min,
               "wall_s": round(wall, 1), "x_rt": round(audio_s / wall, 1), "gpu_busy_pct": busy,
               "errors": errors, "cues": sum(v[0] for v in out.values())}
        print("RESULT " + json.dumps(res), flush=True)
        with open(ROOT / "outputs/halo_throughput.jsonl", "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
