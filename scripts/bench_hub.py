"""Time mimi-hub's pipeline stages on one episode, alone and with encode + transcription overlapped.

    uv run python scripts/bench_hub.py data/ref/ref_24k.wav [--minutes 44] [--out outputs/bench_hub.json]

Uses the hub's own models and settings (hub.mimi_hub.models(), diarize()), on the device the hub picks.
"""

import argparse
import json
import os
import sys
import threading
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hub import mimi_hub  # noqa: E402
import mimi_file  # noqa: E402  (mimi_hub put scripts/ on sys.path)


def timed(f, *a):
    t = time.perf_counter()
    out = f(*a)
    return out, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--minutes", type=float, default=0)
    ap.add_argument("--out", default="outputs/bench_hub.json")
    a = ap.parse_args()

    import librosa
    wav, sr = sf.read(a.wav, dtype="float32")
    assert sr == mimi_file.SR, sr
    if wav.ndim > 1:
        wav = wav.mean(1)
    if a.minutes:
        wav = wav[: int(a.minutes * 60 * sr)]
    dur = len(wav) / sr
    m = mimi_hub.models()
    asr = m["asr"]

    def encode():
        return mimi_file.encode_array(m["mimi"], wav)

    def resample():
        return librosa.resample(wav, orig_sr=sr, target_sr=16000)

    def transcribe(w16):
        return [c for s in asr.recognize(w16, sample_rate=16000) for c in mimi_hub.cues(s)]

    res = {"audio_s": dur, "torch_threads": mimi_hub.THREADS, "cpus": os.cpu_count()}
    # warm-up (first-call graph setup) on 20 s so it doesn't count against a stage
    mimi_file.encode_array(m["mimi"], wav[: 20 * sr])
    transcribe(librosa.resample(wav[: 20 * sr], orig_sr=sr, target_sr=16000))

    print(f"audio {dur / 60:.1f} min; sequential stages:", flush=True)
    _, res["encode_s"] = timed(encode)
    print(f"  mimi encode   {res['encode_s']:7.1f} s", flush=True)
    w16, res["resample_s"] = timed(resample)
    _, res["asr_s"] = timed(transcribe, w16)
    print(f"  parakeet asr  {res['asr_s']:7.1f} s", flush=True)
    _, res["diarize_s"] = timed(mimi_hub.diarize, w16)
    print(f"  diarization   {res['diarize_s']:7.1f} s", flush=True)

    print("overlapped: encode || (asr -> diarize)", flush=True)
    times = {}

    def run(name, f, *args):
        _, times[name] = timed(f, *args)

    def asr_then_diar():
        run("asr", transcribe, w16)
        run("diarize", mimi_hub.diarize, w16)

    t0 = time.perf_counter()
    th = [threading.Thread(target=run, args=("encode", encode)), threading.Thread(target=asr_then_diar)]
    [t.start() for t in th]
    [t.join() for t in th]
    res["overlap_wall_s"] = time.perf_counter() - t0
    res["overlap"] = times
    for k, v in times.items():
        print(f"  {k:12s}  {v:7.1f} s", flush=True)
    seq = res["encode_s"] + res["resample_s"] + res["asr_s"] + res["diarize_s"]
    res["sequential_wall_s"] = seq
    print(f"sequential {seq:.1f} s = {dur / seq:.1f}x real time; overlapped {res['overlap_wall_s']:.1f} s = "
          f"{dur / res['overlap_wall_s']:.1f}x", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
