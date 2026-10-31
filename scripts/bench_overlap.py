"""Mimi encode vs Parakeet ASR on the CPU: alone, then overlapped, for one torch / onnxruntime thread split.

    uv run python scripts/bench_overlap.py data/ref/ref_24k.wav --minutes 15 --torch 8 --ort 8 [--no-spin]

Prints one JSON line: encode_s, asr_s (alone), overlap_wall_s and each stage's time when overlapped.
"""

import argparse
import json
import os
import sys
import threading
import time

import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import mimi_file  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--minutes", type=float, default=15)
    ap.add_argument("--torch", type=int, default=16)
    ap.add_argument("--ort", type=int, default=0, help="0 = onnxruntime default")
    ap.add_argument("--no-spin", action="store_true", help="onnxruntime threads sleep instead of spin-waiting")
    a = ap.parse_args()

    import librosa
    import onnx_asr
    import onnxruntime as ort
    import torch
    torch.set_num_threads(a.torch)
    so = ort.SessionOptions()
    if a.ort:
        so.intra_op_num_threads = a.ort
    if a.no_spin:
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    asr = (onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v2", sess_options=so)
           .with_vad(onnx_asr.load_vad("silero")).with_timestamps())
    mimi = mimi_file.load_model()

    wav, sr = sf.read(a.wav, dtype="float32")
    wav = wav[: int(a.minutes * 60 * sr)]
    w16 = librosa.resample(wav, orig_sr=sr, target_sr=16000)

    def encode():
        mimi_file.encode_array(mimi, wav)

    def transcribe():
        list(asr.recognize(w16, sample_rate=16000))

    mimi_file.encode_array(mimi, wav[: 20 * sr])  # warm-up
    list(asr.recognize(w16[: 20 * 16000], sample_rate=16000))

    res = dict(minutes=a.minutes, torch=a.torch, ort=a.ort or "default", spin=not a.no_spin)
    for name, f in (("encode_s", encode), ("asr_s", transcribe)):
        t = time.perf_counter()
        f()
        res[name] = round(time.perf_counter() - t, 1)
    times = {}

    def run(name, f):
        t = time.perf_counter()
        f()
        times[name] = round(time.perf_counter() - t, 1)

    t0 = time.perf_counter()
    th = [threading.Thread(target=run, args=("encode", encode)), threading.Thread(target=run, args=("asr", transcribe))]
    [t.start() for t in th]
    [t.join() for t in th]
    res["overlap_wall_s"] = round(time.perf_counter() - t0, 1)
    res["overlap"] = times
    res["sequential_s"] = round(res["encode_s"] + res["asr_s"], 1)
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
