"""Where a single episode's audio prep goes (hub/mimi_hub._StreamPrep does it all in one thread), and sharded versions:

  decode  ffmpeg to 24 kHz float, a WAV and an MP3 of the 43.8 min reference; and the MP3 in N parallel ranges
  resample  soxr HQ 24 -> 16 kHz, streamed in 1 s chunks as the hub does
  vad     Silero, one ONNX call per 32 ms frame as the hub does; and in K shards stepped together as one batch,
          each shard starting WARM frames early from a zero state (those frames dropped)

Prints the times and how far the sharded results are from the hub's: max sample difference of the parallel decode,
max probability difference, frames on the other side of 0.5, and the speech segments, of the batched VAD.

    python scripts/prep_profile.py [--shards 32] [--warm 64] [--procs 8]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("MIMI_HUB_DATA", tempfile.mkdtemp(prefix="mimi-prep-"))
HOP, CTX, SR = 512, 64, 16000


def decode(src, start=None, dur=None, sr=24000):
    cmd = ["ffmpeg", "-nostdin", "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.6f}"]
    cmd += ["-i", str(src)]
    if dur is not None:
        cmd += ["-t", f"{dur:.6f}"]
    cmd += ["-f", "f32le", "-ac", "1", "-ar", str(sr), "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.float32)


def frames(w):
    """The hub's VAD frames of w: the first left-padded, then w[448 + 512 k:][:576], the last right-padded."""
    out = [np.pad(w[:HOP], (CTX, 0))]
    k = 0
    while HOP - CTX + HOP * k + CTX + HOP <= len(w):
        a = HOP - CTX + HOP * k
        out.append(w[a:a + CTX + HOP])
        k += 1
    if last := len(w) % HOP:
        out.append(np.pad(w[len(w) - last - CTX:], (0, HOP - last)))
    return np.stack(out).astype(np.float32)


def vad_sequential(sess, fr):
    state, srs, probs = np.zeros((2, 1, 128), np.float32), np.array([SR], np.int64), []
    for f in fr:
        o, state = sess.run(["output", "stateN"], {"input": f[None], "state": state, "sr": srs})
        probs.append(o[0, 0])
    return np.array(probs, np.float32)


def vad_batched(sess, fr, shards, warm):
    """Shards of the frames stepped together; shard k covers frames [k L, (k+1) L), starting `warm` frames early."""
    n = len(fr)
    L = -(-n // shards)
    starts = [max(0, k * L - warm) for k in range(shards)]
    steps = max(min(n, (k + 1) * L) - s for k, s in enumerate(starts))
    state, srs = np.zeros((2, shards, 128), np.float32), np.array([SR], np.int64)
    out = np.zeros((shards, steps), np.float32)
    for j in range(steps):
        idx = np.minimum([s + j for s in starts], n - 1)
        o, state = sess.run(["output", "stateN"], {"input": fr[idx], "state": state, "sr": srs})
        out[:, j] = o[:, 0]
    probs = np.empty(n, np.float32)
    for k, s in enumerate(starts):
        a, b = k * L, min(n, (k + 1) * L)
        if a < b:
            probs[a:b] = out[k, a - s:b - s]
    return probs


def vad_threaded(sess, fr, shards, warm):
    """Shards of the frames, each run as the hub does in its own thread, starting `warm` frames early."""
    n = len(fr)
    L = -(-n // shards)

    def shard(k):
        a, b = k * L, min(n, (k + 1) * L)
        s = max(0, a - warm)
        return a, vad_sequential(sess, fr[s:b])[a - s:] if a < b else None
    probs = np.empty(n, np.float32)
    with ThreadPoolExecutor(shards) as ex:
        for a, p in ex.map(shard, range(shards)):
            if p is not None:
                probs[a:a + len(p)] = p
    return probs


def segments(vad, probs, length):
    from hub import mimi_hub
    return list(mimi_hub._merge_stream(vad, vad._find_segments(iter(probs), HOP), lambda: length, SR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=32)
    ap.add_argument("--warm", type=int, default=64, help="warm-up frames per shard (32 ms each)")
    ap.add_argument("--procs", type=int, default=8, help="parallel ffmpeg range decodes")
    a = ap.parse_args()
    import onnx_asr
    import onnxruntime as ort
    import soxr

    res = {}
    src = ROOT / "data/ref/ref_24k.wav"
    mp3 = Path(tempfile.mkdtemp()) / "ref.mp3"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(src), "-b:a", "128k", str(mp3)], check=True)
    for name, f in (("wav", src), ("mp3", mp3)):
        t = time.perf_counter()
        w24 = decode(f)
        res[f"decode_{name}_s"] = round(time.perf_counter() - t, 2)
    dur = len(w24) / 24000
    res["audio_min"] = round(dur / 60, 1)

    # the MP3 in parallel ranges, each from a second early (the bit reservoir), trimmed to its range
    def part(k, n=a.procs, pre=1.0):
        s = k * dur / n
        x = decode(mp3, max(0.0, s - pre), dur / n + (s - max(0.0, s - pre)))
        skip = round((s - max(0.0, s - pre)) * 24000)
        return x[skip:skip + round((k + 1) * dur / n * 24000) - round(s * 24000)]
    t = time.perf_counter()
    with ThreadPoolExecutor(a.procs) as ex:
        par = np.concatenate(list(ex.map(part, range(a.procs))))
    res["decode_mp3_parallel_s"] = round(time.perf_counter() - t, 2)
    m = min(len(par), len(w24))
    res["parallel_len_diff"] = int(len(par) - len(w24))
    res["parallel_max_abs_diff"] = round(float(np.abs(par[:m] - w24[:m]).max()), 5)

    t = time.perf_counter()
    rs = soxr.ResampleStream(24000, SR, 1, dtype="float32", quality="HQ")
    w16 = np.concatenate([rs.resample_chunk(w24[i:i + 24000], last=i + 24000 >= len(w24))
                          for i in range(0, len(w24), 24000)])
    res["resample_s"] = round(time.perf_counter() - t, 2)

    so = ort.SessionOptions()
    so.intra_op_num_threads = so.inter_op_num_threads = 1
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    vad = onnx_asr.load_vad("silero", sess_options=so)
    sess = vad._model
    fr = frames(w16)
    t = time.perf_counter()
    p_seq = vad_sequential(sess, fr)
    res["vad_sequential_s"] = round(time.perf_counter() - t, 2)
    res["vad_frames"] = len(fr)
    seg_a = segments(vad, p_seq, len(w16))
    runs = [(f"batched_{k}_w{a.warm}", lambda k=k: vad_batched(sess, fr, k, a.warm)) for k in (a.shards,)]
    runs += [(f"threads_{a.shards}_w{w}", lambda w=w: vad_threaded(sess, fr, a.shards, w)) for w in (64, 312, 1250)]
    for name, fn in runs:
        t = time.perf_counter()
        p_bat = fn()
        el = time.perf_counter() - t
        seg_b = segments(vad, p_bat, len(w16))
        res[f"vad_{name}"] = {
            "s": round(el, 2), "max_prob_diff": round(float(np.abs(p_bat - p_seq).max()), 4),
            "frames_flipped": int(((p_bat > 0.5) != (p_seq > 0.5)).sum()),
            "segments": [len(seg_a), len(seg_b)], "segments_identical": seg_a == seg_b,
            "segments_differing": len(set(seg_a) ^ set(seg_b))}
    print("PREP " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
