"""Mimi encode on the server's GPU: the hub's overlapping windows vs streaming with the encoder's KV / padding caches (E6).

    .venv-rocm/bin/python scripts/mimi_streaming_server.py [--chunk-s 60]

Reference = one uninterrupted encode of a 5 min clip (exactly what a causal streaming encoder should reproduce).
Reports code agreement with it (all 32 codebooks, and the first 8) for the hub's windows (60 s + 20 s left context)
and for streaming, then times both on the full 43.8 min episode.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

import mimi_file  # noqa: E402

SPF = mimi_file.SPF


def timed(f):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = f()
    torch.cuda.synchronize()
    return out, round(time.perf_counter() - t, 2)


def full(m, wav):
    with torch.inference_mode():
        x = torch.from_numpy(wav).float()[None, None].cuda()
        return m.encode(x, num_quantizers=32).audio_codes[0].T.cpu().numpy()


def streaming(m, wav, chunk_s):
    """Encode chunk_s pieces in order, carrying the transformer KV cache and the convs' padding cache."""
    chunk = int(chunk_s * mimi_file.SR) // SPF * SPF
    kv, pad, out = None, None, []
    with torch.inference_mode():
        for s in range(0, len(wav), chunk):
            x = torch.from_numpy(wav[s:s + chunk]).float()[None, None].cuda()
            o = m.encode(x, num_quantizers=32, encoder_past_key_values=kv, padding_cache=pad, use_streaming=True)
            kv, pad = o.encoder_past_key_values, o.padding_cache
            out.append(o.audio_codes[0].T.cpu().numpy())
    return np.concatenate(out), kv


def agree(a, b):
    n = min(len(a), len(b))
    return {"frames": [len(a), len(b)], "all": round(float((a[:n] == b[:n]).mean()), 5),
            "q1_8": round(float((a[:n, :8] == b[:n, :8]).mean()), 5),
            "q1": round(float((a[:n, 0] == b[:n, 0]).mean()), 5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-s", type=float, default=60.0)
    a = ap.parse_args()
    wav, sr = sf.read(ROOT / "data/ref/ref_24k.wav", dtype="float32")
    m = mimi_file.load_model("cuda")
    clip = wav[: 300 * sr]
    ref = full(m, clip)
    res = {"chunk_s": a.chunk_s}
    res["windows_vs_full"] = agree(mimi_file.encode_array(m, clip), ref)
    codes, kv = streaming(m, clip, a.chunk_s)
    res["streaming_vs_full"] = agree(codes, ref)
    res["kv_cache"] = type(kv).__name__
    try:
        res["kv_cache_len_frames"] = int(kv.get_seq_length())
    except Exception as e:
        res["kv_cache_len_frames"] = f"? {type(e).__name__}"
    print("AGREE " + json.dumps(res), flush=True)

    streaming(m, wav[: 120 * sr], a.chunk_s)  # warm-up shapes
    mimi_file.encode_array(m, wav[: 120 * sr])
    _, res["windows_full_episode_s"] = timed(lambda: mimi_file.encode_array(m, wav))
    (codes_s, kv), res["streaming_full_episode_s"] = timed(lambda: streaming(m, wav, a.chunk_s))
    # per-chunk time early vs late: does the KV cache make later chunks slower?
    chunk = int(a.chunk_s * sr) // SPF * SPF
    times, kvc, pad = [], None, None
    with torch.inference_mode():
        for s in range(0, len(wav), chunk):
            x = torch.from_numpy(wav[s:s + chunk]).float()[None, None].cuda()
            torch.cuda.synchronize()
            t = time.perf_counter()
            o = m.encode(x, num_quantizers=32, encoder_past_key_values=kvc, padding_cache=pad, use_streaming=True)
            torch.cuda.synchronize()
            times.append(round(time.perf_counter() - t, 3))
            kvc, pad = o.encoder_past_key_values, o.padding_cache
    res["chunk_times_first_last"] = [times[:3], times[-3:]]
    print("RESULT " + json.dumps(res), flush=True)
    with open(ROOT / "outputs/throughput_server.jsonl", "a") as fh:
        fh.write(json.dumps({"label": "E6-mimi-streaming", **res}) + "\n")


if __name__ == "__main__":
    main()
