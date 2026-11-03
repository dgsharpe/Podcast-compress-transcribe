"""Mimi encode of one episode as P pieces streamed in lockstep as one batch, vs the hub's single stream.

    python scripts/mimi_pieces_check.py [P:warmup_frames ...]      (default 4:250 8:250 16:250 8:375)

Every piece streams the same number of frames, so the batch keeps one chunk schedule and identical cache shapes:
piece k covers its S = ceil(F / P) frames plus the W frames before it as warm-up (piece 0: from frame 0, so it
matches the single stream exactly, and its W extra frames at the end are dropped); the last is zero-padded.
Reports codes agreement with the single stream (all 32 codebooks / the first 8 / the first), separately for the W
frames after each piece boundary, and the encode time of the 43.8 min reference.
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("MIMI_HUB_DATA", "/tmp/mimi-pieces")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def main():
    import numpy as np
    import soundfile as sf
    import torch
    from hub import mimi_hub
    import mimi_file

    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    sync = torch.cuda.synchronize if dev == "cuda" else torch.mps.synchronize if dev == "mps" else (lambda: None)
    wav, sr = sf.read(ROOT / "data/ref/ref_24k.wav", dtype="float32")
    spf = mimi_file.SPF
    F = len(wav) // spf
    wav = wav[:F * spf]
    model = mimi_file.load_model(dev)

    def single():
        kv, pad, out = None, None, []
        for f0, f1 in mimi_hub.chunk_bounds(F):
            x = torch.from_numpy(wav[f0 * spf:f1 * spf].copy())[None, None].to(dev)
            o = mimi_hub.mimi_encode(model, x, encoder_past_key_values=kv, padding_cache=pad, use_streaming=True)
            kv, pad = o.encoder_past_key_values, o.padding_cache
            out.append(o.audio_codes[0].T.cpu().numpy())
        return np.concatenate(out)[:F]

    def pieces(P, W, C=750):
        S = -(-F // P)
        L = S + W
        x = np.zeros((P, L * spf), np.float32)
        for k in range(P):
            a = 0 if k == 0 else k * S - W
            seg = wav[a * spf:min(F, a + L) * spf]
            x[k, :len(seg)] = seg
        xt = torch.from_numpy(x).to(dev)[:, None]
        kv, pad, out = None, None, []
        for c0 in range(0, L, C):
            c1 = min(L, c0 + C)
            o = mimi_hub.mimi_encode(model, xt[..., c0 * spf:c1 * spf], encoder_past_key_values=kv, padding_cache=pad,
                                     use_streaming=True)
            kv, pad = o.encoder_past_key_values, o.padding_cache
            out.append(o.audio_codes.transpose(1, 2).cpu().numpy())  # (P, frames, 32)
        codes = np.concatenate(out, axis=1)
        keep = [codes[0, :S]] + [codes[k, W:W + S] for k in range(1, P)]
        return np.concatenate(keep)[:F], S

    single()  # warm-up
    sync()
    t = time.perf_counter()
    ref = single()
    sync()
    res = {"device": torch.cuda.get_device_name(0) if dev == "cuda" else dev, "frames": F,
           "single_s": round(time.perf_counter() - t, 2)}
    print("PIECES " + json.dumps(res), flush=True)
    for v in (sys.argv[1:] or ["4:250", "8:250", "16:250", "8:375"]):
        P, W = map(int, v.split(":"))
        pieces(P, W)  # warm-up
        sync()
        t = time.perf_counter()
        codes, S = pieces(P, W)
        sync()
        dt = time.perf_counter() - t
        near = np.zeros(F, bool)
        for k in range(1, P):
            near[k * S:k * S + W] = True
        same = codes == ref
        print("PIECES " + json.dumps({
            "P": P, "warmup": W, "s": round(dt, 2), "speedup": round(res["single_s"] / dt, 2),
            "all32": round(float(same.mean()), 4), "first8": round(float(same[:, :8].mean()), 4),
            "q1": round(float(same[:, 0].mean()), 4),
            "near_boundary_all32": round(float(same[near].mean()), 4) if near.any() else None,
            "elsewhere_all32": round(float(same[~near].mean()), 4)}), flush=True)


if __name__ == "__main__":
    main()
