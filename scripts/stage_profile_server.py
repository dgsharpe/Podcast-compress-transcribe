"""Where the server's GPU time goes inside each pipeline stage, on one 43.8 min episode (E1-E3 groundwork).

    .venv-rocm/bin/python scripts/stage_profile_server.py

Mimi: window batch size 1 (the hub) vs 4 / 8, and bf16, with code agreement vs fp32 batch 1.
Parakeet: VAD / preprocessing / encoder / decoder time (hub settings: fp16 torch encoder on the GPU).
pyannote: embedding step split into fbank, ResNet frames and pooling, plus segmentation.
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MIMI_HUB_DATA", "/tmp/mimi-stage-profile")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402


def sync_time(f):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = f()
    torch.cuda.synchronize()
    return out, round(time.perf_counter() - t, 2)


def mimi_windows(wav, chunk_s=60.0, ctx_s=20.0, sr=24000, spf=1920):
    """The hub's windows: chunk_s of new audio with ctx_s of left context; [(lo, s, e)] in samples."""
    chunk, ctx = int(chunk_s * sr) // spf * spf, int(ctx_s * sr) // spf * spf
    return [(max(0, s - ctx), s, min(len(wav), s + chunk)) for s in range(0, len(wav), chunk)]


def mimi_batched(m, wav, batch, dtype=None, spf=1920):
    """Encode the hub's windows [batch] at a time (zero-padded to the longest), codes (frames, 32)."""
    wins = mimi_windows(wav)
    out = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype or torch.float32, enabled=dtype is not None):
        for i in range(0, len(wins), batch):
            group = wins[i:i + batch]
            n = max(e - lo for lo, s, e in group)
            x = torch.zeros(len(group), 1, n)
            for j, (lo, s, e) in enumerate(group):
                x[j, 0, :e - lo] = torch.from_numpy(wav[lo:e])
            codes = m.encode(x.cuda(), num_quantizers=32).audio_codes.cpu().numpy()  # (B, 32, F)
            for j, (lo, s, e) in enumerate(group):
                c = codes[j].T
                out.append(c[(s - lo) // spf:][: (e - s + spf - 1) // spf])
    return np.concatenate(out)


def main():
    import librosa
    from hub import mimi_hub, pyannote_gpu
    import mimi_file

    wav, sr = sf.read(ROOT / "data/ref/ref_24k.wav", dtype="float32")
    w16 = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    m = mimi_hub.models()
    res = {"audio_s": len(wav) / sr}

    # --- Mimi ---
    mimi = m["mimi"]
    mimi_batched(mimi, wav[: 300 * sr], 8)  # warm-up shapes
    ref, res["mimi_b1_s"] = sync_time(lambda: mimi_file.encode_array(mimi, wav))
    for b in (4, 8):
        mimi_batched(mimi, wav[: 600 * sr], b)
        codes, res[f"mimi_b{b}_s"] = sync_time(lambda: mimi_batched(mimi, wav, b))
        res[f"mimi_b{b}_agree"] = round(float((codes == ref).mean()), 5)
    mimi_batched(mimi, wav[: 600 * sr], 8, torch.bfloat16)
    codes, res["mimi_b8_bf16_s"] = sync_time(lambda: mimi_batched(mimi, wav, 8, torch.bfloat16))
    res["mimi_b8_bf16_agree"] = round(float((codes == ref).mean()), 5)
    res["mimi_b8_bf16_agree_q1_8"] = round(float((codes[:, :8] == ref[:, :8]).mean()), 5)
    print("MIMI", json.dumps({k: v for k, v in res.items() if k.startswith("mimi")}), flush=True)

    # --- Parakeet ---
    asr = m["asr"]
    inner = asr
    while hasattr(inner, "asr"):
        inner = inner.asr
    vad = asr.vad if hasattr(asr, "vad") else None
    totals = {}

    def wrap(obj, attr, key, method="run"):
        target = getattr(obj, attr)
        f = getattr(target, method) if method else target

        def timed(*a, **k):
            torch.cuda.synchronize()
            t = time.perf_counter()
            try:
                return f(*a, **k)
            finally:
                torch.cuda.synchronize()
                totals[key] = totals.get(key, 0.0) + time.perf_counter() - t
                totals[key + "_calls"] = totals.get(key + "_calls", 0) + 1
        if method:
            setattr(target, method, timed)
        else:
            setattr(obj, attr, timed)

    wrap(inner, "_encoder", "encoder")
    wrap(inner, "_decoder_joint", "decoder")
    wrap(inner, "_preprocessor", "preprocess", method=None)
    if vad is not None and hasattr(vad, "_model"):
        wrap(vad, "_model", "vad")
    _, total = sync_time(lambda: list(asr.recognize(w16, sample_rate=16000)))
    res["asr"] = {"total_s": round(total, 1), **{k: round(v, 2) for k, v in totals.items()}}
    print("ASR", json.dumps(res["asr"]), flush=True)

    # --- pyannote ---
    pipe = pyannote_gpu.load("cuda")
    pipe({"waveform": torch.from_numpy(w16[: 120 * 16000])[None], "sample_rate": 16000})
    emb = pipe._embedding.model_
    parts = {}

    def timed_method(obj, name, key):
        f = getattr(obj, name)

        def g(*a, **k):
            torch.cuda.synchronize()
            t = time.perf_counter()
            r = f(*a, **k)
            torch.cuda.synchronize()
            parts[key] = parts.get(key, 0.0) + time.perf_counter() - t
            return r
        setattr(obj, name, g)

    timed_method(emb, "compute_fbank", "fbank")
    timed_method(emb.resnet, "forward_frames", "resnet_frames")
    timed_method(emb.resnet, "forward_embedding", "pooling")
    from pyannote.audio.pipelines.utils.hook import TimingHook
    f = {"waveform": torch.from_numpy(w16)[None], "sample_rate": 16000}
    with TimingHook() as hook:
        _, total = sync_time(lambda: pipe(f, hook=hook))
    res["diar"] = {"total_s": total, **{k: round(v, 2) for k, v in f.get("timing", {}).items()},
                   **{k: round(v, 2) for k, v in parts.items()}}
    print("DIAR", json.dumps(res["diar"]), flush=True)
    (ROOT / "outputs/stage_profile_server.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
