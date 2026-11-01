"""Native transformers ParakeetEncoder vs the hub's onnx2torch encoder on the server's GPU (E7).

    .venv-rocm/bin/python scripts/parakeet_native_server.py [--minutes 43.8] [--compile]

Captures the encoder batches of a real recognize() (length-bucketed, as the hub runs), checks the native encoder
against onnx2torch in fp32 (relative max |diff| over valid frames), then times: onnx2torch fp16 (the hub today),
native fp16 / bf16, and with --compile native fp16 under torch.compile.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MIMI_HUB_DATA", "/tmp/mimi-native")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402


def timed(f):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = f()
    torch.cuda.synchronize()
    return out, round(time.perf_counter() - t, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=43.8)
    ap.add_argument("--compile", action="store_true")
    a = ap.parse_args()

    import librosa
    import soundfile as sf
    from hub import mimi_hub, parakeet_native

    wav, sr = sf.read(ROOT / "data/ref/ref_24k.wav", dtype="float32")
    w16 = librosa.resample(wav[: int(a.minutes * 60 * sr)], orig_sr=sr, target_sr=16000)
    m = mimi_hub.models()
    inner = m["asr"]
    while hasattr(inner, "asr"):
        inner = inner.asr
    current = inner._encoder
    batches = []
    run = current.run

    def capture(names, feeds):
        batches.append({"audio_signal": np.array(feeds["audio_signal"]), "length": np.asarray(feeds["length"]).copy()})
        return run(names, feeds)

    current.run = capture
    list(m["asr"].recognize(w16, sample_rate=16000))
    current.run = run
    res = {"minutes": a.minutes, "batches": len(batches)}

    t = time.perf_counter()
    model_dir = parakeet_native.convert(mimi_hub.ASR_MODEL, mimi_hub.MODELS_DIR / f"{mimi_hub.ASR_MODEL}-hf")
    native = parakeet_native.load(model_dir, "cuda")
    res["convert_load_s"] = round(time.perf_counter() - t, 1)

    # 1. correctness, fp32 both
    def onnx2torch_fp32(feeds):
        with torch.inference_mode():
            out, lens = current.m(torch.from_numpy(feeds["audio_signal"]).cuda(), torch.from_numpy(feeds["length"]).cuda())
        return out.float().cpu().numpy(), lens.cpu().numpy()

    nat32 = parakeet_native.NativeEncoder(native, torch.float32)
    worst, len_ok = 0.0, True
    for feeds in batches[:6]:
        ref, rl = onnx2torch_fp32(feeds)
        got, gl = nat32.run(None, feeds)
        len_ok &= bool(np.array_equal(rl, gl))
        for r in range(ref.shape[0]):
            k = int(rl[r])
            d = np.abs(got[r, :, :k] - ref[r, :, :k]).max() / (np.sqrt((ref[r, :, :k] ** 2).mean()) + 1e-9)
            worst = max(worst, float(d))
    res["fp32_lengths_equal"], res["fp32_max_rel_diff"] = len_ok, round(worst, 5)
    print("CHECK " + json.dumps(res), flush=True)

    # 2. speed on all captured batches
    def all_batches(enc):
        return [enc.run(None, f) for f in batches]

    variants = [("onnx2torch_fp16", current), ("native_fp16", parakeet_native.NativeEncoder(native, torch.float16)),
                ("native_bf16", parakeet_native.NativeEncoder(native, torch.bfloat16))]
    if a.compile:
        variants.append(("native_fp16_compiled", parakeet_native.NativeEncoder(native, torch.float16, compile=True)))
    for name, enc in variants:
        t = time.perf_counter()
        for f in batches[:3]:
            enc.run(None, f)  # warm-up (and compilation)
        res[f"{name}_warmup_s"] = round(time.perf_counter() - t, 1)
        _, res[f"{name}_s"] = timed(lambda: all_batches(enc))
        print(f"TIME {name}: {res[f'{name}_s']} s (warm-up {res[f'{name}_warmup_s']} s)", flush=True)
    with open(ROOT / "outputs/halo_throughput.jsonl", "a") as fh:
        fh.write(json.dumps({"label": "E7-native-encoder", **res}) + "\n")
    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
