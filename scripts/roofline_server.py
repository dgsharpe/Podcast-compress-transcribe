"""How close each GPU stage gets to the GPU's peak: FLOPs (torch.utils.flop_counter) / measured time.

    .venv-rocm/bin/python scripts/roofline_server.py

Ceilings are measured here too (large fp16 / fp32 matmuls, a big device-to-device copy), next to the datasheet ones
for the Radeon 8060S (gfx1151: 40 CUs at 2.9 GHz -> ~59 TFLOP/s fp16 WMMA, ~30 fp32 with dual issue; LPDDR5X-8000 on
a 256-bit bus -> 256 GB/s). Stages use the hub's own models and representative inputs:
  Mimi encode: one 60 s chunk, streaming with caches (fp32)
  Parakeet encoder: a batch of 16 x 20 s segments (native transformers encoder, fp16 autocast)
  WeSpeaker ResNet34 (pyannote embeddings): 32 x 10 s chunks, fbank + frames (fp32, batch norms frozen)
  pyannote segmentation (SincNet + LSTM, MIOpen off): 32 x 10 s chunks (the FLOP counter may miss the LSTM)
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("MIMI_HUB_DATA", "/tmp/mimi-roofline")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402


def timed(f, reps=5):
    f()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        f()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return sorted(ts)[len(ts) // 2]


def flops(f):
    with FlopCounterMode(display=False) as fc:
        f()
    return fc.get_total_flops()


def main():
    import soundfile as sf
    from hub import mimi_hub, parakeet_native, pyannote_gpu

    res = {}
    # --- ceilings
    for dt, name in ((torch.float16, "fp16"), (torch.float32, "fp32")):
        a = torch.randn(8192, 8192, device="cuda", dtype=dt)
        t = timed(lambda: a @ a, reps=3)
        res[f"peak_matmul_{name}_tflops"] = round(2 * 8192 ** 3 / t / 1e12, 1)
    x = torch.empty(2 ** 30, dtype=torch.uint8, device="cuda")
    y = torch.empty_like(x)
    t = timed(lambda: y.copy_(x), reps=3)
    res["copy_bandwidth_gbs"] = round(2 * x.numel() / t / 1e9, 1)  # read + write
    del a, x, y
    print("CEILINGS " + json.dumps(res), flush=True)

    wav, sr = sf.read(ROOT / "data/ref/ref_24k.wav", dtype="float32")
    m = mimi_hub.models()

    def stage(name, f, audio_s, peak_key):
        fl = flops(f)
        t = timed(f)
        tflops = fl / t / 1e12
        res[name] = {"gflop_per_audio_s": round(fl / audio_s / 1e9, 2), "time_s": round(t, 3),
                     "tflops": round(tflops, 2), "pct_of_measured_peak": round(100 * tflops / res[peak_key], 1),
                     "x_realtime": round(audio_s / t)}
        print(f"STAGE {name}: " + json.dumps(res[name]), flush=True)

    # --- Mimi: a 60 s chunk in steady-state streaming (caches warm from the previous minute)
    mimi = m["mimi"]
    with torch.inference_mode():
        x0 = torch.from_numpy(wav[: 60 * sr]).cuda()[None, None]
        o = mimi.encode(x0, num_quantizers=32, use_streaming=True)
        kv, pad = o.encoder_past_key_values, o.padding_cache
    x1 = torch.from_numpy(wav[60 * sr: 120 * sr]).cuda()[None, None]

    def mimi_chunk():
        import copy
        with torch.inference_mode():
            mimi.encode(x1, num_quantizers=32, encoder_past_key_values=copy.deepcopy(kv),
                        padding_cache=copy.deepcopy(pad), use_streaming=True)
    stage("mimi_encode_fp32", mimi_chunk, 60.0, "peak_matmul_fp32_tflops")

    # --- Parakeet encoder: 16 segments of 20 s (128 mel bins at 100 frames/s), fp16
    enc = parakeet_native.load(mimi_hub.MODELS_DIR / f"{mimi_hub.ASR_MODEL}-hf", "cuda")
    feats = torch.randn(16, 2000, 128, device="cuda")
    mask = torch.ones(16, 2000, dtype=torch.bool, device="cuda")

    def parakeet():
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            enc(input_features=feats, attention_mask=mask)
    stage("parakeet_encoder_fp16", parakeet, 16 * 20.0, "peak_matmul_fp16_tflops")
    del enc

    # --- pyannote: WeSpeaker ResNet34 over 32 chunks of 10 s, and the segmentation model
    pipe = pyannote_gpu.load("cuda")
    w16 = torch.randn(32, 1, 160000, device="cuda") * 0.1
    emb = pipe._embedding.model_

    def resnet():
        with torch.inference_mode():
            emb.resnet.forward_frames(emb.compute_fbank(w16))
    # 32 chunks at the 2 s step cover 64 s of new audio
    stage("wespeaker_resnet34_fp32", resnet, 32 * 2.0, "peak_matmul_fp32_tflops")
    seg = pipe._segmentation.model

    def segmentation():
        with torch.inference_mode():
            seg(w16)
    stage("pyannote_segmentation_fp32", segmentation, 32 * 2.0, "peak_matmul_fp32_tflops")
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs/roofline_server.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
