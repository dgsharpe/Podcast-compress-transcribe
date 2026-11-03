"""hub/autocast_weights.half_weights: the Parakeet encoder, Mimi's encoder and pyannote's ResNet frames with their
autocast layers stored in fp16 vs fp32 weights under the same autocast: identical outputs expected.

    python scripts/fp16_weights_check.py
"""
import os, sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("MIMI_HUB_DATA", "/tmp/mimi-aw")
import numpy as np, soundfile as sf, librosa, torch, onnx_asr
from hub import parakeet_native, autocast_weights, mimi_hub, pyannote_gpu
import mimi_file
res = {}
wav, sr = sf.read("data/ref/ref_24k.wav", dtype="float32")
# Parakeet encoder
w16 = librosa.resample(wav[: 400 * sr], orig_sr=sr, target_sr=16000)
B, L = 16, 20 * 16000
x = np.stack([w16[k * L:(k + 1) * L] for k in range(B)]); xl = np.full(B, L, dtype=np.int64); xl[B // 2:] = L * 3 // 4
feats, flens = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v2").asr._preprocessor(x, xl)
enc = parakeet_native.load(mimi_hub.MODELS_DIR / f"{mimi_hub.ASR_MODEL}-hf", "cuda")
a = parakeet_native.NativeEncoder(enc).run(None, {"audio_signal": feats, "length": flens})
torch.cuda.reset_peak_memory_stats(); n = autocast_weights.half_weights(enc)
b = parakeet_native.NativeEncoder(enc).run(None, {"audio_signal": feats, "length": flens})
res["parakeet"] = {"converted": n, "identical": bool(np.array_equal(a[0], b[0])),
                   "maxdiff": float(np.abs(a[0] - b[0]).max()),
                   "weights_gb": round(sum(p.numel() * p.element_size() for p in enc.parameters()) / 2**30, 2)}
del enc
# Mimi encoder (streaming, as the hub)
m = mimi_file.load_model("cuda")
clip = wav[: 300 * sr]
def codes():
    kv, pad, out = None, None, []
    F = len(clip) // mimi_file.SPF
    for f0, f1 in mimi_hub.chunk_bounds(F):
        xx = torch.from_numpy(clip[f0 * mimi_file.SPF:f1 * mimi_file.SPF].copy())[None, None].cuda()
        o = mimi_hub.mimi_encode(m, xx, encoder_past_key_values=kv, padding_cache=pad, use_streaming=True)
        kv, pad = o.encoder_past_key_values, o.padding_cache
        out.append(o.audio_codes[0].T.cpu().numpy())
    return np.concatenate(out)
c0 = codes(); n = autocast_weights.half_weights(m.encoder); c1 = codes()
res["mimi"] = {"converted": n, "codes_identical": float((c0 == c1).mean())}
# pyannote ResNet frames (fp16 autocast as on NVIDIA)
pyannote_gpu.FRAMES_DTYPE = "fp32"  # load with fp32 weights; autocast below as the hub's fp16 mode does
pipe = pyannote_gpu.load("cuda", path=os.environ.get("TPUT_PYANNOTE_PATH")); r = pipe._embedding.model_.resnet
w = torch.randn(32, 1, 160000, device="cuda") * 0.1
with torch.inference_mode():
    fb = pipe._embedding.model_.compute_fbank(w)
    with torch.autocast("cuda", dtype=torch.float16):
        f0 = r.forward_frames(fb.clone()).float()
    n = autocast_weights.half_weights(r.conv1, r.layer1, r.layer2, r.layer3, r.layer4)
    with torch.autocast("cuda", dtype=torch.float16):
        f1 = r.forward_frames(fb.clone()).float()
res["resnet"] = {"converted": n, "identical": bool(torch.equal(f0, f1)), "maxdiff": float((f0 - f1).abs().max())}
print("AW " + json.dumps(res), flush=True)
