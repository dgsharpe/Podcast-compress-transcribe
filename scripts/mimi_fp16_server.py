"""Mimi encoder in fp16 (quantizer kept fp32) vs fp32 on the server's GPU: speed, code agreement, decoded quality.

    .venv-rocm/bin/python scripts/mimi_fp16_server.py

On a 5 min clip, one uninterrupted encode each way. Quality: the fp32 decoder applied to both code sets, compared
with the original by log-mel L1 (lower is better) and with each other by SNR.
"""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, soundfile as sf, torch, librosa
import mimi_file

def t(f):
    f(); torch.cuda.synchronize(); a = time.perf_counter(); r = f(); torch.cuda.synchronize(); return r, round(time.perf_counter() - a, 3)

def main():
    wav, sr = sf.read(ROOT / "data/ref/ref_24k.wav", dtype="float32")
    clip = wav[: 300 * sr]
    m = mimi_file.load_model("cuda")
    x = torch.from_numpy(clip)[None, None].cuda()
    def enc(dtype):
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                lat = m.encoder(x).latents
            return m.quantizer.encode(lat.float(), 32).transpose(0, 1)  # (B, 32, F), fp32 distances
    c32, t32 = t(lambda: enc(None))
    c16, t16 = t(lambda: enc(torch.float16))
    a, b = c32[0].T.cpu().numpy(), c16[0].T.cpu().numpy()
    def dec(c):
        with torch.inference_mode():
            return m.decode(c).audio_values[0, 0].float().cpu().numpy()[: len(clip)]
    y32, y16 = dec(c32), dec(c16)
    def logmel(y):
        return np.log(librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=80) + 1e-5)
    ref = logmel(clip)
    res = {"fp32_s": t32, "fp16_s": t16, "speedup": round(t32 / t16, 2),
           "codes_identical": round(float((a == b).mean()), 4), "first8_identical": round(float((a[:, :8] == b[:, :8]).mean()), 4),
           "q1_identical": round(float((a[:, 0] == b[:, 0]).mean()), 4),
           "logmel_L1_vs_original_fp32": round(float(np.abs(logmel(y32) - ref).mean()), 4),
           "logmel_L1_vs_original_fp16": round(float(np.abs(logmel(y16) - ref).mean()), 4),
           "snr_fp16_vs_fp32_decode_db": round(float(10 * np.log10((y32 ** 2).sum() / ((y32 - y16) ** 2).sum())), 1)}
    print("MIMIFP16 " + json.dumps(res), flush=True)
    sf.write("/tmp/claude-1000/mimi_fp32.wav", y32, sr); sf.write("/tmp/claude-1000/mimi_fp16.wav", y16, sr)

if __name__ == "__main__":
    main()
