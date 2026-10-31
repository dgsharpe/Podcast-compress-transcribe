"""Encode/decode the reference episode with neural codecs, segment by segment.

Each SEG-second segment is coded independently (cold start), which is also the
"seek to a point and start decoding" case. Outputs per codec variant:
  outputs/<variant>/<variant>.wav    decoded episode, 24 kHz mono (segments concatenated)
  outputs/<variant>/tokens.npz       codes per segment
  outputs/<variant>/timing.json      encode/decode wall time, threads, bitrate

usage: python scripts/run_codecs.py {mimi,qwen12hz} [--seg 30] [--threads 16]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "data/ref/ref_24k.wav"
OUT = ROOT / "outputs"
SR = 24000


def segments(wav, seg):
    n = int(seg * SR)
    return [wav[i : i + n] for i in range(0, len(wav), n)]


def save(variant, wavs, codes, enc_s, dec_s, threads, kbps, extra=None):
    d = OUT / variant
    d.mkdir(parents=True, exist_ok=True)
    sf.write(d / f"{variant}.wav", np.concatenate(wavs), SR, subtype="FLOAT")
    np.savez_compressed(d / "tokens.npz", *codes)
    dur = sum(len(w) for w in wavs) / SR
    info = dict(variant=variant, audio_s=dur, encode_s=enc_s, decode_s=dec_s,
                encode_rtf=enc_s / dur, decode_rtf=dec_s / dur, threads=threads,
                nominal_kbps=kbps, **(extra or {}))
    (d / "timing.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)


def fit(y, n):
    """Trim/pad decoded segment to the input length (codecs round to frames)."""
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    return y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))


def run_mimi(segs, threads):
    from transformers import MimiModel

    model = MimiModel.from_pretrained("kyutai/mimi").eval()
    card = model.config.codebook_size
    fps = model.config.frame_rate
    codes, enc_s = [], 0.0
    with torch.inference_mode():
        for s in segs:
            x = torch.from_numpy(s).float()[None, None]
            t = time.perf_counter()
            c = model.encode(x, num_quantizers=32).audio_codes  # (1, 32, frames)
            enc_s += time.perf_counter() - t
            codes.append(c)
        for k in (8, 16, 32):
            wavs, dec_s = [], 0.0
            for s, c in zip(segs, codes):
                t = time.perf_counter()
                y = model.decode(c[:, :k]).audio_values
                dec_s += time.perf_counter() - t
                wavs.append(fit(y.numpy(), len(s)))
            kbps = fps * k * np.log2(card) / 1000
            save(f"mimi_q{k}", wavs, [c[0, :k].numpy().astype(np.int16) for c in codes],
                 enc_s, dec_s, threads, kbps,
                 dict(frame_rate=fps, codebooks=k, codebook_size=card))


def run_qwen12hz(segs, threads):
    import librosa
    from qwen_tts import Qwen3TTSTokenizer

    tok = Qwen3TTSTokenizer.from_pretrained("Qwen/Qwen3-TTS-Tokenizer-12Hz", device_map="cpu")
    out_sr = tok.get_output_sample_rate()
    codes, wavs, enc_s, dec_s = [], [], 0.0, 0.0
    for s in segs:
        t = time.perf_counter()
        enc = tok.encode(s, sr=SR)
        enc_s += time.perf_counter() - t
        c = enc.audio_codes[0]  # (frames, K)
        t = time.perf_counter()
        y, _ = tok.decode({"audio_codes": c})
        dec_s += time.perf_counter() - t
        y = y[0]
        if out_sr != SR:
            y = librosa.resample(y=y, orig_sr=out_sr, target_sr=SR)
        codes.append(c.numpy().astype(np.int16))
        wavs.append(fit(y, len(s)))
    K = codes[0].shape[1]
    card = int(max(c.max() for c in codes)) + 1
    card = 1 << int(np.ceil(np.log2(card)))  # assume power-of-two codebook
    fps = SR / tok.get_encode_downsample_rate() if tok.get_input_sample_rate() == SR else \
        tok.get_input_sample_rate() / tok.get_encode_downsample_rate()
    save("qwen12hz", wavs, codes, enc_s, dec_s, threads, fps * K * np.log2(card) / 1000,
         dict(frame_rate=fps, codebooks=K, codebook_size_inferred=card, model_out_sr=out_sr))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("codec", choices=["mimi", "qwen12hz"])
    ap.add_argument("--seg", type=float, default=30.0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="only first N segments (smoke test)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    wav, sr = sf.read(REF, dtype="float32")
    assert sr == SR
    segs = segments(wav, a.seg)
    if a.limit:
        segs = segs[: a.limit]
    {"mimi": run_mimi, "qwen12hz": run_qwen12hz}[a.codec](segs, a.threads)
