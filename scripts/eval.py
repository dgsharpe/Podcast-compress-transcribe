"""Per-segment metrics for every decoded variant against the 24 kHz mono reference.

Per SEG-second segment (same boundaries as run_codecs.py):
  ASR word errors vs the reference's own transcript by the same ASR (dWER numerator/denominator),
  speaker-embedding cosine (WavLM-base-plus-sv), PESQ-WB (16 kHz), log-mel L1 (24 kHz).
Writes outputs/eval/<asr>/seg_<variant>.csv (one row per segment; bb_summary.jl reads them all),
and caches transcripts in outputs/eval/<asr>/asr_<variant>.json.

usage: python scripts/eval.py [--asr parakeet|whisper] [--seg 30] [--threads 16] [variants...]
"""

import argparse
import csv
import json
from pathlib import Path

import jiwer
import librosa
import numpy as np
import soundfile as sf
import torch
from whisper_normalizer.english import EnglishTextNormalizer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
EV = OUT / "eval"
SR = 24000
norm = EnglishTextNormalizer()


def variants():
    v = {"ref": ROOT / "data/ref/ref_24k.wav"}
    for p in sorted((OUT / "opus").glob("opus_*k.wav")):
        v[p.stem] = p
    for d in sorted(OUT.iterdir()):
        if (d / f"{d.name}.wav").exists():
            v[d.name] = d / f"{d.name}.wav"
    return v


def load(path, n):
    y, sr = sf.read(path, dtype="float32")
    assert sr == SR, (path, sr)
    if y.ndim > 1:
        y = y.mean(1)
    return y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))


def segs(y, seg):
    k = int(seg * SR)
    return [y[i : i + k] for i in range(0, len(y), k)]


class ASR:
    """Cached per-segment transcripts. `parakeet` (CTC, default) cannot skip or invent whole
    sentences, so its dWER tracks the acoustics; Whisper on 30 s windows sometimes drops half a
    segment or appends the next ad, which swamped codec differences in the first run."""

    def __init__(self, kind, threads):
        self.kind = kind
        if kind == "whisper":
            from faster_whisper import WhisperModel
            self.m = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8", cpu_threads=threads)
        else:
            from transformers import AutoProcessor, ParakeetForCTC
            self.p = AutoProcessor.from_pretrained("nvidia/parakeet-ctc-1.1b")
            self.m = ParakeetForCTC.from_pretrained("nvidia/parakeet-ctc-1.1b").eval()

    @torch.inference_mode()
    def one(self, x):
        if self.kind == "whisper":
            s, _ = self.m.transcribe(x, language="en", beam_size=5, condition_on_previous_text=False,
                                     vad_filter=False, without_timestamps=True)
            return " ".join(t.text.strip() for t in s)
        i = self.p(x, sampling_rate=16000, return_tensors="pt")
        return self.p.batch_decode(self.m.generate(**i), skip_special_tokens=True)[0]

    def __call__(self, name, chunks16):
        cache = EV / f"asr_{name}.json"
        if cache.exists():
            return json.loads(cache.read_text())
        out = []
        for i, x in enumerate(chunks16):
            out.append(self.one(x))
            print(f"asr {name} {i + 1}/{len(chunks16)}", flush=True)
        cache.write_text(json.dumps(out, indent=0))
        return out


class Spk:
    def __init__(self):
        from transformers import AutoFeatureExtractor, WavLMForXVector
        self.fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        self.m = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval()

    @torch.inference_mode()
    def __call__(self, x16):
        i = self.fe(x16, sampling_rate=16000, return_tensors="pt")
        e = self.m(**i).embeddings[0]
        return torch.nn.functional.normalize(e, dim=-1).numpy()


def logmel(x):
    m = librosa.feature.melspectrogram(y=x, sr=SR, n_fft=1024, hop_length=256, n_mels=80)
    return np.log(m + 1e-5)


def pesq_wb(r16, d16):
    from pesq import pesq
    try:
        return pesq(16000, r16, d16, "wb")
    except Exception:  # silent / no-utterance segments
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", type=float, default=30.0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--asr", choices=["parakeet", "whisper"], default="parakeet")
    ap.add_argument("only", nargs="*")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    global EV
    EV = OUT / "eval" / a.asr
    EV.mkdir(parents=True, exist_ok=True)
    V = variants()
    ref = sf.read(V["ref"], dtype="float32")[0]
    n = len(ref)
    rs = segs(ref, a.seg)
    rs16 = [librosa.resample(s, orig_sr=SR, target_sr=16000) for s in rs]
    asr, spk = ASR(a.asr, a.threads), Spk()
    rtxt = [norm(t) for t in asr("ref", rs16)]
    remb = [spk(s) for s in rs16]
    rmel = [logmel(s) for s in rs]

    names = [k for k in V if k != "ref" and (not a.only or k in a.only)
             and not (EV / f"seg_{k}.csv").exists()]
    for name in names:
        ds = segs(load(V[name], n), a.seg)
        ds16 = [librosa.resample(s, orig_sr=SR, target_sr=16000) for s in ds]
        dtxt = [norm(t) for t in asr(name, ds16)]
        rows = []
        for i, (r16, d16, d) in enumerate(zip(rs16, ds16, ds)):
            if rtxt[i].strip() and not dtxt[i].strip():
                words = errs = len(rtxt[i].split())  # all deletions
            elif rtxt[i].strip():
                o = jiwer.process_words(rtxt[i], dtxt[i])
                words, errs = len(rtxt[i].split()), o.substitutions + o.deletions + o.insertions
            else:
                words, errs = 0, len(dtxt[i].split())
            cos = float(remb[i] @ spk(d16))
            ml = float(np.mean(np.abs(rmel[i] - logmel(d))))
            rows.append([name, i, i * a.seg, words, errs, f"{cos:.5f}", f"{pesq_wb(r16, d16):.4f}", f"{ml:.5f}"])
        with open(EV / f"seg_{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["variant", "seg", "t0", "ref_words", "errors", "spk_cos", "pesq_wb", "mel_l1"])
            w.writerows(rows)
            print(f"done {name}", flush=True)


if __name__ == "__main__":
    main()
