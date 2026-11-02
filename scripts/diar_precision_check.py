"""pyannote community-1 (hub/pyannote_gpu.py) with the embedding ResNet's frames in fp32 / tf32 / fp16 / bf16 /
coreml, batch sizes, memory layout and the segmentation model's precision, on the reference episode (43.8 min):
time and DER against the first variant's run (how much the output changes, not accuracy against ground truth).

    python scripts/diar_precision_check.py [fp32:32 fp16:32:cl1 ...]    (dtype:batch[:cl1][:seg16|segbf16][:fold])

TPUT_PYANNOTE_PATH: a local copy of the model instead of the HF cache.
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import soundfile as sf  # noqa: E402
import torch  # noqa: E402


def main():
    import librosa
    from hub import pyannote_gpu
    from pyannote.audio.pipelines.utils.hook import TimingHook
    from pyannote.metrics.diarization import DiarizationErrorRate

    variants = [v.split(":") for v in (sys.argv[1:] or ["fp32:32", "fp16:32", "bf16:32"])]
    wav, sr = sf.read(ROOT / "data/ref/ref_24k.wav", dtype="float32")
    w16 = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    sync = torch.cuda.synchronize if device == "cuda" else torch.mps.synchronize if device == "mps" else (lambda: None)
    pipes = {}
    ref = None
    for dtype, batch, *rest in variants:
        cl = "cl1" in rest  # the ResNet in channels-last layout (set up at load)
        fold = "fold" in rest  # its batch norms folded into the convolutions
        seg = next((f[3:] for f in rest if f.startswith("seg")), "fp32")  # seg16 / segbf16: segmentation autocast
        seg = {"16": "fp16", "bf16": "bf16"}.get(seg, seg)
        pyannote_gpu.FRAMES_DTYPE, pyannote_gpu.BATCH_CHUNKS = dtype, int(batch)
        pyannote_gpu.CHANNELS_LAST, pyannote_gpu.SEG_DTYPE, pyannote_gpu.FOLD_BN = "1" if cl else "0", seg, fold
        key = (dtype, cl, seg, fold)  # all set up at load (the ResNet's weights are stored in its dtype)
        if key not in pipes:
            pipes[key] = pyannote_gpu.load(device, path=os.environ.get("TPUT_PYANNOTE_PATH"))
        pipe = pipes[key]
        pipe({"waveform": torch.from_numpy(w16[: 120 * 16000])[None], "sample_rate": 16000})  # warm-up
        f = {"waveform": torch.from_numpy(w16)[None], "sample_rate": 16000}
        sync()
        t = time.perf_counter()
        with TimingHook() as hook:
            out = pipe(f, hook=hook)
        sync()
        wall = time.perf_counter() - t
        ann = getattr(out, "speaker_diarization", out)
        ref = ann if ref is None else ref
        top = sorted((round(ann.label_duration(k) / 60, 1) for k in ann.labels()), reverse=True)[:4]
        name = torch.cuda.get_device_name(0) if device == "cuda" else device
        res = {"device": name, "dtype": dtype, "batch": int(batch), "channels_last": cl, "seg": seg, "fold": fold,
               "wall_s": round(wall, 2),
               **{k: round(v, 2) for k, v in f["timing"].items() if k != "total"},
               "der_vs_first": round(DiarizationErrorRate()(ref, ann), 4), "speakers_min": top}
        print("DIAR " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
