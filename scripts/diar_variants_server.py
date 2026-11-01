"""pyannote community-1 speed / quality variants on the server's GPU (43.8 min): segmentation step and MIOpen find mode.

    MIOPEN_FIND_MODE=FAST|NORMAL .venv-rocm/bin/python scripts/diar_variants_server.py [steps as fractions of the 10 s window]

DER is against this run's default-step result (pyannote's own 0.1 = 1 s step), i.e. how much the variant changes the
current output, not accuracy against ground truth.
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

    steps = [float(x) for x in (sys.argv[1:] or ["0.1", "0.2", "0.25"])]
    wav, sr = sf.read(ROOT / "data/ref/ref_24k.wav", dtype="float32")
    w16 = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    pipe = pyannote_gpu.load("cuda")
    seg = pipe._segmentation
    ref = None
    for frac in steps:
        seg.step = frac * seg.duration
        pipe({"waveform": torch.from_numpy(w16[: 120 * 16000])[None], "sample_rate": 16000})  # warm-up
        f = {"waveform": torch.from_numpy(w16)[None], "sample_rate": 16000}
        torch.cuda.synchronize()
        t = time.perf_counter()
        with TimingHook() as hook:
            out = pipe(f, hook=hook)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t
        ann = getattr(out, "speaker_diarization", out)
        ref = ann if ref is None else ref
        top = sorted((round(ann.label_duration(k) / 60, 1) for k in ann.labels()), reverse=True)[:4]
        res = {"find_mode": os.environ.get("MIOPEN_FIND_MODE"), "step_frac": frac, "wall_s": round(wall, 1),
               **{k: round(v, 1) for k, v in f["timing"].items() if k != "total"},
               "der_vs_first": round(DiarizationErrorRate()(ref, ann), 4), "speakers_min": top}
        print("DIAR " + json.dumps(res), flush=True)
        with open(ROOT / "outputs/halo_throughput.jsonl", "a") as fh:
            fh.write(json.dumps({"label": "E3-diar", **res}) + "\n")


if __name__ == "__main__":
    main()
