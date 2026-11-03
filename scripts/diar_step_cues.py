"""pyannote's window step vs the speaker each transcript cue gets (what the hub's VTT uses): the share of cue time
labelled as with pyannote's default 1 s step, after the best one-to-one matching of speaker numbers, per step.

    python scripts/diar_step_cues.py [steps in s ...]       (default 1 2 3 4 5)
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MIMI_HUB_DATA", "/tmp/mimi-diar-cues")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def main():
    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from scipy.optimize import linear_sum_assignment
    from hub import mimi_hub, pyannote_gpu

    steps = [float(x) for x in (sys.argv[1:] or ["1", "2", "3", "4", "5"])]
    wav, sr = sf.read(ROOT / "data/ref/ref_24k.wav", dtype="float32")
    w16 = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    m = mimi_hub.models()
    cues = [c for s in m["asr"].recognize(w16, sample_rate=16000) for c in mimi_hub.cues(s)]
    pipe = pyannote_gpu.load(m["device"], path=os.environ.get("TPUT_PYANNOTE_PATH"))
    seg = pipe._segmentation
    labels, times = {}, {}
    for step in steps:
        seg.step = step
        f = {"waveform": torch.from_numpy(w16)[None], "sample_rate": 16000}
        t = time.perf_counter()
        ann = getattr(pipe(f), "speaker_diarization", None)
        times[step] = time.perf_counter() - t
        turns = mimi_hub.number_turns([(s.start, s.end, k) for s, _, k in ann.itertracks(yield_label=True)])
        labels[step] = [mimi_hub.speaker_for(a, b, turns) or 0 for a, b, _ in cues]
    ref = labels[steps[0]]
    dur = np.array([b - a for a, b, _ in cues])
    out = []
    for step in steps:
        got = labels[step]
        ks, kr = sorted(set(got)), sorted(set(ref))
        overlap = np.zeros((len(ks), len(kr)))
        for g, r, d in zip(got, ref, dur, strict=True):
            overlap[ks.index(g), kr.index(r)] += d
        rows, cols = linear_sum_assignment(-overlap)
        agree = overlap[rows, cols].sum() / dur.sum()
        out.append({"step_s": step, "diar_s": round(times[step], 2), "speakers": len(ks),
                    "cue_time_same_speaker": round(float(agree), 4),
                    "cues_changed": int(round(len(cues) * (1 - agree)))})
        print("CUES " + json.dumps(out[-1]), flush=True)


if __name__ == "__main__":
    main()
