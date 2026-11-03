"""What sharding Silero VAD (scripts/prep_profile.py: K shards, each started W frames early from a zero state) does to
the transcript: the hub's ASR of the 43.8 min reference with the sequential VAD's probabilities and with sharded
ones, words compared.

    python scripts/vad_shard_asr_check.py [--shards 16] [--warm 1250 312]
"""
import argparse
import difflib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("MIMI_HUB_DATA", tempfile.mkdtemp(prefix="mimi-vadshard-"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=16)
    ap.add_argument("--warm", type=int, nargs="+", default=[1250, 312])
    a = ap.parse_args()
    import soundfile as sf
    import soxr
    import prep_profile as pp
    from hub import mimi_hub

    w24, sr = sf.read(ROOT / "data/ref/ref_24k.wav", dtype="float32")
    w16 = soxr.resample(w24, sr, 16000, quality="HQ")
    m = mimi_hub.models()
    sess = m["asr"].vad._model
    fr = pp.frames(w16)

    def words(probs):
        mimi_hub._vad_pre.value = (len(w16), probs)
        try:
            cs = [c for s in m["asr"].recognize(w16, sample_rate=16000) for c in mimi_hub.cues(s)]
        finally:
            mimi_hub._vad_pre.value = None
        return " ".join(t for _, _, t in cs).split(), len(cs)

    ref, ref_cues = words(pp.vad_sequential(sess, fr))
    again, _ = words(pp.vad_sequential(sess, fr))  # the same probabilities again: ASR's own run-to-run variation
    res = {"words": len(ref), "cues": ref_cues,
           "same_vad_again": round(difflib.SequenceMatcher(None, ref, again, autojunk=False).ratio(), 5)}
    for w in a.warm:
        got, n = words(pp.vad_threaded(sess, fr, a.shards, w))
        res[f"shards_{a.shards}_warm_{w}"] = {"cues": n, "word_similarity": round(
            difflib.SequenceMatcher(None, ref, got, autojunk=False).ratio(), 5)}
    print("VADSHARD " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
