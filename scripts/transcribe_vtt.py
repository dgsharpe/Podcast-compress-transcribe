"""Full-episode WebVTT transcript from the original audio (the sidecar a feed proxy would serve
via <podcast:transcript type="text/vtt">). Cues are Whisper segments; word timestamps on.

usage: python scripts/transcribe_vtt.py [in.wav] [out.vtt] [--model large-v3-turbo]
"""

import argparse
import time
from pathlib import Path

import soundfile as sf
from faster_whisper import WhisperModel

ROOT = Path(__file__).resolve().parent.parent


def ts(t):
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp", nargs="?", default=str(ROOT / "data/ref/ref_16k.wav"))
    ap.add_argument("out", nargs="?", default=str(ROOT / "outputs/transcript/episode.vtt"))
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    m = WhisperModel(a.model, device="cpu", compute_type="int8", cpu_threads=a.threads)
    t0 = time.perf_counter()
    audio, sr = sf.read(a.inp, dtype="float32")  # faster-whisper's PyAV loader breaks on new PyAV
    assert sr == 16000 and audio.ndim == 1, (sr, audio.shape)
    segs, info = m.transcribe(audio, language="en", beam_size=5, vad_filter=True,
                              word_timestamps=True, condition_on_previous_text=True)
    with open(a.out, "w") as f:
        f.write("WEBVTT\n\n")
        for s in segs:
            f.write(f"{ts(s.start)} --> {ts(s.end)}\n{s.text.strip()}\n\n")
    el = time.perf_counter() - t0
    print(f"wrote {a.out}: {info.duration:.0f} s audio in {el:.0f} s (RTF {el / info.duration:.3f})")


if __name__ == "__main__":
    main()
