"""One episode at a time through the in-process hub (no network listener), models warm: where the wall time goes.

    python scripts/episode_timeline.py [audio URL]      (ffmpeg on PATH; the logged-in agent CLIs for ad detection)

For a real episode URL (download + mp3 decode) and the 43.8 min reference as an upload: the record's stage times
(seconds from the job's start: probed, decoded, asr, diar, mimi, ready, segments), the time to the first playable
audio (the first .mimi chunk), and the wall time from the request.
"""
import io
import json
import os
import sys
import tempfile
import threading
import time

os.environ.setdefault("MIMI_HUB_DATA", tempfile.mkdtemp(prefix="mimi-timeline-"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, ".")

URL = ("https://www.podtrac.com/pts/redirect.mp3/pdst.fm/e/pscrb.fm/rss/p/mgln.ai/e/257/traffic.megaphone.fm/"
       "VMP7855915272.mp3")


def main():
    import numpy as np
    import soundfile as sf
    from fastapi.testclient import TestClient
    from hub import mimi_hub

    url = sys.argv[1] if len(sys.argv) > 1 else URL
    wav, sr = sf.read("data/ref/ref_24k.wav", dtype="float32")
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    with TestClient(mimi_hub.app) as c:
        m = mimi_hub.models()  # warm, as in service: models loaded, kernels picked
        w16 = np.ascontiguousarray(wav[: 60 * sr:3])  # ~16 kHz
        list(m["asr"].recognize(w16, sample_rate=16000))
        mimi_hub.diarize(w16)
        for label in ("url", "upload"):
            t0 = time.time()
            if label == "url":
                i = c.post("/v1/convert", json={"url": url, "id": "timeline", "priority": 0}).json()["id"]
            else:
                i = c.post("/v1/upload", params={"title": "reference"},
                           files={"file": ("ref.wav", buf.getvalue(), "audio/wav")}).json()["id"]
            first = {}

            def first_audio():
                with c.stream("GET", f"/v1/episodes/{i}.mimi", headers={"Range": "bytes=0-2047"}) as resp:
                    b"".join(resp.iter_bytes())
                first["s"] = round(time.time() - t0, 2)
            th = threading.Thread(target=first_audio)
            th.start()
            while True:
                r = c.get(f"/v1/episodes/{i}").json()
                if r["state"] == "failed" or (r["state"] == "ready" and r.get("segments_state") in
                                              ("ready", "failed", "off")):
                    break
                time.sleep(0.25)
            th.join()
            res = {"label": label, "state": r["state"], "duration_min": round(r.get("duration_s", 0) / 60, 1),
                   "wall_s": round(time.time() - t0, 1), "first_audio_s": first.get("s"),
                   "queue_s": round(r.get("timing_start", t0) - t0, 2), "timing": r.get("timing"),
                   "segments": r.get("segments"), "segments_model": r.get("segments_model"), "error": r.get("error")}
            print("TIMELINE " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
