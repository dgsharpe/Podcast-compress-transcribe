"""In-process test of mimi-hub with worker processes (MIMI_HUB_PROCS=2; no network listener): upload three 5 min
clips, seek near the end of one while it's still encoding (the wanted chunk is forwarded to its worker), stream it
whole as it encodes, and check what was streamed equals the finished files.

usage: python hub/test_procs.py      (in a GPU environment; a temporary MIMI_HUB_DATA)
"""

import io
import os
import sys
import tempfile
import time

os.environ.setdefault("MIMI_HUB_PROCS", "2")
os.environ.setdefault("MIMI_HUB_WORKERS", "1")
os.environ.setdefault("MIMI_HUB_SEGMENTER", "off")
os.environ.setdefault("MIMI_HUB_DATA", tempfile.mkdtemp(prefix="mimi-procs-"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, ".")


def main():
    import soundfile as sf
    from fastapi.testclient import TestClient
    from hub import mimi_hub

    wav, sr = sf.read("data/ref/ref_24k.wav", dtype="float32")
    clips = []
    for k in range(3):
        buf = io.BytesIO()
        sf.write(buf, wav[k * 300 * sr:(k + 1) * 300 * sr], sr, format="WAV")
        clips.append(buf.getvalue())
    with TestClient(mimi_hub.app) as c:
        t0 = time.time()
        ids = [c.post("/v1/upload", params={"title": f"clip {k}"},
                      files={"file": (f"clip{k}.wav", b, "audio/wav")}).json()["id"] for k, b in enumerate(clips)]
        print("uploaded", ids, flush=True)
        i = ids[0]
        with c.stream("GET", f"/v1/episodes/{i}.mimi", headers={"Range": "bytes=150000-150999"}) as resp:
            seek = b"".join(resp.iter_bytes())
            print(f"seek: {resp.status_code} {len(seek)} B after {time.time() - t0:.1f} s", flush=True)
        with c.stream("GET", f"/v1/episodes/{i}.mimi") as resp:
            whole = b"".join(resp.iter_bytes())
            print(f"stream: {resp.status_code} {len(whole)} B after {time.time() - t0:.1f} s", flush=True)
        states = {}
        while True:
            states = {j: c.get(f"/v1/episodes/{j}").json() for j in ids}
            if all(s["state"] in ("ready", "failed") for s in states.values()):
                break
            time.sleep(1)
        print(f"all done after {time.time() - t0:.1f} s:",
              {j: (s["state"], s.get("device"), s.get("speakers"), s.get("error")) for j, s in states.items()})
        final = c.get(f"/v1/episodes/{i}.mimi").content
        vtts = [c.get(f"/v1/episodes/{j}.vtt").text.count("-->") for j in ids]
        ok = (all(s["state"] == "ready" for s in states.values()) and whole == final
              and seek == final[150000:151000] and all(v > 50 for v in vtts))
        print("cues per clip", vtts, "| streamed == final:", whole == final, "| seek == final:",
              seek == final[150000:151000])
        print("PASS" if ok else "FAIL", flush=True)


if __name__ == "__main__":
    main()
