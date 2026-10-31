"""In-process test of mimi-hub (no network listener): resolve a Pocket Casts link, convert an
uploaded 90 s clip end to end, then fetch the .mimi (incl. a Range request) and the .vtt.

usage: MIMI_HUB_DATA=/tmp/x uv run python hub/test_hub.py [pca.st link]
"""

import io
import sys
import time

import soundfile as sf
from fastapi.testclient import TestClient

sys.path.insert(0, ".")
from hub import mimi_hub  # noqa: E402

link = sys.argv[1] if len(sys.argv) > 1 else "https://pca.st/episode/74146999-c265-4a89-9ebf-9ba3112140d7"
print("resolve:", mimi_hub.resolve(link))

wav, sr = sf.read("data/ref/ref_24k.wav", dtype="float32", start=600 * 24000, stop=690 * 24000)
buf = io.BytesIO()
sf.write(buf, wav, sr, format="WAV")

with TestClient(mimi_hub.app) as c:
    r = c.post("/v1/upload", params={"title": "test clip"},
               files={"file": ("clip.wav", buf.getvalue(), "audio/wav")}).json()
    i = r["id"]
    print("enqueued", r["id"], r["state"])
    t0 = time.time()
    while True:
        s = c.get(f"/v1/episodes/{i}").json()
        if s["state"] in ("ready", "failed"):
            break
        time.sleep(2)
    print(f"{s['state']} after {time.time() - t0:.0f} s:", {k: s.get(k) for k in ("error", "mimi_bytes", "duration_s", "asr")})
    m = c.get(f"/v1/episodes/{i}.mimi")
    part = c.get(f"/v1/episodes/{i}.mimi", headers={"Range": "bytes=100-199"})
    print("mimi", m.status_code, len(m.content), m.headers.get("content-type"),
          "| range", part.status_code, len(part.content), part.content == m.content[100:200])
    v = c.get(f"/v1/episodes/{i}.vtt")
    print("vtt", v.status_code, v.text[:400])
    print("list", [(x["id"], x["state"]) for x in c.get("/v1/episodes").json()])
    print("dup upload ->", c.post("/v1/upload", files={"file": ("clip.wav", buf.getvalue(), "audio/wav")}).json()["state"])
