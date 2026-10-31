"""In-process test of progressive conversion: request a stream of a real episode, then
  1. seek: a Range request near the end while encoding is still at the start (should be served next),
  2. stream the whole .mimi from byte 0 as it is being encoded,
and check the streamed bytes equal the finished file. Run with ffmpeg on PATH:
  MIMI_HUB_DATA=/tmp/x nix shell nixpkgs#ffmpeg -c uv run python hub/test_stream.py [audio URL]
"""

import sys
import time

from fastapi.testclient import TestClient

sys.path.insert(0, ".")
from hub import mimi_hub  # noqa: E402

url = sys.argv[1] if len(sys.argv) > 1 else (
    "https://www.podtrac.com/pts/redirect.mp3/pdst.fm/e/pscrb.fm/rss/p/mgln.ai/e/257/traffic.megaphone.fm/"
    "VMP7855915272.mp3")
I = "streamtest"

with TestClient(mimi_hub.app) as c:
    mimi_hub.models()  # load models first so timings show the pipeline, not model loading
    t0 = time.time()
    r = c.post("/v1/convert", json={"url": url, "id": I, "priority": 0}).json()
    print("convert ->", r["state"])

    # 1. seek near the end (~40 min) as soon as the header exists
    with c.stream("GET", f"/v1/episodes/{I}.mimi", headers={"Range": "bytes=1300000-1300999"}) as resp:
        t_hdr = time.time() - t0
        body = b"".join(resp.iter_bytes())
    print(f"seek @40 min: HTTP {resp.status_code} {resp.headers.get('content-range')} "
          f"{len(body)} bytes; request accepted {t_hdr:.1f} s, bytes after {time.time() - t0:.1f} s")
    s = c.get(f"/v1/episodes/{I}").json()
    print("state then:", s["state"], "progress", s["progress"])

    # 2. full stream from 0
    t1 = time.time()
    got, marks = bytearray(), {}
    with c.stream("GET", f"/v1/episodes/{I}.mimi") as resp:
        total = int(resp.headers["content-length"])
        for chunk in resp.iter_bytes():
            got += chunk
            for name, n in (("header", 64), ("first 10 s", 64 + 256 + 125 * 44), ("first 70 s", 64 + 256 + 875 * 44)):
                if name not in marks and len(got) >= n:
                    marks[name] = time.time() - t1
    print(f"full stream: {len(got)}/{total} bytes in {time.time() - t1:.1f} s;",
          ", ".join(f"{k} after {v:.2f} s" for k, v in marks.items()))

    while (s := c.get(f"/v1/episodes/{I}").json())["state"] not in ("ready", "failed"):
        time.sleep(2)
    disk = open(mimi_hub.DATA / f"{I}.mimi", "rb").read()
    print("final:", s["state"], s.get("error"), "| streamed == file:", bytes(got) == disk,
          "| seek slice == file:", body == disk[1300000:1301000], f"| all done after {time.time() - t0:.0f} s")
    v = c.get(f"/v1/episodes/{I}.vtt").text
    print(v[:300])
