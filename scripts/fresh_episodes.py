"""Transcripts of new episodes for checking ad detection on shows it wasn't tuned on: each feed's latest episode of
20-100 min, converted by the hub's code in-process (its own data directory, MIMI_HUB_DATA; no ad detection), so
they don't join the running hub's library.

    MIMI_HUB_DATA=data/fresh_hub python scripts/fresh_episodes.py feeds.txt     (lines: name|feed url)
"""
import json
import os
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ["MIMI_HUB_SEGMENTER"] = "off"
NS = {"itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd"}


def minutes(s):
    if not s:
        return None
    sec = 0.0
    for x in s.strip().split(":"):
        sec = sec * 60 + float(x)
    return sec / 60 if ":" in s else float(s) / 60


def latest(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (mimi-hub)"})
    root = ET.fromstring(urllib.request.urlopen(req, timeout=30).read())
    for it in root.findall("./channel/item")[:15]:
        enc, d = it.find("enclosure"), minutes(it.findtext("itunes:duration", default="", namespaces=NS))
        if enc is not None and d and 20 <= d <= 100:
            return it.findtext("title"), enc.get("url")
    return None


def main():
    from fastapi.testclient import TestClient
    from hub import mimi_hub
    feeds = [ln.strip().split("|", 1) for ln in open(sys.argv[1]) if ln.strip()]
    with TestClient(mimi_hub.app) as c:
        for show, url in feeds:
            try:
                got = latest(url)
            except Exception as e:  # noqa: BLE001
                print(f"FRESH {json.dumps({'show': show, 'error': f'{type(e).__name__}: {e}'[:200]})}", flush=True)
                continue
            if not got:
                continue
            title, audio = got
            t = time.time()
            i = c.post("/v1/convert", json={"url": audio, "title": f"{show}: {title}", "priority": 0}).json()["id"]
            while (r := c.get(f"/v1/episodes/{i}").json())["state"] not in ("ready", "failed"):
                time.sleep(1)
            print("FRESH " + json.dumps({"show": show, "title": title[:80], "id": i, "state": r["state"],
                                         "minutes": round(r.get("duration_s", 0) / 60, 1),
                                         "wall_s": round(time.time() - t), "error": r.get("error")}), flush=True)


if __name__ == "__main__":
    main()
