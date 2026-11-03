"""One episode through a running hub over HTTP (e.g. hub/modal_app.py on Modal), timed from a cold start:
POST /v1/warm until the models are ready, then a conversion as a client about to stream it.

    python scripts/remote_episode.py https://<server>/t/<token> [audio URL] [--no-warm]

Prints the warm-up time, then for the episode: time to the first audio bytes, to "ready" and to its ad segments,
and the record's stage timings. --no-warm: request the episode straight away (a cold start without warm-up).
With OPENROUTER_API_KEY in the environment, it's sent with the request (X-OpenRouter-Key) for the ad detection.
"""
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

if not os.environ.get("SSL_CERT_FILE") and os.path.exists("/etc/ssl/certs/ca-certificates.crt"):
    os.environ["SSL_CERT_FILE"] = "/etc/ssl/certs/ca-certificates.crt"
URL = ("https://www.podtrac.com/pts/redirect.mp3/pdst.fm/e/pscrb.fm/rss/p/mgln.ai/e/257/traffic.megaphone.fm/"
       "VMP7855915272.mp3")


def call(base, path, body=None, timeout=600):
    headers = {"Content-Type": "application/json"}
    if body is not None and os.environ.get("OPENROUTER_API_KEY"):
        headers["X-OpenRouter-Key"] = os.environ["OPENROUTER_API_KEY"]
    req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                 headers=headers,
                                 method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    base, audio = args[0].rstrip("/"), (args[1:] or [URL])[0]
    t0 = time.time()
    if "--no-warm" not in sys.argv:
        first, errors = None, 0
        while True:
            try:
                state = call(base, "/v1/warm", {})["models"]
            except urllib.error.HTTPError as e:  # e.g. the container died and the next one is starting
                errors += 1
                print(f"warm: HTTP {e.code} at {time.time() - t0:.0f} s", flush=True)
                if errors > 3:
                    raise
                time.sleep(5)
                continue
            first = first or round(time.time() - t0, 1)
            if state == "ready":
                break
            if state == "failed":
                raise SystemExit("warm-up failed (see the server's log)")
            time.sleep(2)
        print(f"WARM first answer {first or round(time.time() - t0, 1)} s, models ready {time.time() - t0:.1f} s",
              flush=True)
    t1 = time.time()
    i = f"check-{int(t1)}"
    r = call(base, "/v1/convert", {"url": audio, "id": i, "priority": 0})
    print(f"convert answered in {time.time() - t1:.1f} s: {r['state']}", flush=True)
    got = {}

    def first_audio():
        req = urllib.request.Request(f"{base}/v1/episodes/{i}.mimi", headers={"Range": "bytes=0-2047"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            resp.read()
        got["s"] = round(time.time() - t1, 1)
    th = threading.Thread(target=first_audio, daemon=True)
    th.start()
    ready = None
    while True:
        r = call(base, f"/v1/episodes/{i}")
        if r["state"] == "ready" and ready is None:
            ready = round(time.time() - t1, 1)
        if r["state"] == "failed" or (r["state"] == "ready" and r.get("segments_state") in ("ready", "failed", "off")):
            break
        time.sleep(0.5)
    th.join(timeout=60)
    print("EPISODE " + json.dumps({
        "state": r["state"], "minutes": round(r.get("duration_s", 0) / 60, 1), "first_audio_s": got.get("s"),
        "ready_s": ready, "segments_s": round(time.time() - t1, 1), "segments": r.get("segments"),
        "segments_state": r.get("segments_state"), "segments_error": r.get("segments_error"),
        "diarization": r.get("diarization"), "timing": r.get("timing"), "error": r.get("error"),
        "total_from_start_s": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
