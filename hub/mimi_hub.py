"""mimi-hub: on-demand podcast conversion on the server -> `.mimi` audio + WebVTT transcript.

Serve on the Tailscale address only, e.g.
    uv run uvicorn hub.mimi_hub:app --host <tailscale-ip> --port 8765
Optional auth: set MIMI_HUB_TOKEN; clients then send `Authorization: Bearer <token>`.

API (JSON unless noted):
  POST /v1/convert {"url": ...[, "id", "title"]}  enqueue. The URL may be a direct audio URL or a Pocket Casts
                                                share link (pca.st / pocketcasts.com), which is resolved to the
                                                episode's enclosure; the id is then the Pocket Casts episode UUID.
  POST /v1/upload?title=...  (multipart "file")  enqueue an uploaded audio file
  GET  /v1/episodes/{id}                        status: queued | downloading | encoding | transcribing | ready | failed
  GET  /v1/episodes/{id}.mimi                   the audio (Range requests supported)
  GET  /v1/episodes/{id}.vtt                    the transcript
  GET  /v1/episodes                             all jobs, newest first
Files live in $MIMI_HUB_DATA (default neural_audio/hub_data/): <id>.mimi, <id>.vtt, <id>.json.
"""

import gzip
import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import mimi_file  # noqa: E402

# uv's Python doesn't know NixOS's CA bundle; systemd-run services don't inherit SSL_CERT_FILE.
if not os.environ.get("SSL_CERT_FILE") and os.path.exists("/etc/ssl/certs/ca-certificates.crt"):
    os.environ["SSL_CERT_FILE"] = "/etc/ssl/certs/ca-certificates.crt"

DATA = Path(os.environ.get("MIMI_HUB_DATA", ROOT / "hub_data"))
TOKEN = os.environ.get("MIMI_HUB_TOKEN", "")
ASR_MODEL = os.environ.get("MIMI_HUB_ASR", "nemo-parakeet-tdt-0.6b-v2")  # -v3 for multilingual
THREADS = int(os.environ.get("MIMI_HUB_THREADS", "16"))
MAX_BYTES = 1 << 30
UA = {"User-Agent": "Mozilla/5.0 (mimi-hub)"}
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
PC_EPISODE = re.compile(r"pocketcasts\.com/podcast/[^/]+/([0-9a-f-]{36})/[^/]+/([0-9a-f-]{36})")

DATA.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="mimi-hub")
jobs: "queue.Queue[str]" = queue.Queue()
lock = threading.Lock()


# ---------- job records ----------

def rec_path(i):
    return DATA / f"{i}.json"


def load(i):
    p = rec_path(i)
    return json.loads(p.read_text()) if p.exists() else None


def save(r):
    r["updated"] = time.time()
    tmp = rec_path(r["id"]).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(r, indent=1))
    tmp.replace(rec_path(r["id"]))
    return r


def update(i, **kw):
    with lock:
        r = load(i)
        r.update(kw)
        return save(r)


def enqueue(i, **fields):
    with lock:
        r = load(i)
        if r and r["state"] != "failed":
            return r  # already queued, running or done
        r = save(dict(id=i, state="queued", progress=0.0, error=None, created=time.time(), **fields))
    jobs.put(i)
    return r


# ---------- URL resolution ----------

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def location(url):
    """The Location header of one GET to `url` (None if it does not redirect)."""
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        opener.open(urllib.request.Request(url, headers=UA), timeout=30)
        return None
    except urllib.error.HTTPError as e:
        return e.headers.get("Location") if 300 <= e.code < 400 else None


def resolve(url):
    """Returns (id, audio_url, title). Pocket Casts links -> episode UUID + enclosure."""
    if "pca.st/" in url or "pocketcasts.com/" in url:
        final = url
        for _ in range(5):  # follow short-link redirects by hand: pocketcasts.com 403s non-browsers
            if PC_EPISODE.search(final):
                break
            final = location(final) or final
        m = PC_EPISODE.search(final)
        if not m:
            raise ValueError(f"not a Pocket Casts episode link: {final}")
        pod, ep = m.groups()
        req = urllib.request.Request(f"https://podcast-api.pocketcasts.com/podcast/full/{pod}", headers=UA)
        raw = urllib.request.urlopen(req, timeout=60).read()
        data = json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
        for e in data["podcast"]["episodes"]:
            if e["uuid"] == ep:
                return ep, e["url"], f'{data["podcast"].get("title", "")}: {e["title"]}'
        raise ValueError(f"episode {ep} not in podcast {pod}")
    return hashlib.sha1(url.encode()).hexdigest()[:16], url, ""


# ---------- worker ----------

_models = {}


def models():
    if not _models:
        import onnx_asr
        import torch
        torch.set_num_threads(THREADS)
        _models["mimi"] = mimi_file.load_model()
        _models["asr"] = (onnx_asr.load_model(ASR_MODEL).with_vad(onnx_asr.load_vad("silero"))
                          .with_timestamps())
    return _models


def candidates(url):
    """`url`, then the same file fetched from each host embedded in a tracking-prefix chain
    (podtrac/pdst/pscrb/... URLs end in `.../real.host/path`), innermost host first."""
    yield url
    parts = url.split("://", 1)[-1].split("/")
    for k in range(len(parts) - 2, 0, -1):
        if "." in parts[k] and not re.search(r"\.(mp3|m4a|aac|ogg|opus|wav)$", parts[k], re.I):
            yield "https://" + "/".join(parts[k:])


def download(url, dst):
    err = None
    for u in candidates(url):
        try:
            return _download(u, dst)
        except (urllib.error.URLError, OSError) as e:
            err = e
    raise err


def _download(url, dst):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r, open(dst, "wb") as f:
        n = 0
        while chunk := r.read(1 << 20):
            n += len(chunk)
            if n > MAX_BYTES:
                raise ValueError("source larger than 1 GiB")
            f.write(chunk)


def load_audio(path, sr):
    """Mono float32 at `sr`. ffmpeg if available (any container), else soundfile (mp3/wav/flac/ogg)."""
    import numpy as np
    ff = shutil.which("ffmpeg")
    if ff:
        out = subprocess.run([ff, "-v", "error", "-i", str(path), "-map", "0:a:0", "-ac", "1", "-ar",
                              str(sr), "-f", "f32le", "-"], capture_output=True, check=True).stdout
        return np.frombuffer(out, np.float32).copy()
    import librosa
    return librosa.load(path, sr=sr, mono=True)[0]


def cues(seg, max_s=7.0, max_chars=110):
    """Split one timestamped ASR segment into sentence-sized (start, end, text) cues.

    Token times are relative to the segment start; a token with a leading space starts a word. A cue
    ends after sentence punctuation, or (when it gets long) at a comma or word boundary.
    """
    toks, ts = seg.tokens, [seg.start + t for t in seg.timestamps]
    out, cur, t0 = [], "", None
    for k, (tok, t) in enumerate(zip(toks, ts)):
        if t0 is None:
            t0 = t
        cur += tok
        nxt = toks[k + 1] if k + 1 < len(toks) else None
        at_word_end = nxt is None or nxt.startswith(" ")
        long = ts[k] - t0 >= max_s or len(cur) >= max_chars
        sentence = tok.rstrip().endswith((".", "?", "!")) and ts[k] - t0 >= 1.0
        if nxt is None or (at_word_end and (sentence or (long and (tok.endswith(",") or len(cur) >= max_chars)))):
            end = ts[k + 1] if nxt is not None else seg.end
            out.append((t0, max(end, t0 + 0.2), cur.strip()))
            cur, t0 = "", None
    return [c for c in out if c[2]]


def vtt_time(t):
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def run(i):
    r = load(i)
    work = Path(tempfile.mkdtemp(prefix=f"mimi-{i}-"))
    try:
        src = work / "source"
        if r.get("upload"):
            shutil.move(r["upload"], src)
        else:
            update(i, state="downloading")
            download(r["audio_url"], src)
        m = models()
        update(i, state="encoding", progress=0.0)
        wav24 = load_audio(src, mimi_file.SR)
        codes = mimi_file.encode_array(m["mimi"], wav24, progress=lambda p: update(i, progress=round(p, 3)))
        meta = dict(title=r.get("title", ""), source=r.get("url", ""), id=i,
                    encoder="transformers MimiModel", duration_s=len(wav24) / mimi_file.SR)
        mimi_file.write(DATA / f"{i}.mimi.tmp", codes, 32, meta)
        (DATA / f"{i}.mimi.tmp").replace(DATA / f"{i}.mimi")
        update(i, state="transcribing", progress=0.0, duration_s=meta["duration_s"],
               mimi_bytes=(DATA / f"{i}.mimi").stat().st_size)
        del wav24
        wav16 = load_audio(src, 16000)
        with open(DATA / f"{i}.vtt.tmp", "w") as f:
            f.write("WEBVTT\n\n")
            for s in m["asr"].recognize(wav16, sample_rate=16000):
                for a, b, text in cues(s):
                    f.write(f"{vtt_time(a)} --> {vtt_time(b)}\n{text}\n\n")
        (DATA / f"{i}.vtt.tmp").replace(DATA / f"{i}.vtt")
        update(i, state="ready", progress=1.0, asr=ASR_MODEL)
    except Exception as e:
        traceback.print_exc()
        update(i, state="failed", error=f"{type(e).__name__}: {e}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def worker():
    while True:
        i = jobs.get()
        try:
            run(i)
        finally:
            jobs.task_done()


@app.on_event("startup")
def start():
    for p in sorted(DATA.glob("*.json"), key=lambda p: p.stat().st_mtime):
        r = json.loads(p.read_text())
        if r["state"] not in ("ready", "failed"):  # resume interrupted jobs
            save(dict(r, state="queued", progress=0.0))
            jobs.put(r["id"])
    threading.Thread(target=worker, daemon=True, name="mimi-worker").start()


# ---------- HTTP ----------

def auth(authorization):
    if TOKEN and authorization != f"Bearer {TOKEN}":
        raise HTTPException(401, "bad token")


def check_id(i):
    if not ID_RE.match(i):
        raise HTTPException(400, "bad id")


class ConvertRequest(BaseModel):
    url: str
    id: str | None = None
    title: str | None = None


@app.post("/v1/convert", status_code=202)
def convert(req: ConvertRequest, authorization: str | None = Header(None)):
    auth(authorization)
    try:
        i, audio, title = resolve(req.url.strip())
    except Exception as e:
        raise HTTPException(422, f"cannot resolve URL: {e}")
    if req.id:
        check_id(req.id)
        i = req.id
    return enqueue(i, url=req.url, audio_url=audio, title=req.title or title)


@app.post("/v1/upload", status_code=202)
def upload(file: UploadFile = File(...), title: str = "", authorization: str | None = Header(None)):
    auth(authorization)
    h = hashlib.sha1()
    fd, tmp = tempfile.mkstemp(prefix="mimi-upload-", dir=DATA)
    with os.fdopen(fd, "wb") as f:
        while chunk := file.file.read(1 << 20):
            h.update(chunk)
            f.write(chunk)
    i = "u" + h.hexdigest()[:15]
    r = enqueue(i, url="upload:" + (file.filename or ""), upload=tmp, title=title or file.filename or i)
    if r.get("upload") != tmp:
        os.unlink(tmp)  # duplicate of an existing job
    return r


@app.get("/v1/episodes")
def list_jobs(authorization: str | None = Header(None)):
    auth(authorization)
    rs = [json.loads(p.read_text()) for p in DATA.glob("*.json")]
    return sorted(rs, key=lambda r: -r.get("created", 0))


@app.get("/v1/episodes/{name}")
def episode(name: str, authorization: str | None = Header(None)):
    auth(authorization)
    stem, _, ext = name.partition(".")
    check_id(stem)
    if ext in ("mimi", "vtt"):
        p = DATA / f"{stem}.{ext}"
        if not p.exists():
            raise HTTPException(404, "not ready")
        return FileResponse(p, media_type="audio/x-mimi" if ext == "mimi" else "text/vtt",
                            filename=f"{stem}.{ext}")
    r = load(stem)
    if r is None:
        raise HTTPException(404, "unknown id")
    return r
