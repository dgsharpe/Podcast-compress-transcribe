"""mimi-hub: on-demand podcast conversion on the server -> `.mimi` audio + WebVTT transcript.

Serve on the Tailscale address only, e.g.
    uv run uvicorn hub.mimi_hub:app --host <tailscale-ip> --port 8765
Optional auth: set MIMI_HUB_TOKEN; clients then send `Authorization: Bearer <token>`.

API (JSON unless noted):
  POST /v1/convert {"url": ...[, "id", "title", "priority"]}  enqueue (priority 0 = about to be streamed,
                                                jumps the queue). The URL may be a direct audio URL or a Pocket
                                                Casts share link (pca.st / pocketcasts.com), which is resolved to
                                                the episode's enclosure; the id is then the Pocket Casts episode UUID.
  POST /v1/upload?title=...  (multipart "file")  enqueue an uploaded audio file
  GET  /v1/episodes/{id}                        status: queued | starting | downloading | encoding | transcribing
                                                | ready | failed
  GET  /v1/episodes/{id}.mimi                   the audio, Range requests supported. Available from the moment the
                                                job is "encoding": bytes are served as their chunk is encoded, so a
                                                client can play (and seek in) a .mimi that is still being made.
  GET  /v1/episodes/{id}.vtt                    the transcript
  GET  /v1/episodes/{id}.segments.json          ads / self-promotion to skip, found by an LLM after transcription
                                                ({"model", "segments": [{"start", "end", "kind": "ad" | "promo",
                                                "label"}]}, seconds); the record's "segments_state" says
                                                pending | ready | failed | off. See hub/segments.py.
  POST /v1/episodes/{id}/segments               re-run that detection
  GET  /v1/episodes                             all jobs, newest first
Files live in $MIMI_HUB_DATA (default neural_audio/hub_data/): <id>.mimi, <id>.vtt, <id>.segments.json, <id>.json.

GPU: run in .venv-rocm (hub/setup_rocm.sh; hub/serve.sh picks it) and the Mimi encoder and Parakeet's encoder run on
the Radeon 8060S (MIMI_HUB_DEVICE=auto | cuda | cpu). Diarization and Parakeet's decoding loop stay on the CPU.
"""

import gzip
import hashlib
import itertools
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
import types
import urllib.error
import urllib.request
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import mimi_file  # noqa: E402

from hub import segments  # noqa: E402

# uv's Python doesn't know NixOS's CA bundle; systemd-run services don't inherit SSL_CERT_FILE.
if not os.environ.get("SSL_CERT_FILE") and os.path.exists("/etc/ssl/certs/ca-certificates.crt"):
    os.environ["SSL_CERT_FILE"] = "/etc/ssl/certs/ca-certificates.crt"

# MIOpen (ROCm's conv library) otherwise tunes every new input shape exhaustively; Parakeet's batches all differ
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

DATA = Path(os.environ.get("MIMI_HUB_DATA", ROOT / "hub_data"))
DEVICE = os.environ.get("MIMI_HUB_DEVICE", "auto")
TOKEN = os.environ.get("MIMI_HUB_TOKEN", "")
ASR_MODEL = os.environ.get("MIMI_HUB_ASR", "nemo-parakeet-tdt-0.6b-v2")  # -v3 for multilingual
THREADS = int(os.environ.get("MIMI_HUB_THREADS", "16"))
UA = {"User-Agent": "Mozilla/5.0 (mimi-hub)"}
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
PC_EPISODE = re.compile(r"pocketcasts\.com/podcast/[^/]+/([0-9a-f-]{36})/[^/]+/([0-9a-f-]{36})")

DATA.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="mimi-hub")
jobs: "queue.PriorityQueue" = queue.PriorityQueue()  # (priority, seq, id); 0 = streaming
lock = threading.Lock()


# ---------- job records ----------

def rec_path(i):
    return DATA / f"{i}.json"


def record_paths():
    return [p for p in DATA.glob("*.json") if not p.name.endswith(".segments.json")]


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


def enqueue(i, priority=1, **fields):
    """Queue job i (priority 0 = someone is streaming it, 1 = download / background)."""
    with lock:
        r = load(i)
        if r and r["state"] != "failed":
            if r["state"] == "queued" and priority < r.get("priority", 1):  # bump: re-queue at higher priority
                r = save(dict(r, priority=priority))
                jobs.put((priority, next(_seq), i))
            return r  # already queued, running or done
        r = save(dict(id=i, state="queued", priority=priority, progress=0.0, error=None, created=time.time(),
                      **fields))
    jobs.put((priority, next(_seq), i))
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
        import onnxruntime as ort
        import torch
        torch.set_num_threads(THREADS)
        dev = DEVICE if DEVICE != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        _models["device"] = dev
        _models["mimi"] = mimi_file.load_model(dev)
        # onnxruntime's default spin-waiting threads starve torch's: on a 15 min clip (scripts/bench_overlap.py)
        # ASR alone went 50 -> 31 s with spinning off, and encode || ASR 87 -> 39 s
        so = ort.SessionOptions()
        so.intra_op_num_threads = THREADS
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        asr = onnx_asr.load_model(ASR_MODEL, sess_options=so)
        if dev == "cuda":
            try:
                asr.asr._encoder = TorchEncoder(parakeet_encoder())
            except Exception:
                traceback.print_exc()  # the ONNX encoder on the CPU still works
        vad = onnx_asr.load_vad("silero")
        if ASR_BUCKET > 1:
            vad.recognize_batch = types.MethodType(_bucketed_recognize_batch, vad)
        _models["asr"] = asr.with_vad(vad).with_timestamps()
    return _models


# onnx_asr sends VAD segments to the ASR model 8 at a time in time order, each batch padded to its longest segment:
# 61% of the encoder's input was padding on a 43.8 min episode. Batching all of an episode's segments by length
# (ASR_BUCKET at a time) cut the encoder from 14.3 to 8.3 s on the server (scripts/parakeet_encoder_server.py).
ASR_BUCKET = int(os.environ.get("MIMI_HUB_ASR_BUCKET", "16"))  # 0 / 1: onnx_asr's own batching


def _bucketed_recognize_batch(self, asr, waveforms, waveforms_len, sample_rate, asr_kwargs, batch_size=8,
                              **kwargs):
    """Drop-in for onnx_asr BaseVad.recognize_batch: segment each waveform fully, recognize the segments in
    length-sorted batches of ASR_BUCKET, and yield the results in time order."""
    from onnx_asr.utils import pad_list
    from onnx_asr.vad import TimestampedSegmentResult

    def recognize(waveform, segments):
        segments = list(segments)
        order = sorted(range(len(segments)), key=lambda i: segments[i][1] - segments[i][0])
        results = [None] * len(segments)
        for i in range(0, len(order), ASR_BUCKET):
            idx = order[i:i + ASR_BUCKET]
            batch = [segments[j] for j in idx]
            for res, j in zip(asr.recognize_batch(*pad_list([waveform[s:e] for s, e in batch]), **asr_kwargs), idx,
                              strict=True):
                s, e = segments[j]
                results[j] = TimestampedSegmentResult(s / sample_rate, e / sample_rate, res.text, res.timestamps,
                                                      res.tokens, res.logprobs)
        return iter(results)

    return map(recognize, waveforms, self.segment_batch(waveforms, waveforms_len, sample_rate, **kwargs))


# ---------- Parakeet's encoder on the GPU ----------
#
# onnxruntime has no working GPU provider for gfx1151 here (no MIGraphX; WebGPU was slower than the CPU and changed
# the output), so the encoder ONNX graph is converted to torch once (onnx2torch) and run on ROCm in fp16, behind the
# InferenceSession interface onnx_asr calls. The encoder is ~85% of Parakeet's CPU time; preprocessing, VAD and the
# TDT decoding loop stay in onnxruntime on the CPU. On a 15 min clip: 33 s -> 10 s, 190/191 segments identical (one
# comma differs). bf16 changed more words; fp32 is identical but half the speed.

PARAKEET_ONNX = {"nemo-parakeet-tdt-0.6b-v2": "istupakov/parakeet-tdt-0.6b-v2-onnx"}


def parakeet_encoder():
    """Path of the torch-converted encoder of ASR_MODEL, converting it on first use (~2.4 GB, a few seconds)."""
    out = MODELS_DIR / f"{ASR_MODEL}-encoder.pt"
    if out.exists():
        return out
    import onnx2torch
    import torch
    from huggingface_hub import snapshot_download
    src = Path(snapshot_download(PARAKEET_ONNX[ASR_MODEL], allow_patterns=["encoder-model.onnx*"]))
    with tempfile.TemporaryDirectory(dir=MODELS_DIR) as tmp:
        for f in src.glob("encoder-model.onnx*"):  # onnx refuses the HF cache's symlinked weight file
            shutil.copyfile(f, Path(tmp) / f.name)
        module = onnx2torch.convert(str(Path(tmp) / "encoder-model.onnx"))
    torch.save(module, out.with_suffix(".tmp"))
    out.with_suffix(".tmp").replace(out)
    return out


class TorchEncoder:
    """Stands in for onnx_asr's encoder InferenceSession: run(names, feeds) -> [outputs, encoded_lengths]."""

    def __init__(self, path):
        import torch
        self.torch = torch
        self.m = torch.load(path, weights_only=False).eval().to("cuda")

    def run(self, names, feeds):
        import numpy as np
        torch = self.torch
        x = torch.from_numpy(feeds["audio_signal"]).cuda()
        n = torch.from_numpy(np.asarray(feeds["length"])).cuda()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            out, lens = self.m(x, n)
        return [out.float().cpu().numpy(), lens.cpu().numpy()]


def candidates(url):
    """Ways to fetch `url`'s file: for a tracking-prefix chain (podtrac/pdst/pscrb/... URLs end in
    `.../real.host/path`) the real host first (fast, and skips blocked tracker hops), then the outer
    hosts, then the original URL; any other URL is tried as is first."""
    parts = url.split("://", 1)[-1].split("/")
    inner = ["https://" + "/".join(parts[k:]) for k in range(len(parts) - 2, 0, -1)
             if "." in parts[k] and not re.search(r"\.(mp3|m4a|aac|ogg|opus|wav)$", parts[k], re.I)]
    yield from (inner + [url]) if inner else [url]


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


# ---------- speaker diarization (sherpa-onnx: pyannote segmentation-3.0 + WeSpeaker ResNet34 + clustering) ----------
#
# Tuned on a 44-min two-host episode with ad reads. 0.7 looked right on one decode of it (hosts 18.6 / 15.9 min)
# but merged the two hosts on another (different dynamic ads; 24 kHz -> 16 kHz resampling), so clustering near 0.7
# is unstable. 0.6 kept the hosts apart on every version tried (~19 / ~16 min) and only adds small ad/news voices;
# over-splitting is the lesser evil. 0.9 merged hosts; TitaNet over-split. ~115 s per 44 min on 8 threads. Speakers
# are numbered by first appearance; ones with < 30 s in total are folded into their neighbours.

# pyannote's speaker-diarization-community-1 (VBx clustering) splits the hosts the same way (19.3 / 16.6 min on the
# same episode). Stock, it took 931 s on the CPU and 142 s on the GPU; with hub/pyannote_gpu.py's batched embedding
# step (identical output: DER 0.0000 against stock) it takes 33 s on the GPU, so it's the default there
# (MIMI_HUB_DIARIZER=auto; or pyannote / sherpa). Gated model: fetch it once into the HF cache with a token
# (a Hugging Face token); the hub then loads it from the cache, offline, with no token.
MODELS_DIR = Path(os.environ.get("MIMI_HUB_MODELS", ROOT / "hub_models"))
DIARIZER = os.environ.get("MIMI_HUB_DIARIZER", "auto")
PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"
SHERPA_NAME = "pyannote-seg-3.0 + wespeaker-resnet34 @0.6 (sherpa-onnx)"
DIAR_THRESHOLD, DIAR_MIN_TALK_S = 0.6, 30.0
_diar_lock = threading.Lock()


def _pyannote_turns(wav16):
    import torch
    from hub import pyannote_gpu
    if "diar_pyannote" not in _models:  # from the HF cache; no token at runtime
        _models["diar_pyannote"] = pyannote_gpu.load(_models.get("device", "cpu"),
                                                     segmentation_device=os.environ.get("MIMI_HUB_SEG_DEVICE") or None)
    out = _models["diar_pyannote"]({"waveform": torch.from_numpy(wav16)[None], "sample_rate": 16000})
    ann = getattr(out, "speaker_diarization", out)
    return [(t.start, t.end, spk) for t, _, spk in ann.itertracks(yield_label=True)]


_diar_pool = None


def _sherpa_turns(wav16):
    """sherpa-onnx diarization in a child process (hub/diarize_worker.py), which keeps its models loaded."""
    global _diar_pool
    import multiprocessing
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool
    from hub import diarize_worker
    if _diar_pool is None:
        _diar_pool = ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"))
    with tempfile.NamedTemporaryFile(suffix=".npy", dir=DATA) as f:
        np.save(f, wav16)
        f.flush()
        try:
            return _diar_pool.submit(diarize_worker.segments, f.name, str(MODELS_DIR), DIAR_THRESHOLD).result()
        except BrokenProcessPool:
            _diar_pool = None  # the child died (e.g. OOM); start a fresh one next time
            raise


def diarize(wav16):
    """([(start_s, end_s, speaker_number)], diarizer name), speakers numbered 1.. by first appearance."""
    with _diar_lock:  # one diarizer, not shared across threads
        segs = None
        if DIARIZER == "pyannote" or (DIARIZER == "auto" and _models.get("device") == "cuda"):
            try:
                segs, name = _pyannote_turns(wav16), PYANNOTE_MODEL
            except Exception:
                if DIARIZER == "pyannote":
                    raise
                traceback.print_exc()  # e.g. pyannote missing or its model not cached: use sherpa
        if segs is None:
            segs, name = _sherpa_turns(wav16), SHERPA_NAME
    total = {}
    for a, b, k in segs:
        total[k] = total.get(k, 0.0) + b - a
    keep = [(a, b, k) for a, b, k in segs if total[k] >= DIAR_MIN_TALK_S]
    if not keep:
        return [], name
    out = []
    for a, b, k in segs:
        if total[k] < DIAR_MIN_TALK_S:  # fragment: take the speaker of the nearest kept turn
            k = min(keep, key=lambda t: max(t[0] - b, a - t[1], 0.0))[2]
        out.append((a, b, k))
    number = {}
    for _, _, k in out:
        number.setdefault(k, len(number) + 1)
    return [(a, b, number[k]) for a, b, k in out], name


def speaker_for(a, b, turns):
    """Speaker number overlapping cue [a, b] most (nearest turn if none overlaps); None without turns."""
    if not turns:
        return None
    best, best_ov = None, 0.0
    for s, e, k in turns:
        ov = min(b, e) - max(a, s)
        if ov > best_ov:
            best, best_ov = k, ov
    if best is None:
        best = min(turns, key=lambda t: max(t[0] - b, a - t[1], 0.0))[2]
    return best


def vtt_time(t):
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


# ---------- progressive conversion ----------
#
# A job probes the source's duration, writes the .mimi header and preallocates the file at its final size,
# then encodes it in chunks (10 s first for a fast start, then 60 s; each with 20 s of left context) while
# ffmpeg is still decoding the source. Readers of <id>.mimi get bytes as their chunk lands; a chunk somebody
# is waiting for (e.g. after a seek) is encoded next. Every byte's position is fixed (44-byte frames), so
# Range requests work against the final file size from the start.

FPS = 12.5
FIRST_CHUNK, CHUNK, CTX = 125, 750, 250  # frames
FRAME_BYTES = 44  # 32 codebooks x 11 bits
WORKERS = int(os.environ.get("MIMI_HUB_WORKERS", "2"))
STREAMABLE = ("encoding", "transcribing", "ready")
FFMPEG, FFPROBE = shutil.which("ffmpeg"), shutil.which("ffprobe")


def chunk_bounds(n_frames):
    out, f, size = [], 0, FIRST_CHUNK
    while f < n_frames:
        out.append((f, min(n_frames, f + size)))
        f, size = f + size, CHUNK
    return out


class Live:
    """A conversion in progress; its .mimi on disk fills in chunk by chunk."""

    def __init__(self, i, n_frames, header):
        self.i, self.n_frames, self.header = i, n_frames, header
        self.data_offset = len(header)
        self.total = self.data_offset + n_frames * FRAME_BYTES
        self.bounds = chunk_bounds(n_frames)
        self.ready = [False] * len(self.bounds)
        self.wanted = []  # chunks readers are waiting for, most recent first
        self.cond = threading.Condition()
        self.failed = False

    def chunk_of(self, frame):
        c = 0 if frame < FIRST_CHUNK else 1 + (frame - FIRST_CHUNK) // CHUNK
        return min(c, len(self.bounds) - 1)

    def wait_ready(self, c, timeout=300):
        with self.cond:
            if not self.ready[c]:
                if c in self.wanted:
                    self.wanted.remove(c)
                self.wanted.insert(0, c)
                self.cond.notify_all()
            end = time.time() + timeout
            while not self.ready[c] and not self.failed and time.time() < end:
                self.cond.wait(1.0)
            return self.ready[c]


live: dict = {}


class Pcm:
    """24 kHz mono float32 decoded by ffmpeg in the background, usable as it arrives."""

    def __init__(self, src, n_samples, cond):
        import numpy as np
        self.buf = np.zeros(n_samples, np.float32)
        self.avail, self.done, self.error, self.cond = 0, False, None, cond
        http = ["-user_agent", UA["User-Agent"]] if src.startswith("http") else []
        self.proc = subprocess.Popen([FFMPEG, "-v", "error", *http, "-i", src, "-map", "0:a:0", "-ac", "1",
                                      "-ar", "24000", "-f", "f32le", "-"],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        threading.Thread(target=self._read, daemon=True, name="mimi-pcm").start()

    def _read(self):
        import numpy as np
        rest = b""
        try:
            while self.avail < len(self.buf):
                b = self.proc.stdout.read(1 << 18)
                if not b:
                    break
                b = rest + b
                k = len(b) // 4 * 4
                x = np.frombuffer(b[:k], np.float32)
                rest = b[k:]
                n = min(len(x), len(self.buf) - self.avail)
                self.buf[self.avail:self.avail + n] = x[:n]
                with self.cond:
                    self.avail += n
                    self.cond.notify_all()
        except Exception as e:
            self.error = e
        finally:
            self.proc.kill()
            err = self.proc.stderr.read().decode(errors="replace").strip()
            if self.avail == 0 and not self.error:
                self.error = RuntimeError(f"ffmpeg produced no audio: {err[-300:]}")
            with self.cond:
                self.done = True
                self.cond.notify_all()


def probe(src):
    http = ["-user_agent", UA["User-Agent"]] if src.startswith("http") else []
    out = subprocess.run([FFPROBE, "-v", "error", *http, "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", src], capture_output=True, text=True, timeout=60)
    return float(out.stdout.strip())  # raises on failure / "N/A"


def pick(L, pcm):
    """Next chunk to encode: a waited-for one if its audio is in, else the first missing (or None: wait)."""
    def has_audio(c):
        return pcm.done or pcm.avail >= L.bounds[c][1] * mimi_file.SPF
    L.wanted = [c for c in L.wanted if not L.ready[c]]
    for c in L.wanted:
        if has_audio(c):
            return c
    for c, r in enumerate(L.ready):
        if not r:
            return c if has_audio(c) else None
    return None


# Chunks are encoded in order as one stream, carrying the encoder's KV and conv padding caches: the codes match a
# single uninterrupted encode (99.99% on 5 min, vs 98.7% for the old windows with 20 s of left context) and it's 27%
# less GPU work, since no context is re-encoded (43.8 min: 17.2 -> 12.6 s on the server; scripts/mimi_streaming_server.py).
# A chunk a reader is waiting for ahead of the stream is encoded at once with 20 s of left context, as before; the
# stream later passes over it to keep its state, without rewriting bytes that may have been served.
MIMI_STREAMING = os.environ.get("MIMI_HUB_MIMI_STREAMING", "1") != "0"


def encode_live(model, L, pcm, fd, i):
    import numpy as np
    import torch
    spf = mimi_file.SPF
    dev = next(model.parameters()).device
    n = len(L.bounds)

    def has_audio(c):
        return pcm.done or pcm.avail >= L.bounds[c][1] * spf

    def next_job(seq):
        """("stream" | "window", chunk), or None to wait for audio."""
        if not MIMI_STREAMING:
            c = pick(L, pcm)
            return None if c is None else ("window", c)
        L.wanted = [c for c in L.wanted if not L.ready[c]]
        ahead = next((c for c in L.wanted if c != seq and has_audio(c)), None)
        if ahead is not None:
            return "window", ahead
        return ("stream", seq) if has_audio(seq) else None

    seq, kv, pad, written = 0, None, None, 0  # the stream's next chunk and caches
    while not all(L.ready):
        with L.cond:
            while (job := next_job(seq)) is None:
                if pcm.error:
                    raise pcm.error
                L.cond.wait(0.5)
        kind, c = job
        f0, f1 = L.bounds[c]
        if kind == "stream":
            x = torch.from_numpy(pcm.buf[f0 * spf:f1 * spf].copy())[None, None].to(dev)
            with torch.inference_mode():
                o = model.encode(x, num_quantizers=32, encoder_past_key_values=kv, padding_cache=pad,
                                 use_streaming=True)
            kv, pad, seq = o.encoder_past_key_values, o.padding_cache, seq + 1
            if L.ready[c]:
                continue  # encoded early for a reader: keep its bytes, the pass just advanced the caches
            seg = o.audio_codes[0].T.cpu().numpy()[: f1 - f0]
        else:
            lo = max(0, (f0 - CTX) * spf)
            x = torch.from_numpy(pcm.buf[lo:f1 * spf].copy())[None, None].to(dev)
            with torch.inference_mode():
                codes = model.encode(x, num_quantizers=32).audio_codes[0].T.cpu().numpy()
            seg = codes[(f0 * spf - lo) // spf:][: f1 - f0]
        if len(seg) < f1 - f0:  # pad (only at the very end)
            seg = np.concatenate([seg, np.repeat(seg[-1:], f1 - f0 - len(seg), axis=0)])
        os.pwrite(fd, mimi_file.pack(seg, 32).tobytes(), L.data_offset + f0 * FRAME_BYTES)
        with L.cond:
            L.ready[c] = True
            L.cond.notify_all()
        written += 1
        if written % 4 == 0 or written == n:
            update(i, progress=round(written / n, 3))


def transcribe(m, pcm, cond, out, on_cues=None):
    """ASR and diarization of the full decoded episode, concurrently; results (or "error") go into [out].
    [on_cues] gets the ASR cues as soon as they exist, before diarization is done."""
    try:
        import librosa
        with cond:
            while not pcm.done:
                cond.wait(1.0)
        if pcm.error:
            return  # the encoder reports it
        wav16 = librosa.resample(pcm.buf[:pcm.avail], orig_sr=mimi_file.SR, target_sr=16000)

        def speakers():
            try:
                out["turns"], out["diarization"] = diarize(wav16)
            except Exception:
                traceback.print_exc()  # a transcript without speakers is still useful
                out["turns"], out["diarization"] = [], None

        # pyannote on the GPU waits for ASR: the GPU is the bottleneck either way (the stages' times add up), and
        # this way the cues, and so ad detection, are ready ~30 s sooner. sherpa (CPU) runs alongside.
        diar_on_gpu = m.get("device") == "cuda" and DIARIZER in ("auto", "pyannote")
        d = threading.Thread(target=speakers, daemon=True, name="mimi-diarize")
        if not diar_on_gpu:
            d.start()
        out["cues"] = [c for s in m["asr"].recognize(wav16, sample_rate=16000) for c in cues(s)]
        if on_cues:
            on_cues(out["cues"])
        if diar_on_gpu:
            d.start()
        d.join()
    except Exception as e:
        out["error"] = e


def run(i):
    r = load(i)
    upload = r.get("upload")
    L = None
    try:
        if not (FFMPEG and FFPROBE):
            raise RuntimeError("ffmpeg/ffprobe not on PATH (start the hub with hub/run.sh)")
        m = models()
        update(i, state="downloading", progress=0.0)
        if upload:
            src, dur = upload, probe(upload)
        else:
            src, dur, err = None, None, None
            for u in candidates(r["audio_url"]):  # skip tracking hops that are unreachable
                try:
                    src, dur = u, probe(u)
                    break
                except Exception as e:
                    err = e
            if src is None or dur is None:
                raise RuntimeError(f"cannot read source: {err}")
        n_frames = int(dur * FPS) + 13  # + ~1 s: ffprobe estimates can run short; the tail is padded
        head, _ = mimi_file.header(n_frames, 32, dict(title=r.get("title", ""), source=r.get("url", ""), id=i,
                                                      encoder="transformers MimiModel", duration_s=dur))
        path = DATA / f"{i}.mimi"
        with open(path, "wb") as f:
            f.write(head)
            f.truncate(len(head) + n_frames * FRAME_BYTES)
        L = Live(i, n_frames, head)
        live[i] = L
        update(i, state="encoding", progress=0.0, duration_s=dur, mimi_bytes=L.total)
        pcm = Pcm(src, n_frames * mimi_file.SPF, L.cond)
        # transcription and diarization start once the whole episode is decoded (ffmpeg is far ahead of the
        # encoder) and run alongside the encode
        tx = {}
        # ad detection starts from the ASR cues, alongside diarization: speaker labels add little to it (episodes
        # without any were found as well), and diarization takes as long as everything else
        early = lambda cs: queue_segments(i, [(a, b, None, text) for a, b, text in cs])  # noqa: E731
        t = threading.Thread(target=transcribe, args=(m, pcm, L.cond, tx, early), daemon=True,
                             name=f"mimi-asr-{i}")
        t.start()
        fd = os.open(path, os.O_WRONLY)
        try:
            encode_live(m["mimi"], L, pcm, fd, i)
        finally:
            os.close(fd)
        live.pop(i, None)  # complete: served as a plain file from now on
        update(i, state="transcribing", progress=0.0)
        t.join()
        if "error" in tx:
            raise tx["error"]
        if "cues" not in tx:
            raise pcm.error or RuntimeError("transcription did not run")
        all_cues, turns = tx["cues"], tx["turns"]
        with open(DATA / f"{i}.vtt.tmp", "w") as f:
            f.write("WEBVTT\n\n")
            for a, b, text in all_cues:
                text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                who = speaker_for(a, b, turns)
                f.write(f"{vtt_time(a)} --> {vtt_time(b)}\n{f'<v Speaker {who}>' if who else ''}{text}\n\n")
        (DATA / f"{i}.vtt.tmp").replace(DATA / f"{i}.vtt")
        update(i, state="ready", progress=1.0, asr=ASR_MODEL, device=m["device"],
               speakers=len({t[2] for t in turns}), diarization=tx["diarization"] if turns else None)
    except Exception as e:
        traceback.print_exc()
        if L is not None:
            with L.cond:
                L.failed = True
                L.cond.notify_all()
        live.pop(i, None)
        update(i, state="failed", error=f"{type(e).__name__}: {e}")
    finally:
        if upload and os.path.exists(upload):
            os.unlink(upload)


# ---------- ad / self-promotion segments (an LLM over the transcript; one at a time, after "ready") ----------

seg_jobs: "queue.Queue" = queue.Queue()


def queue_segments(i, cues=None):
    """Detect [i]'s ads / self-promotion from [cues] ([(start_s, end_s, speaker or None, text)]), else its VTT."""
    if segments.SEGMENTER.strip().lower() in ("", "off"):
        update(i, segments_state="off")
        return
    update(i, segments_state="pending")
    seg_jobs.put((i, cues))


def segment_worker():
    while True:
        i, cues = seg_jobs.get()
        try:
            if cues is None:
                cues = segments.read_cues((DATA / f"{i}.vtt").read_text())
            res = segments.detect(cues, log=lambda m: print(f"[{i}] {m}", flush=True))
            tmp = DATA / f"{i}.segments.json.tmp"
            tmp.write_text(json.dumps(res, indent=1))
            tmp.replace(DATA / f"{i}.segments.json")
            update(i, segments_state="ready", segments=len(res["segments"]), segments_model=res["model"],
                   skippable_s=round(sum(s["end"] - s["start"] for s in res["segments"]), 1))
        except Exception as e:
            traceback.print_exc()
            update(i, segments_state="failed", segments_error=f"{type(e).__name__}: {e}"[:500])


_seq = itertools.count()


def worker():
    while True:
        _, _, i = jobs.get()
        with lock:
            r = load(i)
            if not r or r["state"] != "queued":
                continue  # duplicate entry (re-queued at a higher priority) or already handled
            save(dict(r, state="starting"))
        run(i)


@app.on_event("startup")
def start():
    for p in sorted(record_paths(), key=lambda p: p.stat().st_mtime):
        r = json.loads(p.read_text())
        if r["state"] not in ("ready", "failed"):  # resume interrupted jobs from scratch
            (DATA / f"{r['id']}.mimi").unlink(missing_ok=True)  # partial, preallocated: never serve it
            save(dict(r, state="queued", progress=0.0))
            jobs.put((r.get("priority", 1), next(_seq), r["id"]))
        elif r["state"] == "ready" and r.get("segments_state") not in ("ready", "failed", "off"):
            queue_segments(r["id"])  # converted before segment detection existed, or interrupted
    for k in range(WORKERS):
        threading.Thread(target=worker, daemon=True, name=f"mimi-worker-{k}").start()
    threading.Thread(target=segment_worker, daemon=True, name="mimi-segments").start()


@app.on_event("shutdown")
def stop():
    if _diar_pool is not None:
        _diar_pool.shutdown(wait=False, cancel_futures=True)


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
    priority: int = 1  # 0: a client is about to stream it


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
    return enqueue(i, priority=min(1, max(0, req.priority)), url=req.url, audio_url=audio,
                   title=req.title or title)


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


@app.post("/v1/episodes/{i}/segments", status_code=202)
def redo_segments(i: str, authorization: str | None = Header(None)):
    auth(authorization)
    check_id(i)
    r = load(i)
    if not r or r["state"] != "ready":
        raise HTTPException(409, "episode not ready")
    queue_segments(i)
    return load(i)


@app.get("/v1/episodes")
def list_jobs(authorization: str | None = Header(None)):
    auth(authorization)
    rs = [json.loads(p.read_text()) for p in record_paths()]
    return sorted(rs, key=lambda r: -r.get("created", 0))


def stream_live(L, rng):
    """Serve (a Range of) a .mimi that is still being encoded, waiting for each chunk as needed."""
    start, end, status = 0, L.total - 1, 200
    m = re.match(r"bytes=(\d*)-(\d*)$", (rng or "").strip())
    if m and (m.group(1) or m.group(2)):
        a, b = m.groups()
        if a == "":
            start = max(0, L.total - int(b))
        else:
            start, end = int(a), min(int(b), end) if b else end
        if start > end:
            return Response(status_code=416, headers={"Content-Range": f"bytes */{L.total}"})
        status = 206
    path = DATA / f"{L.i}.mimi"

    def body():
        # os.pread, not a buffered file: read-ahead would cache bytes of chunks that aren't written yet
        fd = os.open(path, os.O_RDONLY)
        try:
            pos = start
            while pos <= end:
                if pos < L.data_offset:
                    n = min(end + 1, L.data_offset) - pos
                    yield L.header[pos:pos + n]
                    pos += n
                    continue
                c = L.chunk_of((pos - L.data_offset) // FRAME_BYTES)
                if not L.wait_ready(c):
                    return  # job failed or stalled; the client sees a short read
                stop = min(end, L.data_offset + L.bounds[c][1] * FRAME_BYTES - 1)
                data = os.pread(fd, stop - pos + 1, pos)
                if not data:
                    return
                yield data
                pos += len(data)
        finally:
            os.close(fd)

    headers = {"Accept-Ranges": "bytes", "Content-Length": str(end - start + 1)}
    if status == 206:
        headers["Content-Range"] = f"bytes {start}-{end}/{L.total}"
    return StreamingResponse(body(), status_code=status, media_type="audio/x-mimi", headers=headers)


@app.get("/v1/episodes/{name}")
def episode(name: str, request: Request, authorization: str | None = Header(None)):
    auth(authorization)
    stem, _, ext = name.partition(".")
    check_id(stem)
    if ext == "segments.json":
        p = DATA / f"{stem}.segments.json"
        if not p.exists():
            raise HTTPException(404, "no segments yet")
        return FileResponse(p, media_type="application/json")
    if ext == "mimi":
        deadline = time.time() + 30  # a just-requested job writes its header within a few seconds
        while (L := live.get(stem)) is None and time.time() < deadline:
            r = load(stem)
            if not r or r["state"] not in ("queued", "starting", "downloading"):
                break
            time.sleep(0.25)
        if L is not None:
            return stream_live(L, request.headers.get("range"))
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
