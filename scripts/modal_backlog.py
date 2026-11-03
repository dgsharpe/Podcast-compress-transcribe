"""A podcast backlog converted in bulk on a Modal H100 (hub/bulk.py) straight into a Modal hub's library
(hub/modal_app.py's data Volume), each episode under its Pocket Casts episode UUID with a hub record, so the app
finds it ready. Ad detection is left to scripts/backlog_ads.py (run locally, with your own key).

    uvx --from modal modal run scripts/modal_backlog.py --manifest episodes.json [--limit 8] [--chunk 300]

The manifest: [{"uuid", "url", "title", "duration_s"}, ...] (e.g. from Pocket Casts' podcast/full API). Episodes the
library already has are skipped, so a stopped run resumes. It runs in chunks (one hub/bulk.py process each, on the
free-threaded Python), each chunk's files and records committed to the Volume as it ends. Guards: a chunk is
killed after CHUNK_TIMEOUT_S, the container after 4 h, and the local entrypoint cancels the call when stopped.
"""

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import modal

if modal.is_local():  # scripts/modal_hub_bench.py's image: free-threaded Python, TensorRT, the CUDA toolchain
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import modal_hub_bench
    image = modal_hub_bench.image
else:  # (in the container the image is built; that module isn't there)
    image = modal.Image.debian_slim()
NA, MODELS, FT_PYTHON = "/root/na", "/models/hub", "/ft/bin/python"  # as in scripts/modal_hub_bench.py
FT_ENV = {"PYTHON_GIL": "0", "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python"}

app = modal.App("mimi-backlog", image=image)
DATA = modal.Volume.from_name(os.environ.get("MIMI_MODAL_APP", "mimi-hub") + "-data", version=2)
CACHE = modal.Volume.from_name("mimi-hub-inductor-cache")
CHUNK_TIMEOUT_S = 2 * 3600
ASR_MODEL = "nemo-parakeet-tdt-0.6b-v2"


def _record(e, stats, mimi_bytes):
    now = time.time()
    return {"id": e["uuid"], "state": "ready", "priority": 1, "progress": 1.0, "error": None, "created": now,
            "updated": now, "keep_source": False, "compressed": True, "url": e["url"], "audio_url": e["url"],
            "title": e["title"], "duration_s": stats["duration_s"], "mimi_bytes": mimi_bytes, "asr": ASR_MODEL,
            "device": "cuda (hub/bulk.py)", "speakers": stats.get("speakers"),
            "diarization": "pyannote/speaker-diarization-community-1" if stats.get("speakers") else None,
            # (scripts/backlog_ads.py fills these in; until then the app's own key can have the hub redo them)
            "segments_state": "failed", "segments_error": "batch conversion: ad detection not run yet",
            "batch": True}


@app.function(gpu="H100", cpu=32.0, memory=96 * 1024, timeout=4 * 3600,
              volumes={"/data": DATA, "/cache": CACHE})
def backlog(manifest: list, chunk: int = 300) -> list:
    os.environ.update({"TORCHINDUCTOR_CACHE_DIR": "/tmp/inductor-cache", "MIMI_HUB_AOTI": "/cache/aoti",
                       "MIMI_HUB_AOTI_MARCH": "x86-64-v3"})
    sys.path.insert(0, NA)
    from hub import parakeet_native
    model_dir = Path(MODELS) / f"{ASR_MODEL}-hf"
    parakeet_native.convert(ASR_MODEL, model_dir)
    parakeet_native.convert_decoder(ASR_MODEL, model_dir)
    data = Path("/data")
    todo = [e for e in manifest if not (data / f"{e['uuid']}.mimi").exists()]
    print(f"BACKLOG {len(manifest)} episodes, {len(manifest) - len(todo)} already in the library, {len(todo)} to do",
          flush=True)
    env = {**os.environ, **FT_ENV,
           "MIMI_HUB_DIAR_STEP": "0.5", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
           "MIMI_BULK_SPOOL": "/tmp"}
    out, results = Path("/tmp/out"), []
    for k in range(0, len(todo), chunk):
        part, t0 = todo[k:k + chunk], time.time()
        by_id = {e["uuid"]: e for e in part}
        out.mkdir(exist_ok=True)
        (Path("/tmp") / "chunk.json").write_text(json.dumps([{"id": e["uuid"], "url": e["url"]} for e in part]))
        proc = subprocess.Popen([FT_PYTHON, "-m", "hub.bulk", str(out), "--manifest", "/tmp/chunk.json",
                                 "--prep", "28"], cwd=NA, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, start_new_session=True)
        try:
            stdout, stderr = proc.communicate(timeout=CHUNK_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate()
            stderr += f"\nkilled after {CHUNK_TIMEOUT_S} s"
        rows = [json.loads(ln.split(" ", 1)[1]) for ln in stdout.splitlines() if re.match(r"^EPISODE \{", ln)]
        summary = next((json.loads(ln.split(" ", 1)[1]) for ln in stdout.splitlines() if ln.startswith("BULK ")), None)
        ok = 0
        for s in rows:
            e = by_id.get(s["name"])
            m, v = out / f"{s['name']}.mimi", out / f"{s['name']}.vtt"
            if e is None or s.get("error") or not (m.exists() and v.exists()):
                results.append({"uuid": s["name"], "error": (s.get("error") or "no output")[:300]})
                continue
            shutil.move(v, data / v.name)
            shutil.move(m, data / m.name)  # (the record last: a record means the files are there)
            rec = data / f"{s['name']}.json"
            rec.write_text(json.dumps(_record(e, s, (data / m.name).stat().st_size), indent=1))
            results.append({"uuid": s["name"], "duration_s": s["duration_s"], "speakers": s.get("speakers")})
            ok += 1
        DATA.commit()
        print("CHUNK " + json.dumps({"from": k, "episodes": len(part), "written": ok, "wall_s": round(time.time() - t0),
                                     "bulk": summary, "exit": proc.returncode}), flush=True)
        if proc.returncode or summary is None:
            print(f"CHUNK STDERR:\n{stderr[-4000:]}", flush=True)
        shutil.rmtree(out, ignore_errors=True)
    return results


@app.local_entrypoint()
def main(manifest: str, limit: int = 0, chunk: int = 300, wait_min: float = 230):
    eps = json.loads(Path(manifest).read_text())
    eps = eps[:limit] if limit else eps
    call = backlog.spawn(eps, chunk)

    def stop(*_):  # stopped from outside: don't leave the GPU running
        call.cancel()
        raise SystemExit("stopped; remote call cancelled")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGHUP, stop)
    try:
        results = call.get(timeout=wait_min * 60)
    except BaseException:
        call.cancel()
        raise
    failed = [r for r in results if r.get("error")]
    out = Path(manifest).with_suffix(".results.json")
    out.write_text(json.dumps(results, indent=1))
    print(f"done: {len(results) - len(failed)} written, {len(failed)} failed -> {out}")
    for r in failed[:10]:
        print("  failed", r["uuid"], r["error"][:200])
