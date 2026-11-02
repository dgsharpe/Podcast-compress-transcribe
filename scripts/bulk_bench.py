"""hub/bulk.py on a backlog of clips of the reference episode: throughput, batch sizes, and the output of the first
--check clips against the hub's one-episode path (transcript words, Mimi codes, DER of the speaker turns).

    python scripts/bulk_bench.py [--clips 48] [--clip-min 5] [--check 3] [--prep N]
"""
import argparse
import difflib
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("MIMI_HUB_DATA", tempfile.mkdtemp(prefix="mimi-bulk-data-"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def vtt_words(path):
    text = [re.sub(r"<v [^>]*>", "", ln) for ln in Path(path).read_text().splitlines()[2:]
            if ln and "-->" not in ln]
    return " ".join(text).split()


def reference(m, w24, w16):
    """The hub's one-episode results for a clip: Mimi codes, transcript words, speaker turns."""
    import numpy as np
    import throughput_server
    from hub import mimi_hub
    codes = np.concatenate(throughput_server.hub_encode(m["mimi"], w24))
    words = " ".join(t for _, _, t in (c for s in m["asr"].recognize(w16, sample_rate=16000)
                                       for c in mimi_hub.cues(s))).split()
    turns, _ = mimi_hub.diarize(w16)
    return codes, words, turns


def der(a, b):
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate
    ref, hyp = Annotation(), Annotation()
    for s, e, k in a:
        ref[Segment(s, e)] = k
    for s, e, k in b:
        hyp[Segment(s, e)] = k
    return DiarizationErrorRate()(ref, hyp)


def _bulk_proc(k, sources, work, prep, barrier, results):
    """One of --procs bulk processes: its own models, its share of the sources, timed after a common start."""
    from hub import bulk
    models = bulk.Models()
    bulk.convert(sources[:2], work / f"warmup{k}", prep, models)
    barrier.wait(timeout=900)
    t = time.time()
    summary, stats = bulk.convert(sources, work / "out", prep, models)
    results.put((t, time.time(), summary, sum(s.get("duration_s", 0) for s in stats)))


def multi(a, sources, work):
    import multiprocessing as mp
    import queue
    ctx = mp.get_context("spawn")
    barrier, q = ctx.Barrier(a.procs + 1), ctx.Queue()
    prep = max(1, (a.prep or (os.cpu_count() or 2) // 2) // a.procs)
    ps = [ctx.Process(target=_bulk_proc, args=(k, sources[k::a.procs], work, prep, barrier, q))
          for k in range(a.procs)]
    [p.start() for p in ps]
    got = []
    try:
        barrier.wait(timeout=900)
        while len(got) < len(ps):
            try:
                got.append(q.get(timeout=5))
            except queue.Empty:
                if all(not p.is_alive() for p in ps):
                    break
    finally:
        [p.kill() for p in ps if p.is_alive()]
    if len(got) < len(ps):
        print("BULK " + json.dumps({"procs": a.procs, "error": f"{len(ps) - len(got)} processes died"}), flush=True)
        return
    wall = max(g[1] for g in got) - min(g[0] for g in got)
    audio = sum(g[3] for g in got)
    print("BULK " + json.dumps({"procs": a.procs, "episodes": len(sources), "audio_min": round(audio / 60, 1),
                                "wall_s": round(wall, 1), "x_rt": round(audio / wall, 1),
                                "failed": sum(g[2].get("failed", 0) for g in got),
                                "first_error": next((g[2]["first_error"] for g in got if g[2].get("first_error")),
                                                    None),
                                "per_proc": [{k: g[2].get(k) for k in ("x_rt", "main_cpu_cores", "busy_frac", "failed",
                                                                        "gpu_reserved_peak_gb")}
                                             for g in got]}), flush=True)


def _watchdog(seconds):
    """After [seconds], print every thread's stack (a hang shows where; works with the GIL off, unlike faulthandler's
    all-threads dump) and exit."""
    import threading
    import traceback

    def run():
        time.sleep(seconds)
        names = {t.ident: t.name for t in threading.enumerate()}
        frames = sorted(sys._current_frames().items(), key=lambda kv: "diar" in str(names.get(kv[0], "")))
        for ident, frame in frames:  # the innermost frames, the diarization threads (many, alike) last
            print(f"--- thread {names.get(ident, ident)}", file=sys.stderr)
            for fs in traceback.extract_stack(frame)[-6:]:
                print(f"  {fs.filename.split('site-packages/')[-1]}:{fs.lineno} {fs.name}: {fs.line}", file=sys.stderr)
        sys.stderr.flush()
        os._exit(3)
    threading.Thread(target=run, daemon=True, name="watchdog").start()


def main():
    if os.environ.get("BULK_WATCHDOG_S"):
        _watchdog(float(os.environ["BULK_WATCHDOG_S"]))
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=48)
    ap.add_argument("--clip-min", type=float, default=5.0)
    ap.add_argument("--check", type=int, default=3)
    ap.add_argument("--prep", type=int, default=None)
    ap.add_argument("--procs", type=int, default=1, help="bulk processes, each with its own models (e.g. under MPS)")
    a = ap.parse_args()
    import soundfile as sf
    import throughput_server
    import mimi_file
    from hub import bulk

    work = Path(tempfile.mkdtemp(prefix="mimi-bulk-"))
    clips = throughput_server.load_clips(str(ROOT / "data/ref/ref_24k.wav"), a.clips, a.clip_min)
    sources = []
    for k, (w24, _) in enumerate(clips):
        sources.append(work / f"clip{k:03d}.wav")
        sf.write(sources[-1], w24, mimi_file.SR, subtype="FLOAT")
    if a.procs > 1:
        return multi(a, sources, work)
    m = throughput_server.setup(False)  # the hub's models (and its pyannote, for the reference)
    refs = [reference(m, *clips[k]) for k in range(min(a.check, len(clips)))]
    models = bulk.Models()
    bulk.convert(sources[:2], work / "warmup", a.prep, models)  # kernels for every shape, pool started
    t = time.perf_counter()
    summary, stats = bulk.convert(sources, work / "out", a.prep, models)
    summary["wall_incl_s"] = round(time.perf_counter() - t, 1)
    checks = []
    for k, (codes, words, turns) in enumerate(refs):
        name = sources[k].stem
        _, got = mimi_file.read(work / "out" / f"{name}.mimi")
        n = min(len(got), len(codes))
        got_words = vtt_words(work / "out" / f"{name}.vtt")
        st = next(s for s in stats if s["name"] == name)
        checks.append({"clip": name, "codes_all32": round(float((got[:n] == codes[:n]).mean()), 4),
                       "codes_first8": round(float((got[:n, :8] == codes[:n, :8]).mean()), 4),
                       "words": len(got_words), "word_similarity": round(difflib.SequenceMatcher(
                           None, words, got_words, autojunk=False).ratio(), 5),
                       "speakers": st.get("speakers"), "ref_speakers": len({t[2] for t in turns}),
                       "der_vs_hub": round(der(turns, st["turns"]), 4) if turns and st.get("turns") else None,
                       "errors": st.get("errors")})
    errors = [{k: v for k, v in s.items() if k != "turns"} for s in stats if s.get("error") or s.get("errors")]
    print("BULK " + json.dumps({**summary, "errors": errors[:3]}), flush=True)
    print("BULKCHECK " + json.dumps({"checks": checks}), flush=True)


if __name__ == "__main__":
    main()
