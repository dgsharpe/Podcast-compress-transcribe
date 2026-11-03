"""hub/bulk.py on a backlog of clips of the reference episode: throughput, batch sizes, and the output of the first
--check clips against the hub's one-episode path (transcript words, Mimi codes, DER of the speaker turns).

    python scripts/bulk_bench.py [--clips 48] [--clip-min 5] [--check 3] [--prep N] [--profile 60:5]

--profile START:SECONDS: torch.profiler over SECONDS of the run from START s in, then the GPU kernel time launched
inside each stage (hub/bulk.py's stage:* labels) and the top kernels.
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


def _profiler(start_s, seconds):
    """Profile [seconds] of the run from [start_s] in (a thread; the run blocks the main one). -> the profiler."""
    import threading
    import torch
    from torch._C._profiler import _ExperimentalConfig
    prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                              torch.profiler.ProfilerActivity.CUDA],
                                  experimental_config=_ExperimentalConfig(profile_all_threads=True))  # the workers

    def run():
        time.sleep(start_s)
        prof.start()
        time.sleep(seconds)
        prof.stop()
    threading.Thread(target=run, daemon=True, name="profiler").start()
    return prof


def profile_summary(prof, seconds):
    """GPU time per stage. Each kernel goes to the stage:* span on its own stream that contains it (kineto puts
    record_function spans on the GPU timeline too); a kernel outside any span (CUDA-graph replays and the TDT
    search's kernels got none) goes to its stream's stage: every stage runs on streams of its own, so a stream is
    the stage most of its labelled kernel time is, and unlabelled time on an ASR stream is the decoder. GPU busy =
    the union of kernel intervals (streams overlap)."""
    import bisect
    from collections import defaultdict
    from torch.autograd import DeviceType
    gpu = [e for e in prof.events() if e.device_type != DeviceType.CPU]
    spans, kernels = defaultdict(list), []
    for e in gpu:
        r = (e.time_range.start, e.time_range.end)
        if e.is_user_annotation:
            if e.name.startswith("stage:"):
                spans[e.device_resource_id].append((*r, e.name[6:]))
        else:
            kernels.append((*r, e.device_resource_id, e.name))
    for v in spans.values():
        v.sort()
    labelled, by_name = [], defaultdict(lambda: [0.0, 0])
    per_stream = defaultdict(lambda: defaultdict(float))
    for s0, s1, stream, name in kernels:
        v = spans.get(stream, [])
        k = bisect.bisect_right(v, (s0, float("inf"), "")) - 1
        while k >= 0 and v[k][1] < s0 and k > bisect.bisect_right(v, (s0 - 5e6, 0, "")) - 1:
            k -= 1  # spans nest (fbank inside the ResNet batcher's): the innermost one still open at s0
        label = v[k][2] if k >= 0 and v[k][1] >= s0 else None
        labelled.append((stream, label, (s1 - s0) / 1e6))
        if label:
            per_stream[stream][label] += (s1 - s0) / 1e6
        by_name[name][0] += (s1 - s0) / 1e6
        by_name[name][1] += 1
    owner = {st: max(v, key=v.get) for st, v in per_stream.items()}
    stage, streams = defaultdict(float), defaultdict(float)
    for stream, label, t in labelled:
        if label is None:
            label = owner.get(stream, "other")
            label = "asr-decode" if label == "asr-encode" else label
        stage[label] += t
        streams[f"{stream}:{owner.get(stream, 'other')}"] += t
    busy, end = 0.0, None
    for s0, s1, _, _ in sorted(kernels):
        if end is None or s0 > end:
            busy, end = busy + (s1 - s0), s1
        elif s1 > end:
            busy, end = busy + (s1 - end), s1
    total = sum(stage.values())
    top = sorted(by_name.items(), key=lambda kv: -kv[1][0])[:15]
    window = (max(k[1] for k in kernels) - min(k[0] for k in kernels)) / 1e6  # kernels finish after stop()
    return {"window_s": round(window, 2), "gpu_busy_frac": round(busy / 1e6 / window, 3), "kernel_s": round(total, 2),
            "stage_share": {k: round(v / total, 3) for k, v in sorted(stage.items(), key=lambda kv: -kv[1])},
            "stage_kernel_s": {k: round(v, 2) for k, v in sorted(stage.items(), key=lambda kv: -kv[1])},
            "streams": {k: round(v, 2) for k, v in sorted(streams.items(), key=lambda kv: -kv[1])[:12]},
            "top_kernels": [(name[:90], round(t, 3), n) for name, (t, n) in top]}


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
    ap.add_argument("--subtype", default="FLOAT", help="the clips' WAV sample format (PCM_16: half the disk)")
    ap.add_argument("--profile", default=None, help="START:SECONDS of the timed run to profile")
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
        sf.write(sources[-1], w24, mimi_file.SR, subtype=a.subtype)
    if a.procs > 1:
        return multi(a, sources, work)
    m = throughput_server.setup(False)  # the hub's models (and its pyannote, for the reference)
    refs = [reference(m, *clips[k]) for k in range(min(a.check, len(clips)))]
    models = bulk.Models()
    bulk.convert(sources[:2], work / "warmup", a.prep, models)  # kernels for every shape, pool started
    prof = None
    if a.profile:
        start_s, seconds = map(float, a.profile.split(":"))
        prof = _profiler(start_s, seconds)
    t = time.perf_counter()
    summary, stats = bulk.convert(sources, work / "out", a.prep, models)
    summary["wall_incl_s"] = round(time.perf_counter() - t, 1)
    if prof is not None:
        print("PROFILE " + json.dumps(profile_summary(prof, seconds)), flush=True)
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
