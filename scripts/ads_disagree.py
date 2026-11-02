"""Where ad-detection models disagree (scripts/ads_bench.py --save): per episode, every model's breaks clustered by
overlap; a cluster all models have, with starts and ends within TOL s of each other, is unanimous; the others are
printed with their transcript (cue index, time, speaker, text) from CONTEXT s before to CONTEXT s after, and each
model's cue range, to be judged by reading.

    python scripts/ads_disagree.py runs.json [--tol 8] [--context 25] [--episode ID]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def mmss(t):
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def clusters(by_spec):
    """[(start, end, {spec: segment})] of overlapping breaks across specs."""
    items = sorted((s["start"], s["end"], spec, s) for spec, segs in by_spec.items() for s in segs)
    out = []
    for start, end, spec, s in items:
        if out and start < out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
            out[-1][2].setdefault(spec, s)
        else:
            out.append([start, end, {spec: s}])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--tol", type=float, default=8.0)
    ap.add_argument("--context", type=float, default=25.0)
    ap.add_argument("--episode", default=None)
    ap.add_argument("--data", default="hub_data")
    a = ap.parse_args()
    from hub import segments
    runs = json.loads(Path(a.runs).read_text())
    specs = list(runs)
    episodes = sorted({i for v in runs.values() for i in v})
    total = disputed = 0
    for i in episodes:
        if a.episode and not i.startswith(a.episode):
            continue
        rec = json.loads((ROOT / a.data / f"{i}.json").read_text())
        cues = segments.read_cues((ROOT / a.data / f"{i}.vtt").read_text())
        have = [s for s in specs if i in runs[s]]
        cs = clusters({s: runs[s][i] for s in have})
        print(f"\n######## {i[:8]} {rec.get('title', '')[:70]} ({rec.get('duration_s', 0) / 60:.0f} min, "
              f"{len(cs)} clusters, models: {len(have)})")
        for k, (start, end, members) in enumerate(cs):
            total += 1
            unanimous = len(members) == len(have) and \
                max(m["start"] for m in members.values()) - min(m["start"] for m in members.values()) <= a.tol and \
                max(m["end"] for m in members.values()) - min(m["end"] for m in members.values()) <= a.tol
            if unanimous:
                print(f"== C{k} {mmss(start)}-{mmss(end)} unanimous: {next(iter(members.values()))['label'][:80]}")
                continue
            disputed += 1
            print(f"\n== C{k} {mmss(start)}-{mmss(end)} DISPUTED ({len(members)}/{len(have)} models)")
            for s in have:
                m = members.get(s)
                print(f"   {s:34s} " + (f"cues {m['first']}-{m['last']} {mmss(m['start'])}-{mmss(m['end'])} "
                                         f"{m['kind']}: {m['label'][:60]}" if m else "-"))
            for j, c in enumerate(cues):
                if start - a.context <= c[0] <= end + a.context:
                    print(f"   [{j}] {mmss(c[0])} {c[2] or '?'}: {c[3]}")
    print(f"\n{total} clusters, {disputed} disputed")


if __name__ == "__main__":
    main()
