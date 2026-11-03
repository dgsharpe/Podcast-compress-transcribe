"""Ad / promo detection (hub/segments.py) with other models and settings, on episodes the hub has already
labelled: time per episode, and how well each agrees with the hub's labels (hub_data/<id>.segments.json, from
codex:gpt-6.1-sol at high effort, which matched hand labels on six episodes).

Agreement: skipped seconds in common / in either (IoU) over all episodes, and per reference break whether a
candidate break covers half of it (found) plus candidate breaks covering nothing of the reference (extra).

    python scripts/ads_bench.py [--episodes 6] [--prompt v2] [--batch N] [spec ...]
        (spec as MIMI_HUB_SEGMENTER, e.g. codex:gpt-6.1-sol+low; --batch: N episodes per request, hub/segments.py's
        detect_many; an episode's time is then its request's divided by N)
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT = ["codex:gpt-6.1-sol", "codex:gpt-6.1-sol+low", "codex:gpt-6.1-sol+low+fast", "codex:gpt-6.1-terra+low",
           "codex:gpt-5.6-terra+low", "codex:gpt-5.6-luna+low", "claude:claude-haiku-5-5"]


def overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def union_len(rs):
    total, end = 0.0, None
    for s, e in sorted(rs):
        if end is None or s > end:
            total, end = total + e - s, e
        elif e > end:
            total, end = total + e - end, e
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="*", default=DEFAULT)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--parallel", type=int, default=3, help="episodes at once per spec")
    ap.add_argument("--save", default=None, help="JSON file: every spec's segments per episode (merged into it)")
    ap.add_argument("--prompt", default="v1", choices=["v1", "v2"])
    ap.add_argument("--batch", type=int, default=1, help="episodes per request (v2 rules)")
    ap.add_argument("--label", default="", help="appended to the spec's name in the output / saved file")
    ap.add_argument("--data", default="hub_data", help="the episodes' directory (fresh ones have no hub labels)")
    ap.add_argument("--truth", default=None, help="only the episodes of this ground truth (scripts/ads_truth.py)")
    a = ap.parse_args()
    from hub import segments
    data = ROOT / a.data
    eps = []
    for vtt in sorted(data.glob("*.vtt")):
        i = vtt.name.split(".")[0]
        rec = json.loads((data / f"{i}.json").read_text()) if (data / f"{i}.json").exists() else {}
        labels = data / f"{i}.segments.json"
        ref = [(s["start"], s["end"]) for s in json.loads(labels.read_text())["segments"]] if labels.exists() else None
        if ref is not None or a.data != "hub_data":  # hub_data: the labelled ones only, as before
            eps.append((i, rec.get("title", "")[:40], segments.read_cues(vtt.read_text()), ref or []))
    if a.truth:  # (hub_data also holds backlogs now, with labels of their own)
        keep = set(json.loads(Path(a.truth).read_text()))
        eps = [e for e in eps if e[0] in keep]
    eps = sorted(eps, key=lambda e: -len(e[3]))[:a.episodes]  # the ones with the most breaks first
    print(f"{len(eps)} episodes: " + "; ".join(f"{t} ({len(r)} breaks)" for _, t, _, r in eps), flush=True)
    for spec in a.specs:
        def group(g):
            t = time.time()
            try:
                if a.batch > 1:
                    res = segments.detect_many([e[2] for e in g], spec, log=lambda m: None)
                else:
                    res = [segments.detect(g[0][2], spec, log=lambda m: None, prompt=a.prompt)]
            except Exception as e:  # noqa: BLE001
                return [{"id": ep[0], "error": f"{type(e).__name__}: {str(e)[:200]}"} for ep in g]
            el = round((time.time() - t) / len(g), 1)
            return [{"id": ep[0], "s": el, "ref": ep[3], "got": [(s["start"], s["end"]) for s in r["segments"]],
                     "segments": r["segments"]} for ep, r in zip(g, res, strict=True)]
        groups = [eps[k:k + a.batch] for k in range(0, len(eps), a.batch)]
        t_all = time.time()
        with ThreadPoolExecutor(a.parallel) as ex:
            rows = [r for rs in ex.map(group, groups) for r in rs]
        wall = round(time.time() - t_all, 1)
        name = spec + a.label
        ok = [r for r in rows if "error" not in r]
        inter = sum(sum(overlap(x, y) for x in r["ref"] for y in r["got"]) for r in ok)
        either = sum(union_len(r["ref"] + r["got"]) for r in ok)
        found = sum(sum(any(overlap(x, y) >= 0.5 * (x[1] - x[0]) for y in r["got"]) for x in r["ref"]) for r in ok)
        refs = sum(len(r["ref"]) for r in ok)
        extra = sum(sum(all(overlap(x, y) == 0 for x in r["ref"]) for y in r["got"]) for r in ok)
        times = sorted(r["s"] for r in ok)
        if a.save:
            saved = json.loads(Path(a.save).read_text()) if Path(a.save).exists() else {}
            saved[name] = {r["id"]: r["segments"] for r in ok}
            Path(a.save).write_text(json.dumps(saved, indent=1))
        print("ADS " + json.dumps({
            "spec": name, "episodes": len(ok), "requests": len(groups), "wall_s": wall,
            "errors": [r["error"] for r in rows if "error" in r][:2],
            "iou": round(inter / either, 3) if either else None, "found": f"{found}/{refs}", "extra": extra,
            "median_s": times[len(times) // 2] if times else None, "max_s": times[-1] if times else None}),
            flush=True)


if __name__ == "__main__":
    main()
