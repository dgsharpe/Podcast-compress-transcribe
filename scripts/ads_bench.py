"""Ad / promo detection (hub/segments.py) with other models and settings, on episodes the hub has already
labelled: time per episode, and how well each agrees with the hub's labels (hub_data/<id>.segments.json, from
codex:gpt-6.1-sol at high effort, which matched hand labels on six episodes).

Agreement: skipped seconds in common / in either (IoU) over all episodes, and per reference break whether a
candidate break covers half of it (found) plus candidate breaks covering nothing of the reference (extra).

    python scripts/ads_bench.py [--episodes 6] [spec ...]       (spec as MIMI_HUB_SEGMENTER, e.g. codex:gpt-6.1-sol+low)
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
    a = ap.parse_args()
    from hub import segments
    data = ROOT / "hub_data"
    eps = []
    for p in sorted(data.glob("*.segments.json")):
        i = p.name.split(".")[0]
        vtt = data / f"{i}.vtt"
        rec = json.loads((data / f"{i}.json").read_text()) if (data / f"{i}.json").exists() else {}
        if vtt.exists():
            eps.append((i, rec.get("title", "")[:40], segments.read_cues(vtt.read_text()),
                        [(s["start"], s["end"]) for s in json.loads(p.read_text())["segments"]]))
    eps = sorted(eps, key=lambda e: -len(e[3]))[:a.episodes]  # the ones with the most breaks first
    print(f"{len(eps)} episodes: " + "; ".join(f"{t} ({len(r)} breaks)" for _, t, _, r in eps), flush=True)
    for spec in a.specs:
        def one(ep):
            i, title, cues, ref = ep
            t = time.time()
            try:
                got = [(s["start"], s["end"]) for s in segments.detect(cues, spec, log=lambda m: None)["segments"]]
            except Exception as e:  # noqa: BLE001
                return {"id": i, "error": f"{type(e).__name__}: {str(e)[:200]}"}
            return {"id": i, "s": round(time.time() - t, 1), "ref": ref, "got": got}
        with ThreadPoolExecutor(a.parallel) as ex:
            rows = list(ex.map(one, eps))
        ok = [r for r in rows if "error" not in r]
        inter = sum(sum(overlap(x, y) for x in r["ref"] for y in r["got"]) for r in ok)
        either = sum(union_len(r["ref"] + r["got"]) for r in ok)
        found = sum(sum(any(overlap(x, y) >= 0.5 * (x[1] - x[0]) for y in r["got"]) for x in r["ref"]) for r in ok)
        refs = sum(len(r["ref"]) for r in ok)
        extra = sum(sum(all(overlap(x, y) == 0 for x in r["ref"]) for y in r["got"]) for r in ok)
        times = sorted(r["s"] for r in ok)
        print("ADS " + json.dumps({
            "spec": spec, "episodes": len(ok), "errors": [r["error"] for r in rows if "error" in r][:2],
            "iou": round(inter / either, 3) if either else None, "found": f"{found}/{refs}", "extra": extra,
            "median_s": times[len(times) // 2] if times else None, "max_s": times[-1] if times else None}),
            flush=True)


if __name__ == "__main__":
    main()
