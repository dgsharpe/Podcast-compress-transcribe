"""Ground truth for ad / promo detection, and every model scored against it.

The truth: the breaks all models agreed on (scripts/ads_disagree.py: unanimous clusters, the models' median cue
range), plus a judgment, from reading the transcript, of each disputed cluster (JUDGED below: a cue range, or None
for not a skip). Written to data/ads_truth.json; then each spec in the runs file is scored: breaks found (half of a
true break covered), missed, extra (candidate breaks overlapping no true break), and IoU of the skipped seconds.

    python scripts/ads_truth.py runs.json
    python scripts/ads_truth.py --score other_runs.json     (score against the saved data/ads_truth.json)
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# (episode id prefix, cluster index): (first cue, last cue) or None. Judged 2026-11-02 against hub/segments.PROMPT:
# ads (incl. sponsored news inserts like "Bloomberg Tech Minute brought to you by ChatGPT" and the sponsor teasers
# that open a break) and the show's self-promotion are skips; network idents, credits, short "email us"
# invitations and on-topic chat about the hosts' other shows aren't.
JUDGED = {
    ("1fc05d70", 0): (0, 33),       # Barclays Brief promo, Tech Minute (ChatGPT), Comcast; not the Bloomberg ident
    ("1fc05d70", 3): (936, 970),    # newsletter / Discord / review / ad-free pitch, then Hex, Delta, Comcast
    ("397b1417", 3): (314, 317),    # Vox membership; the listener call-in request before it isn't
    ("739ac934", 3): None,          # hosts chatting about next week's Decoder guest and coverage
    ("739ac934", 4): (2084, 2090),  # The Verge subscription pitch
    ("885a00a4", 0): (0, 13),       # Anthropic, Monday.com
    ("885a00a4", 2): (743, 768),    # Machine Gods promo from its cold open, Choiceology, Toast
    ("885a00a4", 3): None,          # a 5.5 s "rate and review" in the outro
    ("885a00a4", 4): (923, 945),    # YouTube / Vox membership pitch, then post-roll Anthropic and Fetch ads
    ("a79cf4d0", 3): (835, 839),    # Galaxy Brain / Atlantic subscription; not "that's it for us"
    ("add43a5b", 0): (0, 33),       # Economy Scale teaser, Tech Minute (ChatGPT), Cincinnati Insurance
    ("add43a5b", 3): (1024, 1032),  # Discord / newsletter / review / ad-free pitch
    ("caf2226c", 0): None,          # the guest on how to apply for funding: content
    ("caf2226c", 1): None,          # the same, in the closing
    ("cd2d488e", 0): (0, 30),       # Economy Scale teaser, Tech Minute (ChatGPT), Boomi
    ("cd2d488e", 1): (304, 366),    # Economy Scale promo, Tech Minute, Chase, Virgin Voyages
    ("cd2d488e", 2): (706, 753),    # Tech Minute, Chase Sapphire, Virgin Voyages, AutoTrader
    ("cd2d488e", 3): (1130, 1137),  # Discord / newsletter / review / ad-free pitch
}


def overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def main():
    import ads_bench
    import ads_disagree
    from hub import segments
    if sys.argv[1] == "--score":
        runs = json.loads(Path(sys.argv[2]).read_text())
        return score(json.loads((ROOT / "data/ads_truth.json").read_text()), runs)
    runs = json.loads(Path(sys.argv[1]).read_text())
    specs = list(runs)
    truth, judged_used = {}, set()
    for i in sorted({i for v in runs.values() for i in v}):
        cues = segments.read_cues((ROOT / "hub_data" / f"{i}.vtt").read_text())
        times = lambda f, l: (cues[f][0], cues[l + 1][0] if l + 1 < len(cues) else cues[l][1])  # noqa: E731
        have = [s for s in specs if i in runs[s]]
        breaks = []
        for k, (start, end, members) in enumerate(ads_disagree.clusters({s: runs[s][i] for s in have})):
            key = (i[:8], k)
            if key in JUDGED:
                judged_used.add(key)
                if JUDGED[key] is not None:
                    f, l = JUDGED[key]
                    breaks.append({"first": f, "last": l, "judged": True})
            elif len(members) == len(have):
                breaks.append({"first": round(statistics.median(m["first"] for m in members.values())),
                               "last": round(statistics.median(m["last"] for m in members.values())),
                               "judged": False})
            else:
                raise SystemExit(f"disputed cluster {key} has no judgment")
        for b in breaks:
            b["start"], b["end"] = (round(x, 2) for x in times(b["first"], b["last"]))
        truth[i] = breaks
    if missing := set(JUDGED) - judged_used:
        raise SystemExit(f"judgments for clusters that don't exist: {sorted(missing)}")
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data/ads_truth.json").write_text(json.dumps(truth, indent=1))
    print(f"truth: {sum(len(v) for v in truth.values())} breaks in {len(truth)} episodes "
          f"({sum(b['judged'] for v in truth.values() for b in v)} judged) -> data/ads_truth.json")
    score(truth, runs)


def score(truth, runs):
    import ads_bench
    n_true = sum(len(v) for v in truth.values())
    for spec in runs:
        found = extra = 0
        inter = either = 0.0
        for i, tb in truth.items():
            t = [(b["start"], b["end"]) for b in tb]
            g = [(s["start"], s["end"]) for s in runs[spec].get(i, [])]
            found += sum(any(overlap(x, y) >= 0.5 * (x[1] - x[0]) for y in g) for x in t)
            extra += sum(all(overlap(x, y) == 0 for x in t) for y in g)
            inter += sum(overlap(x, y) for x in t for y in g)
            either += ads_bench.union_len(t + g)
        print(f"  {spec:34s} found {found}/{n_true}  extra {extra}  IoU {inter / either:.3f}")


if __name__ == "__main__":
    main()
