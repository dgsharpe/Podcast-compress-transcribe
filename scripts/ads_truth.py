"""Ground truth for ad / promo detection, and every model scored against it.

The truth: the breaks all models agreed on (scripts/ads_disagree.py: unanimous clusters, the models' median cue
range), plus a judgment, from reading the transcript, of each disputed cluster (JUDGED below: a cue range, or None
for not a skip). Written to data/ads_truth.json; then each spec in the runs file is scored: breaks found (half of a
true break covered), missed, extra (candidate breaks overlapping no true break), and IoU of the skipped seconds.

    python scripts/ads_truth.py runs.json
    python scripts/ads_truth.py --score other_runs.json     (score against the saved data/ads_truth.json)
    python scripts/ads_truth.py --fresh runs.json / --fresh --score runs.json
        (the episodes in data/fresh_hub, scripts/fresh_episodes.py: JUDGED_FRESH, data/ads_truth_fresh.json)
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


# The same for the fresh episodes (data/fresh_hub: scripts/fresh_episodes.py, shows the v2 prompt wasn't written
# from), judged 2026-11-02 by the same rules; a cluster can hold several breaks (a list).
JUDGED_FRESH = {
    ("12f7b588", 3): (1110, 1111),  # Freakonomics' video on YouTube / Instagram; not the next-episode preview
    ("12f7b588", 4): (1144, 1187),  # post-roll Ozempic, Mazda, Nurtec, Southwest
    ("41ed1f09", 3): (1504, 1518),  # live-event and Incognito Mode pitch; not the host's tip for a friend's show
    ("41ed1f09", 4): [(1541, 1549), (1552, 1569)],  # Incognito Mode paid feed and merch; post-roll Vistaprint
    ("96e6ecb4", 2): (666, 669),    # the producer's other show, Hyperfixed
    ("96e6ecb4", 3): (695, 717),    # Cohesity, SimpliSafe, SiriusXM Podcast Plus pitch
    ("b1048773", 2): (790, 820),    # FM, iREN, OutSystems; not "we're going to go to break"
    ("c031ca05", 3): (1926, 1928),  # post-roll Pfizer
    ("db0b7a74", 1): (493, 583),    # host-read Hyundai (from its award banter), T-Mobile, Pluto TV
    ("db0b7a74", 3): (1553, 1574),  # rate and review / SiriusXM / subscribe pitch, then Old Navy, SimpliSafe
    ("e3f77b52", 0): (0, 26),       # Happy Hour trailer, Harvey, Deepgram; not the cold-open quote
    ("e3f77b52", 1): None,          # the hosts chatting about their event
    ("e3f77b52", 2): [(274, 296), (339, 390)],  # tour and Pivot Post pitch (not the book banter after); ads
    ("e3f77b52", 3): [(859, 914), (1135, 1173)],  # Pivot Post promo, DeleteMe, Today Explained; Vox / Criminal promos
    ("e3f77b52", 4): (1318, 1330),  # YouTube / Pivot Post plug and tour invitation
    ("e3f77b52", 5): None,          # "follow Pivot" inside the credits
    ("f203f5ef", 0): (348, 389),    # Capital One, Bombas, Ear Hustle; not "we'll be right back"
    ("f203f5ef", 1): (599, 626),    # BetterHelp, Ear Hustle
    ("f203f5ef", 2): (1125, 1143),  # ElevenLabs, Mercedes-Benz
}


def overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def main():
    import ads_bench
    import ads_disagree
    from hub import segments
    args = sys.argv[1:]
    fresh = args[:1] == ["--fresh"]
    args = args[1:] if fresh else args
    data, judged = ("data/fresh_hub", JUDGED_FRESH) if fresh else ("hub_data", JUDGED)
    truth_file = ROOT / ("data/ads_truth_fresh.json" if fresh else "data/ads_truth.json")
    if args[0] == "--score":
        return score(json.loads(truth_file.read_text()), json.loads(Path(args[1]).read_text()))
    runs = json.loads(Path(args[0]).read_text())
    specs = list(runs)
    truth, judged_used = {}, set()
    for i in sorted({i for v in runs.values() for i in v}):
        cues = segments.read_cues((ROOT / data / f"{i}.vtt").read_text())
        times = lambda f, l: (cues[f][0], cues[l + 1][0] if l + 1 < len(cues) else cues[l][1])  # noqa: E731
        have = [s for s in specs if i in runs[s]]
        breaks = []
        for k, (start, end, members) in enumerate(ads_disagree.clusters({s: runs[s][i] for s in have})):
            key = (i[:8], k)
            if key in judged:
                judged_used.add(key)
                ranges = judged[key] if isinstance(judged[key], list) else [judged[key]] * (judged[key] is not None)
                for f, l in ranges:
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
    if missing := set(judged) - judged_used:
        raise SystemExit(f"judgments for clusters that don't exist: {sorted(missing)}")
    (ROOT / "data").mkdir(exist_ok=True)
    truth_file.write_text(json.dumps(truth, indent=1))
    print(f"truth: {sum(len(v) for v in truth.values())} breaks in {len(truth)} episodes "
          f"({sum(b['judged'] for v in truth.values() for b in v)} judged) -> {truth_file.relative_to(ROOT)}")
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
