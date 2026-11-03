"""Copy episodes from a Modal hub's library (hub/modal_app.py's data Volume) into this hub's, e.g. a backlog
converted there by scripts/modal_backlog.py: each episode's .mimi, .vtt, .segments.json and record, read from the
Volume directly (no container started), the record written last. Episodes this hub already has are skipped.

    uvx --python 3.12 --with modal python scripts/import_modal_library.py RESULTS.json [...] [--data hub_data]

RESULTS: modal_backlog.py's results ([{"uuid", ...}]); the episodes to copy.
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    import modal
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--data", default=os.environ.get("MIMI_HUB_DATA", str(ROOT / "hub_data")))
    ap.add_argument("--volume", default=os.environ.get("MIMI_MODAL_APP", "mimi-hub") + "-data")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    data, vol = Path(a.data), modal.Volume.from_name(a.volume, version=2)
    ids = [r["uuid"] for f in a.results for r in json.loads(Path(f).read_text()) if not r.get("error")]
    have = {e.path.lstrip("/") for e in vol.listdir("/")}
    stats = {"copied": 0, "skipped": 0, "missing": 0, "bytes": 0}

    def one(i):
        if (data / f"{i}.json").exists():
            stats["skipped"] += 1
            return
        if not {f"{i}.json", f"{i}.mimi", f"{i}.vtt"} <= have:
            stats["missing"] += 1
            return
        for ext in ("mimi", "vtt", "segments.json", "json"):  # the record last
            if f"{i}.{ext}" not in have:
                continue
            body = b"".join(vol.read_file(f"{i}.{ext}"))
            if ext == "json":
                rec = json.loads(body)
                rec.update(imported_from=f"modal:{a.volume}", updated=time.time())
                body = json.dumps(rec, indent=1).encode()
            tmp = data / f"{i}.{ext}.import"
            tmp.write_bytes(body)
            tmp.replace(data / f"{i}.{ext}")
            stats["bytes"] += len(body)
        stats["copied"] += 1

    t0 = time.time()
    with ThreadPoolExecutor(a.threads) as ex:
        list(ex.map(one, ids))
    stats["mb"] = round(stats.pop("bytes") / 1e6)
    print(f"IMPORT {json.dumps(stats)} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
