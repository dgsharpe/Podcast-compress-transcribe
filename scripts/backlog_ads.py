"""Ad detection for a backlog converted in bulk (scripts/modal_backlog.py), as the hub would have done it: each
episode's transcript through hub/segments.py, then its .segments.json and its record updated. Episodes whose record
already says segments_state "ready" are skipped, so it can be rerun.

    python scripts/backlog_ads.py RESULTS.json [--data hub_data] [--segmenter claude:claude-haiku-5-5] [--threads 8]
        (this hub's library, e.g. after scripts/import_modal_library.py; the agent CLI's own login, no API key)
    OPENROUTER_API_KEY=... uvx --python 3.12 --with modal python scripts/backlog_ads.py RESULTS.json --volume
        mimi-hub-data --segmenter openrouter:anthropic/claude-haiku-5.5
        (a Modal hub's library, read and written through Modal's API without starting a container)

RESULTS: modal_backlog.py's results ([{"uuid", ...}]). Haiku 5.5 through the claude CLI found all 98 judged breaks of
scripts/ads_truth.py's 24 episodes.

    --recheck 240 [--segmenter claude:claude-opus-5-5]   redo, with that model, episodes (over 20 min) whose found
        segments include one over 240 s: Haiku now and then ran a break on through the content after it (a Weeds
        episode: a 40 s break marked as the last 17 min), on 2 of the first 559 backlog episodes
"""
import argparse
import fcntl
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class Local:
    """This hub's data directory; records changed under the hub's own lock file (hub/mimi_hub.py _RecordLock)."""

    def __init__(self, data):
        self.data = Path(data)

    def read(self, name):
        return (self.data / name).read_bytes()

    def update(self, i, seg_json, change):
        tmp = self.data / f"{i}.segments.json.tmp"
        tmp.write_text(seg_json)
        tmp.replace(self.data / f"{i}.segments.json")
        fd = os.open(self.data / ".records.lock", os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            rec = json.loads(self.read(f"{i}.json"))
            rec.update(change)
            tmp = self.data / f"{i}.json.tmp"
            tmp.write_text(json.dumps(rec, indent=1))
            tmp.replace(self.data / f"{i}.json")
        finally:
            os.close(fd)

    def flush(self):
        pass


class Volume:
    """A Modal hub's data Volume: written in one batch at the end."""

    def __init__(self, name):
        import modal
        import tempfile
        self.vol, self.out, self.done = modal.Volume.from_name(name, version=2), Path(tempfile.mkdtemp()), []

    def read(self, name):
        return b"".join(self.vol.read_file(name))

    def update(self, i, seg_json, change):
        rec = json.loads(self.read(f"{i}.json"))
        rec.update(change)
        (self.out / f"{i}.segments.json").write_text(seg_json)
        (self.out / f"{i}.json").write_text(json.dumps(rec, indent=1))
        self.done.append(i)

    def flush(self):
        with self.vol.batch_upload(force=True) as b:
            for i in self.done:
                b.put_file(str(self.out / f"{i}.segments.json"), f"/{i}.segments.json")
                b.put_file(str(self.out / f"{i}.json"), f"/{i}.json")


def main():
    from hub import segments
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--data", default=os.environ.get("MIMI_HUB_DATA", str(ROOT / "hub_data")))
    ap.add_argument("--volume", default=None, help="a Modal hub's data Volume instead of --data")
    ap.add_argument("--segmenter", default="claude:claude-haiku-5-5")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--recheck", type=float, default=0, help="redo episodes with a segment over this many seconds")
    a = ap.parse_args()
    lib = Volume(a.volume) if a.volume else Local(a.data)
    ids = [r["uuid"] for f in a.results for r in json.loads(Path(f).read_text()) if not r.get("error")]
    lock, stats = threading.Lock(), {"ads": 0, "skipped": 0, "failed": 0, "skippable_s": 0.0}

    def one(i):
        try:
            rec = json.loads(lib.read(f"{i}.json"))
        except (FileNotFoundError, Exception) as e:  # noqa: BLE001 - not in this library
            with lock:
                stats["failed"] += 1
            print(f"{i}: no record ({type(e).__name__})", flush=True)
            return
        if a.recheck:
            found = json.loads(lib.read(f"{i}.segments.json"))["segments"] if rec.get("segments_state") == "ready" else []
            if rec.get("duration_s", 0) < 20 * 60 or not any(x["end"] - x["start"] > a.recheck for x in found):
                with lock:
                    stats["skipped"] += 1
                return
            before = [(round(x["start"]), round(x["end"])) for x in found]
        elif rec.get("segments_state") == "ready":
            with lock:
                stats["skipped"] += 1
            return
        t0 = time.time()
        try:
            res = segments.detect(segments.read_cues(lib.read(f"{i}.vtt").decode()), a.segmenter, log=lambda m: None)
        except Exception as e:  # noqa: BLE001 - left failed; the app's key can have the hub redo it
            with lock:
                stats["failed"] += 1
            print(f"{i}: {type(e).__name__}: {e}"[:300], flush=True)
            return
        skippable = round(sum(s["end"] - s["start"] for s in res["segments"]), 1)
        if a.recheck:
            print(f"{i} {rec.get('title', '')[:50]}: {before} -> "
                  f"{[(round(x['start']), round(x['end'])) for x in res['segments']]}", flush=True)
        lib.update(i, json.dumps(res, indent=1), dict(
            segments_state="ready", segments=len(res["segments"]), segments_model=res["model"], skippable_s=skippable,
            segments_error=None, updated=time.time(), timing={**(rec.get("timing") or {}),
                                                              "segments_backlog": round(time.time() - t0, 1)}))
        with lock:
            stats["ads"] += 1
            stats["skippable_s"] += skippable
            n = stats["ads"] + stats["skipped"] + stats["failed"]
            if n % 50 == 0:
                print(f"{n}/{len(ids)} {json.dumps({k: round(v) for k, v in stats.items()})}", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(a.threads) as ex:
        list(ex.map(one, ids))
    lib.flush()
    stats["skippable_h"] = round(stats.pop("skippable_s") / 3600, 1)
    print(f"ADS {json.dumps(stats)} in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
