"""A podcast's episodes as a manifest for scripts/modal_backlog.py: each with its Pocket Casts episode UUID (the id the
app asks the hub for), audio URL, title and duration, from Pocket Casts' public podcast/full API. The podcast is
found by name through Apple's podcast search (pca.st/itunes/<id> maps an Apple id to Pocket Casts').

    python scripts/podcast_manifest.py "revolutions mike duncan" > revolutions.json
    python scripts/podcast_manifest.py "the weeds vox" --title "The Weeds" --before 2024-05-31 >> ...
    python scripts/podcast_manifest.py --apple-id 703889772 --after 2020-01-01

Prints the matches it chose from on stderr; --pick N takes another. Several podcasts: run it once each and join the
lists (python -c 'import json,sys; print(json.dumps(sum((json.load(open(f)) for f in sys.argv[1:]), [])))' a b).
Feeds that keep their back catalog behind a subscription (e.g. NYT's) list only their public episodes.
"""
import argparse
import gzip
import json
import os
import re
import sys
import urllib.parse
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (mimi-hub)"}
if not os.environ.get("SSL_CERT_FILE") and os.path.exists("/etc/ssl/certs/ca-certificates.crt"):
    os.environ["SSL_CERT_FILE"] = "/etc/ssl/certs/ca-certificates.crt"


def get(url, **kw):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60, **kw)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **kw):
        return None


def pocketcasts_uuid(apple_id):
    """Pocket Casts' podcast UUID for an Apple podcast id (pca.st redirects to pocketcasts.com/podcast/<slug>/<uuid>)."""
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        opener.open(urllib.request.Request(f"https://pca.st/itunes/{apple_id}", headers=UA), timeout=30)
    except urllib.error.HTTPError as e:
        loc = e.headers.get("Location", "")
    else:
        raise SystemExit(f"pca.st/itunes/{apple_id}: no redirect")
    m = re.search(r"/podcast/[^/]+/([0-9a-f-]{36})", loc)
    if not m:
        raise SystemExit(f"pca.st/itunes/{apple_id}: not a Pocket Casts podcast ({loc!r})")
    return m[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("search", nargs="?", help="the podcast's name (Apple's podcast search)")
    ap.add_argument("--apple-id", type=int)
    ap.add_argument("--pick", type=int, default=0, help="which search result (0: the first)")
    ap.add_argument("--after", default="", help="only episodes published on or after this date (YYYY-MM-DD)")
    ap.add_argument("--before", default="", help="only episodes published before this date")
    ap.add_argument("--title", default=None, help="the show's name in the episode titles (default: Pocket Casts')")
    a = ap.parse_args()
    apple_id = a.apple_id
    if apple_id is None:
        if not a.search:
            raise SystemExit("give a podcast name or --apple-id")
        q = urllib.parse.quote(a.search)
        found = json.load(get(f"https://itunes.apple.com/search?media=podcast&limit=8&term={q}"))["results"]
        for k, r in enumerate(found):
            print(f"{'>' if k == a.pick else ' '} {k}: {r['collectionName']} | {r.get('artistName')} | "
                  f"{r.get('trackCount')} episodes | apple id {r['collectionId']}", file=sys.stderr)
        if not found:
            raise SystemExit("no podcast found")
        apple_id = found[a.pick]["collectionId"]
    pod = pocketcasts_uuid(apple_id)
    raw = get(f"https://podcast-api.pocketcasts.com/podcast/full/{pod}").read()
    data = json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)["podcast"]
    show = a.title or data.get("title", "")
    eps = [{"uuid": e["uuid"], "url": e["url"], "title": f"{show}: {e['title']}", "published": e["published"],
            "duration_s": e.get("duration") or 0} for e in data["episodes"]
           if e.get("url") and (not a.after or e["published"][:10] >= a.after)
           and (not a.before or e["published"][:10] < a.before)]
    eps.sort(key=lambda e: e["published"])
    hours = sum(e["duration_s"] for e in eps) / 3600
    print(f"{data.get('title')}: {len(eps)} of {len(data['episodes'])} episodes, {hours:.0f} h "
          f"({eps[0]['published'][:10] if eps else '-'} .. {eps[-1]['published'][:10] if eps else '-'})",
          file=sys.stderr)
    print(json.dumps(eps, indent=1))


if __name__ == "__main__":
    main()
