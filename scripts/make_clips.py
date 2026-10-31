"""Cut matching listening excerpts from the reference and every decoded variant.

usage: python scripts/make_clips.py LABEL:START_S:DUR_S [...]
Writes outputs/listening/<LABEL>/<variant>.wav (24 kHz mono 16-bit PCM, loudness untouched) and
outputs/listening/index.html, a local page with one player per variant (shuffled
order + hidden names button, for rough blind listening).
"""

import base64
import json
import random
import sys
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from eval import SR, variants  # noqa: E402

L = ROOT / "outputs/listening"


def main(specs):
    V = variants()
    page = {}
    for spec in specs:
        label, t0, dur = spec.split(":")
        a, b = int(float(t0) * SR), int((float(t0) + float(dur)) * SR)
        d = L / label
        d.mkdir(parents=True, exist_ok=True)
        for name, path in V.items():
            y, sr = sf.read(path, dtype="float32", start=a, stop=b)
            sf.write(d / f"{name}.wav", y, sr, subtype="PCM_16")
        names = list(V)
        random.Random(label).shuffle(names)
        page[label] = dict(start=float(t0), names=names)
    def render(src):
        rows = []
        for label, p in page.items():
            rows.append(f"<h2>{label} <small>@ {p['start']:.0f} s</small></h2>")
            for i, n in enumerate(p["names"]):
                rows.append(f'<div><span class="k">#{i + 1}</span> <span class="n">{n}</span> '
                            f'<audio controls preload="metadata" src="{src(label, n)}"></audio></div>')
        return ("<!doctype html><meta charset=utf-8><title>codec listening</title>"
                "<style>body{font:15px sans-serif;max-width:720px;margin:2em auto}.n{display:none}"
                "body.show .n{display:inline}audio{vertical-align:middle;height:32px}</style>"
                "<button onclick=\"document.body.classList.toggle('show')\">show/hide names</button>"
                + "\n".join(rows))

    def embed(label, n):  # self-contained: no file:// subresource loads, works from any machine
        return "data:audio/wav;base64," + base64.b64encode((L / label / f"{n}.wav").read_bytes()).decode()

    (L / "index.html").write_text(render(lambda label, n: f"{label}/{n}.wav"))
    (L / "standalone.html").write_text(render(embed))
    (L / "key.json").write_text(json.dumps(page, indent=1))
    print(f"wrote {L}/index.html and {L}/standalone.html")


if __name__ == "__main__":
    main(sys.argv[1:])
