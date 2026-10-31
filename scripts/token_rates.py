"""Actual storage rates: Opus file sizes, and for token codecs the nominal rate, the
per-codebook unigram entropy (what a static arithmetic/rANS coder would reach), and
xz-compressed packed tokens. Writes outputs/eval/rates.json.
"""

import json
import lzma
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"


def main():
    res = {}
    for p in sorted((OUT / "opus").glob("opus_*k.opus")):
        dur = 2625.28
        res[p.stem] = dict(kbps_file=p.stat().st_size * 8 / dur / 1000)
    for t in sorted(OUT.glob("*/timing.json")):
        info = json.loads(t.read_text())
        z = np.load(t.parent / "tokens.npz")
        segs = [z[k] for k in z.files]
        # codes as (K, frames) for every codec
        segs = [s if s.shape[0] < s.shape[1] else s.T for s in segs]
        allc = np.concatenate(segs, axis=1)
        K, F = allc.shape
        dur = info["audio_s"]
        H = 0.0
        for k in range(K):
            _, c = np.unique(allc[k], return_counts=True)
            q = c / c.sum()
            H += -(q * np.log2(q)).sum()
        bits_per_code = int(np.ceil(np.log2(info.get("codebook_size") or info.get("codebook_size_inferred"))))
        packed = allc.T.astype(np.uint16).tobytes()  # frame-major, 16 bit per code
        xz = len(lzma.compress(packed, preset=9 | lzma.PRESET_EXTREME))
        res[info["variant"]] = dict(
            kbps_nominal=info["nominal_kbps"], frames=F, codebooks=K, bits_per_code=bits_per_code,
            kbps_unigram_entropy=H * F / dur / 1000, kbps_xz=xz * 8 / dur / 1000,
            MB_per_hour_nominal=info["nominal_kbps"] * 3600 / 8 / 1000,
            MB_per_hour_entropy=H * F / dur * 3600 / 8 / 1e6)
    for k, v in res.items():
        if "kbps_file" in v:
            v["MB_per_hour"] = v["kbps_file"] * 3600 / 8 / 1000
        print(k, {a: round(b, 3) if isinstance(b, float) else b for a, b in v.items()})
    (OUT / "eval/rates.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
