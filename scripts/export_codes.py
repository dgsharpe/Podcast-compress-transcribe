"""Export a token codec's codes as raw int16, frame-major [F][32], for phone/mimi_dec.c.

usage: python scripts/export_codes.py outputs/mimi_q32/tokens.npz OUT.i16
Segments (30 s each, coded independently) are concatenated in order.
"""

import sys

import numpy as np

z = np.load(sys.argv[1])
segs = [z[f"arr_{i}"] for i in range(len(z.files))]
c = np.concatenate(segs, axis=1)  # (32, F)
assert c.shape[0] == 32, c.shape
np.ascontiguousarray(c.T.astype("<i2")).tofile(sys.argv[2])
print(f"{c.shape[1]} frames ({c.shape[1] / 12.5:.1f} s) -> {sys.argv[2]}")
