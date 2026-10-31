"""Convert litert-community/Mimi's mimi_rvq.bin (float32) to the decoder-only fp16 table file.

mimi_rvq.bin   = sem_Win[256,512], aco_Win[256,512], sem_Wout[512,256], aco_Wout[512,256], sem_CB[2048,256],
                 31 x aco_CB[2048,256]  (float32 LE; the *_Win projections are only used for encoding)
mimi_rvq_f16.bin = sem_Wout, aco_Wout, sem_CB, aco_CB  (float16 LE), 34 MB instead of 69 MB.
fp16 tables change decoded audio by -72 dB relative (71.7 dB SNR vs fp32 tables, 60 s test; bakeoff notes).

usage: python scripts/rvq_fp16.py IN/mimi_rvq.bin OUT/mimi_rvq_f16.bin
"""

import sys

import numpy as np

D, H = 256, 512
w = np.fromfile(sys.argv[1], "<f4")
assert w.size == 4 * D * H + 32 * 2048 * D, w.size
w[2 * D * H:].astype("<f2").tofile(sys.argv[2])
print(f"{sys.argv[2]}: {(w.size - 2 * D * H) * 2 / 1e6:.1f} MB")
