"""CUDA streams for capturing CUDA graphs, each one no other capture got.

cuBLAS's workspace is per (handle, stream), and a captured graph keeps the one it was captured with. Graphs that
can replay at the same time must not share one: on a shared workspace hub/bulk.py's Mimi graph and TDT search
graphs hung the GPU, split-K GEMMs waiting on each other's counters (scripts/graph_hang_repro.py; with
CUBLAS_WORKSPACE_CONFIG=:0:0 they didn't). torch.cuda.Stream() hands out a pool's 32 streams round-robin, so two
of them can be the same stream; this skips the ones already taken.
"""
import threading

import torch

_taken, _lock = set(), threading.Lock()


def capture_stream():
    with _lock:
        for _ in range(64):
            s = torch.cuda.Stream()
            if s.cuda_stream not in _taken:
                _taken.add(s.cuda_stream)
                return s
    raise RuntimeError("no CUDA stream left that no other graph was captured on")
