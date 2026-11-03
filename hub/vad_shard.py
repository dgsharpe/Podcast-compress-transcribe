"""Silero VAD in shards across processes (MIMI_HUB_VAD_SHARDS): the VAD is one ONNX call per 32 ms frame, each on the
previous one's state, ~55 us of work a call: 4.6 s for a 43.8 min episode on the server, ~12 s on a Modal H100's
CPUs, the slowest part of one episode there. Here shard k (SHARD_FRAMES frames) runs in a worker process from a
zero state WARM_FRAMES frames early, and those frames are dropped. Not identical frame by frame (the state has a
long memory): on the reference episode 12 of 550 speech segments moved a little, the transcript was word for word
the same (902 cues; with 10 s of warm-up 99.86% of words; scripts/vad_shard_asr_check.py).

Kept small: the workers import this module only.
"""
import numpy as np

HOP, CTX, SR = 512, 64, 16000
SHARD_FRAMES = 5000  # 160 s
WARM_FRAMES = 1250  # 40 s

_sess = None


def _init():
    global _sess
    import onnx_asr
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = so.inter_op_num_threads = 1
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    _sess = onnx_asr.load_vad("silero", sess_options=so)._model


def _run(frames, skip):
    """Probabilities of [frames] (n, 576) from a zero state, without the first [skip]."""
    state, srs, out = np.zeros((2, 1, 128), np.float32), np.array([SR], np.int64), np.empty(len(frames), np.float32)
    for k, f in enumerate(frames):
        o, state = _sess.run(["output", "stateN"], {"input": f[None], "state": state, "sr": srs})
        out[k] = o[0, 0]
    return out[skip:]


def pool(workers):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    return ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_init)


def n_frames(n):
    """How many VAD frames hub/mimi_hub._StreamPrep makes of n samples: the first (left-padded), w[448 + 512 k:][:576]
    while it fits, and a right-padded last one."""
    return 0 if n < HOP else 1 + (n // HOP - 1) + (1 if n % HOP else 0)


def frames(w, n, i0, i1):
    """Frames i0..i1-1 of w[:n] (n: the final length if frame i1 - 1 may be the padded last one)."""
    out = np.empty((i1 - i0, CTX + HOP), np.float32)
    full = n // HOP - 1  # frames 1..full are whole windows
    for j, i in enumerate(range(i0, i1)):
        if i == 0:
            out[j] = np.pad(w[:HOP], (CTX, 0))
        elif i <= full:
            a = HOP - CTX + HOP * (i - 1)
            out[j] = w[a:a + CTX + HOP]
        else:  # the padded last one
            last = n % HOP
            out[j] = np.pad(w[n - last - CTX:n], (0, HOP - last))
    return out


def samples_for(i):
    """Samples frame i (whole) needs."""
    return HOP if i == 0 else HOP * (i + 1)
