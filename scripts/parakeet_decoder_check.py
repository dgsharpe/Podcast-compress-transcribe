"""hub/parakeet_native.NativeTdtDecoder vs onnx_asr's TDT decoding loop, on the same encoder outputs.

    .venv-rocm/bin/python scripts/parakeet_decoder_check.py [--seconds N]

Runs the hub's ASR (VAD, length buckets, native encoder) over the reference episode; every batch's encoder output
is decoded both ways. Reports segments with identical tokens / timestamps and each decoder's total time.
DECODER_GRAPH=1: the CUDA-graph search; DECODER_PRECAPTURE=1 also captures its graphs up front in the main thread
and replays them from a worker thread on its own stream, as hub/bulk.py does.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MIMI_HUB_DATA", "/tmp/mimi-decoder-check")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=0, help="only the first N s (0: all)")
    a = ap.parse_args()
    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from hub import mimi_hub, parakeet_native

    wav, sr = sf.read(ROOT / "data/ref/ref_24k.wav", dtype="float32")
    if a.seconds:
        wav = wav[: int(a.seconds * sr)]
    w16 = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    asr = mimi_hub.models()["asr"].asr
    path = parakeet_native.convert_decoder(mimi_hub.ASR_MODEL, mimi_hub.MODELS_DIR / f"{mimi_hub.ASR_MODEL}-hf")
    native = parakeet_native.NativeTdtDecoder(path, "cuda", asr._blank_idx, asr._vocab_size,
                                              max_symbols=asr._max_tokens_per_step)
    onnx = asr._decoding
    precapture = os.environ.get("DECODER_PRECAPTURE") == "1"
    if precapture:
        from concurrent.futures import ThreadPoolExecutor
        t = time.perf_counter()
        native.capture_graphs(0, max_batch=64)
        capture_s = time.perf_counter() - t
        worker, worker_stream = ThreadPoolExecutor(1), torch.cuda.Stream()

    def graphed(enc, lens):
        enc, lens = torch.from_numpy(np.ascontiguousarray(enc)).cuda(), torch.from_numpy(np.asarray(lens))
        if not precapture:
            return list(native.decode_graphed(enc, lens))

        def run():
            with torch.cuda.stream(worker_stream):
                worker_stream.wait_stream(torch.cuda.default_stream())
                return list(native.decode_graphed(enc, lens))
        return worker.submit(run).result()
    stats = {"segments": 0, "tokens_same": 0, "times_same": 0, "tokens": 0, "onnx_s": 0.0, "native_s": 0.0,
             "examples": []}

    def both(enc, lens, **kw):
        t = time.perf_counter()
        ref = list(onnx(enc, lens, **kw))
        stats["onnx_s"] += time.perf_counter() - t
        torch.cuda.synchronize()
        t = time.perf_counter()
        if os.environ.get("DECODER_GRAPH") == "1":  # the CUDA-graph search (hub/bulk.py)
            new = graphed(enc, lens)
        else:
            new = list(native(enc, lens, **kw))
        torch.cuda.synchronize()
        stats["native_s"] += time.perf_counter() - t
        for (rt, rs, _), (nt, ns, _) in zip(ref, new, strict=True):
            rt, rs = list(map(int, rt)), list(map(int, rs))
            stats["segments"] += 1
            stats["tokens"] += len(rt)
            stats["tokens_same"] += rt == nt
            stats["times_same"] += rs == ns
            if rt != nt and len(stats["examples"]) < 3:
                stats["examples"].append({"onnx": asr._decode_tokens(rt, None, None).text,
                                          "native": asr._decode_tokens(nt, None, None).text})
        return iter(ref)

    asr._decoding = both
    list(mimi_hub.models()["asr"].recognize(w16[: 60 * 16000], sample_rate=16000))  # warm-up
    for k in ("segments", "tokens_same", "times_same", "tokens"):
        stats[k] = 0
    stats["onnx_s"] = stats["native_s"] = 0.0
    stats["examples"] = []
    list(mimi_hub.models()["asr"].recognize(w16, sample_rate=16000))
    stats["audio_s"] = round(len(w16) / 16000, 1)
    stats["onnx_s"], stats["native_s"] = round(stats["onnx_s"], 2), round(stats["native_s"], 2)
    stats["graph"] = os.environ.get("DECODER_GRAPH") == "1"
    stats["precision"] = native.precision
    stats["precapture"] = precapture
    if precapture:
        stats["capture_s"] = round(capture_s, 1)
    print("DECODER " + json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
