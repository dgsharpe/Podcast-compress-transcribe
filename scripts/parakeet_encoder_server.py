"""Parakeet's encoder on the server's GPU: replay one episode's real encoder batches through variants (E2).

    .venv-rocm/bin/python scripts/parakeet_encoder_server.py [--minutes 43.8] [--variants eager,half,compile,bucket16]

The hub's encoder (hub.mimi_hub.TorchEncoder: onnx2torch graph, fp16 autocast) is wrapped to capture every
(features, lengths) batch onnx_asr feeds it during a real recognize(); each variant then runs exactly those inputs.
Outputs are compared with the eager-autocast baseline (max |diff| over valid frames, relative to its RMS).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MIMI_HUB_DATA", "/tmp/mimi-enc-profile")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402


def timed(f):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = f()
    torch.cuda.synchronize()
    return out, round(time.perf_counter() - t, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=43.8)
    ap.add_argument("--variants", default="eager,half,compile,bucket16")
    a = ap.parse_args()

    import librosa
    import soundfile as sf
    from hub import mimi_hub

    wav, sr = sf.read(ROOT / "data/ref/ref_24k.wav", dtype="float32")
    w16 = librosa.resample(wav[: int(a.minutes * 60 * sr)], orig_sr=sr, target_sr=16000)
    m = mimi_hub.models()
    inner = m["asr"]
    while hasattr(inner, "asr"):
        inner = inner.asr
    enc = inner._encoder
    assert isinstance(enc, mimi_hub.TorchEncoder), type(enc)
    batches = []
    run = enc.run

    def capture(names, feeds):
        batches.append((np.array(feeds["audio_signal"]), np.asarray(feeds["length"]).copy()))
        return run(names, feeds)

    enc.run = capture
    list(m["asr"].recognize(w16, sample_rate=16000))
    enc.run = run
    frames = sum(int(n.sum()) for _, n in batches)
    padded = sum(x.shape[0] * x.shape[2] for x, _ in batches)
    res = {"minutes": a.minutes, "batches": len(batches), "batch_size": int(max(x.shape[0] for x, _ in batches)),
           "padding_waste": round(1 - frames / padded, 3)}
    print("SETUP " + json.dumps(res), flush=True)

    module = enc.m  # the onnx2torch encoder, fp32 weights, on cuda

    def run_batches(fn, groups):
        outs = []
        for x, n in groups:
            outs.append(fn(torch.from_numpy(x).cuda(), torch.from_numpy(n).cuda()))
        return outs

    def autocast_fn(mod):
        def f(x, n):
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                out, lens = mod(x, n)
            return out.float().cpu().numpy(), lens.cpu().numpy()
        return f

    variants = a.variants.split(",")
    base_fn = autocast_fn(module)
    run_batches(base_fn, batches[:3])  # warm-up
    base, res["eager_s"] = timed(lambda: run_batches(base_fn, batches))
    base_items = base

    def compare(outs):
        worst = 0.0
        for (out, lens), (ref, rlens) in zip(outs, base_items):
            for r in range(ref.shape[0]):
                k = int(rlens[r])
                a_, b_ = out[r, :, :k], ref[r, :, :k]
                worst = max(worst, float(np.abs(a_ - b_).max() / (np.sqrt((b_ ** 2).mean()) + 1e-9)))
        return round(worst, 4)

    if "half" in variants:
        import copy
        half = copy.deepcopy(module).half()  # keep the fp32 module intact

        def half_fn(x, n):
            with torch.inference_mode():
                out, lens = half(x.half(), n)
            return out.float().cpu().numpy(), lens.cpu().numpy()
        run_batches(half_fn, batches[:3])
        outs, res["half_s"] = timed(lambda: run_batches(half_fn, batches))
        res["half_maxdiff_rel"] = compare(outs)
        del half
        print("HALF " + json.dumps({k: v for k, v in res.items() if k.startswith(("eager", "half"))}), flush=True)

    if "compile" in variants:
        compiled = torch.compile(module, dynamic=True)
        comp_fn = autocast_fn(compiled)
        t = time.perf_counter()
        run_batches(comp_fn, batches[:3])  # compile + warm-up
        res["compile_warmup_s"] = round(time.perf_counter() - t, 1)
        outs, res["compile_s"] = timed(lambda: run_batches(comp_fn, batches))
        res["compile_s_2nd"] = timed(lambda: run_batches(comp_fn, batches))[1]
        res["compile_maxdiff_rel"] = compare(outs)
        print("COMPILE " + json.dumps({k: v for k, v in res.items() if k.startswith("compile")}), flush=True)

    if "bucket16" in variants:
        # regroup every segment by length into batches of 16 (less padding, fewer calls); outputs per segment
        segs = []
        for x, n in batches:
            for r in range(x.shape[0]):
                segs.append((x[r, :, : int(n[r])], int(n[r])))
        order = sorted(range(len(segs)), key=lambda i: segs[i][1])
        groups = []
        for i in range(0, len(order), 16):
            idx = order[i:i + 16]
            t_max = max(segs[j][1] for j in idx)
            x = np.zeros((len(idx), segs[0][0].shape[0], t_max), np.float32)
            for r, j in enumerate(idx):
                x[r, :, : segs[j][1]] = segs[j][0]
            groups.append((x, np.array([segs[j][1] for j in idx], np.int64)))
        run_batches(base_fn, groups[:2])
        _, res["bucket16_s"] = timed(lambda: run_batches(base_fn, groups))
        res["bucket16_calls"] = len(groups)
        res["bucket16_padding_waste"] = round(1 - sum(int(n.sum()) for _, n in groups)
                                              / sum(x.shape[0] * x.shape[2] for x, _ in groups), 3)
        print("BUCKET " + json.dumps({k: v for k, v in res.items() if k.startswith("bucket")}), flush=True)

    with open(ROOT / "outputs/throughput_server.jsonl", "a") as fh:
        fh.write(json.dumps({"label": "E2-encoder", **res}) + "\n")
    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
