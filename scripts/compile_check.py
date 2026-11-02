"""The hub's compiled Parakeet encoder and pyannote ResNet (MIMI_HUB_COMPILE: aot / jit) against eager, on the shapes they get
in the pipeline (varied batch sizes and segment lengths, under inference_mode): output difference, time for the
mixed sequence after warm-up, the CPU time spent issuing it (kernel launches: what a launch-bound process such as
hub/bulk.py pays), and how many graphs were compiled (a recompile per shape would show up here).

    python scripts/compile_check.py [aot|jit]
"""
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MIMI_HUB_DATA", "/tmp/mimi-compile-check")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
HOW = sys.argv[1] if len(sys.argv) > 1 else "aot"
os.environ["MIMI_HUB_COMPILE"] = "0"  # models load eager; each is compiled here, timed


def mixed(fn, inputs):
    import torch
    with torch.inference_mode():
        for x in inputs:
            fn(*x)
        torch.cuda.synchronize()
        t, cpu = time.perf_counter(), time.thread_time()
        outs = [fn(*x) for x in inputs]
        cpu = time.thread_time() - cpu
        torch.cuda.synchronize()
    return time.perf_counter() - t, outs, cpu


def rel(a, b):
    a, b = a.float(), b.float()
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-6))


def main():
    import torch
    import pyannote.audio  # noqa: F401 - before anything is compiled
    from torch._dynamo.utils import counters
    from hub import mimi_hub, parakeet_native, pyannote_gpu

    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    res = {"device": torch.cuda.get_device_name(0)}
    rng = random.Random(0)

    # Parakeet: batches as the hub's ASR batches them (1-64 segments of 1-20 s, 100 frames / s)
    encoder = parakeet_native.load(mimi_hub.MODELS_DIR / f"{mimi_hub.ASR_MODEL}-hf", "cuda", half=True)
    eager = parakeet_native.NativeEncoder(encoder)
    t = time.perf_counter()
    comp = parakeet_native.NativeEncoder(encoder, compile=HOW)
    res["how"] = HOW
    res["parakeet_compile_s"] = round(time.perf_counter() - t, 1)
    inputs = []
    for b in (1, 2, 5, 16, 33, 64, 64, 40):
        t_max = rng.randint(100, 2000)
        lens = torch.tensor([rng.randint(50, t_max) for _ in range(b - 1)] + [t_max], device="cuda")
        inputs.append((torch.randn(b, t_max, 128, device="cuda"), lens))
    loaded = counters["stats"]["unique_graphs"]
    te, oe, ce = mixed(lambda x, n: eager.encode(x, n), inputs)
    tc, oc, cc = mixed(lambda x, n: comp.encode(x, n), inputs)
    res["parakeet_graphs_at_load"], res["parakeet_graphs_mid_run"] = loaded, counters["stats"]["unique_graphs"] - loaded
    diffs = []
    for (he, le), (hc, lc), (_, lens) in zip(oe, oc, inputs, strict=True):
        assert torch.equal(le, lc) and he.shape == hc.shape, (he.shape, hc.shape)
        m = (torch.arange(he.shape[1], device="cuda")[None] < le[:, None])[..., None]
        diffs.append(rel(hc * m, he * m))
    res["parakeet"] = {"eager_ms": round(1000 * te, 1), "compiled_ms": round(1000 * tc, 1),
                       "speedup": round(te / tc, 2), "max_rel_diff": round(max(diffs), 5),
                       "cpu_ms_eager": round(1000 * ce, 1), "cpu_ms_compiled": round(1000 * cc, 1),
                       "audio_s": round(sum(float(n.sum()) for _, n in inputs) / 100, 1)}
    del encoder, eager, comp

    # pyannote's ResNet frames: the embedding batches (up to 256 windows of 10 s in bulk mode, 32 in the hub)
    pipe = pyannote_gpu.load("cuda", path=os.environ.get("TPUT_PYANNOTE_PATH"))
    model = pipe._embedding.model_
    mode = pyannote_gpu._frames_mode(torch.device("cuda"))
    eager_frames = model.resnet.forward_frames
    t = time.perf_counter()
    pyannote_gpu.compile_frames(model, torch.device("cuda"), HOW)
    res["resnet_compile_s"] = round(time.perf_counter() - t, 1)
    compiled_frames = model.resnet.forward_frames
    assert compiled_frames is not eager_frames, "compile failed"
    inputs = []
    with torch.inference_mode():
        for n in (32, 7, 256, 100, 32, 1, 256, 13):
            inputs.append((model.compute_fbank(torch.randn(n, 1, 160000, device="cuda") * 0.1),))

    def frames(f):
        def g(fb):
            model.resnet.forward_frames = f
            return pyannote_gpu.resnet_frames(model, fb, mode)
        return g
    loaded = counters["stats"]["unique_graphs"]
    te, oe, ce = mixed(frames(eager_frames), inputs)
    tc, oc, cc = mixed(frames(compiled_frames), inputs)
    res["resnet_graphs_mid_run"] = counters["stats"]["unique_graphs"] - loaded
    model.resnet.forward_frames = compiled_frames
    res["resnet"] = {"eager_ms": round(1000 * te, 1), "compiled_ms": round(1000 * tc, 1),
                     "speedup": round(te / tc, 2),
                     "cpu_ms_eager": round(1000 * ce, 1), "cpu_ms_compiled": round(1000 * cc, 1),
                     "max_rel_diff": round(max(rel(c, e) for c, e in zip(oc, oe, strict=True)), 5),
                     "windows": sum(len(x[0]) for x in inputs)}
    res["graphs_compiled"] = counters["stats"]["unique_graphs"]
    print("COMPILECHECK " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
