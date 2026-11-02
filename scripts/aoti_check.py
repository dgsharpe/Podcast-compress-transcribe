"""hub/aoti.py: Parakeet's encoder (one 2048-frame bucket, dynamic batch) and pyannote's ResNet frames (dynamic
batch) through torch.export + AOTInductor, the packages without weights (the loaded model uses the module's own
GPU tensors), against eager: output difference, GPU time, CPU time spent issuing calls.

    python scripts/aoti_check.py [parakeet] [resnet]
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MIMI_HUB_DATA", "/tmp/mimi-aoti-check")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def timed(fn, inputs):
    import torch
    with torch.inference_mode():
        for x in inputs:
            fn(*x)
        torch.cuda.synchronize()
        t, cpu = time.perf_counter(), time.thread_time()
        outs = [fn(*x) for x in inputs]
        cpu = time.thread_time() - cpu
        torch.cuda.synchronize()
    return time.perf_counter() - t, cpu, outs


def build(module, args, dynamic_shapes, path):
    """hub/aoti.py's package of [module] (built here into a fresh cache, so the time is the build's)."""
    from hub import aoti
    aoti.DIR = path.parent / "aoti"
    t = time.perf_counter()
    model = aoti.load(path.stem, module, args, dynamic_shapes)
    pkg = next(aoti.DIR.glob(f"{path.stem}-*.pt2"))
    return model, {"build_s": round(time.perf_counter() - t, 1), "package_mb": round(pkg.stat().st_size / 2**20, 1)}


def rel(a, b):
    a, b = a.float(), b.float()
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-6))


def main():
    import torch
    import pyannote.audio  # noqa: F401
    from hub import mimi_hub, parakeet_native, pyannote_gpu
    which = sys.argv[1:] or ["parakeet", "resnet"]
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    nvidia = not torch.version.hip
    dtype = torch.float16
    res = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__}
    work = Path(tempfile.mkdtemp(prefix="aoti-"))

    if "parakeet" in which:
        encoder = parakeet_native.load(mimi_hub.MODELS_DIR / f"{mimi_hub.ASR_MODEL}-hf", "cuda", half=True).eval()

        class Enc(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = encoder

            def forward(self, x, mask):
                with torch.autocast("cuda", dtype=dtype):
                    out = self.encoder(input_features=x, attention_mask=mask)
                return out.last_hidden_state, out.attention_mask

        T = 2048
        x = torch.randn(4, T, 128, device="cuda")
        mask = torch.ones(4, T, dtype=torch.bool, device="cuda")
        batch = torch.export.Dim("batch", min=2, max=256)
        try:
            model, info = build(Enc().eval(), (x, mask), ({0: batch}, {0: batch}), work / "parakeet.pt2")
            res["parakeet_build"] = info
            inputs = []
            for b in (2, 16, 64, 40):
                lens = torch.randint(200, T + 1, (b,), device="cuda")
                lens[0] = T
                inputs.append((torch.randn(b, T, 128, device="cuda"),
                               torch.arange(T, device="cuda")[None] < lens[:, None]))
            eager = Enc().eval()
            te, ce, oe = timed(lambda a, m: eager(a, m), inputs)
            ta, ca, oa = timed(lambda a, m: model(a, m), inputs)
            diffs = []
            for (he, me), (ha, ma) in zip(oe, oa, strict=True):
                assert torch.equal(me, ma)
                diffs.append(rel(ha * me[..., None], he * me[..., None]))
            res["parakeet"] = {"eager_ms": round(1000 * te, 1), "aoti_ms": round(1000 * ta, 1),
                               "speedup": round(te / ta, 2), "cpu_ms_eager": round(1000 * ce, 1),
                               "cpu_ms_aoti": round(1000 * ca, 1), "max_rel_diff": round(max(diffs), 5),
                               "gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            res["parakeet"] = f"{type(e).__name__}: {str(e)[:500]}"
        print("AOTICHECK " + json.dumps(res), flush=True)
        del encoder

    if "resnet" in which:
        pipe = pyannote_gpu.load("cuda", path=os.environ.get("TPUT_PYANNOTE_PATH"))
        model_ = pipe._embedding.model_
        resnet = model_.resnet
        frames_fn = getattr(resnet.forward_frames, "_torchdynamo_orig_callable", resnet.forward_frames)
        fdtype = pyannote_gpu._DTYPES.get(pyannote_gpu._frames_mode(torch.device("cuda")))

        class Frames(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.resnet = resnet

            def forward(self, fbank):
                with torch.autocast("cuda", dtype=fdtype or torch.float32, enabled=fdtype is not None):
                    return frames_fn(fbank)

        with torch.inference_mode():
            fb = model_.compute_fbank(torch.randn(8, 1, 160000, device="cuda") * 0.1)
        fb = fb.clone()  # a normal tensor (export rejects inference tensors)
        n = torch.export.Dim("n", min=1, max=1024)
        try:
            model, info = build(Frames().eval(), (fb,), ({0: n},), work / "resnet.pt2")
            res["resnet_build"] = info
            inputs = []
            with torch.inference_mode():
                for k in (32, 7, 256, 100):
                    inputs.append((model_.compute_fbank(torch.randn(k, 1, 160000, device="cuda") * 0.1),))
            eager = Frames().eval()
            te, ce, oe = timed(lambda f: eager(f), inputs)
            ta, ca, oa = timed(lambda f: model(f), inputs)
            res["resnet"] = {"eager_ms": round(1000 * te, 1), "aoti_ms": round(1000 * ta, 1),
                             "speedup": round(te / ta, 2), "cpu_ms_eager": round(1000 * ce, 1),
                             "cpu_ms_aoti": round(1000 * ca, 1),
                             "max_rel_diff": round(max(rel(a, e) for a, e in zip(oa, oe, strict=True)), 5)}
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            res["resnet"] = f"{type(e).__name__}: {str(e)[:500]}"
        print("AOTICHECK " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
