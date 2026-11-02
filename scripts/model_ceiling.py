"""Each model alone at a large batch, as the hub runs it (eager) and through torch.compile: how much audio per
second each could process if fed perfectly - the pipeline's GPU ceiling, and which model to work on.

    python scripts/model_ceiling.py [--compile parakeet|resnet|segmentation|mimi_seanet] [--mode max-autotune]

  Parakeet encoder: 64 segments x 20 s (fp16 weights, autocast), audio = 64 x 20 s
  pyannote ResNet frames: 256 windows of 10 s (channels-last, folded batch norms, fp16) - at the hub's 2 s step a
      window is 2 s of new audio
  pyannote segmentation: 256 windows of 10 s (fp16 LSTM), likewise 2 s each
  Mimi: 32 streams x 20 s chunks through hub/mimi_static.py (eager step and CUDA graph)
Prints x real time per model and the serialized total (1 / sum of 1 / x). --compile: one model, eager and through
torch.compile, under no_grad (compiling under inference_mode failed on inference tensors; after one failed compile
the next one hit a torch assert, so one model per process).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("MIMI_HUB_DATA", "/tmp/mimi-ceiling")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def timed(f, reps=5):
    import torch
    f()
    f()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        f()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return sorted(ts)[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile", choices=["parakeet", "resnet", "segmentation", "mimi_seanet"],
                    help="only this model, eager vs torch.compile")
    ap.add_argument("--mode", default=None, help="torch.compile mode (default, max-autotune-no-cudagraphs, ...)")
    a = ap.parse_args()
    import torch
    import pyannote.audio  # noqa: F401 - before anything is compiled (importing it under dynamo hit a torch assert)
    import mimi_file
    from hub import mimi_hub, mimi_static, parakeet_native, pyannote_gpu
    from hub.autocast_weights import half_weights

    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # as pyannote's pipeline sets
    res = {"device": torch.cuda.get_device_name(0)}
    # every model loaded before anything is compiled (loading pyannote's checkpoint after a compile hit a torch assert)
    enc = parakeet_native.NativeEncoder(parakeet_native.load(mimi_hub.MODELS_DIR / f"{mimi_hub.ASR_MODEL}-hf", "cuda",
                                                             half=True))
    pipe = pyannote_gpu.load("cuda", path=os.environ.get("TPUT_PYANNOTE_PATH"))
    model = mimi_file.load_model("cuda")
    half_weights(model.encoder)

    def report(name, seconds, audio_s):
        res[name] = {"ms": round(1000 * seconds, 2), "x_rt": round(audio_s / seconds)}
        print("CEIL " + json.dumps({"model": name, **res[name]}), flush=True)

    if a.compile:
        return compiled(a, res, report, enc, pipe, model)

    def variants(name, fn, audio_s):
        report(name, timed(fn), audio_s)

    # Parakeet encoder
    x = torch.randn(64, 2000, 128, device="cuda")
    lens = torch.full((64,), 2000, device="cuda")

    def parakeet():
        with torch.inference_mode():
            return enc.encode(x, lens)
    variants("parakeet_encoder_b64", parakeet, 64 * 20.0)

    # pyannote (as the hub loads it on NVIDIA)
    emb = pipe._embedding.model_
    w = torch.randn(256, 1, 160000, device="cuda") * 0.1
    with torch.inference_mode():
        fb = emb.compute_fbank(w)
    mode = pyannote_gpu._frames_mode(torch.device("cuda"))

    def resnet():
        with torch.inference_mode():
            return pyannote_gpu.resnet_frames(emb, fb, mode)
    variants(f"resnet_frames_{mode}_b256", resnet, 256 * 2.0)

    def fbank():
        with torch.inference_mode():
            return emb.compute_fbank(w)
    report("fbank_b256", timed(fbank), 256 * 2.0)
    seg = pipe._segmentation.model

    def segmentation():
        with torch.inference_mode():
            return seg(w)
    variants("segmentation_b256", segmentation, 256 * 2.0)

    # Mimi, 32 streams of 20 s chunks
    chunk = torch.randn(32, 1, 250 * mimi_file.SPF, device="cuda") * 0.1
    for graph in (False, True):
        s = mimi_static.StaticMimiEncoder(model, 32, 250, graph=graph)
        report(f"mimi_b32_{'graph' if graph else 'eager'}", timed(lambda s=s: s.encode(chunk)), 32 * 20.0)
    total = sum(1 / v["x_rt"] for k, v in res.items() if isinstance(v, dict) and "x_rt" in v and k in (
        "parakeet_encoder_b64", f"resnet_frames_{mode}_b256", "fbank_b256", "segmentation_b256", "mimi_b32_graph"))
    res["serialized_x_rt"] = round(1 / total)
    print("CEILING " + json.dumps(res), flush=True)


def compiled(a, res, report, enc, pipe, model):
    """One model's module, eager and compiled, same inputs and precision as above (no_grad, autocast fp16)."""
    import torch
    from hub import pyannote_gpu
    fp16 = torch.autocast("cuda", dtype=torch.float16)
    if a.compile == "parakeet":
        x = torch.randn(64, 2000, 128, device="cuda")
        mask = torch.ones(64, 2000, dtype=torch.bool, device="cuda")
        module, args, autocast, audio_s = enc.encoder, (), fp16, 64 * 20.0
        kwargs = {"input_features": x, "attention_mask": mask}
    elif a.compile == "resnet":
        emb = pipe._embedding.model_
        w = torch.randn(256, 1, 160000, device="cuda") * 0.1
        with torch.no_grad():
            fb = emb.compute_fbank(w)
        if pyannote_gpu.CHANNELS_LAST == "1":
            fb = fb.contiguous()
        module, args, kwargs, autocast, audio_s = emb.resnet.forward_frames, (fb,), {}, fp16, 256 * 2.0
    elif a.compile == "segmentation":
        w = torch.randn(256, 1, 160000, device="cuda") * 0.1
        module, args, kwargs, audio_s = pipe._segmentation.model, (w,), {}, 256 * 2.0
        autocast = torch.autocast("cuda", enabled=False)  # its LSTM runs in fp16 by its own wrapper
    else:  # Mimi's convolutional encoder (most of its time), 32 x 20 s, not streamed
        chunk = torch.randn(32, 1, 250 * 1920, device="cuda") * 0.1
        module, args, kwargs, autocast, audio_s = model.encoder.encoder_seanet, (chunk,), {}, fp16, 32 * 20.0

    def run(f):
        def g():
            with torch.no_grad(), autocast:
                return f(*args, **kwargs)
        return g
    name = a.compile
    report(name, timed(run(module)), audio_s)
    t = time.perf_counter()
    try:
        f = torch.compile(module, mode=a.mode)
        run(f)()
        torch.cuda.synchronize()
        res["compile_s"] = round(time.perf_counter() - t, 1)
        report(f"{name}_compiled_{a.mode or 'default'}", timed(run(f)), audio_s)
    except Exception as e:  # noqa: BLE001
        res[name + "_compiled"] = f"{type(e).__name__}: {str(e)[:300]}"
        print("CEIL " + json.dumps({"model": name + "_compiled", "error": res[name + "_compiled"]}), flush=True)
    print("CEILING " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
