"""hub/mimi_static.StaticMimiEncoder (fixed-size state, one CUDA graph) vs transformers' streaming encode
(hub/mimi_hub.mimi_encode), on B streams of the reference episode chunk by chunk: codes agreement and time per chunk.

    python scripts/mimi_static_check.py [--batch 4] [--chunk 250] [--chunks 8] [--trt]    (--trt: also with the SEANet
                                                                                        as a TensorRT engine)
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
os.environ.setdefault("MIMI_HUB_DATA", "/tmp/mimi-static-check")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def quality(model, xt, codes, streams=4):
    """Each encoder's codes decoded back to audio: log-mel L1 distance to the original (lower is better), mean
    over the first [streams] streams."""
    import librosa
    import numpy as np
    import torch
    out = {}
    orig = xt[:streams, 0].float().cpu().numpy()

    def logmel(w):
        return np.log(librosa.feature.melspectrogram(y=w, sr=24000, n_fft=1024, hop_length=256, n_mels=80) + 1e-5)
    ref = [logmel(o) for o in orig]
    for name, c in codes.items():
        with torch.inference_mode():
            audio = model.decode(c[:streams]).audio_values[:, 0].float().cpu().numpy()
        d = []
        for a, o, r in zip(audio, orig, ref, strict=True):
            m = logmel(a[:len(o)])
            d.append(np.abs(m - r[:, :m.shape[1]]).mean())
        out[name] = round(float(np.mean(d)), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--trt", action="store_true")
    a = ap.parse_args()
    import numpy as np
    import soundfile as sf
    import torch
    import mimi_file
    from hub import mimi_hub, mimi_static
    from hub.autocast_weights import half_weights

    wav, sr = sf.read(ROOT / "data/ref/ref_24k.wav", dtype="float32")
    spf, n = mimi_file.SPF, a.chunk * a.chunks
    x = np.stack([wav[k * n * spf:(k + 1) * n * spf] for k in range(a.batch)])
    xt = torch.from_numpy(x).cuda()[:, None]
    model = mimi_file.load_model("cuda")
    half_weights(model.encoder)  # as the hub
    chunks = [xt[..., c * a.chunk * spf:(c + 1) * a.chunk * spf] for c in range(a.chunks)]

    def hf():
        kv, pad, out = None, None, []
        for c in chunks:
            o = mimi_hub.mimi_encode(model, c, encoder_past_key_values=kv, padding_cache=pad, use_streaming=True)
            kv, pad = o.encoder_past_key_values, o.padding_cache
            out.append(o.audio_codes)
        return torch.cat(out, dim=2)

    def static(enc):
        enc.reset()
        return torch.cat([enc.encode(c) for c in chunks], dim=2)

    def timed(f, *args):
        f(*args)
        torch.cuda.synchronize()
        t = time.perf_counter()
        r = f(*args)
        torch.cuda.synchronize()
        return r, (time.perf_counter() - t) / a.chunks

    ref, t_hf = timed(hf)
    eager, t_eager = timed(static, mimi_static.StaticMimiEncoder(model, a.batch, a.chunk, graph=False))
    graph, t_graph = timed(static, mimi_static.StaticMimiEncoder(model, a.batch, a.chunk, graph=True))
    res = {"device": torch.cuda.get_device_name(0), "batch": a.batch, "chunk_frames": a.chunk, "chunks": a.chunks,
           "ms_per_chunk": {"transformers": round(1000 * t_hf, 2), "static_eager": round(1000 * t_eager, 2),
                            "static_graph": round(1000 * t_graph, 2)}}
    variants = [("eager", eager), ("graph", graph)]
    if a.trt:
        t0 = time.perf_counter()
        enc = mimi_static.StaticMimiEncoder(model, a.batch, a.chunk, graph=False, trt=True)
        res["trt_load_s"] = round(time.perf_counter() - t0, 1)
        trt_eager, t = timed(static, enc)
        res["ms_per_chunk"]["static_trt"] = round(1000 * t, 2)
        trt_graph, t = timed(static, mimi_static.StaticMimiEncoder(model, a.batch, a.chunk, graph=True, trt=True))
        res["ms_per_chunk"]["static_trt_graph"] = round(1000 * t, 2)
        variants += [("trt", trt_eager), ("trt_graph", trt_graph)]
        res["trt_graph_vs_graph"] = {"all32": round((trt_graph == graph).float().mean().item(), 4),
                                     "first8": round((trt_graph == graph)[:, :8].float().mean().item(), 4)}
        res["mel_l1_vs_original"] = quality(model, xt, {"transformers": ref, "static_graph": graph,
                                                         "static_trt_graph": trt_graph})
    for name, c in variants:
        same = (c == ref).float()
        res[f"{name}_vs_transformers"] = {"all32": round(same.mean().item(), 4),
                                          "first8": round(same[:, :8].mean().item(), 4),
                                          "first_chunk_all32": round(same[..., :a.chunk].mean().item(), 4)}
    res["graph_vs_eager_all32"] = round((graph == eager).float().mean().item(), 4)
    print("MIMISTATIC " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
