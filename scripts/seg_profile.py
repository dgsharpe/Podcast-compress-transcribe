"""pyannote's segmentation model (PyanNet: SincNet + LSTM + linear) in parts, 256 windows of 10 s on the GPU, as
the hub runs it: where its time goes, with SincNet in fp32 / TF32.

    python scripts/seg_profile.py
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def main():
    import torch
    import torch.nn.functional as F
    from hub import pyannote_gpu

    pipe = pyannote_gpu.load("cuda", path=os.environ.get("TPUT_PYANNOTE_PATH"))
    m = pipe._segmentation.model
    s = m.sincnet
    x = torch.randn(256, 1, 160000, device="cuda") * 0.1

    def timed(f):
        with torch.inference_mode():
            f()
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(5):
                out = f()
            torch.cuda.synchronize()
        return (time.perf_counter() - t) / 5, out

    res = {"device": torch.cuda.get_device_name(0)}
    for tf32 in (False, True):
        with torch.backends.cudnn.flags(enabled=True, allow_tf32=tf32):
            first, y = timed(lambda: s.conv1d[0](s.wav_norm1d(x)))

            def rest(y=y):
                out = y
                for c, (conv, pool, norm) in enumerate(zip(s.conv1d, s.pool1d, s.norm1d, strict=True)):
                    if c > 0:
                        out = conv(out)
                    if c == 0:
                        out = torch.abs(out)
                    out = F.leaky_relu(norm(pool(out)))
                return out
            rest_s, feats = timed(rest)
            whole, _ = timed(lambda: m(x))
        res["tf32" if tf32 else "fp32"] = {"sinc_first_conv_ms": round(1000 * first, 1),
                                           "sincnet_rest_ms": round(1000 * rest_s, 1),
                                           "whole_model_ms": round(1000 * whole, 1)}
    with torch.inference_mode():
        seq = feats.transpose(1, 2)  # (batch, frames, features), as PyanNet's rearrange
    lstm_s, _ = timed(lambda: m.lstm(seq))
    res["lstm_ms"] = round(1000 * lstm_s, 1)
    res["first_conv_shape"] = list(y.shape)
    print("SEGPROF " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
