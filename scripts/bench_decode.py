"""Phone-relevant decoder cost on the server's CPU (a rough proxy, not a phone measurement).

For each decoder: FLOPs per second of audio (torch FlopCounterMode, matmul/conv only),
parameter count, single-thread and 4-thread real-time factor on a 10 s chunk, and the
wall time to decode a 2 s chunk cold (a stand-in for "time to first audio after a seek").
Each timing is repeated REPS times; writes outputs/eval/bench_decode.json.
"""

import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parent.parent
SR, REPS = 24000, 5


def timed(fn, reps=REPS):
    fn()  # warm-up
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return ts


def flops(fn):
    with FlopCounterMode(display=False) as fc:
        fn()
    return fc.get_total_flops()


def bench(name, dec, codes10, codes2, nparams):
    r = dict(name=name, params_M=nparams / 1e6, gflop_per_audio_s=flops(lambda: dec(codes10)) / 10 / 1e9)
    for th in (1, 4):
        torch.set_num_threads(th)
        r[f"rtf_{th}t"] = [t / 10 for t in timed(lambda: dec(codes10))]
        r[f"seek2s_{th}t_s"] = timed(lambda: dec(codes2))
    print(json.dumps(r), flush=True)
    return r


def main():
    wav = sf.read(ROOT / "data/ref/ref_24k.wav", dtype="float32")[0]
    x10 = wav[600 * SR : 610 * SR]  # a conversational stretch
    res = []
    with torch.inference_mode():
        from transformers import MimiModel
        m = MimiModel.from_pretrained("kyutai/mimi").eval()
        c = m.encode(torch.from_numpy(x10)[None, None], num_quantizers=32).audio_codes
        dparams = sum(p.numel() for n, p in m.named_parameters() if not n.startswith("encoder"))
        for k in (8, 16, 32):
            ck, c2 = c[:, :k], c[:, :k, :25]
            res.append(bench(f"mimi_q{k}", lambda z: m.decode(z).audio_values, ck, c2, dparams))

        from qwen_tts import Qwen3TTSTokenizer
        tok = Qwen3TTSTokenizer.from_pretrained("Qwen/Qwen3-TTS-Tokenizer-12Hz", device_map="cpu")
        q = tok.encode(x10, sr=SR).audio_codes[0]
        allp = sum(p.numel() for p in tok.model.parameters())
        dec = getattr(tok.model, "decoder", None)
        qp = sum(p.numel() for p in dec.parameters()) if dec is not None else allp
        res.append(bench("qwen12hz", lambda z: tok.decode({"audio_codes": z}), q, q[:25], qp))
    (ROOT / "outputs/eval").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs/eval/bench_decode.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
