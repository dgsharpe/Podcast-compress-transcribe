"""Minimal reproduction of bulk mode's hang with the TDT search as CUDA graphs: Mimi's static encoder (a CUDA graph,
hub/mimi_static.py) and the decoder's graphs (hub/parakeet_native.NativeTdtDecoder.decode_graphed), each replayed
in a loop from its own thread on its own stream, as hub/bulk.py does, for --seconds. A loop iteration taking over
--stall s counts as the hang: every thread's stack is printed and the process exits 3.

    python scripts/graph_hang_repro.py [--mimi graph|eager|off] [--decoder graph|eager|off] [--decoders 1]
                                       [--order mimi-first|decoder-first] [--seconds 60]
"""
import argparse
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimi", default="graph", choices=["graph", "eager", "off"])
    ap.add_argument("--decoder", default="graph", choices=["graph", "eager", "off"])
    ap.add_argument("--decoders", type=int, default=1, help="decoder threads")
    ap.add_argument("--order", default="mimi-first", choices=["mimi-first", "decoder-first"])
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--stall", type=float, default=30)
    a = ap.parse_args()
    import torch
    import mimi_file
    from hub import mimi_hub, parakeet_native
    from hub.autocast_weights import half_weights
    from hub.mimi_static import StaticMimiEncoder

    mimi = mimi_file.load_model("cuda")
    half_weights(mimi.encoder)
    path = parakeet_native.convert_decoder(mimi_hub.ASR_MODEL, mimi_hub.MODELS_DIR / f"{mimi_hub.ASR_MODEL}-hf")
    dec = parakeet_native.NativeTdtDecoder(path, "cuda", 1024, 1025, max_symbols=10)  # blank, vocab (+ blank)
    static = None

    def capture_mimi():
        nonlocal static
        if a.mimi != "off":
            static = StaticMimiEncoder(mimi, 16, 250, graph=a.mimi == "graph")
            if a.mimi == "graph":
                static.capture()

    def capture_decoder():
        if a.decoder == "graph":
            dec.capture_graphs(0, max_batch=64)
    for f in ((capture_mimi, capture_decoder) if a.order == "mimi-first" else (capture_decoder, capture_mimi)):
        f()
    torch.cuda.synchronize()

    beat, counts, stop = {}, {}, threading.Event()

    def mimi_loop():
        s = torch.cuda.Stream()
        x = torch.randn(16, 1, 250 * mimi_file.SPF, device="cuda") * 0.1
        with torch.cuda.stream(s):
            while not stop.is_set():
                beat["mimi"] = time.time()
                static.reset()
                for _ in range(16):  # a piece: 16 chunks of one batch
                    codes = static.encode(x)
                codes.cpu()
                counts["mimi"] = counts.get("mimi", 0) + 1

    def decoder_loop(k):
        s = torch.cuda.Stream()
        g = torch.Generator(device="cuda").manual_seed(k)
        name = f"decoder{k}"
        with torch.cuda.stream(s):
            while not stop.is_set():
                beat[name] = time.time()
                B = int(torch.randint(8, 65, (1,)))
                T = int(torch.randint(60, 252, (1,)))
                enc = torch.randn(B, T, 1024, device="cuda", generator=g) * 0.5
                lens = torch.randint(T // 2, T + 1, (B,), device="cuda", generator=g)
                if a.decoder == "graph":
                    list(dec.decode_graphed(enc, lens, k))
                else:
                    list(dec(enc, lens))
                counts[name] = counts.get(name, 0) + 1

    threads = []
    if a.mimi != "off":
        threads.append(threading.Thread(target=mimi_loop, daemon=True, name="mimi"))
    if a.decoder != "off":
        threads += [threading.Thread(target=decoder_loop, args=(k,), daemon=True, name=f"decoder{k}")
                    for k in range(a.decoders)]
    t0 = time.time()
    for th in threads:
        th.start()
    hung = None
    while time.time() - t0 < a.seconds:
        time.sleep(1)
        now = time.time()
        late = [n for n, b in beat.items() if now - b > a.stall]
        if late:
            hung = late
            break
    res = {"config": vars(a), "seconds": round(time.time() - t0, 1), "iterations": counts, "hung": hung}
    print("REPRO " + json.dumps(res), flush=True)
    if hung:
        names = {t.ident: t.name for t in threading.enumerate()}
        for ident, frame in sys._current_frames().items():
            if names.get(ident) in [t.name for t in threads]:
                print(f"--- {names[ident]}", file=sys.stderr)
                for fs in traceback.extract_stack(frame)[-3:]:
                    print(f"  {fs.filename.split('/')[-1]}:{fs.lineno} {fs.name}: {fs.line}", file=sys.stderr)
        sys.stderr.flush()
        os._exit(3)
    stop.set()
    for th in threads:
        th.join(timeout=30)


if __name__ == "__main__":
    main()
