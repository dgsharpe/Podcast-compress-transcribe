"""Encode audio to a `.mimi` file (FORMAT.md) on the server, and read one back.

Encoding is continuous, unlike the 30 s cold-start segments used in the bake-off. Chunks of CHUNK s
are encoded with CTX s of left context, which matches Mimi's 250-frame (20 s) transformer window,
and only the codes for the chunk itself are kept.

usage:
  python scripts/mimi_file.py encode IN_AUDIO OUT.mimi [--nq 32] [--title T] [--source URL]
  python scripts/mimi_file.py decode IN.mimi OUT.wav [--seconds N]   (PyTorch reference decode)
  python scripts/mimi_file.py info IN.mimi
"""

import argparse
import json
import struct
import time

import numpy as np
import torch

SR, FPS_NUM, FPS_DEN, BITS = 24000, 25, 2, 11
SPF = SR * FPS_DEN // FPS_NUM  # 1920 samples per frame
HDR = struct.Struct("<4sBBBBIHHIIII32s")  # 64 bytes


def pack(codes, nq, bits=BITS):
    """codes (F, nq) int -> (F, frame_bytes) uint8, little-endian bit packing."""
    F = codes.shape[0]
    fb = (nq * bits + 7) // 8
    acc = np.zeros((F, fb * 8), np.uint8)
    for i in range(nq):
        for b in range(bits):
            acc[:, i * bits + b] = (codes[:, i] >> b) & 1
    return np.packbits(acc, axis=1, bitorder="little")


def unpack(frames, nq, bits=BITS):
    bitsarr = np.unpackbits(frames, axis=1, bitorder="little")
    codes = np.zeros((frames.shape[0], nq), np.int64)
    for i in range(nq):
        for b in range(bits):
            codes[:, i] |= bitsarr[:, i * bits + b].astype(np.int64) << b
    return codes


def write(path, codes, nq, meta):
    F = codes.shape[0]
    fb = (nq * BITS + 7) // 8
    mj = json.dumps(meta, separators=(",", ":")).encode()
    data_offset = (64 + len(mj) + 63) // 64 * 64
    hdr = HDR.pack(b"MIMI", 1, nq, BITS, 0, SR, FPS_NUM, FPS_DEN, F, fb, data_offset, len(mj),
                   b"kyutai/mimi".ljust(32, b"\0"))
    with open(path, "wb") as f:
        f.write(hdr)
        f.write(mj)
        f.write(b"\0" * (data_offset - 64 - len(mj)))
        f.write(pack(codes, nq).tobytes())


def read(path):
    with open(path, "rb") as f:
        buf = f.read()
    (magic, ver, nq, bits, flags, sr, fn, fd, F, fb, off, ml, model) = HDR.unpack_from(buf, 0)
    assert magic == b"MIMI" and ver == 1 and bits == BITS, (magic, ver, bits)
    meta = json.loads(buf[64 : 64 + ml])
    frames = np.frombuffer(buf, np.uint8, count=F * fb, offset=off).reshape(F, fb)
    return dict(nq=nq, sr=sr, fps=fn / fd, n_frames=F, frame_bytes=fb, data_offset=off,
                model=model.rstrip(b"\0").decode(), meta=meta), unpack(frames, nq)


def load_model():
    from transformers import MimiModel
    return MimiModel.from_pretrained("kyutai/mimi").eval()


def encode_array(m, wav, nq=32, chunk_s=60.0, ctx_s=20.0, progress=None):
    """24 kHz mono float array -> codes (frames, nq), chunked with left context (see module doc)."""
    chunk, ctx = int(chunk_s * SR) // SPF * SPF, int(ctx_s * SR) // SPF * SPF
    out = []
    with torch.inference_mode():
        for s in range(0, len(wav), chunk):
            lo = max(0, s - ctx)
            x = torch.from_numpy(wav[lo : s + chunk]).float()[None, None]
            c = m.encode(x, num_quantizers=nq).audio_codes[0].T.numpy()  # (frames, nq)
            out.append(c[(s - lo) // SPF :][: (min(len(wav), s + chunk) - s + SPF - 1) // SPF])
            if progress:
                progress(min(1.0, (s + chunk) / len(wav)))
    return np.concatenate(out)


def encode(a):
    import librosa
    torch.set_num_threads(a.threads)
    wav, _ = librosa.load(a.inp, sr=SR, mono=True)
    m = load_model()
    t = time.perf_counter()
    codes = encode_array(m, wav, a.nq, a.chunk, a.ctx)
    meta = dict(title=a.title, source=a.source, encoder="transformers MimiModel",
                chunk_s=a.chunk, ctx_s=a.ctx, duration_s=len(wav) / SR)
    write(a.out, codes, a.nq, meta)
    print(f"{len(codes)} frames ({len(codes) / 12.5:.1f} s) in {time.perf_counter() - t:.0f} s -> {a.out}")


def decode(a):
    import soundfile as sf
    torch.set_num_threads(a.threads)
    info, codes = read(a.inp)
    if a.seconds:
        codes = codes[: int(a.seconds * 12.5)]
    m = load_model()
    with torch.inference_mode():
        y = m.decode(torch.from_numpy(codes.T[None])).audio_values[0, 0].numpy()
    sf.write(a.out, y, SR)
    print(f"decoded {len(y) / SR:.1f} s -> {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("encode")
    e.add_argument("inp"); e.add_argument("out")
    e.add_argument("--nq", type=int, default=32)
    e.add_argument("--chunk", type=float, default=60.0)
    e.add_argument("--ctx", type=float, default=20.0)
    e.add_argument("--title", default=""); e.add_argument("--source", default="")
    e.add_argument("--threads", type=int, default=16)
    d = sp.add_parser("decode")
    d.add_argument("inp"); d.add_argument("out")
    d.add_argument("--threads", type=int, default=16)
    d.add_argument("--seconds", type=float, default=0, help="decode only the first N seconds")
    i = sp.add_parser("info"); i.add_argument("inp")
    a = ap.parse_args()
    if a.cmd == "encode":
        encode(a)
    elif a.cmd == "decode":
        decode(a)
    else:
        info, codes = read(a.inp)
        print(json.dumps(info, indent=1), codes.shape, codes.min(), codes.max())
