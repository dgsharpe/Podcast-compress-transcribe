# mimi_dec on the Fold 8 (Termux)

`mimi_dec.c` is a Mimi decoder written against the LiteRT C API (CompiledModel). It uses the
`litert-community/Mimi` fp16 graphs (2 s windows), with RVQ on the host and overlap/crossfade between windows.
It is built and run inside Termux over ssh (`ssh -p 8022 <phone-ip>`, files in `~/nac/v2`).

## Layout on the phone (`~/nac/v2`)

- `include/` — LiteRT 2.3.x C headers (`litert_cc_sdk.zip` → `litert/c`, plus `litert/c/internal`) and an empty
  `litert/build_common/build_config.h`.
- `lib/` — `libLiteRt.so` (Maven `com.google.ai.edge.litert:litert:2.3.2`, `jni/arm64-v8a`) and
  `libLiteRtClGlAccelerator.so` (`litert-gpu:2.3.2`).
- `npu/` contains three groups of libraries:
  - `libLiteRtDispatch_Qualcomm.so` and `libLiteRtCompilerPlugin_Qualcomm.so` (`litert-npu-runtime-qualcomm:2.3.2`).
  - The QAIRT 2.50.0.260828 libraries: `libQnnHtp.so`, `libQnnHtpPrepare.so`, `libQnnHtpV81Stub.so`,
    `libQnnSystem.so`, plus the `hexagon-v81/unsigned/` files `libQnnHtpV81Skel.so` and `libQairtHtpV81.so`. Fetch them
    with `scripts/remote_zip_extract.py` (range requests; the full zip is 2.6 GB).
  - **Copies** of `/vendor/lib64/{libcdsprpc.so, libvmmem.so, vendor.qti.hardware.dsp@1.0.so,
    vendor.qti.hardware.dsp-V1-ndk.so}`. Termux's linker namespace can't load from `/vendor`, and a symlink resolves
    back to `/vendor` and gets refused too. Putting `/vendor/lib64` on `LD_LIBRARY_PATH` breaks linking (an old
    vendor `libc++`).
- `mimi_dec_tx_fp16.tflite`, `mimi_deconly_fp16.tflite`, `mimi_rvq.bin` (litert-community/Mimi);
  `episode_q32.i16` (from `scripts/export_codes.py`); `cache/` (compiled NPU graphs).

## Build and run

```sh
pkg install clang
clang -O3 -mcpu=native -std=c11 -DLITERT_DISABLE_OPENGL_SUPPORT -Iinclude mimi_dec.c \
  -Llib -lLiteRt -lm -Wl,-rpath,'$ORIGIN/lib' -o mimi_dec
export LD_LIBRARY_PATH=$PWD/npu:$PWD/lib ADSP_LIBRARY_PATH="$PWD/npu;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/vendor/dsp"
./mimi_dec --codes episode_q32.i16 --tx npu --conv gpu --libdir $PWD/lib --npudir $PWD/npu \
  --cache $PWD/cache --out out.wav
```

## Backend choice (measured 2026-10-30, ms per 2 s window = 1.52 s of new audio)

| transformer | conv | per window | quality vs the server decode |
|---|---|---|---|
| **NPU** | **GPU** | **41.5** (tx 7.9, conv 26, rvq 7.5) | same as CPU path (48 dB between them) |
| CPU, 2 threads | GPU | 48.6 | reference phone path, 19.6 dB vs the server fp32 |
| GPU fp16 or fp32 | GPU | 36.3 | **broken**: 12 dB, the transformer residual overflows GPU precision |
| CPU, 4 threads | CPU | 295 | fine, but slow; fallback without a GPU |

The conv decoder on the NPU isn't usable: JIT compiling it on the phone ran for more than 400 s single-threaded
(1.2 GB RSS) and never finished. Published S26 numbers also put it slower than the GPU (69 ms vs 24 ms).

## libmimi (`phone/libmimi/`): the reusable decoder

`mimi.h` / `mimi.c` provide `.mimi` parsing (FORMAT.md) and block decoding. Block k covers output frames
[18k, 18k+18) and is decoded from the window [18k−6, 18k+19): 6 frames of left context and 1 frame of lookahead,
whose audio is crossfaded (20 ms) into block k+1. `mimi_play.c` is a CLI that decodes a whole file or a
`--start/--dur` range.

```sh
cd libmimi && clang -O3 -mcpu=native -std=c11 -DLITERT_DISABLE_OPENGL_SUPPORT -I../include mimi.c mimi_play.c \
  -L../lib -lLiteRt -lm -Wl,-rpath,'$ORIGIN/../lib' -o mimi_play
./libmimi/mimi_play episode.mimi out.wav --start 600 --dur 60 --models . --libdir $PWD/lib \
  --npudir $PWD/npu --cache $PWD/cache        # (LD_LIBRARY_PATH / ADSP_LIBRARY_PATH as above)
```

Measured 2026-10-31, Vergecast episode encoded with `scripts/mimi_file.py encode` (1.44 MB for 43:45):

- Full decode: RTF 0.0305, mean 43.8 ms per 1.44 s block.
- First audio after a seek: 48–50 ms at 0 s, 600 s and 2000 s. Decoder setup takes 2.2–3.0 s with the NPU cache warm.
- Seeking at 600 s gives audio identical to the continuous decode from the second block on. The first block is
  31.6 dB from it, because it has no crossfade.
- Whole-episode scores: dWER 1.68% [1.39, 2.00], speaker cosine 0.997, PESQ-WB 3.45. That matches the bake-off
  PyTorch decode (1.67%).
