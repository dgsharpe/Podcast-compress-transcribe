# neural_audio: podcasts as Mimi, with transcripts and ad skipping

A podcast pipeline. A server (**mimi-hub**) re-encodes episodes with Kyutai's
[Mimi](https://huggingface.co/kyutai/mimi) neural codec (32 codebooks, 4.4 kb/s, about 2 MB per hour), transcribes
them with speaker labels (Parakeet TDT 0.6B v2 + pyannote) and marks ads and self-promotion with an LLM. A patched
[Pocket Casts for Android](https://github.com/joshjob42/pocket-casts-android) (branch `mimi`) plays the `.mimi`
files, decoding on the phone's NPU / GPU, and shows the transcripts, highlights ad breaks on the seek bar and skips
them. Each part is optional per podcast: compression, and transcripts + ad marks.

Background and measurements: `reports/Neural audio codecs SOTA 2026.md`, `bakeoff_2026-10-30.md`. Timings in code
comments and `scripts/*_server.py` were measured on an AMD Ryzen AI Max+ 395 server (Radeon 8060S, gfx1151) and a
Galaxy Z Fold 8 (Snapdragon, Hexagon V81 NPU).

| where | what |
|---|---|
| `hub/` | **mimi-hub** (FastAPI, port 8765): progressive `.mimi` conversion and streaming, transcripts with speakers, LLM ad detection (`segments.py`); setup and service files |
| `FORMAT.md`, `scripts/mimi_file.py` | the `.mimi` container |
| `phone/libmimi/` | the C decoder on LiteRT (NPU transformer, GPU conv decoder) |
| `android/` | Media3 extension (`:mimi`), demo player, Mimi Bridge (see `android/README.md`) |
| `pocket-casts-android/` | the patched Pocket Casts, a separate repository cloned here (see below) |
| `scripts/`, `julia/` | the codec bake-off, benchmarks, Modal GPU tests |

## Set up a server

Linux with Python 3.12 ([uv](https://docs.astral.sh/uv/)) and `ffmpeg`. A GPU makes it much faster but isn't
required: a 44 min episode takes about 45 s end to end on the Radeon 8060S, about 5 minutes on its CPU alone.

```sh
git clone https://github.com/joshjob42/Podcast-compress-transcribe neural_audio
cd neural_audio
uv sync                    # CPU environment (.venv)
hub/setup_rocm.sh          # or, AMD GPU: .venv-rocm (AMD's ROCm torch wheels; tested on gfx1151)
hub/setup_cuda.sh          # or, NVIDIA GPU: .venv-cuda (CUDA 12.8 torch wheels)
hub/fetch_models.sh        # CPU speaker diarization models (sherpa-onnx), into hub_models/
```

- **Models:** Mimi, Parakeet and Silero VAD download from Hugging Face on first use. The GPU diarizer, pyannote
  speaker-diarization-community-1, is gated: accept its conditions on huggingface.co, then run once
  `HF_TOKEN=... hf download pyannote/speaker-diarization-community-1`. Without it the hub uses the CPU diarizer.
- **Ad detection** runs an agent CLI in print mode with your own login: install and sign in to `codex` and/or `claude`
  on the server (`MIMI_HUB_SEGMENTER`, below), or set `MIMI_HUB_SEGMENTER=off`.
- **Try it:** `hub/serve.sh --host 127.0.0.1 --port 8765`, then
  `curl -X POST localhost:8765/v1/convert -H 'Content-Type: application/json' -d '{"url": "<episode audio URL or pca.st link>"}'`
  and poll `GET /v1/episodes/<id>`. `hub/serve.sh` uses `.venv-rocm` or `.venv-cuda` if present, else the uv `.venv`.
- **Network:** the patched app sends no token, so bind the hub to a private address (a Tailscale IP works well)
  rather than the internet.

### Run it as a service

`hub/mimi-hub.service` is a systemd user unit:

```sh
cp hub/mimi-hub.service ~/.config/systemd/user/   # edit WorkingDirectory, --host and Environment
systemctl --user daemon-reload
systemctl --user enable --now mimi-hub
loginctl enable-linger "$USER"                     # keep running while you're logged out
journalctl --user -u mimi-hub -f                   # logs
```

On NixOS with home-manager, `hub/mimi-hub.service.nix` is the same unit as a module snippet. `hub/run.sh` starts a
one-off transient unit instead.

### Server settings

Environment variables, e.g. `Environment=` lines in the unit:

| variable | default | |
|---|---|---|
| `MIMI_HUB_COMPRESS` | `1` | `0`: no `.mimi` at all, only transcripts and ad marks (the app then plays originals via the hub) |
| `MIMI_HUB_SOURCES` | `fetch` | originals for transcript-only episodes: `fetch` = not stored, re-fetched from the publisher when asked (409 if the publisher's file changed); `keep` = stored on disk |
| `MIMI_HUB_SEGMENTER` | `codex:gpt-6.1-sol,claude:claude-haiku-5-5` | ad detection via logged-in agent CLIs, tried in order; `off` to disable |
| `MIMI_HUB_DEVICE` | `auto` | `cuda` / `cpu` |
| `MIMI_HUB_DIARIZER` | `auto` | `pyannote` (GPU; gated HF model, fetch once with a token) / `sherpa` (CPU) |
| `MIMI_HUB_ASR_ENCODER` | `native` | `onnx2torch` for the converted ONNX encoder |
| `MIMI_HUB_MIMI_FP16` | `1` | `0`: Mimi encoder in fp32 on the GPU (fp16 is ~1.7x faster; the coarse codes nearly all match) |
| `MIMI_HUB_TOKEN` | (none) | require `Authorization: Bearer <token>` |
| `MIMI_HUB_DATA` | `hub_data/` | where `.mimi` / `.vtt` / records live |

## Build the app

The patched Pocket Casts includes this repository's `android/mimi` module (`../android/mimi`), so clone it inside
the checkout:

```sh
cd neural_audio
git clone -b mimi https://github.com/joshjob42/pocket-casts-android
android/fetch_deps.sh      # LiteRT, Qualcomm QNN runtime, Mimi decoder graphs + tables (not in git)
echo "mimiHubUrl=http://<your-hub>:8765" >> ~/.gradle/gradle.properties   # optional: the app's default hub URL
cd pocket-casts-android && ./gradlew :app:assembleDebugProd
```

- **Toolchain:** JDK 21, Android SDK 37 (36 for the standalone `android/` apps) and NDK 29.0.14206865 with CMake
  (the decoder is native).
  `android/pc_build.sh` runs the same build in a container.
- **Phone:** arm64 only. The NPU path needs the transformer precompiled for that phone's SoC
  (`mimi_dec_tx_npu.tflite`; see `phone/README.md`). Without it, decoding runs on the CPU and GPU, still well
  above real time.
- **In the app:** Settings > Advanced > Mimi server sets the hub URL and the defaults (compression, transcripts and
  ad marks, ad and self-promotion skipping, trimmed timeline, download when streaming). Each podcast's settings
  page overrides compression and transcripts.

`android/` also builds Mimi Bridge, which lets any player play `.mimi` files, and a demo player
(`android/README.md`).

## Bake-off pipeline

Python env: `uv sync` (CPU torch). Needs `opusenc` (libopus 1.6.1) and `ffmpeg` on PATH (on NixOS:
`nix shell nixpkgs#opus-tools nixpkgs#ffmpeg`, and nix-ld for the Python wheels).

1. `data/raw/` episode MP3 → `data/ref/ref_{16,24,48}k.wav` (mono, ffmpeg).
2. Opus: `opusenc --bitrate N` on `ref_48k.wav` → `outputs/opus/opus_Nk.{opus,wav}` (stock decoder,
   no 1.6 neural enhancement).
3. `scripts/run_codecs.py {mimi,qwen12hz}` — 30 s segments coded independently (cold start = seek case).
4. `scripts/eval.py` — per-segment dWER vs the reference's Whisper transcript, WavLM speaker cosine,
   PESQ-WB, log-mel L1 → `outputs/eval/seg_<variant>.csv` (one process per variant runs fine in parallel
   once `asr_ref.json` is cached).
5. `julia --project=julia julia/bb_summary.jl outputs/eval` — Bayesian-bootstrap table.
6. `scripts/bench_decode.py` — decoder GFLOP per audio second, 1/4-thread RTF, 2 s cold decode.
7. `scripts/mimi_file.py encode IN OUT.mimi` — production encoder (FORMAT.md); `phone/` — the Android decoder.
8. `scripts/transcribe_vtt.py` — sidecar WebVTT for the episode; `scripts/make_clips.py` — listening page.

`data/` and `outputs/` are gitignored (copyrighted audio, large files).

## mimi-hub internals

- **Progressive:** a job writes the header and preallocates the `.mimi` within about a second, then encodes 10 s
  and then 60 s chunks as one stream (the encoder's caches carried across), while ffmpeg is still decoding.
  `GET <id>.mimi` serves bytes as they land (Range requests work; a chunk a reader seeks to is encoded first).
- **Transcript:** Parakeet TDT 0.6B v2: transformers' native encoder in fp16 on the GPU (checked against NVIDIA's
  NeMo), onnx-asr's decoding, segments batched by length; sentence-sized cues with `<v Speaker N>` voices.
- **Speakers:** pyannote community-1 on the GPU with a batched embedding step (`hub/pyannote_gpu.py`), or
  sherpa-onnx on the CPU. Speakers with under 30 s in total are folded into their neighbours.
- **Ads:** `hub/segments.py` asks an LLM for ad and self-promotion cue ranges as soon as the ASR cues exist.
- **Originals:** with compression off for a podcast, the hub hands the client the exact audio it processed (`.src`),
  since dynamically inserted ads differ between downloads.
- **Tests:** `hub/test_hub.py` (upload), `hub/test_stream.py` (progressive streaming); in-process, ffmpeg on PATH.
  The API is documented in the `hub/mimi_hub.py` docstring.
