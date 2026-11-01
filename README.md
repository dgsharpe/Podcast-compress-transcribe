# neural_audio

Neural speech codecs for a personal podcast pipeline: the server re-encodes episodes with Mimi (32 codebooks,
4.4 kb/s, ~2 MB/hour), transcribes them with speaker labels and marks ads / self-promotion; the phone decodes
`.mimi` on its NPU + GPU (see `reports/Neural audio codecs SOTA 2026.md`, `bakeoff_2026-10-30.md`).

| where | what |
|---|---|
| `hub/` | **mimi-hub** on the server (FastAPI, :8765, tailnet only): progressive `.mimi` conversion + streaming, Parakeet transcripts, diarization, LLM ad detection (`segments.py`). GPU env: `hub/setup_rocm.sh`; runs via `hub/serve.sh` (systemd unit in nix-config `home/server.nix`). |
| `FORMAT.md`, `scripts/mimi_file.py` | the `.mimi` container |
| `phone/libmimi/` | C decoder on LiteRT (NPU transformer, GPU conv decoder) |
| `android/` | Media3 extension, demo player, Mimi Bridge (see `android/README.md`) |
| `pocket-casts-android/` | the patched Pocket Casts: separate repo, private [joshjob42/pocket-casts-android](https://github.com/joshjob42/pocket-casts-android) branch `mimi` (`origin` there = Automattic upstream); build with `android/pc_build.sh` |

## Bake-off pipeline

Python env: `uv sync` (CPU torch). In Claude's sandboxed shell, prefix commands with
`LD_LIBRARY_PATH=/run/current-system/sw/share/nix-ld/lib SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt`.
Opus/ffmpeg come from `nix shell nixpkgs#opus-tools nixpkgs#ffmpeg` (libopus 1.6.1).

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

## mimi-hub (`hub/`): on-demand conversion on the server

`hub/run.sh` starts `hub/mimi_hub.py` (FastAPI) on the server's Tailscale IP, port 8765, as the `mimi-hub` user unit.
It converts a URL, a Pocket Casts share link (resolved to the episode UUID and its enclosure) or an uploaded file
into `hub_data/<id>.mimi` plus `<id>.vtt`. Mimi Bridge's "Convert" feature and the patched Pocket Casts are its
clients. The API is documented in the module docstring.

- **Progressive:** a job writes the header and preallocates the file within about 1 s. It then encodes 10 s and
  then 60 s chunks while ffmpeg is still decoding the source, and `GET <id>.mimi` serves bytes as their chunk lands
  (Range requests work, and a waited-for chunk jumps the queue). Clients can stream or download a `.mimi` that is
  still being made. In `hub/test_stream.py` the whole 44 min arrived in 73 s and matched the final file exactly.
- **Transcript:** Parakeet TDT 0.6B v2 via onnx-asr (44 min in 140 s on CPU, punctuated), split into
  sentence-sized cues using token timestamps.
- **Speakers:** sherpa-onnx diarization (pyannote segmentation-3.0, WeSpeaker ResNet34, clustering threshold 0.6,
  fragments < 30 s folded; about 115 s per 44 min). 0.7 sometimes merged the two hosts; see the note in mimi_hub.py. Each cue gets `<v Speaker N>`, which Pocket Casts shows as speaker labels. Models are in
  `hub_models/` (gitignored, from the sherpa-onnx GitHub releases). One person can still end up with two numbers,
  e.g. a host over intro music.
- **Running it:** `hub/run.sh` (a transient unit), or the home-manager service in `hub/mimi-hub.service.nix`.
- **Tests:** `hub/test_hub.py` (upload) and `hub/test_stream.py` (progressive streaming); both run in-process,
  with ffmpeg on PATH.

## Running your own hub

`hub/serve.sh --host <ip> --port 8765` (GPU: run `hub/setup_rocm.sh` first on an AMD box; otherwise the uv `.venv`
on the CPU). In the patched Pocket Casts, point Settings > Advanced > Mimi > Mimi converter URL at it. Server
settings (environment):

| variable | default | |
|---|---|---|
| `MIMI_HUB_COMPRESS` | `1` | `0`: no `.mimi` at all, only transcripts and ad marks (the app then plays originals via the hub) |
| `MIMI_HUB_SOURCES` | `fetch` | originals for transcript-only episodes: `fetch` = not stored, re-fetched from the publisher when asked (409 if the publisher's file changed); `keep` = stored on disk |
| `MIMI_HUB_SEGMENTER` | `codex:gpt-6.1-sol,claude:claude-haiku-5-5` | ad detection via logged-in agent CLIs, tried in order; `off` to disable |
| `MIMI_HUB_DEVICE` | `auto` | `cuda` / `cpu` |
| `MIMI_HUB_DIARIZER` | `auto` | `pyannote` (GPU; gated HF model, fetch once with a token) / `sherpa` (CPU) |
| `MIMI_HUB_ASR_ENCODER` | `native` | `onnx2torch` for the converted ONNX encoder |
| `MIMI_HUB_TOKEN` | (none) | require `Authorization: Bearer <token>` |
| `MIMI_HUB_DATA` | `hub_data/` | where `.mimi` / `.vtt` / records live |
