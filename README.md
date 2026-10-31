# neural_audio

Neural speech codecs for a personal podcast pipeline: re-encode episodes on the server, serve them
with Whisper transcripts, decode on the phone (see `reports/Neural audio codecs SOTA 2026.md`).

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
into `hub_data/<id>.mimi` plus `<id>.vtt`. The transcript comes from Parakeet TDT 0.6B v2 via onnx-asr: 44 min of
audio in 140 s on CPU, punctuated. Mimi Bridge's "Convert" feature and the planned Pocket Casts patch are its
clients. The API is documented in the module docstring, and `hub/test_hub.py` is an in-process end-to-end test.
