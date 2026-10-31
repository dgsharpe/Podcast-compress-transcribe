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
into `hub_data/<id>.mimi` plus `<id>.vtt`. Mimi Bridge's "Convert" feature and the patched Pocket Casts are its
clients. The API is documented in the module docstring.

- **Progressive:** a job writes the header and preallocates the file within about 1 s. It then encodes 10 s and
  then 60 s chunks while ffmpeg is still decoding the source, and `GET <id>.mimi` serves bytes as their chunk lands
  (Range requests work, and a waited-for chunk jumps the queue). Clients can stream or download a `.mimi` that is
  still being made. In `hub/test_stream.py` the whole 44 min arrived in 73 s and matched the final file exactly.
- **Transcript:** Parakeet TDT 0.6B v2 via onnx-asr (44 min in 140 s on CPU, punctuated), split into
  sentence-sized cues using token timestamps.
- **Speakers:** sherpa-onnx diarization (pyannote segmentation-3.0, WeSpeaker ResNet34, clustering threshold 0.7;
  about 112 s per 44 min). Each cue gets `<v Speaker N>`, which Pocket Casts shows as speaker labels. Models are in
  `hub_models/` (gitignored, from the sherpa-onnx GitHub releases). One person can still end up with two numbers,
  e.g. a host over intro music.
- **Running it:** `hub/run.sh` (a transient unit), or the home-manager service in `hub/mimi-hub.service.nix`.
- **Tests:** `hub/test_hub.py` (upload) and `hub/test_stream.py` (progressive streaming); both run in-process,
  with ffmpeg on PATH.
