# AGENTS.md

A guide to this repository for AI coding agents (and people): what's where, how to set a server up — at home, on
Modal, or both — and the rules for working on it. `README.md` has the details and measurements.

## What it is

A personal podcast pipeline. **mimi-hub** (`hub/`) takes an episode (its URL, a Pocket Casts link or an upload)
and makes:

- a `.mimi` file: the audio in Kyutai's Mimi neural codec, ~2 MB per hour, playable while it's still being made;
- a WebVTT transcript (Parakeet TDT ASR, Silero VAD) with speaker labels (pyannote, or sherpa-onnx on the CPU);
- ad and self-promotion marks (`<id>.segments.json`), found by an LLM reading the transcript.

The patched Pocket Casts app (`pocket-casts-android/`, its own repository, branch `mimi`) downloads and streams
those, shows the transcript and skips the ads. It finds an episode on the hub by its Pocket Casts episode UUID.

## Layout

| Path | What |
|---|---|
| `hub/mimi_hub.py` | the server: HTTP API (module docstring), job queue, live `.mimi` streaming, remote GPU + relay |
| `hub/segments.py` | ad detection: backends (`claude` / `codex` / `agy` CLIs, OpenRouter, Inception), the prompt, `A>B` refine |
| `hub/modal_app.py` | the hub on Modal: scale-to-zero GPU, GPU memory snapshots, Volumes, keep-alive while busy |
| `hub/bulk.py` | batch conversion: one process, every model's GPU work batched across episodes |
| `hub/aoti.py`, `parakeet_native.py`, `pyannote_gpu.py`, `mimi_static.py`, `trt_seanet.py` | GPU speedups (AOTInductor, CUDA graphs, TensorRT) |
| `hub/serve.sh`, `hub/mimi-hub.service`, `hub/setup_{rocm,cuda,mac}.sh`, `hub/fetch_models.sh`, `hub/prefetch.py` | running it at home |
| `scripts/podcast_manifest.py`, `modal_backlog.py`, `import_modal_library.py`, `backlog_ads.py` | a podcast's back catalog in bulk on Modal (below) |
| `scripts/remote_episode.py` | times a cold start and one episode against a running hub |
| `scripts/ads_bench.py`, `ads_truth.py`, `ads_disagree.py` | ad detection scored against judged ground truth (`data/ads_truth*.json`) |
| `scripts/modal_hub_bench.py`, `bulk_bench.py`, ... | GPU benchmarks; `modal_hub_bench.py` also defines the bulk image |
| `android/`, `phone/` | the on-phone Mimi decoder (the app includes `android/mimi`) |
| `data/`, `hub_data/`, `hub_models/`, `outputs/` | local state, git-ignored (`hub_data/` is the hub's library) |

## Pick a setup

| Setup | For | Ad detection |
|---|---|---|
| **A. Home server** | a decent GPU at home (Radeon 8060S: ~97x real time for a backlog, a 44 min episode in ~45 s) | the home server's `claude` login (free per episode) |
| **B. Modal only** | no machine of your own; billed per second while up | OpenRouter (Haiku 5.5, ~$0.004 per hour of audio), the key in the server's secret or sent by the app |
| **C. Home + Modal** | a small or slow home GPU, or none: the home server keeps the library, answers the app and runs the LLM; Modal converts | the home server's `claude` login |

For a weak GPU, C with `MIMI_HUB_REMOTE_WHEN=always` sends every conversion to Modal (uploads and episodes that keep
their source still convert at home); `busy` (the default) only what the home GPU can't start right away.

## A: home server

```sh
uv sync && hub/setup_cuda.sh          # or setup_rocm.sh / setup_mac.sh; hub/serve.sh picks the GPU environment
hub/fetch_models.sh                   # the CPU diarizer's models
# speaker labels on the GPU (gated): accept pyannote/speaker-diarization-community-1's conditions on huggingface.co
HF_TOKEN=<token> hf download pyannote/speaker-diarization-community-1
claude                                # log in once: ad detection uses this login (or codex)
hub/serve.sh --host <tailscale-ip> --port 8765    # or as a systemd user service: hub/mimi-hub.service, README
```

The app: Settings > Advanced > Mimi server: `http://<tailscale-ip>:8765`.

## B: Modal

1. A Modal account and its CLI: `pip install modal` (or `uvx --from modal modal ...`), then `modal setup`.
2. Speaker labels (optional; without them the CPU diarizer): after the `hf download` above,
   `cp -rL ~/.cache/huggingface/hub/models--pyannote--speaker-diarization-community-1/snapshots/*/ data/modal_pyannote`
   (baked into the images), or `HF_TOKEN` in the secret (the hub's image only).
3. A token for the server (its URL is public) and the secret:
   ```sh
   mkdir -p ~/.config/mimi-hub && python3 -c 'import secrets; print(secrets.token_urlsafe(32))' > ~/.config/mimi-hub/modal-token
   chmod 600 ~/.config/mimi-hub/modal-token
   modal secret create mimi-hub MIMI_HUB_TOKEN=$(cat ~/.config/mimi-hub/modal-token) [OPENROUTER_API_KEY=<key>]
   ```
4. `modal deploy hub/modal_app.py` prints `https://<workspace>--mimi-hub-serve.<region>.modal.run`.
5. Wake it once: `python scripts/remote_episode.py https://<...>.modal.run/t/<token>`. The first start converts the
   models and compiles their kernels (5-10 min, kept on the `mimi-hub-models` Volume); the next 2-3 cold starts
   (~1 min) make Modal's memory snapshots; after that a cold start has the models ready in ~8 s, and a 44 min
   episode plays after ~2 s and is transcribed in ~12 s.
6. The app: the server URL `https://<...>.modal.run/t/<token>` (the app sends no header: the token goes in the
   path), and Settings > Advanced > Mimi server > OpenRouter key if the secret has none.

Cost (L40S + 16 cores, ~$3/h while up): ~$0.02 a launch (~$0.05 when the app wakes it on opening), ~$0.02 per hour
of audio; Modal's free plan includes $30 a month. Settings at deploy time (`MIMI_MODAL_GPU`, `_CPU`, `_IDLE_S`, ...):
`hub/modal_app.py`'s docstring.

## C: home server + Modal

Set up A and B (B's secret needs no OpenRouter key), then on the home server:

```sh
cat >> ~/.config/mimi-hub/hub.env <<'END'
MIMI_HUB_REMOTE=https://<workspace>--mimi-hub-serve.<region>.modal.run
MIMI_HUB_REMOTE_TOKEN_FILE=~/.config/mimi-hub/modal-token
MIMI_HUB_REMOTE_WHEN=busy        # or always (a weak home GPU), priority, off
END
systemctl --user restart mimi-hub
```

The app points at the home server only. A job on Modal is converted there without ad detection, copied back
(`.vtt`, `.mimi`), and its ads found at home; the home server relays Modal's `.mimi` while it's being made, and redoes
at home anything Modal fails. Check: submit three episodes at once (2 local workers) and one goes to Modal
(`"device": "remote: cuda"` in its record).

## Backlogs: a back catalog in bulk

From the app: a podcast's settings > Convert episodes on Mimi server (all, the latest N, or a date range; it shows
the episodes, hours, where it runs and a rough cost first, then progress), or Send to Mimi server on more than 10
selected episodes. The hub (`hub/backlog.py`, `POST /v1/backlog`) puts under 3 h of new audio in its own queue, and
anything bigger, where the `modal` CLI is set up on it, through the steps below by itself: one Modal bulk run, each
chunk imported as it lands, ads and summaries with the hub's own logins. By hand:

```sh
python scripts/podcast_manifest.py "revolutions mike duncan" > /tmp/revolutions.json   # --before / --after dates
uvx --from modal modal run scripts/modal_backlog.py --manifest /tmp/revolutions.json   # H100, ~1,350x real time
#   --no-compress: transcripts only (no .mimi; the app plays the original); --keep-source: the originals too (.src)
uvx --python 3.12 --with modal python scripts/import_modal_library.py /tmp/revolutions.results.json   # into hub_data/
python scripts/backlog_ads.py /tmp/revolutions.results.json                            # ads: claude's Haiku 5.5
python scripts/backlog_ads.py /tmp/revolutions.results.json --refine claude:claude-opus-5-5   # Opus on the edges
```

2026-11-02: 1,131 episodes (912 h) converted in ~45 min (~$5), none failed; ads in 12 min (Haiku) + 20 min (Opus,
7.75M input tokens). A run resumes where it stopped (episodes already in the Modal library are skipped). Without
`data/modal_pyannote` the backlog has no speaker labels. In setup B, skip the import and run `backlog_ads.py
--volume mimi-hub-data --segmenter openrouter:anthropic/claude-haiku-5.5` with `OPENROUTER_API_KEY` set.

## Ad detection

`MIMI_HUB_SEGMENTER` is a comma-separated list tried in order; a CLI that isn't installed and an API without a key
are skipped. Default: `claude:claude-haiku-5-5>claude:claude-opus-5-5` (Haiku reads the transcript, Opus re-reads
60 s around each break it found: Opus's scores for 13.5% of its reading), then OpenRouter Haiku, then `codex`. Before
changing the prompt, the models or the merging, score the change: `scripts/ads_bench.py --truth data/ads_truth.json
--prompt v2 <spec>` then `scripts/ads_truth.py --score <runs.json>` (and `--fresh` on `data/fresh_hub`).

## Rules

- **Cloud GPUs cost money.** Every Modal script has timeouts and cancels its calls when stopped; keep it that way.
  After any Modal work, check `modal container list` is empty (a deployed hub scales to zero by itself, ~20 s after
  its last request) and `modal app stop` ephemeral apps you started. Never set `min_containers`.
- **Secrets.** Never commit or print tokens or API keys. The Modal server's token lives in its secret and in
  `~/.config/mimi-hub/modal-token` (0600); a URL with `/t/<token>` shows up in Modal's request logs, so don't paste
  logs that contain one. The app's OpenRouter key is held in the hub's memory only.
- **The library.** `hub_data/<id>.json` records are changed under the hub's lock (`hub_data/.records.lock`, as
  `scripts/backlog_ads.py` does); write a record last, after the files it describes.
- **Services.** The home hub is a systemd user service (`systemctl --user restart mimi-hub`); its unit may be
  generated (home-manager), so put settings in `~/.config/mimi-hub/hub.env`, not the unit.
- **Measure before claiming.** Speed and quality numbers in docs and commits come from a run (`bulk_bench.py`,
  `remote_episode.py`, `ads_bench.py`), with the date.
