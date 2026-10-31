#!/usr/bin/env bash
# Start mimi-hub on the server's Tailscale address as a transient user service (memory-capped).
#   hub/run.sh            start      |  journalctl --user -u mimi-hub -f   logs
#   systemctl --user stop mimi-hub   stop
set -euo pipefail
cd "$(dirname "$0")/.."
# ffmpeg lets the hub read any container (m4a/aac podcasts); without it only mp3/wav/flac/ogg work
command -v ffmpeg >/dev/null || exec nix shell nixpkgs#ffmpeg -c "$0" "$@"
IP=$(tailscale ip -4 | head -1)
exec systemd-run --user --unit=mimi-hub --collect -p MemoryMax=24G -p MemorySwapMax=0 \
  --working-directory="$PWD" \
  --setenv=PATH="$PATH" --setenv=MIMI_HUB_THREADS=16 \
  --setenv=SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
  "$(command -v uv)" run uvicorn hub.mimi_hub:app --host "$IP" --port 8765
