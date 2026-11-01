#!/usr/bin/env bash
# Run mimi-hub (uvicorn args pass through), in .venv-rocm (GPU, see hub/setup_rocm.sh) if it exists, else the uv
# project .venv (CPU). The systemd unit (nix-config home/server.nix) execs this.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -x .venv-rocm/bin/uvicorn ]; then
  exec .venv-rocm/bin/uvicorn hub.mimi_hub:app "$@"
fi
exec uv run uvicorn hub.mimi_hub:app "$@"
