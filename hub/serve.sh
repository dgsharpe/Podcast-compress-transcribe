#!/usr/bin/env bash
# Run mimi-hub (uvicorn args pass through), in a GPU environment (.venv-rocm / -cuda / -mac) if one exists, else the uv
# project .venv (CPU). The systemd unit hub/mimi-hub.service (see README) execs this.
set -euo pipefail
cd "$(dirname "$0")/.."
for venv in .venv-rocm .venv-cuda .venv-mac; do  # GPU environments: hub/setup_{rocm,cuda,mac}.sh
  if [ -x "$venv/bin/uvicorn" ]; then
    exec "$venv/bin/uvicorn" hub.mimi_hub:app "$@"
  fi
done
exec uv run uvicorn hub.mimi_hub:app "$@"
