#!/usr/bin/env bash
# Build .venv-mac: mimi-hub's GPU environment on Apple silicon (PyTorch's Metal backend, "mps"), next to the CPU
# .venv. PyPI's macOS arm64 torch wheels include Metal; onnxruntime's CPU wheel runs Parakeet's preprocessing, VAD
# and decoder on the CPU, as on the other GPUs.   usage: hub/setup_mac.sh
set -euo pipefail
cd "$(dirname "$0")/.."
uv venv -q --allow-existing .venv-mac -p 3.12
export VIRTUAL_ENV=$PWD/.venv-mac
uv pip install -q torch torchaudio
uv pip freeze | grep -E '^torch(audio)?==' > .venv-mac/constraints.txt  # keep these torch builds
uv pip install -q -c .venv-mac/constraints.txt "transformers>=4.50" soundfile librosa scipy "fastapi>=0.162" \
  python-multipart httpx uvicorn "onnx-asr[cpu,hub]>=0.12" "sherpa-onnx>=2.0.1" "pyannote-audio>=4" safetensors pyyaml coremltools
.venv-mac/bin/python -c \
  "import torch; assert torch.backends.mps.is_available(), 'no Metal GPU'; print(torch.__version__, 'mps')"
