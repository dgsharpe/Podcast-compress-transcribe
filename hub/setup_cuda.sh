#!/usr/bin/env bash
# Build .venv-cuda: mimi-hub's GPU environment for an NVIDIA card (CUDA 12.8 torch wheels), next to the CPU .venv.
# The AMD counterpart is hub/setup_rocm.sh; hub/serve.sh uses whichever exists. The GPU paths here were developed on
# ROCm; the same torch code (Mimi, the native Parakeet encoder, batched pyannote) also ran on NVIDIA in
# scripts/modal_bench.py.   usage: hub/setup_cuda.sh
set -euo pipefail
cd "$(dirname "$0")/.."
IDX=https://download.pytorch.org/whl/cu128
uv venv -q --allow-existing .venv-cuda -p 3.12
export VIRTUAL_ENV=$PWD/.venv-cuda
uv pip install -q --index-url "$IDX" torch torchaudio torchvision
uv pip freeze | grep -E '^torch(audio|vision)?==' > .venv-cuda/constraints.txt  # keep these torch builds
uv pip install -q -c .venv-cuda/constraints.txt "transformers>=4.50" soundfile librosa scipy "fastapi>=0.162" \
  python-multipart httpx uvicorn "onnx-asr[cpu,hub]>=0.12" "sherpa-onnx>=2.0.1" onnx onnx2torch "pyannote-audio>=4"
.venv-cuda/bin/python -c "import torch; assert torch.cuda.is_available(), 'no CUDA GPU'; print(torch.__version__, torch.cuda.get_device_name(0))"
