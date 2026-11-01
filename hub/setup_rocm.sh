#!/usr/bin/env bash
# Build .venv-rocm: mimi-hub's GPU environment for the server's Radeon 8060S (gfx1151), next to the CPU .venv.
# AMD's TheRock nightly torch wheels bundle their own ROCm, so nothing is needed from NixOS beyond /dev/kfd and
# /dev/dri (user in the render + video groups). torch is pinned through a constraints file: any PyPI package that
# depends on torch would otherwise swap in PyPI's CUDA build.   usage: hub/setup_rocm.sh
set -euo pipefail
cd "$(dirname "$0")/.."
IDX=https://rocm.nightlies.amd.com/v2/gfx1151/
uv venv -q --allow-existing .venv-rocm -p 3.12
export VIRTUAL_ENV=$PWD/.venv-rocm
cat > .venv-rocm/constraints.txt <<'PINS'
torch==2.12.0a0+rocm7.13.0a20260411
torchaudio==2.11.0a0+rocm7.13.0a20260411
torchvision==0.27.0a0+rocm7.13.0a20260411
PINS
uv pip install -q --prerelease=allow --index-url "$IDX" -c .venv-rocm/constraints.txt torch torchaudio torchvision numpy
uv pip install -q --prerelease=allow --extra-index-url "$IDX" --index-strategy unsafe-best-match \
  -c .venv-rocm/constraints.txt "transformers>=4.50" soundfile librosa scipy "fastapi>=0.162" python-multipart httpx \
  uvicorn "onnx-asr[cpu,hub]>=0.12" "sherpa-onnx>=2.0.1" onnx onnx2torch "pyannote-audio>=4"
.venv-rocm/bin/python -c "import torch; assert torch.cuda.is_available(), 'no ROCm GPU'; print(torch.__version__, torch.cuda.get_device_name(0))"
