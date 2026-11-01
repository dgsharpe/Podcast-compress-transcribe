#!/usr/bin/env bash
# Models mimi-hub doesn't download by itself, into hub_models/ (gitignored):
#   sherpa-onnx speaker diarization (pyannote segmentation-3.0 + WeSpeaker ResNet34), the CPU diarizer
#   (MIMI_HUB_DIARIZER=sherpa, and the fallback when pyannote's GPU pipeline isn't available).
# Mimi, Parakeet and Silero come from Hugging Face on first use. pyannote speaker-diarization-community-1 (the GPU
# default) is gated: accept its conditions on huggingface.co, then once `HF_TOKEN=... hf download
# pyannote/speaker-diarization-community-1`; the hub reads it from the cache without a token.
set -euo pipefail
cd "$(dirname "$0")/.."
M=hub_models
mkdir -p "$M"
R=https://github.com/k2-fsa/sherpa-onnx/releases/download
if [ ! -f "$M/sherpa-onnx-pyannote-segmentation-3-0/model.onnx" ]; then
  curl -sL "$R/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2" | tar -xj -C "$M"
fi
[ -f "$M/wespeaker_en_voxceleb_resnet34_LM.onnx" ] ||
  curl -sL -o "$M/wespeaker_en_voxceleb_resnet34_LM.onnx" \
    "$R/speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx"
ls -la "$M"
