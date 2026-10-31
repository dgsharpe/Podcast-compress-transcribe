#!/usr/bin/env bash
# Fetch the binaries the :mimi module needs but that are not in git:
#   third_party/litert/lib/arm64-v8a/libLiteRt.so   (link-time copy; the AAR from Maven packages it)
#   mimi/src/main/jniLibs/arm64-v8a/                 Qualcomm QAIRT 2.50 QNN libs for Hexagon V81
#   mimi/src/main/assets/mimi/                       litert-community/Mimi graphs + RVQ tables
# Re-running skips files already present. QAIRT is fetched member-by-member with HTTP range requests
# (scripts/remote_zip_extract.py) instead of downloading the 2.6 GB zip.
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$(cd .. && pwd)
UA="Mozilla/5.0"

mkdir -p third_party/litert/lib/arm64-v8a mimi/src/main/jniLibs/arm64-v8a mimi/src/main/assets/mimi

lib=third_party/litert/lib/arm64-v8a/libLiteRt.so
if [ ! -f "$lib" ]; then
  tmp=$(mktemp -d)
  curl -sL -o "$tmp/litert.aar" \
    https://dl.google.com/android/maven2/com/google/ai/edge/litert/litert/2.3.2/litert-2.3.2.aar
  (cd "$tmp" && unzip -q litert.aar jni/arm64-v8a/libLiteRt.so)
  cp "$tmp/jni/arm64-v8a/libLiteRt.so" "$lib"
  rm -rf "$tmp"
fi

Q=https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.50.0.260828/v2.50.0.260828.zip
J=mimi/src/main/jniLibs/arm64-v8a
need=()
for f in libQnnHtp.so libQnnHtpV81Stub.so libQnnSystem.so; do  # no libQnnHtpPrepare: NPU graph is precompiled
  [ -f "$J/$f" ] || need+=("260828/lib/aarch64-android/$f")
done
for f in libQnnHtpV81Skel.so libQairtHtpV81.so; do
  [ -f "$J/$f" ] || need+=("260828/lib/hexagon-v81/unsigned/$f")
done
if [ ${#need[@]} -gt 0 ]; then
  python3 -I "$ROOT/scripts/remote_zip_extract.py" "$Q" "$J" "${need[@]}"
fi

A=mimi/src/main/assets/mimi
for f in mimi_dec_tx_fp16.tflite mimi_deconly_fp16.tflite; do
  [ -f "$A/$f" ] || curl -sL -A "$UA" -o "$A/$f" "https://huggingface.co/litert-community/Mimi/resolve/main/$f"
done
if [ ! -f "$A/mimi_rvq_f16.bin" ]; then  # fp16 decoder tables (scripts/rvq_fp16.py; -72 dB vs fp32)
  tmp=$(mktemp -d)
  curl -sL -A "$UA" -o "$tmp/mimi_rvq.bin" https://huggingface.co/litert-community/Mimi/resolve/main/mimi_rvq.bin
  (cd "$ROOT" && uv run python scripts/rvq_fp16.py "$tmp/mimi_rvq.bin" "$OLDPWD/$A/mimi_rvq_f16.bin")
  rm -rf "$tmp"
fi
# mimi_dec_tx_npu.tflite (optional, SoC-specific) is the transformer compiled for the Fold 8's Hexagon V81:
# LiteRT's JIT cache from one NPU run of phone/libmimi/mimi_play (see phone/README.md), copied here.
[ -f "$A/mimi_dec_tx_npu.tflite" ] || echo "note: no $A/mimi_dec_tx_npu.tflite; the transformer will run on the CPU"
ls -la "$lib" "$J" "$A"
