# Third-party models and libraries

None of these are stored in this repository: `android/fetch_deps.sh` and mimi-hub download them from their
sources, under their own terms. Builds of the Android apps (`android/mimi`, Mimi Bridge, the patched Pocket Casts)
bundle the items marked **(app)**.

## Models

| model | license | used for |
|---|---|---|
| [Kyutai Mimi](https://huggingface.co/kyutai/mimi) | CC-BY-4.0 | the codec: encoder on the hub; decoder converted to LiteRT (`mimi_dec_tx_*.tflite`, `mimi_deconly_fp16.tflite`, `mimi_rvq_f16.bin`) **(app)** |
| [NVIDIA Parakeet TDT 0.6B v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) | CC-BY-4.0 | transcription (hub); weights mapped onto transformers' ParakeetEncoder |
| [istupakov/parakeet-tdt-0.6b-v2-onnx](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx) | CC-BY-4.0 | ONNX export of the above, via onnx-asr (hub) |
| [pyannote speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) | CC-BY-4.0 (gated: accept its conditions on Hugging Face) | speaker labels (hub) |
| [pyannote segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) | MIT | speaker segmentation, sherpa-onnx path (hub) |
| [WeSpeaker voxceleb ResNet34-LM](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM) | CC-BY-4.0 | speaker embeddings, sherpa-onnx path (hub) |
| [Silero VAD](https://github.com/snakers4/silero-vad) | MIT | speech detection (hub) |

Attribution for the CC-BY-4.0 models: Mimi © Kyutai; Parakeet TDT © NVIDIA; speaker-diarization-community-1 ©
pyannoteAI; WeSpeaker © the WeSpeaker authors. Changes made here: Mimi's decoder converted to LiteRT (fp16, and a
precompiled Qualcomm NPU graph) with its RVQ codebooks exported as fp16 tables; Parakeet's encoder run in fp16.

## Libraries

| library | license | |
|---|---|---|
| [LiteRT](https://ai.google.dev/edge/litert) 2.3.2 (runtime, GPU accelerator, Qualcomm dispatch) | Apache-2.0 | **(app)** |
| [Qualcomm AI Runtime (QAIRT) 2.50](https://softwarecenter.qualcomm.com/) `libQnnHtp*`, `libQnnSystem` | Qualcomm license (in the SDK: `LICENSE.pdf`) | NPU decode **(app)**; may be distributed only in object code as part of an application |
| Pocket Casts Android ([Automattic](https://github.com/Automattic/pocket-casts-android)) | MPL-2.0 | the patched app (separate repository) |

Ad detection runs through agent CLIs (`codex`, `claude`, `agy`) logged in with their user's own account; their
providers' terms apply.
