"""pyannote speaker-diarization-community-1 on the server's Radeon 8060S (ROCm, gfx1151), with a batched embedding step.

pyannote's own embedding step crops each 10 s chunk out of the audio in Python, then runs the whole WeSpeaker
ResNet34 once per (chunk, local speaker): 3x per chunk, only the statistics-pooling weights differing. On a 43.8 min
episode that step took 122 s on this GPU, of which ~17 s was GPU work. get_embeddings() below cuts all chunks out
of the waveform on the GPU, runs the ResNet once per chunk (forward_frames) and pools each local speaker from the
same frames (forward_embedding with (batch, speakers, frames) weights), which is what forward() computes anyway.

gfx1151 workarounds (ROCm 7.13 nightly): MIOpen's LSTM, instance norm and batch norm fail with
miopenStatusUnknownError, so the (small) segmentation model runs with MIOpen off, and the embedding model's batch
norms become fixed per-channel scale + shift (eval mode). fp16 is not used: it merged every speaker into one.
"""

import math
import os
import types

import numpy as np
import torch

MODEL = "pyannote/speaker-diarization-community-1"
BATCH_CHUNKS = 32
# Segmentation / embedding windows (10 s) start every STEP_FRAC x 10 s. pyannote's default 0.1 (1 s) sends every
# moment through the ResNet ~10 times; 0.2 halves diarization time (43.8 min: 34 -> 17 s on the server) and changed the
# output by DER 0.042 vs the default (hosts 17.5 / 14.9 -> 17.5 / 15.4 min). 0.25: 13.7 s, DER 0.057.
# scripts/diar_variants_server.py.
STEP_FRAC = float(os.environ.get("MIMI_HUB_DIAR_STEP", "0.2"))


class FrozenBN(torch.nn.Module):
    """Eval-mode BatchNorm as a per-channel scale + shift."""

    def __init__(self, bn):
        super().__init__()
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        self.register_buffer("scale", scale.detach())
        self.register_buffer("shift", (bn.bias - bn.running_mean * scale).detach())

    def forward(self, x):
        shape = (1, -1) + (1,) * (x.dim() - 2)
        return x * self.scale.view(shape) + self.shift.view(shape)


def _freeze_bn(module):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
            setattr(module, name, FrozenBN(child))
        else:
            _freeze_bn(child)


def _without_miopen(module):
    forward = module.forward

    def wrapped(*a, **k):
        with torch.backends.cudnn.flags(enabled=False):
            return forward(*a, **k)

    module.forward = wrapped


def get_embeddings(self, file, binary_segmentations, exclude_overlap=False, hook=None):
    """Drop-in for SpeakerDiarization.get_embeddings: (num_chunks, num_speakers, dimension) array."""
    sr = self._embedding.sample_rate
    waveform = file.get("waveform")
    if self.training or waveform is None or waveform.shape[0] != 1 or file.get("sample_rate") != sr:
        return type(self).get_embeddings(self, file, binary_segmentations, exclude_overlap=exclude_overlap, hook=hook)

    window = binary_segmentations.sliding_window
    data = binary_segmentations.data
    num_chunks, num_frames, num_speakers = data.shape
    masks = np.nan_to_num(data, nan=0.0).astype(np.float32)
    if exclude_overlap:  # as pyannote: drop overlapped frames unless too little speech would be left
        min_num_frames = math.ceil(num_frames * self._embedding.min_num_samples / (window.duration * sr))
        clean = np.nan_to_num(data * (np.sum(data, axis=2, keepdims=True) < 2), nan=0.0).astype(np.float32)
        masks = np.where(clean.sum(axis=1, keepdims=True) > min_num_frames, clean, masks)
    weights = torch.from_numpy(np.ascontiguousarray(masks.transpose(0, 2, 1)))  # (chunks, speakers, frames)

    # chunk sample ranges exactly as Audio.crop(mode="pad") computes them
    get = self._audio.get_num_samples
    starts = np.array([get(window[c].start, sr) for c in range(num_chunks)])
    length = get(window[0].end, sr) - starts[0]
    device = self._embedding.device
    w = waveform[0].to(device=device, dtype=torch.float32)
    pad_end = max(0, int(starts.max()) + length - w.shape[0])
    w = torch.nn.functional.pad(w, (int(max(0, -starts.min())), pad_end))
    starts = torch.from_numpy(starts - min(0, int(starts.min()))).to(device)
    offsets = torch.arange(length, device=device)

    model = self._embedding.model_
    out = []
    batch_count = math.ceil(num_chunks / BATCH_CHUNKS)
    if hook is not None:
        hook("embeddings", None, total=batch_count, completed=0)
    with torch.inference_mode():
        for k, c0 in enumerate(range(0, num_chunks, BATCH_CHUNKS), 1):
            c1 = min(num_chunks, c0 + BATCH_CHUNKS)
            chunks = w[starts[c0:c1, None] + offsets][:, None]  # (batch, 1, samples)
            frames = model.resnet.forward_frames(model.compute_fbank(chunks))
            emb = model.resnet.forward_embedding(frames, weights=weights[c0:c1].to(device))[1]
            out.append(emb.float().cpu().numpy())  # (batch, speakers, dimension)
            if hook is not None:
                hook("embeddings", None, total=batch_count, completed=k)
    return np.concatenate(out)


def load(device="cuda", path=None, segmentation_device=None):
    """The community-1 pipeline on [device] with the batched embedding step (model from the HF cache, offline).
    [segmentation_device] (default: [device]) moves just the segmentation model, e.g. to the CPU."""
    from huggingface_hub import snapshot_download
    from pyannote.audio import Pipeline
    pipeline = Pipeline.from_pretrained(path or snapshot_download(MODEL, local_files_only=True))
    pipeline.to(torch.device(device))
    if segmentation_device:
        pipeline._segmentation.to(torch.device(segmentation_device))
    if torch.device(device).type == "cuda" and torch.version.hip:  # ROCm only; NVIDIA needs none of this
        if pipeline._segmentation.device.type == "cuda":
            _without_miopen(pipeline._segmentation.model)
        _freeze_bn(pipeline._embedding.model_)
    pipeline.get_embeddings = types.MethodType(get_embeddings, pipeline)
    pipeline._segmentation.step = STEP_FRAC * pipeline._segmentation.duration
    return pipeline
