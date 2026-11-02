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
import sys
import types
from pathlib import Path

import numpy as np
import torch

MODEL = "pyannote/speaker-diarization-community-1"
BATCH_CHUNKS = int(os.environ.get("MIMI_HUB_DIAR_BATCH", "32"))
# The ResNet's frames (its convolutions: nearly all the work) in fp16 / bf16 / tf32 (NVIDIA; pyannote itself turns
# TF32 off), pooling and everything after in fp32. On a 43.8 min episode fp16, bf16 and tf32 all left the
# diarization unchanged (DER 0 vs fp32); fp16 made the embedding step 2x faster on an L40S and 13x on an H100
# (weak at plain fp32), but 2-3x slower on the server's MIOpen, so the default is fp16 on NVIDIA only. (fp16 for
# the whole model, pooling included, once merged every speaker into one.) scripts/diar_precision_check.py.
# On a Mac, "coreml" runs the frames on the Neural Engine (Core ML, fp16), off the GPU: 32 x 10 s in 0.16 s on an
# M5 vs 0.58 s for PyTorch on its GPU (Metal), relative error 1.6%. Converted once, cached in MIMI_HUB_MODELS.
FRAMES_DTYPE = os.environ.get("MIMI_HUB_DIAR_DTYPE", "auto")  # auto: fp16 on NVIDIA, coreml on macOS, else fp32
_DTYPES = {"fp32": None, "tf32": None, "fp16": torch.float16, "bf16": torch.bfloat16}
COREML_DIR = Path(os.environ.get("MIMI_HUB_MODELS", Path(__file__).resolve().parent.parent / "hub_models"))
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


def _fold_bn(module):
    """Fold each eval-mode BatchNorm2d into the Conv2d before it (WeSpeaker's convN / bnN pairs and its
    Sequential(conv, bn) shortcuts): the conv's weights scaled per channel, the norm's shift as its bias."""
    nn = torch.nn

    def fold(conv, bn):
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        bias = bn.bias - bn.running_mean * scale
        if conv.bias is not None:
            bias = bias + conv.bias * scale
        conv.weight.data.mul_(scale.view(-1, 1, 1, 1))
        conv.bias = nn.Parameter(bias.detach())

    for child in module.children():
        _fold_bn(child)
    for k in range(1, 4):
        conv, bn = getattr(module, f"conv{k}", None), getattr(module, f"bn{k}", None)
        if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
            fold(conv, bn)
            setattr(module, f"bn{k}", nn.Identity())
    if isinstance(module, nn.Sequential):
        for k in range(len(module) - 1):
            if isinstance(module[k], nn.Conv2d) and isinstance(module[k + 1], nn.BatchNorm2d):
                fold(module[k], module[k + 1])
                module[k + 1] = nn.Identity()


def _without_miopen(module):
    forward = module.forward

    def wrapped(*a, **k):
        with torch.backends.cudnn.flags(enabled=False):
            return forward(*a, **k)

    module.forward = wrapped


# The ResNet in channels-last memory layout, the one cuDNN's fp16 tensor-core convolutions use natively: the
# embedding step 1.4x faster on an L40S, 1.3x on an H100, diarization unchanged (DER 0). auto: on with NVIDIA.
# (CUDA graphs for it and the segmentation model gained nothing: both are compute-bound, not launch-bound.)
CHANNELS_LAST = os.environ.get("MIMI_HUB_DIAR_CL", "auto")
# The ResNet's batch norms folded into its convolutions: the separate normalization kernels were ~6% of a loaded
# H100's GPU time; on the server the embedding step went 10.9 -> 9.0 s, diarization unchanged (DER 0).
FOLD_BN = os.environ.get("MIMI_HUB_DIAR_FOLD", "1") == "1"
# The segmentation model's LSTM under fp16 / bf16 autocast: 1.3x faster segmentation on an L40S and an H100, DER
# 0.0002 (also on the server, where it's slower without MIOpen). auto: fp16 on NVIDIA, else fp32. (The whole model in
# fp16, its SincNet front end too, changed a 43.8 min episode's diarization by DER 0.11.)
SEG_DTYPE = os.environ.get("MIMI_HUB_SEG_DTYPE", "auto")


def _frames_mode(device):
    if FRAMES_DTYPE != "auto":
        return FRAMES_DTYPE
    if device.type == "cuda" and not torch.version.hip:
        return "fp16"
    if sys.platform == "darwin":
        import importlib.util
        if importlib.util.find_spec("coremltools"):
            return "coreml"
    return "fp32"


def _coreml_frames(resnet, fbank_shape):
    """The ResNet's forward_frames as a Core ML program on the Neural Engine, for fbank batches of [fbank_shape]."""
    import copy

    import coremltools as ct

    path = COREML_DIR / f"wespeaker-resnet34-frames-{'x'.join(map(str, fbank_shape))}.mlpackage"
    if not path.exists():
        class Frames(torch.nn.Module):
            def __init__(self, r):
                super().__init__()
                self.r = r

            def forward(self, x):
                return self.r.forward_frames(x)

        m = Frames(copy.deepcopy(resnet).float().cpu()).eval()
        x = torch.zeros(fbank_shape)
        with torch.no_grad():
            traced = torch.jit.trace(m, x)
        ml = ct.convert(traced, inputs=[ct.TensorType(name="fbank", shape=fbank_shape)],
                        outputs=[ct.TensorType(name="frames")], convert_to="mlprogram",
                        compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS15)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.mlpackage")
        ml.save(str(tmp))
        tmp.rename(path)
    return ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)


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
            fbank = model.compute_fbank(chunks)
            mode = _frames_mode(device)
            if mode == "coreml":  # fixed batch: the last one is zero-padded
                shape = (BATCH_CHUNKS, *fbank.shape[1:])
                if getattr(self, "_coreml", (None,))[0] != shape:
                    self._coreml = (shape, _coreml_frames(model.resnet, shape))
                x = np.zeros(shape, dtype=np.float32)
                x[:len(fbank)] = fbank.float().cpu().numpy()
                frames = torch.from_numpy(self._coreml[1].predict({"fbank": x})["frames"][:len(fbank)]).to(device)
            else:
                dtype = _DTYPES[mode]
                cudnn = torch.backends.cudnn.flags(enabled=torch.backends.cudnn.enabled, allow_tf32=mode == "tf32")
                with torch.autocast(device.type, dtype=dtype or torch.float32, enabled=dtype is not None), cudnn:
                    frames = model.resnet.forward_frames(fbank)
            frames = frames.float()
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
    if FOLD_BN:
        _fold_bn(pipeline._embedding.model_.eval())
    pipeline.to(torch.device(device))
    if segmentation_device:
        pipeline._segmentation.to(torch.device(segmentation_device))
    if torch.device(device).type == "cuda" and torch.version.hip:  # ROCm only; NVIDIA needs none of this
        if pipeline._segmentation.device.type == "cuda":
            _without_miopen(pipeline._segmentation.model)
        _freeze_bn(pipeline._embedding.model_)
    pipeline.get_embeddings = types.MethodType(get_embeddings, pipeline)
    pipeline._segmentation.step = STEP_FRAC * pipeline._segmentation.duration
    nvidia = torch.device(device).type == "cuda" and not torch.version.hip
    from hub.autocast_weights import half_weights
    seg_dtype = _DTYPES.get(SEG_DTYPE if SEG_DTYPE != "auto" else ("fp16" if nvidia else "fp32"))
    if seg_dtype is not None and pipeline._segmentation.device.type != "cpu":
        lstms = pipeline._segmentation.model.lstm
        for lstm in (lstms if isinstance(lstms, torch.nn.ModuleList) else [lstms]):
            half_weights(lstm, dtype=seg_dtype)  # what autocast computes with: no per-call conversion
            def lstm_forward(x, *a, _forward=lstm.forward, **k):
                with torch.autocast(x.device.type, dtype=seg_dtype):
                    out, state = _forward(x, *a, **k)
                return out.float(), state
            lstm.forward = lstm_forward
    resnet = pipeline._embedding.model_.resnet
    if CHANNELS_LAST == "1" or (CHANNELS_LAST == "auto" and nvidia):
        resnet.to(memory_format=torch.channels_last)
        resnet.conv1.register_forward_pre_hook(lambda m, args: (args[0].contiguous(memory_format=torch.channels_last),))
    frames_dtype = _DTYPES.get(_frames_mode(torch.device(device)))
    if frames_dtype is not None:  # the layers forward_frames runs under autocast; the embedding head stays fp32
        half_weights(resnet.conv1, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4, dtype=frames_dtype)
    return pipeline
