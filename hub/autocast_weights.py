"""Store the weights of the layers autocast runs in fp16 (linear, convolutions, recurrent) in fp16 ahead of time.

Under torch.autocast those layers convert their fp32 weights to fp16 on every call (thousands of copy kernels per
episode, ~5% of a loaded H100's GPU time). Converted once, they compute the same numbers; normalization layers,
which autocast runs in fp32, keep their fp32 weights. The converted modules must then only run under autocast.
"""

import os

import torch

ENABLED = os.environ.get("MIMI_HUB_FP16_WEIGHTS", "1") != "0"
_KINDS = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d,
          torch.nn.LSTM, torch.nn.GRU)


def half_weights(*modules, dtype=torch.float16):
    """Convert the autocast-to-[dtype] layers inside [modules], in place; returns the number converted."""
    if not ENABLED:
        return 0
    n = 0
    for module in modules:
        for m in module.modules():
            if isinstance(m, _KINDS):
                m.to(dtype)
                n += 1
    return n
