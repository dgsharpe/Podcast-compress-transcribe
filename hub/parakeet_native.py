"""Parakeet TDT's FastConformer encoder as transformers' native ParakeetEncoder, for onnx_asr on the server's GPU.

The hub's first GPU encoder (mimi_hub.TorchEncoder) is the ONNX graph converted op by op with onnx2torch: ~3,000
small ops (reshape / gather / constant ...), which torch.compile can't trace. transformers ships the same
architecture natively (SDPA attention). NVIDIA publishes only a .nemo for parakeet-tdt-0.6b-v2, so its encoder
weights are mapped onto ParakeetEncoder here: model_config.yaml and model_weights.ckpt are read straight out of the
.nemo tar (weights with torch.load(weights_only=True)), renamed with transformers' NeMo -> HF table
(models/parakeet/convert_nemo_to_hf.py), and cached as a safetensors state dict.
"""

import re
import tarfile
from pathlib import Path

import torch

NEMO_REPO = {"nemo-parakeet-tdt-0.6b-v2": ("nvidia/parakeet-tdt-0.6b-v2", "parakeet-tdt-0.6b-v2.nemo")}

# transformers' NEMO_TO_HF_WEIGHT_MAPPING, encoder entries
KEY_MAP = [
    (r"encoder\.pre_encode\.conv\.", "encoder.subsampling.layers."),
    (r"encoder\.pre_encode\.out\.", "encoder.subsampling.linear."),
    (r"encoder\.pos_enc\.", "encoder.encode_positions."),
    (r"encoder\.layers\.(\d+)\.conv\.batch_norm\.", r"encoder.layers.\1.conv.norm."),
    (r"linear_([kv])", r"\1_proj"),
    (r"linear_out", "o_proj"),
    (r"linear_q", "q_proj"),
    (r"pos_bias_([uv])", r"bias_\1"),
    (r"linear_pos", "relative_k_proj"),
]
CONFIG_MAP = {"d_model": "hidden_size", "n_heads": "num_attention_heads", "n_layers": "num_hidden_layers",
              "feat_in": "num_mel_bins", "conv_kernel_size": "conv_kernel_size",
              "subsampling_factor": "subsampling_factor", "subsampling_conv_channels": "subsampling_conv_channels",
              "pos_emb_max_len": "max_position_embeddings", "dropout": "dropout", "dropout_emb": "dropout_positions",
              "dropout_att": "attention_dropout", "xscaling": "scale_input", "use_bias": "attention_bias",
              "stochastic_depth_drop_prob": "layerdrop"}


def _read_member(tar, name):
    member = next(m for m in tar.getmembers() if m.name.lstrip("./") == name)
    return tar.extractfile(member)


def convert(asr_model, out_dir):
    """Write out_dir/config.json + encoder.safetensors for [asr_model] (once; ~2.4 GB read)."""
    import yaml
    from huggingface_hub import hf_hub_download
    from safetensors.torch import save_file
    from transformers import ParakeetEncoderConfig

    out_dir = Path(out_dir)
    if (out_dir / "encoder.safetensors").exists():
        return out_dir
    repo, filename = NEMO_REPO[asr_model]
    nemo = hf_hub_download(repo, filename)
    with tarfile.open(nemo) as tar:
        cfg = yaml.safe_load(_read_member(tar, "model_config.yaml"))["encoder"]
        state = torch.load(_read_member(tar, "model_weights.ckpt"), map_location="cpu", weights_only=True)
    conf = {"layerdrop": 0.0}
    for key, value in cfg.items():
        if key in CONFIG_MAP:
            conf[CONFIG_MAP[key]] = value
            if key == "use_bias":
                conf["convolution_bias"] = value
    config = ParakeetEncoderConfig(**conf)
    encoder = {}
    for key, value in state.items():
        if not key.startswith("encoder."):
            continue
        for pattern, repl in KEY_MAP:
            key = re.sub(pattern, repl, key)
        encoder[key[len("encoder."):]] = value.contiguous()
    out_dir.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(out_dir)
    save_file(encoder, str(out_dir / "encoder.safetensors"))
    return out_dir


def load(model_dir, device="cuda"):
    """ParakeetEncoder with the converted weights (strict: every weight must map), eval, on [device]."""
    from safetensors.torch import load_file
    from transformers import ParakeetEncoder, ParakeetEncoderConfig

    config = ParakeetEncoderConfig.from_pretrained(model_dir)
    config._attn_implementation = "sdpa"
    encoder = ParakeetEncoder(config)
    state = load_file(str(Path(model_dir) / "encoder.safetensors"))
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    if missing or unexpected:
        raise RuntimeError(f"weights don't map: missing {missing[:5]}, unexpected {unexpected[:5]}")
    encoder = encoder.eval()
    if torch.version.hip:  # MIOpen's batch norm fails to compile on gfx1151 (as in pyannote_gpu)
        from hub.pyannote_gpu import _freeze_bn
        _freeze_bn(encoder)
    return encoder.to(device)


class NativeEncoder:
    """Stands in for onnx_asr's encoder session: run(names, {"audio_signal": (B, mels, T), "length": (B,)})
    -> [encoder outputs (B, D, T'), encoded lengths (B,)], like the exported ONNX encoder."""

    def __init__(self, encoder, dtype=torch.float16, compile=False):
        self.encoder, self.dtype = encoder, dtype
        if compile:
            import torch._inductor.config as inductor
            # its memory-coalescing tiling analysis asserts on these dynamic shapes (torch 2.12 nightly, ROCm)
            inductor.triton.coalesce_tiling_analysis = False
        self.forward = torch.compile(encoder, dynamic=True) if compile else encoder

    def run(self, names, feeds):
        import numpy as np
        device = next(self.encoder.parameters()).device
        x = torch.from_numpy(np.ascontiguousarray(feeds["audio_signal"].transpose(0, 2, 1))).to(device)  # (B, T, F)
        lengths = torch.from_numpy(np.asarray(feeds["length"])).to(device)
        mask = torch.arange(x.shape[1], device=device)[None, :] < lengths[:, None]
        with torch.inference_mode(), torch.autocast("cuda", dtype=self.dtype, enabled=self.dtype != torch.float32):
            out = self.forward(input_features=x, attention_mask=mask)
        hidden = out.last_hidden_state.float().transpose(1, 2)  # (B, D, T')
        return [hidden.cpu().numpy(), out.attention_mask.sum(-1).cpu().numpy().astype(np.int64)]
