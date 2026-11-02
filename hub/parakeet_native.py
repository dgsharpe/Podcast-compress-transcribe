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


def load(model_dir, device="cuda", half=False):
    """ParakeetEncoder with the converted weights (strict: every weight must map), eval, on [device]. [half]: its
    linear and conv weights in fp16 (hub/autocast_weights.py: what autocast computes with anyway, so the same
    output, in half the memory), converted before it's moved."""
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
    if half:
        from hub.autocast_weights import half_weights
        half_weights(encoder)
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
        with torch.inference_mode(), torch.autocast(device.type, dtype=self.dtype, enabled=self.dtype != torch.float32):
            out = self.forward(input_features=x, attention_mask=mask)
        hidden = out.last_hidden_state.float().transpose(1, 2)  # (B, D, T')
        return [hidden.cpu().numpy(), out.attention_mask.sum(-1).cpu().numpy().astype(np.int64)]


# ---------- TDT decoding ----------
#
# onnx_asr decodes one segment at a time, one decoder_joint ONNX call per step on the CPU, and each call re-runs the
# prediction network (2-layer LSTM) on the last token even when nothing new was emitted: ~2,300 calls and ~1.4 s per
# 5 min of audio, the largest CPU cost of ASR on fast GPUs. NativeTdtDecoder runs the same greedy TDT search for a
# whole batch of segments in lockstep on the GPU: the encoder half of the joint once for all frames, then per step
# the joint for every segment and the LSTM only where a token was emitted.

def convert_decoder(asr_model, out_dir):
    """Write out_dir/decoder.safetensors (prediction network + joint weights from the .nemo; once)."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import save_file

    out = Path(out_dir) / "decoder.safetensors"
    if out.exists():
        return out
    repo, filename = NEMO_REPO[asr_model]
    with tarfile.open(hf_hub_download(repo, filename)) as tar:
        state = torch.load(_read_member(tar, "model_weights.ckpt"), map_location="cpu", weights_only=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in state.items() if k.startswith(("decoder.", "joint."))}, str(out))
    return out


class NativeTdtDecoder:
    """Stands in for onnx_asr's _AsrWithTransducerDecoding._decoding (greedy TDT, same rules: a token is emitted if
    the argmax isn't blank; the duration head's argmax moves t, else blank or [max_symbols] emissions move it 1)."""

    def __init__(self, path, device, blank, vocab_size, durations=(0, 1, 2, 3, 4), max_symbols=10):
        from safetensors.torch import load_file
        w = {k: v.to(device) for k, v in load_file(str(path)).items()}
        p = "decoder.prediction."
        self.H = w[p + "dec_rnn.lstm.weight_hh_l0"].shape[1]
        b0 = w[p + "dec_rnn.lstm.bias_ih_l0"] + w[p + "dec_rnn.lstm.bias_hh_l0"]
        # layer 0's input projection for every token, so a step gathers a row instead of a matmul
        self.x0 = torch.addmm(b0, w[p + "embed.weight"], w[p + "dec_rnn.lstm.weight_ih_l0"].T)
        self.whh0 = w[p + "dec_rnn.lstm.weight_hh_l0"].T.contiguous()
        self.wih1 = w[p + "dec_rnn.lstm.weight_ih_l1"].T.contiguous()
        self.whh1 = w[p + "dec_rnn.lstm.weight_hh_l1"].T.contiguous()
        self.b1 = w[p + "dec_rnn.lstm.bias_ih_l1"] + w[p + "dec_rnn.lstm.bias_hh_l1"]
        self.pred_w, self.pred_b = w["joint.pred.weight"].T.contiguous(), w["joint.pred.bias"]
        self.enc_w, self.enc_b = w["joint.enc.weight"].T.contiguous(), w["joint.enc.bias"]
        self.out_w, self.out_b = w["joint.joint_net.2.weight"].T.contiguous(), w["joint.joint_net.2.bias"]
        self.device, self.blank, self.V, self.max_symbols = device, blank, vocab_size, max_symbols
        self.durations = torch.tensor(durations, device=device)

    def _cell(self, gates, c):
        H = self.H
        s = torch.sigmoid(gates)
        c = s[:, H:2 * H] * c + s[:, :H] * torch.tanh(gates[:, 2 * H:3 * H])
        return s[:, 3 * H:] * torch.tanh(c), c

    def _lstm(self, tok, h, c):
        """One prediction-network step from state (h, c) (each (2, B, H)) on tokens [tok]: output, h, c."""
        h0, c0 = self._cell(torch.addmm(self.x0[tok], h[0], self.whh0), c[0])
        h1, c1 = self._cell(torch.addmm(torch.addmm(self.b1, h0, self.wih1), h[1], self.whh1), c[1])
        return h1, torch.stack([h0, h1]), torch.stack([c0, c1])

    def __call__(self, encoder_out, encoder_out_lens, /, **kwargs):
        import numpy as np
        need_logprobs = kwargs.get("need_logprobs")
        with torch.inference_mode():
            enc = torch.from_numpy(np.ascontiguousarray(encoder_out)).to(self.device)  # (B, T, D)
            B, T = enc.shape[:2]
            f_all = torch.addmm(self.enc_b, enc.reshape(B * T, -1), self.enc_w).reshape(B, T, -1)
            lens = torch.from_numpy(np.asarray(encoder_out_lens)).to(self.device)
            rows = torch.arange(B, device=self.device)
            t = torch.zeros(B, dtype=torch.long, device=self.device)
            emitted = torch.zeros_like(t)
            h = torch.zeros(2, B, self.H, device=self.device)
            c = torch.zeros_like(h)
            g, h, c = self._lstm(torch.full((B,), self.blank, device=self.device), h, c)
            pred = torch.addmm(self.pred_b, g, self.pred_w)
            active = t < lens
            toks, times, emits, lps = [], [], [], []
            step = 0
            while step % 16 or bool(active.any()):  # one host sync every 16 steps
                logits = torch.addmm(self.out_b, torch.relu(f_all[rows, t.clamp(max=T - 1)] + pred), self.out_w)
                tok = logits[:, :self.V].argmax(-1)
                dur = self.durations[logits[:, self.V:].argmax(-1)]
                emit = active & (tok != self.blank)
                toks.append(tok)
                times.append(t)
                emits.append(emit)
                if need_logprobs:
                    lps.append(torch.log_softmax(logits[:, :self.V], -1).gather(1, tok[:, None])[:, 0])
                g2, h2, c2 = self._lstm(tok, h, c)
                pred = torch.where(emit[:, None], torch.addmm(self.pred_b, g2, self.pred_w), pred)
                h = torch.where(emit[None, :, None], h2, h)
                c = torch.where(emit[None, :, None], c2, c)
                emitted = emitted + emit
                adv = torch.where(dur > 0, dur, ((tok == self.blank) | (emitted >= self.max_symbols)).long())
                emitted = torch.where(adv > 0, 0, emitted)
                t = t + adv * active
                active = t < lens
                step += 1
            toks, times, emits = (torch.stack(x).cpu().numpy() for x in (toks, times, emits))
            lps = torch.stack(lps).cpu().numpy() if need_logprobs else None
        for b in range(B):
            m = emits[:, b]
            yield toks[m, b].tolist(), times[m, b].tolist(), lps[m, b].tolist() if need_logprobs else None
