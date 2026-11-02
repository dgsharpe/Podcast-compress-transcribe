"""Parakeet TDT's FastConformer encoder as transformers' native ParakeetEncoder, for onnx_asr on the server's GPU.

The hub's first GPU encoder (mimi_hub.TorchEncoder) is the ONNX graph converted op by op with onnx2torch: ~3,000
small ops (reshape / gather / constant ...), which torch.compile can't trace. transformers ships the same
architecture natively (SDPA attention). NVIDIA publishes only a .nemo for parakeet-tdt-0.6b-v2, so its encoder
weights are mapped onto ParakeetEncoder here: model_config.yaml and model_weights.ckpt are read straight out of the
.nemo tar (weights with torch.load(weights_only=True)), renamed with transformers' NeMo -> HF table
(models/parakeet/convert_nemo_to_hf.py), and cached as a safetensors state dict.
"""

import math
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

    # Compiled, the time dim is padded up to a multiple of BUCKET frames (100 / s; segments are <= 20 s) and kept
    # static, the batch dim dynamic: on an H100 (64 x 20 s) static shapes 111 -> 67 ms, a dynamic batch 119 -> 77 ms,
    # a dynamic time dim 119 -> 146 ms (slower than eager), both dynamic 0.84x. The encoder masks padded frames in
    # every layer, so the padding doesn't change the output. A batch of 1 is padded to 2 (one graph less).
    BUCKET = 512

    def __init__(self, encoder, dtype=torch.float16, compile=False):
        self.encoder, self.dtype, self.compiled = encoder, dtype, compile
        if compile and torch.version.hip:
            import torch._inductor.config as inductor
            # its memory-coalescing tiling analysis asserted on dynamic shapes (torch 2.12 nightly, ROCm)
            inductor.triton.coalesce_tiling_analysis = False
        self.forward = torch.compile(encoder) if compile else encoder
        if compile:  # every bucket traced now, not by the workers mid-run (two threads in one recompile failed)
            device = next(encoder.parameters()).device
            try:
                for t in range(self.BUCKET, 2048 + 1, self.BUCKET):
                    for b in (2, 3):  # a second batch size: the recompile with the batch dim dynamic, here
                        self.encode(torch.zeros(b, t, encoder.config.num_mel_bins, device=device),
                                    torch.full((b,), t, device=device))
            except Exception as e:  # noqa: BLE001 - e.g. a Python torch.compile doesn't support: eager
                print(f"parakeet_native: torch.compile failed, eager: {type(e).__name__}: {e}"[:300], flush=True)
                self.forward, self.compiled = encoder, False

    def run(self, names, feeds):
        import numpy as np
        hidden, lens = self.encode(torch.from_numpy(np.ascontiguousarray(feeds["audio_signal"].transpose(0, 2, 1))),
                                   torch.from_numpy(np.asarray(feeds["length"])))
        return [hidden.float().transpose(1, 2).cpu().numpy(), lens.cpu().numpy().astype(np.int64)]

    def encode(self, x, lengths):
        """x (B, T, F) features, lengths (B,): encoder outputs (B, T', D) and their lengths, on the GPU."""
        device = next(self.encoder.parameters()).device
        x, lengths = x.to(device, non_blocking=True), lengths.to(device, non_blocking=True)
        B, T = x.shape[:2]
        # the compiled graphs' inputs always alike (inference tensors, contiguous): a tensor made outside
        # inference_mode, or a stride-0 expand, is a different input to torch.compile, compiled again mid-run
        with torch.inference_mode():
            if self.compiled:
                x = torch.nn.functional.pad(x, (0, 0, 0, -T % self.BUCKET))
                if B == 1:
                    x, lengths = x.repeat(2, 1, 1), lengths.repeat(2)
            mask = torch.arange(x.shape[1], device=device)[None, :] < lengths[:, None]
            if self.compiled:
                for v in (x, mask):
                    torch._dynamo.maybe_mark_dynamic(v, 0)
                    torch._dynamo.mark_static(v, 1)
        with torch.inference_mode(), torch.autocast(device.type, dtype=self.dtype, enabled=self.dtype != torch.float32):
            out = self.forward(input_features=x, attention_mask=mask)
        hidden, lens = out.last_hidden_state[:B], out.attention_mask[:B].sum(-1)
        if self.compiled:  # the frames eager would return for T (each subsampling conv's output length)
            c, t = self.encoder.config, T
            k, stride = c.subsampling_conv_kernel_size, c.subsampling_conv_stride
            for _ in range(int(math.log2(c.subsampling_factor))):
                t = (t + 2 * ((k - 1) // 2) - k) // stride + 1
            hidden = hidden[:, :t]
        return hidden, lens


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
            if torch.is_tensor(encoder_out):  # straight from NativeEncoder.encode, still on the GPU
                enc, encoder_out_lens = encoder_out.to(self.device, torch.float32), encoder_out_lens.cpu().numpy()
            else:
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

    # ---- the same search as CUDA graphs (hub/bulk.py): 16 steps per launch ----

    GRAPH_STEPS = 16  # decoding steps per graph replay (one host check for "any segment still active" per replay)
    _capture_lock = __import__("threading").Lock()  # one capture at a time, across threads

    GRAPH_SIZES = (8, 16, 32, 64)  # batch sizes (segments; a batch pads to the next)
    GRAPH_FRAMES = 64  # lengths pad to a multiple of this many encoder frames

    def capture_graphs(self, slot, max_batch=64, max_frames=256):
        """Capture the graphs [slot] (one per worker thread: a graph replays into its own buffers) will use, now:
        torch.cuda.graph() synchronizes the device and empties the allocator's cache, which mid-run stalls every
        other stream (capturing on demand cost ~25% of bulk mode's throughput)."""
        for B in (b for b in self.GRAPH_SIZES if b <= max_batch):
            for T in range(self.GRAPH_FRAMES, max_frames + 1, self.GRAPH_FRAMES):
                self._graph_state(slot, B, T)

    def _graph_state(self, slot, B, T):
        """Static buffers and a captured GRAPH_STEPS-step graph for batches of B segments of up to T frames."""
        key = (slot, B, T)
        if key in getattr(self, "_graphs", {}):
            return self._graphs[key]
        with self._capture_lock:
            return self._capture(key, B, T)

    def _capture(self, key, B, T):
        import types
        dev, H = self.device, self.H
        with torch.inference_mode():
            g = types.SimpleNamespace(
                f=torch.zeros(B, T, self.enc_w.shape[1], device=dev), lens=torch.zeros(B, dtype=torch.long, device=dev),
                h=torch.zeros(2, B, H, device=dev), c=torch.zeros(2, B, H, device=dev),
                pred=torch.zeros(B, self.pred_w.shape[1], device=dev), t=torch.zeros(B, dtype=torch.long, device=dev),
                emitted=torch.zeros(B, dtype=torch.long, device=dev),
                active=torch.zeros(B, dtype=torch.bool, device=dev),
                s=torch.zeros((), dtype=torch.long, device=dev), rows=torch.arange(B, device=dev))
            S = T * (self.max_symbols + 1) + self.GRAPH_STEPS  # steps never exceed T x (max_symbols + 1)
            g.toks = torch.zeros(S, B, dtype=torch.long, device=dev)
            g.times = torch.zeros(S, B, dtype=torch.long, device=dev)
            g.emits = torch.zeros(S, B, dtype=torch.bool, device=dev)

            def steps():
                for _ in range(self.GRAPH_STEPS):
                    self._graph_step(g)

            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                steps()  # warm-up (on zeros: every segment inactive, nothing recorded that matters)
            torch.cuda.current_stream().wait_stream(side)
            g.s.zero_()
            g.graph = torch.cuda.CUDAGraph()
            # its own memory pool: graphs sharing one must replay in their capture order, batches don't
            with torch.cuda.graph(g.graph, capture_error_mode="thread_local"):
                steps()
        self._graphs = {**getattr(self, "_graphs", {}), key: g}
        return g

    def _graph_step(self, g):
        T = g.f.shape[1]
        logits = torch.addmm(self.out_b, torch.relu(g.f[g.rows, g.t.clamp(max=T - 1)] + g.pred), self.out_w)
        tok = logits[:, :self.V].argmax(-1)
        dur = self.durations[logits[:, self.V:].argmax(-1)]
        emit = g.active & (tok != self.blank)
        at = g.s.view(1)
        g.toks.index_copy_(0, at, tok[None])
        g.times.index_copy_(0, at, g.t[None])
        g.emits.index_copy_(0, at, emit[None])
        g2, h2, c2 = self._lstm(tok, g.h, g.c)
        g.pred.copy_(torch.where(emit[:, None], torch.addmm(self.pred_b, g2, self.pred_w), g.pred))
        g.h.copy_(torch.where(emit[None, :, None], h2, g.h))
        g.c.copy_(torch.where(emit[None, :, None], c2, g.c))
        emitted = g.emitted + emit
        adv = torch.where(dur > 0, dur, ((tok == self.blank) | (emitted >= self.max_symbols)).long())
        g.emitted.copy_(torch.where(adv > 0, 0, emitted))
        g.t.add_(adv * g.active)
        g.active.copy_(g.t < g.lens)
        g.s.add_(1)

    def decode_graphed(self, encoder_out, encoder_out_lens, slot=0):
        """As __call__ (tokens and frame indices per segment; no logprobs) for GPU tensors (B, T, D) / (B,), with the
        search replayed as CUDA graphs: the batch is padded to a GRAPH_SIZES size (inactive segments) and T to a
        multiple of GRAPH_FRAMES (never reached), so a few graphs serve every batch. [slot]: the calling worker's."""
        B0, T0 = encoder_out.shape[:2]
        B = next((b for b in self.GRAPH_SIZES if b >= B0), 1 << max(0, (B0 - 1).bit_length()))
        T = -(-T0 // self.GRAPH_FRAMES) * self.GRAPH_FRAMES
        g = self._graph_state(slot, B, T)
        with torch.inference_mode():
            enc = encoder_out.to(self.device, torch.float32)
            g.f.zero_()
            g.f[:B0, :T0] = torch.addmm(self.enc_b, enc.reshape(B0 * T0, -1), self.enc_w).reshape(B0, T0, -1)
            g.lens.zero_()
            g.lens[:B0] = encoder_out_lens.to(self.device)
            for x in (g.t, g.emitted, g.s):
                x.zero_()
            zeros = torch.zeros(2, B, self.H, device=self.device)
            first, h, c = self._lstm(torch.full((B,), self.blank, device=self.device), zeros, zeros.clone())
            g.h.copy_(h)
            g.c.copy_(c)
            g.pred.copy_(torch.addmm(self.pred_b, first, self.pred_w))
            g.active.copy_(g.t < g.lens)
            for _ in range(g.toks.shape[0] // self.GRAPH_STEPS):  # the most steps a batch can take
                g.graph.replay()
                if not bool(g.active.any()):
                    break
            else:
                raise RuntimeError(f"TDT graph search did not finish ({B} x {T})")
            n = int(g.s.item())
            toks, times, emits = (x[:n, :B0].cpu().numpy() for x in (g.toks, g.times, g.emits))
        for b in range(B0):
            m = emits[:, b]
            yield toks[m, b].tolist(), times[m, b].tolist(), None
