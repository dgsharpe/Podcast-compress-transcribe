"""Mimi's streaming encoder with fixed-size state, captured as one CUDA graph (for hub/bulk.py's lockstep pieces).

transformers' streaming caches (MimiConv1dPaddingCache, DynamicCache with sliding-window layers) are Python objects
that allocate new tensors on every call, and its mask / position logic runs in Python: a 20 s chunk of a batch is
hundreds of kernel launches from Python, which in bulk mode (one process launching for every episode) was the limit.
Here the same modules and weights run with fixed buffers instead:
  - each causal conv's padding state is a (B, C, pad) buffer (zeros at the start, as transformers' constant padding);
  - each transformer layer attends over a (B, H, W, D) key / value buffer of the last W = 250 tokens plus the
    chunk's own, with the sliding-window mask (kv > q - W, kv <= q) and the rotary positions computed from a position
    counter on the GPU (cached keys from before position 0 are masked out, so the first chunk needs no other path);
  - the RVQ quantizer (fp32) as in transformers.
So one graph, captured for a batch size and chunk length, serves every chunk of a stream: replay = one launch.
"""

import torch
from torch.nn import functional as F

from transformers.models.mimi.modeling_mimi import apply_rotary_pos_emb


class _PadCache:
    """Same interface as MimiConv1dPaddingCache (update returns the state, then keeps the input's tail), in buffers
    allocated on the first update."""

    def __init__(self, pads):
        self.pads, self.buf = pads, {}

    def update(self, hidden_states, layer_idx):
        pad = self.pads[layer_idx]
        if layer_idx not in self.buf:
            self.buf[layer_idx] = hidden_states.new_zeros((*hidden_states.shape[:2], pad))
        b = self.buf[layer_idx]
        out = b.clone()
        if pad > 0:
            b.copy_(hidden_states[:, :, -pad:])
        return out

    def reset(self):
        for b in self.buf.values():
            b.zero_()


class StaticMimiEncoder:
    """Streaming encode of [batch] streams [chunk_frames] Mimi frames (12.5 Hz) at a time:
    codes = encoder(chunk) for consecutive chunks of every stream; reset() starts new streams."""

    def __init__(self, model, batch, chunk_frames, num_quantizers=32, dtype=torch.float16, graph=True):
        enc = model.encoder
        self.model, self.enc, self.nq, self.dtype = model, enc, num_quantizers, dtype
        self.seanet, self.tr, self.down = enc.encoder_seanet, enc.encoder_transformer, enc.downsample
        cfg = model.config
        self.W, self.H, self.D = cfg.sliding_window, cfg.num_attention_heads, cfg.head_dim
        self.device = next(model.parameters()).device
        names = list(self.seanet._conv1d_layer_names)
        pads = [int(self.seanet.get_submodule(n).padding_total) for n in names] + [int(self.down.padding_total)]
        self.pad = _PadCache(pads)
        self.B, self.samples = batch, chunk_frames * int(cfg.sampling_rate / cfg.frame_rate)
        with torch.inference_mode():  # all state is inference tensors (updated in place under inference mode)
            self.pos = torch.zeros((), dtype=torch.long, device=self.device)  # tokens (25 Hz) seen per stream
            self.x = torch.zeros((batch, 1, self.samples), device=self.device)
        self.kv = None  # per layer: (keys, values), each (B, H, W, D)
        self.graph, self.use_graph = None, graph

    # ---- one chunk, eager (and as captured) ----

    def _transformer(self, h):
        B, T, _ = h.shape
        W = self.W
        q_pos = self.pos + torch.arange(T, device=h.device)
        k_pos = self.pos + torch.cat([torch.arange(-W, 0, device=h.device), torch.arange(T, device=h.device)])
        mask = (k_pos[None] <= q_pos[:, None]) & (k_pos[None] > q_pos[:, None] - W) & (k_pos[None] >= 0)
        cos, sin = self.tr.rotary_emb(h, q_pos[None])
        if self.kv is None:
            self.kv = [(h.new_zeros((B, self.H, W, self.D)), h.new_zeros((B, self.H, W, self.D)))
                       for _ in self.tr.layers]
        for layer, (kc, vc) in zip(self.tr.layers, self.kv, strict=True):
            attn = layer.self_attn
            residual = h
            x = layer.input_layernorm(h)
            q = attn.q_proj(x).view(B, T, self.H, self.D).transpose(1, 2)
            k = attn.k_proj(x).view(B, T, self.H, self.D).transpose(1, 2)
            v = attn.v_proj(x).view(B, T, self.H, self.D).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            keys, values = torch.cat([kc, k.to(kc.dtype)], dim=2), torch.cat([vc, v.to(vc.dtype)], dim=2)
            out = F.scaled_dot_product_attention(q, keys.to(q.dtype), values.to(q.dtype), attn_mask=mask,
                                                 scale=attn.scaling)
            out = attn.o_proj(out.transpose(1, 2).reshape(B, T, -1))
            h = residual + layer.self_attn_layer_scale(out)
            residual = h
            h = residual + layer.mlp_layer_scale(layer.mlp(layer.post_attention_layernorm(h)))
            kc.copy_(keys[:, :, -W:])
            vc.copy_(values[:, :, -W:])
        self.pos.add_(T)
        return h

    def _step(self):
        with torch.inference_mode(), torch.autocast(self.device.type, dtype=self.dtype, cache_enabled=False):
            h = self.seanet(self.x, padding_cache=self.pad)
            h = self._transformer(h.transpose(1, 2))
            latents = self.down(h.transpose(1, 2), padding_cache=self.pad)
        with torch.inference_mode():
            return self.model.quantizer.encode(latents.float(), self.nq).transpose(0, 1)  # (B, nq, frames)

    # ---- public ----

    @torch.inference_mode()
    def reset(self):
        self.pad.reset()
        self.pos.zero_()
        if self.kv is not None:
            for k, v in self.kv:
                k.zero_()
                v.zero_()

    @torch.inference_mode()
    def capture(self):
        """Allocate the state with two eager chunks, then capture one chunk as a CUDA graph."""
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self._step()
        torch.cuda.current_stream().wait_stream(side)
        self.reset()
        self.graph = torch.cuda.CUDAGraph()
        # thread_local: other threads may keep launching work (and allocating) while this one captures
        with torch.cuda.graph(self.graph, capture_error_mode="thread_local"):
            self.codes = self._step()
        self.reset()

    @torch.inference_mode()
    def encode(self, chunk):
        """chunk: (B, 1, samples) float32 audio on the GPU (the next chunk of each stream) -> (B, nq, frames)."""
        self.x.copy_(chunk)
        if self.use_graph and self.graph is None:
            self.capture()
        if self.graph is None:
            return self._step()
        self.graph.replay()
        return self.codes.clone()
