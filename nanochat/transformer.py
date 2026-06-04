import torch
import torch.nn as nn
import torch.nn.functional as F

# Our custom Flash Attention module that automatically uses
# FA3 on Hopper+ and SDPA fallback elsewhere
from nanochat.flash_attention import flash_attn
from nanochat.common import COMPUTE_DTYPE


def norm(x):
    # note that this will run in bf16, seems ok
    return F.rms_norm(x, (x.size(-1),))


class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


def has_ve(layer_idx, n_layer):
    """
    Returns True if GPT layer should have Value Embedding
    (alternating, last layer always included).
    """
    return layer_idx % 2 == (n_layer - 1) % 2


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()

        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head

        assert self.n_embd % self.n_head == 0
        assert (
            self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        )

        self.c_q = Linear(
            self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(
            self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(
            self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)

        self.ve_gate_channels = 12

        # in case BLT config changes
        use_value_embeddings = getattr(config, "use_value_embeddings", False)
        if use_value_embeddings and has_ve(layer_idx, config.n_layer):
            self.ve_gate = Linear(
                self.ve_gate_channels, self.n_kv_head, bias=False)
        else:
            self.ve_gate = None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer):
        # mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)

            # (B, T, n_kv_head), range (0, 3)
            gate = 3 * torch.sigmoid(
                self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative
        # positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm

        # sharper attention (split scale between Q and K),
        # TODO think through better
        q = q * 1.2
        k = k * 1.2

        # Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)
        # window_size is (left, right) tuple:
        #   (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(
                q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache
            # management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class TransformerTrunk(nn.Module):
    def __init__(self, config, n_value_embeddings=None):
        super().__init__()
        self.config = config

        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple:
        #   (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        self.h = nn.ModuleList([
            Block(config, layer_idx) for layer_idx in range(config.n_layer)
        ])

        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas:
        #   scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas:
        #   blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        # fake inits, real init in init_weights()
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))

        # Backout:
        #   subtract cached mid-layer residual before final norm to remove
        #   low-level features
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))

        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim

        # Value embeddings (ResFormer-style):
        #   alternating layers, last layer always included
        self.use_value_embeddings = getattr(
            config, "use_value_embeddings", False)
        if self.use_value_embeddings:
            assert n_value_embeddings is not None, (
                "n_value_embeddings is None but "
                "config.use_value_embeddings is True"
            )
            self.value_embeds = nn.ModuleDict({
                str(i): nn.Embedding(n_value_embeddings, kv_dim) \
                    for i in range(config.n_layer) if has_ve(i, config.n_layer)
            })
        else:
            self.value_embeds = nn.ModuleDict()

        # To support meta device initialization, we init the rotary embeddings
        # here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap
        # in memory, so let's just over-compute them by 10X, but assert fail if
        # we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        # 10X over-compute should be enough, TODO make nicer?
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(
            self.rotary_seq_len, head_dim)

        # persistent=False means it's not saved to the checkpoint
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)


    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Transformer blocks:
        #   uniform init with bound = sqrt(3) * std
        #       (same standard deviation as normal)
        n_embd = self.config.n_embd
        # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        s = 3**0.5 * n_embd**-0.5
        for block in self.h:
            # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)

            # projections are zero
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

            # 0.4x init scale for c_fc
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)

        # Per-layer scalars
        # Per-layer resid init:
        #   stronger residual at early layers, weaker at deep layers
        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))

        # Decaying x0 init: earlier layers get more input embedding blending
        for i in range(n_layer):
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

        # Backout scalars  must be explicitly initialized
        torch.nn.init.constant_(self.backout_lambda, 0.2)

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init with small positive values so gates start slightly
        # above neutral
        for block in self.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(
            self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        if COMPUTE_DTYPE != torch.float16:
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(
        self,
        seq_len,
        head_dim,
        base=100000,
        device=None
    ):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.get_device()
        # stride the channels
        channel_range = torch.arange(
            0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)

        # add batch and head dims for later broadcasting
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to
            (-1 = unlimited)
        - right: how many tokens after current position to attend to
            (0 for causal)

        Pattern string is tiled across layers.
        Final layer always gets L (full context).
        Characters: L=long (full context), S=short (quarter context)
        """
        pattern = config.window_pattern.upper()
        assert (
            all(c in "SL" for c in pattern)
        ), f"Invalid window_pattern: {pattern}. Use only S and L."

        # Map characters to window sizes
        long_window = config.sequence_len
        # ceil to FA3 tile size (2048 -> 768)
        short_window = -(-long_window // 4 // 128) * 128
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def matrix_params(self):
        return list(self.h.parameters())

    def value_embeds_params(self):
        return list(self.value_embeds.parameters())

    def scalar_params(self):
        return {
            "resid": [self.resid_lambdas],
            "x0": [self.x0_lambdas],
            "backout": [self.backout_lambda]
        }

    def get_device(self):
        return self.resid_lambdas.device

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())

        value_embeds_numel = sum(
            ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (
            value_embeds_numel +
            self.resid_lambdas.numel() +
            self.x0_lambdas.numel() +
            self.backout_lambda.numel()
        )

        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len

        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq

        return 6 * (nparams - nparams_exclude) + attn_flops


    def num_scaling_params(self):
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        transformer_matrices = sum(
            p.numel() for p in self.h.parameters())
        scalars = (
            self.resid_lambdas.numel() +
            self.x0_lambdas.numel() +
            self.backout_lambda.numel()
        )
        total = value_embeds + transformer_matrices + scalars
        assert (
            total == sum(p.numel() for p in self.parameters())
        ), "Parameter count mismatch"
        return {
            'value_embeds': value_embeds,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total
        }

    def forward(self, x, value_ids=None, kv_cache=None):
        B, T, _ = x.size()

        # Grab the rotary embeddings for the current sequence length
        #   (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), (
            "Sequence length grew beyond the rotary embeddings cache: "
            f"{T} > {self.cos.size(1)}"
        )
        assert x.device == self.cos.device, (
            "Rotary embeddings and x are on different devices: "
            f"{x.device} != {self.cos.device}"
        )
        if value_ids is not None:
            assert value_ids.device == self.cos.device, (
                "Rotary embeddings and value ids are on different devices: "
                f"{value_ids.device} != {self.cos.device}"
            )
        assert self.cos.dtype == COMPUTE_DTYPE, (
            "Rotary embeddings must be in "
            f"{COMPUTE_DTYPE}, got {self.cos.dtype}"
        )

        # if kv cache exists, we need to offset the rotary embeddings to the
        # current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        # truncate cache to current sequence length
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        # Forward the trunk of the Transformer
        x0 = x  # save initial normalized embedding for x0 residual
        n_layer = self.config.n_layer
        backout_layer = n_layer // 2  # cache at halfway point
        x_backout = None
        for i, block in enumerate(self.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            if self.use_value_embeddings and str(i) in self.value_embeds:
                ve = self.value_embeds[str(i)](value_ids).to(x.dtype)
            else:
                ve = None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
            if i == backout_layer:
                x_backout = x
        # Subtract mid-layer residual to remove low-level features before logit
        # projection
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)

        return x
