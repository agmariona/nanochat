from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.transformer import (
    norm,
    Linear,
    Block,
    TransformerTrunk,
    precompute_rotary_embeddings
)
from nanochat.patching import (
    static_patch_lengths,
    patch_ids_from_lengths,
    decoder_patch_ids_from_lengths,
    validate_patch_lengths
)
from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW


@dataclass
class BLTConfig:
    # byte-level model
    byte_sequence_len: int = 2048
    patch_mode: str = "static"
    static_patch_size: int = 8 # -1 for not static?

    # global transformer
    @property
    def sequence_len(self):
        # compute effective latent sequence length
        assert self.patch_mode == "static"
        return 1 + math.ceil(self.byte_sequence_len / self.static_patch_size)
    n_layer: int = 12
    n_head: int = 6     # number of query heads
    n_kv_head: int = 6  # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers.
    # Final layer always L.
    # Characters: L=long (full context), S=short (quarter context)
    # Examples:
    #   "L"=all full context,
    #   "SL"=alternating,
    #   "SSL"=two short then one long
    window_pattern: str = "L"
    use_value_embeddings: bool = False

    # local encoder
    n_layer_enc: int = 2
    n_head_enc: int = 2
    n_kv_head_enc: int = 2
    window_size_enc: int | None = None

    # local decoder
    n_layer_dec: int = 2
    n_head_dec: int = 2
    n_kv_head_dec: int = 2
    window_size_dec: int | None = None

    # output
    softcap: float = 15.0

@dataclass
class BlockConfig:
    n_layer: int
    n_head: int
    n_kv_head: int
    n_embd: int
    use_value_embeddings: bool = False

class BLT(nn.Module):
    def __init__(self, config, byte_vocab_size):
        super().__init__()
        self.config = config
        self.byte_vocab_size = byte_vocab_size

        self.local_encoder = LocalEncoder(config, byte_vocab_size)
        self.global_transformer = TransformerTrunk(config)
        self.local_decoder = LocalDecoder(config, byte_vocab_size)
        self.lm_head = Linear(config.n_embd, byte_vocab_size, bias=False)

    @torch.no_grad()
    def init_weights(self):
        self.local_encoder.init_weights()
        self.global_transformer.init_weights()
        self.local_decoder.init_weights()

        # matches GPT
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

    def setup_optimizer(
        self,
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        weight_decay=0.0,
        scalar_lr=0.5
    ):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        embedding_params = (
            self.local_encoder.embedding_params() +
            self.local_decoder.embedding_params()
        )
        lm_head_params = list(self.lm_head.parameters())
        matrix_params = (
            self.local_encoder.matrix_params() +
            self.global_transformer.matrix_params() +
            self.local_decoder.matrix_params()
        )

        scalar_params = self.global_transformer.scalar_params()
        resid_params = scalar_params["resid"]
        x0_params = scalar_params["x0"]
        backout_params = scalar_params["backout"]

        # safety check: BLT does not use value embeddings
        value_embeds_params = self.global_transformer.value_embeds_params()
        assert len(value_embeds_params) == 0

        # coverage check
        all_params =(
            embedding_params +
            lm_head_params +
            matrix_params +
            resid_params +
            x0_params +
            backout_params
        )
        param_ids ={id(p) for p in all_params}
        assert len(param_ids) == len(all_params)
        assert param_ids == {id(p) for p in self.parameters()}

        # Scale the LR for the AdamW parameters by ∝1/√dmodel
        #   (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(
            f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) "
            f"= {dmodel_lr_scale:.6f}"
        )

        # Build param_groups with all required fields explicit
        # Mirrors GPT optimizer setup
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(
                kind='adamw', params=lm_head_params,
                lr=unembedding_lr * dmodel_lr_scale,
                betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01
            ),
            dict(
                kind='adamw', params=embedding_params,
                lr=embedding_lr * dmodel_lr_scale,
                betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001
            ),
            dict(
                kind='adamw', params=resid_params,
                lr=scalar_lr * 0.01,
                betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05
            ),
            # higher beta1 for x0
            dict(
                kind='adamw', params=x0_params,
                lr=scalar_lr,
                betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0
            ),
            dict(
                kind='adamw', params=backout_params,
                lr=0.2,
                betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0
            ),
        ]

        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params,
                lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9,
                weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def estimate_flops(self):
        """
        Return estimated FLOPs per input token position, matching the GPT
        training-loop accounting. For BLT, input token positions are byte IDs,
        while the global transformer positions are latent patches and must be
        scaled by tokens_per_patch.
        """
        tokens_per_patch = (
            self.config.byte_sequence_len / self.config.sequence_len
        )
        head_params = sum(p.numel() for p in self.lm_head.parameters())

        num_flops_per_token = (
            self.local_encoder.estimate_flops() +
            self.global_transformer.estimate_flops() / tokens_per_patch +
            self.local_decoder.estimate_flops() +
            6 * head_params
        )

        return num_flops_per_token

    def num_scaling_params(self):
        local_enc_wte = sum(
            p.numel() for p in self.local_encoder.embedding_params()
        )
        local_enc_matrices = sum(
            p.numel() for p in self.local_encoder.matrix_params()
        )

        local_dec_wte = sum(
            p.numel() for p in self.local_decoder.embedding_params()
        )
        local_dec_matrices = sum(
            p.numel() for p in self.local_decoder.matrix_params()
        )

        lm_head = sum(p.numel() for p in self.lm_head.parameters())

        global_params = self.global_transformer.num_scaling_params()

        total = (
           local_enc_wte + local_enc_matrices +
           local_dec_wte + local_dec_matrices +
           lm_head +
           global_params['total']
        )
        assert (
            total == sum(p.numel() for p in self.parameters())
        ), "Parameter count mismatch"

        transformer_matrices = (
            local_enc_matrices +
            local_dec_matrices +
            global_params["transformer_matrices"]
        )

        return {
            'wte': local_enc_wte + local_dec_wte,
            'transformer_matrices': transformer_matrices,
            'value_embeds': global_params['value_embeds'],
            'lm_head': lm_head,
            'scalars': global_params['scalars'],
            'total': total,
        }


    def get_device(self):
        return self.lm_head.weight.device

    def forward(
        self,
        idx,
        targets=None,
        patch_lengths=None,
        kv_cache=None,
        loss_reduction="mean"
    ):
        """
        Meta BLT Forward Flow:
        0. get BLT input
        1. patching
            1a. generate patch lengths
            1b. generate patch ids from patch_lengths
            1c. cross-attention mask
            1d. hash embeddings
            1e. N-gram embeddings
        2. local encoder
            2a. local encoder
            2b. downsampling
        3. global transformer
        4. local decoder
            4a. unpatching
            4b. generate decoder patch ids
            4c. cross-attention mask
            4d. local decoder

        We start with a much simpler forward flow.
        """

        if kv_cache is not None:
            # these are guardrails for simplified generation-only kv cache
            assert targets is None
            assert patch_lengths is None

            if hasattr(kv_cache, "blt_tokens"):
                past = kv_cache.blt_tokens
                if past.size(0) != idx.size(0):
                    past = past.expand(idx.size(0),-1)
                idx = torch.cat([past, idx], dim=1)

            kv_cache.blt_tokens = idx.detach()
            kv_cache.cache_seqlens.fill_(idx.size(1))

        B, T = idx.size()

        # -----------------------------
        # local encoder: byte-level contextualizer
        h_encoder = self.local_encoder(idx)
        D = h_encoder.shape[2]
        assert h_encoder.shape == (B,T,D)

        # -----------------------------
        # static patching
        assert self.config.patch_mode == "static"
        if patch_lengths is None:
            patch_lengths = static_patch_lengths(
                B, T+1, self.config.static_patch_size, device=idx.device
            )
        patch_lengths = patch_lengths.to(idx.device)
        # validate_patch_lengths(patch_lengths, row_len=T+1)
        P = patch_lengths.shape[1]
        # assert P <= self.config.sequence_len

        enc_patch_ids = patch_ids_from_lengths(patch_lengths, T)
        # assert enc_patch_ids.max() < P  # this shouldn't be necessary since
                                        # calling validate_patch_lengths
        assert enc_patch_ids.shape == (B,T)

        # -----------------------------
        # downsampling: patch reducer
        h_patch = h_encoder.new_zeros(B,P,D)
        h_patch.scatter_add_(
            1, enc_patch_ids.unsqueeze(-1).expand(-1,-1,D), h_encoder
        )

        counts = h_encoder.new_zeros(B, P)
        counts.scatter_add_(1, enc_patch_ids,
            torch.ones(B,T, device=h_encoder.device, dtype=h_encoder.dtype)
        ) # how many tokens contributed to each patch

        # clamp to guard against edge case (empty encoder patch)
        h_patch = h_patch / counts.clamp_min(1).unsqueeze(-1)
        assert h_patch.shape == (B,P,D)

        # -----------------------------
        # global transformer: patch-level long-context model
        h = self.global_transformer(h_patch, value_ids=None)
        assert h.shape == (B,P,D)

        # -----------------------------
        # local decoder: byte-level autoregressive reconstruction
        dec_patch_ids = decoder_patch_ids_from_lengths(patch_lengths, T)
        # assert dec_patch_ids.max() < h.shape[1]
        assert dec_patch_ids.shape == (B,T)

        # gather global patch states back to byte positions
        patch_context = torch.gather(
            h, 1, dec_patch_ids.unsqueeze(-1).expand(-1,-1,D)
        )
        assert patch_context.shape == (B,T,D)

        output = self.local_decoder(
            byte_tokens=idx,
            patch_context=patch_context
        )
        assert output.shape == (B,T,D)

        # -----------------------------
        # lm head
        logits = self.lm_head(output)
        logits = logits[..., :self.byte_vocab_size]
        logits = logits.float()
        logits = self.config.softcap * torch.tanh(logits / self.config.softcap)

        if targets is not None:
            # training
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction
            )
            return loss
        else:
            # inference
            return logits


class LocalEncoder(nn.Module):
    def __init__(self, config, byte_vocab_size):
        super().__init__()
        self.config = config
        self.byte_vocab_size = byte_vocab_size

        assert config.n_embd % config.n_head_enc == 0, (
            f"Number of local encoder heads ({config.n_head_enc}) "
            f"must divide embedding dimension ({config.n_embd})"
        )
        assert config.n_head_enc % config.n_kv_head_enc == 0, (
            f"Number of local encoder kv heads ({config.n_kv_head_enc}) "
            f"must divide number of query heads ({config.n_head_enc})"
        )

        self.enc_config = BlockConfig(
            n_layer     = config.n_layer_enc,
            n_head      = config.n_head_enc,
            n_kv_head   = config.n_kv_head_enc,
            n_embd      = config.n_embd
        )

        self.wte = nn.Embedding(byte_vocab_size, config.n_embd)
        self.h = nn.ModuleList([
            Block(self.enc_config, i) for i in range(self.enc_config.n_layer)
        ])

        self.rotary_seq_len = config.byte_sequence_len * 10
        self.head_dim = self.enc_config.n_embd // self.enc_config.n_head
        cos, sin = precompute_rotary_embeddings(
            self.get_device(), self.rotary_seq_len, self.head_dim)

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        if config.window_size_enc is not None:
            assert config.window_size_enc > 0
            self.window_size = (config.window_size_enc, 0)
        else:
            self.window_size = (config.byte_sequence_len, 0)

    @torch.no_grad()
    def init_weights(self):
        """
        Uses same initialization scheme as TransformerTrunk
        """
        # Embedding
        torch.nn.init.normal_(self.wte.weight, mean=0.0, std=0.8)
        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate
        # reduced-precision embeddings and it saves memory.
        # Exception: fp16 requires fp32 embeddings because GradScaler cannot
        # unscale fp16 gradients.
        if COMPUTE_DTYPE != torch.float16:
            self.wte.to(dtype=COMPUTE_DTYPE)

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

        # Rotary embeddings
        cos, sin = precompute_rotary_embeddings(
            self.get_device(), self.rotary_seq_len, self.head_dim)
        self.cos, self.sin = cos, sin

    def get_device(self):
        return self.wte.weight.device

    def matrix_params(self):
        return list(self.h.parameters())

    def embedding_params(self):
        return list(self.wte.parameters())

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.h.parameters())

        h = self.enc_config.n_head
        q = self.config.n_embd // self.enc_config.n_head
        t = min(self.window_size[0], self.config.byte_sequence_len)

        attn_flops = 12 * h * q * t * self.enc_config.n_layer

        return 6 * nparams + attn_flops

    def forward(self, byte_tokens: torch.Tensor):
        assert byte_tokens.device == self.cos.device, (
            "Byte tokens and rotary embeddings are on different devices: "
            f"{byte_tokens.device} != {self.cos.device}"
        )
        assert self.cos.dtype == COMPUTE_DTYPE, (
            "Rotary embeddings must be in "
            f"{COMPUTE_DTYPE}, got {self.cos.dtype}"
        )

        B, T = byte_tokens.shape

        # embed the tokens and ensure activations are in compute dtype
        x = self.wte(byte_tokens).to(COMPUTE_DTYPE)
        x = norm(x)

        # truncate cache to current sequence length
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        # forward the causal blocks
        for block in self.h:
            x = block(
                x,
                ve=None,
                cos_sin=cos_sin,
                window_size=self.window_size,
                kv_cache=None
            )

        x = norm(x)

        return x


class LocalDecoder(nn.Module):
    def __init__(self, config, byte_vocab_size):
        super().__init__()
        self.config = config
        self.byte_vocab_size = byte_vocab_size

        assert config.n_embd % config.n_head_dec == 0, (
            f"Number of local decoder heads ({config.n_head_dec}) "
            f"must divide embedding dimension ({config.n_embd})"
        )
        assert config.n_head_dec % config.n_kv_head_dec == 0, (
            f"Number of local decoder kv heads ({config.n_kv_head_dec}) "
            f"must divide number of query heads ({config.n_head_dec})"
        )

        self.dec_config = BlockConfig(
            n_layer     = config.n_layer_dec,
            n_head      = config.n_head_dec,
            n_kv_head   = config.n_kv_head_dec,
            n_embd      = config.n_embd
        )

        self.wte = nn.Embedding(byte_vocab_size, config.n_embd)
        self.h = nn.ModuleList([
            Block(self.dec_config, i) for i in range(self.dec_config.n_layer)
        ])

        self.rotary_seq_len = config.byte_sequence_len * 10
        self.head_dim = self.dec_config.n_embd // self.dec_config.n_head
        cos, sin = precompute_rotary_embeddings(
            self.get_device(), self.rotary_seq_len, self.head_dim)

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        if config.window_size_dec is not None:
            assert config.window_size_dec > 0
            self.window_size = (config.window_size_dec, 0)
        else:
            self.window_size = (config.byte_sequence_len, 0)

    @torch.no_grad()
    def init_weights(self):
        """
        Uses same initialization scheme as TransformerTrunk
        """
        # Embedding
        torch.nn.init.normal_(self.wte.weight, mean=0.0, std=0.8)
        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate
        # reduced-precision embeddings and it saves memory.
        # Exception: fp16 requires fp32 embeddings because GradScaler cannot
        # unscale fp16 gradients.
        if COMPUTE_DTYPE != torch.float16:
            self.wte.to(dtype=COMPUTE_DTYPE)

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

        # Rotary embeddings
        cos, sin = precompute_rotary_embeddings(
            self.get_device(), self.rotary_seq_len, self.head_dim)
        self.cos, self.sin = cos, sin

    def get_device(self):
        return self.wte.weight.device

    def matrix_params(self):
        return list(self.h.parameters())

    def embedding_params(self):
        return list(self.wte.parameters())

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.h.parameters())

        h = self.dec_config.n_head
        q = self.config.n_embd // self.dec_config.n_head
        t = min(self.window_size[0], self.config.byte_sequence_len)

        attn_flops = 12 * h * q * t * self.dec_config.n_layer

        return 6 * nparams + attn_flops

    def forward(self, byte_tokens: torch.Tensor, patch_context: torch.Tensor):
        assert byte_tokens.device == self.cos.device, (
            "Byte tokens and rotary embeddings are on different devices: "
            f"{byte_tokens.device} != {self.cos.device}"
        )
        assert byte_tokens.device == patch_context.device, (
            "Byte tokens and patch_context on different devices: "
            f"{byte_tokens.device} != {patch_context.device}"
        )
        assert self.cos.dtype == COMPUTE_DTYPE, (
            "Rotary embeddings must be in "
            f"{COMPUTE_DTYPE}, got {self.cos.dtype}"
        )

        B, T = byte_tokens.shape

        # embed the tokens and ensure activations are in compute dtype
        x = self.wte(byte_tokens).to(COMPUTE_DTYPE)
        x = norm(x)

        # incorporate patch context
        # basic addition follows Meta's default
        assert patch_context.shape == x.shape
        assert patch_context.dtype == x.dtype
        x = x + patch_context

        # truncate cache to current sequence length
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        # forward the causal blocks
        for block in self.h:
            x = block(
                x,
                ve=None,
                cos_sin=cos_sin,
                window_size=self.window_size,
                kv_cache=None
            )

        x = norm(x)

        return x
