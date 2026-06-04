"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

from dataclasses import dataclass
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.transformer import norm, Linear, TransformerTrunk
from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers.
    # Final layer always L.
    # Characters: L=long (full context), S=short (quarter context)
    # Examples:
    #   "L"=all full context,
    #   "SL"=alternating,
    #   "SSL"=two short then one long
    window_pattern: str = "SSSL"
    use_value_embeddings: bool = True


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device
        context (!!)

        Therefore, any calculations inside here are shapes and dtypes only, no
        actual data.

        => We actually initialize all data (parameters, buffers, etc.) in
        init_weights() instead.
        """
        super().__init__()
        self.config = config

        # Pad vocab for efficiency (DDP, tensor cores). This is just an
        # optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/
        #   model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // \
            pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(
                f"Padding vocab_size from {config.vocab_size} "
                f"to {padded_vocab_size} for efficiency"
            )

        self.wte = nn.Embedding(padded_vocab_size, config.n_embd)
        if config.use_value_embeddings:
            n_value_embeddings = padded_vocab_size
        else:
            n_value_embeddings = None
        self.trunk = TransformerTrunk(config, n_value_embeddings)
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)

        # Smear: mix previous token's embedding into current token
        #    (cheap bigram-like info)
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer trunk
        self.trunk.init_weights()

        # Smear scalars and smear gate must be explicitly initialized
        torch.nn.init.zeros_(self.smear_lambda)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)

        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate
        # reduced-precision embeddings and it saves memory.
        # Exception: fp16 requires fp32 embeddings because GradScaler cannot
        # unscale fp16 gradients.
        if COMPUTE_DTYPE != torch.float16:
            self.wte.to(dtype=COMPUTE_DTYPE)

    def get_device(self):
        return self.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs
            (multiply *, accumulate +) in forward, and 2X that in backward
                => 2+4=6.
        Cleanest explanation of this:
        https://medium.com/@dzmitrybahdanau/
            the-flops-calculus-of-language-model-training-3b19c1f025e4

        On top of that, 12 * h * q * effective_seq_len
            accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by
        window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).

        This is ~1% off from the exact formulas of Chinchilla paper, the
        difference is:
            - Chinchilla counts the embedding layer as flops
                (? weird, it's just a lookup => we ignore)
            - Chinchilla counts exp/sum/divide in attention softmax as flops
                (a little sus and very tiny => we ignore)
        """
        nparams = sum(p.numel() for p in self.parameters())
        trunk_params = sum(p.numel() for p in self.trunk.parameters())

        gpt_params = nparams - trunk_params
        gpt_exclude_params = (
            self.wte.weight.numel() +
            self.smear_gate.weight.numel() +
            self.smear_lambda.numel()
        )

        num_flops_per_token = (
            self.trunk.estimate_flops() +
            6 * (gpt_params - gpt_exclude_params)
        )
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
            - Kaplan et al. excluded embedding parameters
            - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361
            (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream
        analysis can experiment with which combination gives the cleanest
        scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.wte.parameters())
        trunk_params = self.trunk.num_scaling_params()
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        scalars = self.smear_gate.weight.numel() + self.smear_lambda.numel()

        total = wte + lm_head + scalars + trunk_params['total']
        assert (
            total == sum(p.numel() for p in self.parameters())
        ), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': trunk_params['value_embeds'],
            'lm_head': lm_head,
            'transformer_matrices': trunk_params['transformer_matrices'],
            'scalars': scalars + trunk_params['scalars'],
            'total': total,
        }

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

        # Separate out all parameters into groups
        embedding_params = list(self.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        matrix_params = self.trunk.matrix_params()
        value_embeds_params = self.trunk.value_embeds_params()
        trunk_scalar_params = self.trunk.scalar_params()
        resid_params = trunk_scalar_params["resid"]
        x0_params = trunk_scalar_params["x0"]
        smear_params = [
            self.smear_gate.weight,
            self.smear_lambda,
        ] + trunk_scalar_params["backout"]    # legacy grouping

        assert len(list(self.parameters())) == (
            len(matrix_params) +
            len(embedding_params) +
            len(lm_head_params) +
            len(value_embeds_params) +
            len(resid_params) +
            len(x0_params) +
            len(smear_params)
        )

        # Scale the LR for the AdamW parameters by ∝1/√dmodel
        #   (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(
            f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) "
            f"= {dmodel_lr_scale:.6f}"
        )

        # Build param_groups with all required fields explicit
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
                kind='adamw', params=value_embeds_params,
                lr=embedding_lr * dmodel_lr_scale * 0.5,
                betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01
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
                kind='adamw', params=smear_params,
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

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        # Embed the tokens
        x = self.wte(idx) # embed current token
        # ensure activations are in compute dtype
        #   (no-op usually, but active for fp16 code path)
        x = x.to(COMPUTE_DTYPE)
        x = norm(x)

        # Smear: mix previous token's embedding into current position
        #   (cheap bigram info)
        if kv_cache is None:
            # Training / naive generate:
            #   full sequence available, use fast slice
            assert T > 1, "Training forward pass should have T > 1"
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(
                self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            # KV cache inference:
            #   read prev embedding from cache, store current for next step
            x_pre_smear = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]
            if T > 1:
                # Prefill: apply smear to positions 1+, same as training
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(
                    self.smear_gate(x[:, 1:, :24]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            elif x_pre_smear is not None:
                # Decode: single token, use cached prev embedding
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(
                    self.smear_gate(x[:, :, :24]))
                x = x + gate * x_pre_smear

        # Forward the trunk of the Transformer
        x = self.trunk(x, value_ids=idx, kv_cache=kv_cache)

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]

        # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = self.lm_head(x)
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        # switch to fp32 for logit softcap and loss computation
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction
            )
            return loss
        else:
            # inference: just return the logits directly
            return logits

    @torch.inference_mode()
    def generate(
        self,
        tokens,
        max_tokens,
        temperature=1.0,
        top_k=None,
        seed=42
    ):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)

        # add batch dim
        ids = torch.tensor([tokens], dtype=torch.long, device=device)

        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(
                    probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token


# ------------------------------------------------------------------------------
# legacy state migration

_EXACT_KEY_MAP = {
    "resid_lambdas": "trunk.resid_lambdas",
    "x0_lambdas": "trunk.x0_lambdas",
    "backout_lambda": "trunk.backout_lambda"
}
_PREFIX_KEY_MAP = (
    ("transformer.wte.", "wte."),
    ("transformer.h.", "trunk.h."),
    ("value_embeds.", "trunk.value_embeds."),
)

def _migrate_gpt_key(key):
    if key in _EXACT_KEY_MAP:
        return _EXACT_KEY_MAP[key]

    for old_prefix, new_prefix in _PREFIX_KEY_MAP:
        if key.startswith(old_prefix):
            return new_prefix + key[len(old_prefix):]

    return key

def migrate_gpt_named_parameters(named_parameters):
    return {
        _migrate_gpt_key(name): param
        for name, param in named_parameters
    }

def migrate_gpt_state_dict(state_dict):
    migrated_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = _migrate_gpt_key(key)
        if new_key in migrated_state_dict:
            raise KeyError(
                f"State dict migration collision: {key} -> {new_key}"
            )
        migrated_state_dict[new_key] = value

    if hasattr(state_dict, "_metadata"):
        migrated_state_dict._metadata = OrderedDict(
            (_migrate_gpt_key(key), value)
            for key, value in state_dict._metadata.items()
        )

    return migrated_state_dict
