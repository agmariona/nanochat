import math
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.common import COMPUTE_DTYPE


def tiny_config(vocab_size=57, sequence_len=16):
    return GPTConfig(
        sequence_len=sequence_len,
        vocab_size=vocab_size,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        window_pattern="L",
    )


def make_model(config=None, device="cpu"):
    config = config or tiny_config()
    model = GPT(config)
    model.to(device)
    model.init_weights()
    model.eval()
    return model, config


def make_batch(config, batch_size=3, device="cpu"):
    g = torch.Generator(device=device).manual_seed(123)
    idx = torch.randint(
        0, config.vocab_size,
        (batch_size, config.sequence_len),
        generator=g,
        device=device
    )
    targets = torch.randint(
        0, config.vocab_size,
        (batch_size, config.sequence_len),
        generator=g,
        device=device
    )
    return idx, targets


def test_forward_logits_shape():
    model, config = make_model()
    idx, _ = make_batch(config)
    with torch.inference_mode():
        logits = model(idx)

    assert logits.shape == (*idx.shape, config.vocab_size)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all().item()


def test_forward_loss_reductions():
    model, config = make_model()
    idx, targets = make_batch(config)
    loss = model(idx, targets)

    assert loss.ndim == 0
    assert torch.isfinite(loss).item()

    loss_none = model(idx, targets, loss_reduction="none")
    assert loss_none.shape == (math.prod(idx.shape),)
    assert torch.isfinite(loss_none).all().item()

    loss_sum = model(idx, targets, loss_reduction="sum")
    assert loss_sum.ndim == 0
    assert torch.isfinite(loss_sum).item()


def assert_finite_grad(param, name):
    assert param.grad is not None, f"{name} grad is None"
    assert (
        torch.isfinite(param.grad).all().item()
    ), f"{name} grad is not finite"


def test_backward_connectivity():
    model, config = make_model()
    model.train()
    idx, targets = make_batch(config)
    loss = model(idx, targets)
    loss.backward()

    assert len(model.transformer.h) == config.n_layer

    # embedding and unembedding
    assert_finite_grad(model.transformer.wte.weight, "wte")
    assert_finite_grad(model.lm_head.weight, "lm_head")

    # transformer blocks
    for block in model.transformer.h:
        assert_finite_grad(block.attn.c_q.weight, "attn.c_q")
        assert_finite_grad(block.attn.c_k.weight, "attn.c_k")
        assert_finite_grad(block.attn.c_v.weight, "attn.c_v")
        assert_finite_grad(block.attn.c_proj.weight, "attn.c_proj")
        assert_finite_grad(block.mlp.c_fc.weight, "mlp.c_fc")
        assert_finite_grad(block.mlp.c_proj.weight, "mlp.c_proj")

    # per-layer scalars
    assert_finite_grad(model.resid_lambdas, "resid_lambdas")
    assert_finite_grad(model.x0_lambdas, "x0_lambdas")

    # smear / backout scalars and smear gate
    assert_finite_grad(model.smear_lambda, "smear_lambda")
    assert_finite_grad(model.backout_lambda, "backout_lambda")
    assert_finite_grad(model.smear_gate.weight, "smear_gate")

    # value embeddings
    assert len(model.value_embeds) > 0
    for ve in model.value_embeds.values():
        assert_finite_grad(ve.weight, "value_embeds")

    for block in model.transformer.h:
        if block.attn.ve_gate is not None:
            assert_finite_grad(block.attn.ve_gate.weight, "ve_gate")


def test_ignore_index_targets():
    model, config = make_model()
    idx, targets = make_batch(config)
    targets = targets.clone()
    targets[:, ::2] = -1

    loss_none = model(idx, targets, loss_reduction="none")
    loss_none = loss_none.view_as(targets)

    assert torch.isfinite(loss_none).all().item()
    assert loss_none[:, ::2].eq(0).all().item()
    assert loss_none[:, 1::2].ge(0).all().item()

    loss_sum = model(idx, targets, loss_reduction="sum")
    assert torch.isfinite(loss_sum).item()
    assert torch.allclose(loss_sum, loss_none.sum())


def test_model_device():
    model, _ = make_model()
    device = model.get_device()

    for name, param in model.named_parameters():
        assert param.device == device, name

    for name, buffer in model.named_buffers():
        assert buffer.device == device, name


def test_init_weights_rotary_buffers():
    config = tiny_config()
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device="cpu")
    model.init_weights()

    assert model.cos.device == model.get_device()
    assert model.sin.device == model.get_device()
    assert model.cos.dtype == COMPUTE_DTYPE
    assert model.sin.dtype == COMPUTE_DTYPE
    assert model.cos.shape[1] >= config.sequence_len
    assert model.sin.shape == model.cos.shape
    assert torch.isfinite(model.cos).all().item()
    assert torch.isfinite(model.sin).all().item()


def test_fwd_short_seq():
    model, config = make_model()
    idx, _ = make_batch(config)
    idx = idx[:, :(config.sequence_len//2)]

    with torch.inference_mode():
        logits = model(idx)

    assert logits.shape == (*idx.shape, config.vocab_size)
    assert torch.isfinite(logits).all().item()


def test_model_accounting():
    model, _ = make_model()

    counts = model.num_scaling_params()
    assert set(counts.keys()) == set((
        "wte",
        "value_embeds",
        "lm_head",
        "transformer_matrices",
        "scalars",
        "total",
    ))
    assert counts["total"] == sum(p.numel() for p in model.parameters())
    assert counts["transformer_matrices"] > 0
    assert counts["lm_head"] > 0

    flops = model.estimate_flops()
    assert isinstance(flops, (int, float))
    assert flops > 0


def test_optimizer_setup():
    model, _ = make_model()
    optimizer = model.setup_optimizer()

    assert len(optimizer.param_groups) > 0
    for group in optimizer.param_groups:
        assert "kind" in group
        assert "initial_lr" in group
