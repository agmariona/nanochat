import pytest
import torch

from nanochat.blt import BLT, BLTConfig
from nanochat.patching import static_patch_lengths

def tiny_config(byte_sequence_len=16, patch_size=4):
    return BLTConfig(
        byte_sequence_len=byte_sequence_len,
        static_patch_size=patch_size,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        n_layer_enc=1,
        n_head_enc=2,
        n_kv_head_enc=2,
        n_layer_dec=1,
        n_head_dec=2,
        n_kv_head_dec=2,
        window_pattern="L",
    )


def make_model(config, byte_vocab_size, device="cpu"):
    config = config
    model = BLT(config, byte_vocab_size=byte_vocab_size).to(device)
    model.init_weights()
    model.eval()
    return model


def make_batch(batch_size, seq_len, byte_vocab_size, device="cpu"):
    g = torch.Generator(device=device).manual_seed(123)
    idx = torch.randint(
        0, byte_vocab_size,
        (batch_size, seq_len),
        generator=g,
        device=device
    )
    targets = torch.randint(
        0, byte_vocab_size,
        (batch_size, seq_len),
        generator=g,
        device=device
    )
    return idx, targets


def test_blt_forward_logits_shape():
    byte_vocab_size = 262
    config = tiny_config()
    model = make_model(config, byte_vocab_size)

    idx, _ = make_batch(
        batch_size=2,
        seq_len=16,
        byte_vocab_size=byte_vocab_size
    )

    with torch.inference_mode():
        logits = model(idx)

    assert logits.shape == (2, 16, byte_vocab_size)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_blt_loss_reductions(reduction):
    byte_vocab_size = 262
    config = tiny_config()
    model = make_model(config, byte_vocab_size)
    idx, targets = make_batch(
        batch_size=2,
        seq_len=16,
        byte_vocab_size=byte_vocab_size
    )
    loss = model(idx, targets, loss_reduction=reduction)

    if reduction == "none":
        assert loss.shape == (idx.numel(),)
    else:
        assert loss.ndim == 0

    assert torch.isfinite(loss).all()


def test_blt_backward_connectivity():
    byte_vocab_size = 262
    config = tiny_config()
    model = make_model(config, byte_vocab_size)
    model.train()
    idx, targets = make_batch(
        batch_size=2,
        seq_len=16,
        byte_vocab_size=byte_vocab_size
    )
    loss = model(idx, targets)
    loss.backward()

    assert model.local_encoder.wte.weight.grad is not None
    assert model.local_encoder.h[0].attn.c_q.weight.grad is not None
    assert model.global_transformer.h[0].attn.c_q.weight.grad is not None
    assert model.local_decoder.wte.weight.grad is not None
    assert model.local_decoder.h[0].attn.c_q.weight.grad is not None
    assert model.lm_head.weight.grad is not None


def test_blt_empty_final_patch():
    """
    T=5, row_len=T+1=6, patch_size=4 -> patch_lengths [1, 4, 1]
    Final patch exists only for the extra target row position, so encoder count
    is zero.
    """
    byte_vocab_size = 262
    config = tiny_config(byte_sequence_len=5, patch_size=4)
    model = make_model(config, byte_vocab_size)
    idx, targets = make_batch(
        batch_size=2,
        seq_len=5,
        byte_vocab_size=byte_vocab_size
    )
    logits = model(idx)
    loss = model(idx, targets)

    assert logits.shape == (2, 5, byte_vocab_size)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)


def test_blt_explicit_patch_lengths():
    byte_vocab_size = 262
    B, T = 2, 16
    patch_size = 4
    config = tiny_config(byte_sequence_len=T, patch_size=patch_size)
    model = make_model(config, byte_vocab_size)
    idx, _ = make_batch(
        batch_size=B,
        seq_len=T,
        byte_vocab_size=byte_vocab_size
    )

    patch_lengths = static_patch_lengths(
        batch_size=B,
        row_len=T+1,
        patch_size=patch_size,
        device=idx.device
    )

    logits = model(idx, patch_lengths=patch_lengths)

    assert logits.shape == (B, T, byte_vocab_size)
    assert torch.isfinite(logits).all()


def test_blt_rejects_malformed_patch_lengths():
    byte_vocab_size = 262
    B, T = 2, 5
    patch_size = 4

    config = tiny_config(byte_sequence_len=T, patch_size=patch_size)
    model = make_model(config, byte_vocab_size)
    idx, _ = make_batch(
        batch_size=B,
        seq_len=T,
        byte_vocab_size=byte_vocab_size
    )

    bad_patch_lengths = torch.tensor([
        [1,0,2],
        [1,0,2],
    ])

    with pytest.raises(ValueError):
        model(idx, patch_lengths=bad_patch_lengths)


def test_blt_optimizer_one_step():
    byte_vocab_size = 262
    config = tiny_config()
    model = make_model(config, byte_vocab_size)
    opt = model.setup_optimizer()

    idx, targets = make_batch(
        batch_size=2,
        seq_len=16,
        byte_vocab_size=byte_vocab_size
    )

    loss = model(idx, targets)
    loss.backward()
    opt.step()
    opt.zero_grad()

    assert torch.isfinite(loss)
