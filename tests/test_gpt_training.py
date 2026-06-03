import pytest
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.engine import KVCache
from tests.reference_gpt import LegacyGPT


def tiny_config(
    sequence_len=16,
    vocab_size=57,
    n_layer=2,
    n_head=2,
    n_kv_head=2,
    n_embd=32,
    window_pattern="L"
):
    return GPTConfig(
        sequence_len=sequence_len,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_kv_head=n_kv_head,
        n_embd=n_embd,
        window_pattern=window_pattern,
    )

TEST_CONFIGS = [
    tiny_config(),
    tiny_config(n_layer=1),
    tiny_config(sequence_len=160, n_layer=3, window_pattern="SL"),
    tiny_config(n_head=4, n_kv_head=2, window_pattern="L"),
]

OPTIMIZER_TEST_CONFIGS = [
    tiny_config(
        sequence_len=160,
        n_layer=3,
        n_head=4,
        n_kv_head=2,
        window_pattern="SL"
    ),
]


def make_model(model_cls=GPT, config=None, device="cpu"):
    config = config or tiny_config()
    model = model_cls(config)
    model.to(device)
    model.init_weights()
    model.eval()
    return model, config


def make_matched_models(config=None, device="cpu"):
    config = config or tiny_config()
    new_model, _ = make_model(GPT, config)
    old_model, _ = make_model(LegacyGPT, config)
    old_model.load_state_dict(new_model.state_dict(), strict=True)
    return new_model, old_model, config


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


def make_kv_cache(model, batch_size, max_seq_len):
    config = model.config
    head_dim = config.n_embd // config.n_head

    if model.get_device().type == "cuda":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    return KVCache(
        batch_size=batch_size,
        num_heads=config.n_kv_head,
        seq_len=max_seq_len,
        head_dim=head_dim,
        num_layers=config.n_layer,
        device=model.get_device(),
        dtype=dtype,
    )


def assert_named_params_close(new_model, old_model):
    new_params = dict(new_model.named_parameters())
    old_params = dict(old_model.named_parameters())
    assert new_params.keys() == old_params.keys()

    for name, new_param in new_params.items():
        torch.testing.assert_close(new_param, old_params[name], msg=name)


def assert_named_grads_close(new_model, old_model):
    new_params = dict(new_model.named_parameters())
    old_params = dict(old_model.named_parameters())
    assert new_params.keys() == old_params.keys()

    for name, new_param in new_params.items():
        old_grad = old_params[name].grad
        new_grad = new_param.grad

        if (new_grad is None) or (old_grad is None):
            assert (new_grad is None) and (old_grad is None), name
        else:
            torch.testing.assert_close(new_grad, old_grad, msg=name)


@pytest.mark.parametrize("config", TEST_CONFIGS)
def test_forward_vs_reference_gpt(config):
    new_model, old_model, config = make_matched_models(config)
    idx, targets = make_batch(config)

    with torch.inference_mode():
        # logits
        torch.testing.assert_close(new_model(idx), old_model(idx))
        # mean loss
        torch.testing.assert_close(
            new_model(idx, targets),
            old_model(idx, targets),
        )
        # sum loss
        torch.testing.assert_close(
            new_model(idx, targets, loss_reduction="sum"),
            old_model(idx, targets, loss_reduction="sum"),
        )
        # unreduced loss
        torch.testing.assert_close(
            new_model(idx, targets, loss_reduction="none"),
            old_model(idx, targets, loss_reduction="none"),
        )


@pytest.mark.parametrize("config", TEST_CONFIGS)
def test_backward_vs_reference_gpt(config):
    new_model, old_model, config = make_matched_models(config)
    new_model.train()
    old_model.train()

    idx, targets = make_batch(config)

    new_loss = new_model(idx, targets)
    old_loss = old_model(idx, targets)

    new_loss.backward()
    old_loss.backward()

    assert_named_grads_close(new_model, old_model)


@pytest.mark.parametrize("config", OPTIMIZER_TEST_CONFIGS)
def test_onestep_vs_reference_gpt(config):
    new_model, old_model, config = make_matched_models(config)
    new_model.train()
    old_model.train()

    idx, targets = make_batch(config)

    new_opt = new_model.setup_optimizer()
    old_opt = old_model.setup_optimizer()

    new_model(idx, targets).backward()
    old_model(idx, targets).backward()

    new_opt.step()
    old_opt.step()

    assert_named_params_close(new_model, old_model)


@pytest.mark.parametrize("config", TEST_CONFIGS)
def test_init_weights_vs_reference_gpt(config):
    new_model, old_model, config = make_matched_models(config)
    torch.manual_seed(123)
    new_model.init_weights()

    torch.manual_seed(123)
    old_model.init_weights()

    assert_named_params_close(new_model, old_model)


@pytest.mark.parametrize("config", TEST_CONFIGS)
def test_state_dict_weights_vs_reference_gpt(config):
    new_model, old_model, config = make_matched_models(config)
    assert new_model.state_dict().keys() == old_model.state_dict().keys()


PREFILL_CASES = ["one", "half", "last"]

def resolve_prefill_len(case, config):
    if case == "one":
        return 1
    if case == "half":
        return config.sequence_len // 2
    if case == "last":
        return config.sequence_len - 1
    raise ValueError(case)


@pytest.mark.parametrize("config", TEST_CONFIGS)
@pytest.mark.parametrize("prefill_case", PREFILL_CASES)
def test_kv_cache_full_forward(config, prefill_case):
    new_model, old_model, config = make_matched_models(config)

    batch_size = 2
    idx, _ = make_batch(config, batch_size)

    prefill_len = resolve_prefill_len(prefill_case, config)

    with torch.inference_mode():
        full_logits = new_model(idx)
        cache = make_kv_cache(
            new_model,
            batch_size=idx.size(0),
            max_seq_len=config.sequence_len,
        )

        prefill_logits = new_model(idx[:, :prefill_len], kv_cache=cache)
        assert cache.get_pos() == prefill_len
        torch.testing.assert_close(
            prefill_logits,
            full_logits[:, :prefill_len],
        )
        assert cache.prev_embedding is not None
        assert cache.prev_embedding.shape == (batch_size, 1, config.n_embd)

        for pos in range(prefill_len, config.sequence_len):
            token = idx[:, pos:pos+1]
            cached_logits = new_model(token, kv_cache=cache)
            assert cache.get_pos() == pos + 1
            torch.testing.assert_close(
                cached_logits,
                full_logits[:, pos:pos+1]
            )


@pytest.mark.parametrize("config", TEST_CONFIGS)
@pytest.mark.parametrize("prefill_case", PREFILL_CASES)
def test_kv_cache_vs_reference_gpt(config, prefill_case):
    new_model, old_model, config = make_matched_models(config)

    batch_size = 2
    idx, _ = make_batch(config, batch_size)

    prefill_len = resolve_prefill_len(prefill_case, config)

    with torch.inference_mode():
        new_cache = make_kv_cache(
            new_model,
            batch_size=idx.size(0),
            max_seq_len=config.sequence_len,
        )
        old_cache = make_kv_cache(
            old_model,
            batch_size=idx.size(0),
            max_seq_len=config.sequence_len,
        )

        prefill_new = new_model(idx[:, :prefill_len], kv_cache=new_cache)
        prefill_old = old_model(idx[:, :prefill_len], kv_cache=old_cache)
        torch.testing.assert_close(prefill_new, prefill_old)
        assert new_cache.get_pos() == prefill_len
        assert old_cache.get_pos() == prefill_len
        torch.testing.assert_close(
            new_cache.prev_embedding,
            old_cache.prev_embedding
        )

        for pos in range(prefill_len, config.sequence_len):
            token = idx[:, pos:pos+1]
            cached_new = new_model(token, kv_cache=new_cache)
            cached_old = old_model(token, kv_cache=old_cache)
            assert new_cache.get_pos() == old_cache.get_pos()
            torch.testing.assert_close(cached_new, cached_old)


@pytest.mark.parametrize("config", TEST_CONFIGS)
def test_short_sequence_vs_reference_gpt(config):
    new_model, old_model, config = make_matched_models(config)
    idx, targets = make_batch(config)

    short_len = max(2, config.sequence_len // 2)
    idx = idx[:, :short_len].contiguous()
    targets = targets[:, :short_len].contiguous()

    with torch.inference_mode():
        torch.testing.assert_close(new_model(idx), old_model(idx))
        torch.testing.assert_close(
            new_model(idx, targets),
            old_model(idx, targets)
        )


@pytest.mark.parametrize("config", TEST_CONFIGS)
def test_ignore_index_loss_vs_reference_gpt(config):
    new_model, old_model, config = make_matched_models(config)
    idx, targets = make_batch(config)
    targets = targets.clone()
    targets[:, ::2] = -1

    with torch.inference_mode():
        for reduction in ["mean", "sum", "none"]:
            torch.testing.assert_close(
                new_model(idx, targets, loss_reduction=reduction),
                old_model(idx, targets, loss_reduction=reduction)
            )
