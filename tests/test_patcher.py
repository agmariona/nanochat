import pytest
import torch

from nanochat.patching import (
    static_patch_lengths,
    patch_ids_from_lengths,
    decoder_patch_ids_from_lengths,
    validate_patch_lengths
)


def test_static_patch_lengths_first_one():
    p_lens = static_patch_lengths(batch_size=2, row_len=11, patch_size=4)
    assert p_lens.tolist() == [
        [1,4,4,2],
        [1,4,4,2],
    ]
    assert p_lens.sum(dim=1).tolist() == [11, 11]


@pytest.mark.parametrize(
    ("row_len", "patch_size", "expected"),
    [
        (1, 4, [1]),
        (2, 4, [1, 1]),
        (5, 4, [1, 4]),
        (9, 4, [1, 4, 4]),
        (10, 4, [1, 4, 4, 1]),
        (11, 4, [1, 4, 4, 2]),
        (4, 8, [1, 3]),
    ]
)
def test_static_patch_lengths_edge_cases(row_len, patch_size, expected):
    patch_lengths = static_patch_lengths(
        batch_size=1,
        row_len=row_len,
        patch_size=patch_size
    )
    assert patch_lengths.tolist() == [expected]
    assert patch_lengths.sum().item() == row_len


def test_patch_ids_encoder():
    patch_lengths = torch.tensor([[1,4,4,2]])
    patch_ids = patch_ids_from_lengths(patch_lengths, seq_len=10)
    assert patch_ids.tolist() == [[0, 1,1,1,1, 2,2,2,2, 3]]


def test_patch_ids_decoder():
    patch_lengths = torch.tensor([[1,4,4,2]])
    patch_ids = decoder_patch_ids_from_lengths(patch_lengths, seq_len=10)
    assert patch_ids.tolist() == [[0,0,0,0, 1,1,1,1, 2,2]]


def test_patch_lengths_match_nanochat_contract():
    T = 10
    row_len = T+1
    patch_lengths = static_patch_lengths(
        batch_size=3,
        row_len=row_len,
        patch_size=4
    )

    validate_patch_lengths(patch_lengths, row_len=row_len)

    assert patch_lengths.shape == (3,4)
    assert patch_lengths[:,0].eq(1).all()
    assert patch_lengths.sum(dim=1).eq(T+1).all()

    encoder_patch_ids = patch_ids_from_lengths(patch_lengths, seq_len=T)
    decoder_patch_ids = decoder_patch_ids_from_lengths(patch_lengths, seq_len=T)

    assert encoder_patch_ids.shape == (3,T)
    assert decoder_patch_ids.shape == (3,T)


def test_patch_ids_support_zero_padding():
    patch_lengths = torch.tensor([
        [1,4,2,0],
        [1,3,3,0],
    ])

    patch_ids = patch_ids_from_lengths(patch_lengths, seq_len=6)

    assert patch_ids.tolist() == [
        [0, 1,1,1,1, 2],
        [0, 1,1,1, 2,2],
    ]


@pytest.mark.parametrize(
    "bad_patch_lengths",
    [
        torch.tensor([[1,0,2]]),    # nonzero after zero
        torch.tensor([[2,3]]),      # first patch not one
        torch.tensor([[1,2,-1]]),   # negative length
    ]
)
def test_validate_patch_lengths(bad_patch_lengths):
    with pytest.raises(ValueError):
        validate_patch_lengths(bad_patch_lengths, row_len=3)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS unavailable"
)
def test_static_patch_lengths_device_dtype():
    patch_lengths = static_patch_lengths(
        batch_size=2,
        row_len=11,
        patch_size=4,
        device=torch.device("mps"),
    )

    assert patch_lengths.device.type == "mps"
    assert patch_lengths.dtype == torch.long
