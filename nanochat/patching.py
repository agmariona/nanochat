import math
import torch

"""
patch_lengths.shape == (B, num_patches)
patch_lengths.sum(dim=1) == T + 1       T = row / seq len
first patch is always length 1
if padding, right zero-pad
encoder patch ids come from positions 0:T
decoder patch ids come from positions 1:(T+1)
"""

def static_patch_lengths(
    batch_size: int,
    row_len: int,
    patch_size: int,
    dtype=torch.long,
    device="cpu"
):
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    if row_len <= 0:
        raise ValueError("Row length must be positive")
    if patch_size <= 0:
        raise ValueError("Patch size must be positive")

    patch_lengths = torch.full(
        (batch_size, 1 + math.ceil((row_len - 1) / patch_size)),
        patch_size,
        dtype=dtype,
        device=device
    )

    patch_lengths[:, 0] = 1  # first patch always length 1
    if (row_len - 1) % patch_size != 0:
        patch_lengths[:, -1] = (row_len - 1) % patch_size

    return patch_lengths

# zero-start for encoder-like behavior
def patch_ids_from_lengths(patch_lengths, seq_len):
    patch_ends = patch_lengths.cumsum(dim=1)
    positions = torch.arange(seq_len, device=patch_lengths.device)

    return (positions[None, :, None] >= patch_ends[:, None, :]).sum(dim=-1)

def decoder_patch_ids_from_lengths(patch_lengths, seq_len):
    return patch_ids_from_lengths(patch_lengths[:, 1:], seq_len)

def validate_patch_lengths(patch_lengths, row_len):
    malformed = 0
    if not patch_lengths.ndim == 2:
        malformed = 1
        raise ValueError("Malformed patch lengths")

    if not patch_lengths[:, 0].eq(1).all():
        malformed = 1
    if not (patch_lengths >= 0).all():
        malformed = 1
    if not patch_lengths.sum(dim=1).eq(row_len).all():
        malformed = 1

    seen_zero = (patch_lengths == 0).cummax(dim=1).values
    if ((patch_lengths != 0) & seen_zero).any():
        malformed = 1

    if malformed:
        raise ValueError("Malformed patch lengths")

