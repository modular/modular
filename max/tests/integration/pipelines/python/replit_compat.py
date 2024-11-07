# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Replit compatibility for newer Transformers versions.

The Replit model only works out of the box before Transformers 4.35.0, which
removed private functions it imported and used.  But downgrading is not an
option for us.  This module contains a function to monkey-patch equivalent
functions the Replit code is looking for back into the places it's looking for
it.
"""

from typing import Optional

import torch


# Code snippets taken from ac5893756bafcd745d93a442cf36f984545dbad8's diff in
# the Transformers repo.
def _bloom_expand_mask(mask: torch.Tensor, tgt_length: int) -> torch.BoolTensor:
    batch_size, src_length = mask.shape
    tgt_length = tgt_length if tgt_length is not None else src_length
    expanded_mask = ~(mask[:, None, None, :].to(torch.bool))
    return expanded_mask.expand(batch_size, 1, tgt_length, src_length)


def _bloom_make_causal_mask(
    input_ids_shape: torch.Size,
    device: torch.device,
    past_key_values_length: int,
) -> torch.BoolTensor:
    batch_size, target_length = input_ids_shape
    mask = torch.empty(
        (target_length, target_length + past_key_values_length),
        dtype=torch.bool,
        device=device,
    )
    seq_ids = torch.arange(target_length, device=device)
    mask[:, past_key_values_length:] = seq_ids[:, None] < seq_ids[None, :]
    if past_key_values_length > 0:
        mask[:, :past_key_values_length] = False
    expanded_mask = mask[None, None, :, :].expand(
        batch_size, 1, target_length, target_length + past_key_values_length
    )
    return expanded_mask


def _opt_make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
) -> torch.Tensor:
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    tgt_len, past_key_values_length, dtype=dtype, device=device
                ),
                mask,
            ],
            dim=-1,
        )
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )


def _opt_expand_mask(
    mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None
) -> torch.Tensor:
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = (
        mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    )
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


def monkeypatch_transformers() -> None:
    """Monkey-patch transformers to be compatible with replit.

    The Replit model only works out of the box before Transformers 4.35.0,
    which removed private functions it imported and used.  But downgrading is
    not an option for us.  This function monkey-patches equivalent functions
    the Replit code is looking for back into the places it's looking for it.
    """

    from transformers.models.bloom import modeling_bloom
    from transformers.models.opt import modeling_opt

    setattr(modeling_bloom, "_expand_mask", _bloom_expand_mask)
    setattr(modeling_bloom, "_make_causal_mask", _bloom_make_causal_mask)
    setattr(modeling_opt, "_expand_mask", _opt_expand_mask)
    setattr(modeling_opt, "_make_causal_mask", _opt_make_causal_mask)
