# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from max.pipelines.dataprocessing import causal_attention_mask_with_alibi

ALIBI_BIAS_MAX = 8
N_HEADS = 4
MAX_BATCH_SIZE = 32
MAX_SEQUENCE_LENGTH = 1024
PAD_MULTIPLES = [2, 1, 128]
batch_sizes = st.shared(st.integers(1, MAX_BATCH_SIZE))
start_positions = st.integers(0, MAX_SEQUENCE_LENGTH // 2)
seq_lens = st.integers(1, MAX_SEQUENCE_LENGTH // 2)

# TODO(KERN-782): This should be -inf but softmax saturates with NaNs.
FILL_VAL = -10000.0


def lists_of_size(strategy, size_strategy):
    return size_strategy.flatmap(
        lambda length: st.lists(strategy, min_size=length, max_size=length)
    )


@settings(deadline=None)
@given(
    start_pos=lists_of_size(start_positions, batch_sizes),
    seq_len=lists_of_size(seq_lens, batch_sizes),
)
def test_causal_mask_with_alibi__shape(
    start_pos: list[int], seq_len: list[int]
):
    assert len(start_pos) == len(seq_len)

    for pad_to_multiple_of in PAD_MULTIPLES:
        mask = causal_attention_mask_with_alibi(
            start_pos, seq_len, ALIBI_BIAS_MAX, N_HEADS, pad_to_multiple_of
        )
        assert len(mask.shape) == 4
        assert mask.shape[0] == len(start_pos)
        assert mask.shape[1] == N_HEADS

        if max(seq_len) == 1:
            padded_length = 1
        else:
            padded_length = (
                math.ceil(max(seq_len) / pad_to_multiple_of)
                * pad_to_multiple_of
            )
            assert mask.shape[2] % pad_to_multiple_of == 0
            assert mask.shape[2] == padded_length

        post_seq_len = max([(pos + padded_length) for pos in start_pos])
        assert mask.shape[-1] == post_seq_len


@settings(deadline=None)
@given(
    start_pos=lists_of_size(start_positions, batch_sizes),
    seq_len=lists_of_size(seq_lens, batch_sizes),
)
def test_causal_mask_with_alibi__masks_padding(
    start_pos: list[int], seq_len: list[int]
):
    mask = causal_attention_mask_with_alibi(
        start_pos, seq_len, ALIBI_BIAS_MAX, N_HEADS
    )
    for i in range(N_HEADS):
        for m, sp, sl in zip(mask[:, i, :, :], start_pos, seq_len):
            post_seq_len = sp + sl
            assert np.all(m[:sl, post_seq_len:] <= FILL_VAL)


@settings(deadline=None)
@given(
    start_pos=lists_of_size(start_positions, batch_sizes),
    seq_len=lists_of_size(seq_lens, batch_sizes),
)
def test_causal_mask_with_alibi__masks_current_and_later_tokens(
    start_pos: list[int], seq_len: list[int]
):
    assert len(start_pos) == len(seq_len)
    mask = causal_attention_mask_with_alibi(
        start_pos, seq_len, ALIBI_BIAS_MAX, N_HEADS
    )
    for i in range(N_HEADS):
        for m, sp, _ in zip(mask[:, i, :, :], start_pos, seq_len):
            for pos, sequence_mask in enumerate(m):
                # Check that all tokens _after_ this one are masked.
                assert np.all(sequence_mask[sp + pos + 1 :] <= FILL_VAL)


@settings(deadline=None)
@given(
    start_pos=lists_of_size(start_positions, batch_sizes),
    seq_len=lists_of_size(seq_lens, batch_sizes),
)
def test_causal_mask_with_alibi__does_not_mask_prior_tokens(
    start_pos: list[int], seq_len: list[int]
):
    assert len(start_pos) == len(seq_len)
    mask = causal_attention_mask_with_alibi(
        start_pos, seq_len, ALIBI_BIAS_MAX, N_HEADS
    )
    for i in range(N_HEADS):
        for m, sp, _ in zip(mask[:, i, :, :], start_pos, seq_len):
            for pos, sequence_mask in enumerate(m):
                assert np.all(sequence_mask[: sp + pos + 1] >= FILL_VAL)
                assert np.all(sequence_mask[: sp + pos + 1] <= 0.0)
