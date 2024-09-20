# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from llama3.causal_attention_mask import causal_attention_mask

from hypothesis import given, strategies as st
import numpy as np

MAX_BATCH_SIZE = 32
MAX_SEQUENCE_LENGTH = 1024
batch_sizes = st.shared(st.integers(1, MAX_BATCH_SIZE))
start_positions = st.integers(0, MAX_SEQUENCE_LENGTH // 2)
seq_lens = st.integers(1, MAX_SEQUENCE_LENGTH // 2)


def lists_of_size(strategy, size_strategy):
    return size_strategy.flatmap(
        lambda length: st.lists(strategy, min_size=length, max_size=length)
    )


@given(
    start_pos=lists_of_size(start_positions, batch_sizes),
    seq_len=lists_of_size(seq_lens, batch_sizes),
)
def test_causal_mask__shape(start_pos: list[int], seq_len: list[int]):
    assert len(start_pos) == len(seq_len)
    mask = causal_attention_mask(start_pos, seq_len)
    assert len(mask.shape) == 3
    assert mask.shape[0] == len(start_pos)
    assert mask.shape[1] == max(seq_len)
    assert mask.shape[2] == max(
        (pos + len) for pos, len in zip(start_pos, seq_len)
    )


@given(
    start_pos=lists_of_size(start_positions, batch_sizes),
    seq_len=lists_of_size(seq_lens, batch_sizes),
)
def test_causal_mask__masks_padding(start_pos: list[int], seq_len: list[int]):
    mask = causal_attention_mask(start_pos, seq_len)
    for m, sp, sl in zip(mask, start_pos, seq_len):
        post_seq_len = sp + sl
        assert np.all(m[:sl, post_seq_len:] == float("-inf"))
        # No expectations for tokens past seq_len
        # assert np.all(m[sl:, :] == float("-inf"))


@given(
    start_pos=lists_of_size(start_positions, batch_sizes),
    seq_len=lists_of_size(seq_lens, batch_sizes),
)
def test_causal_mask__masks_current_and_later_tokens(
    start_pos: list[int], seq_len: list[int]
):
    assert len(start_pos) == len(seq_len)
    mask = causal_attention_mask(start_pos, seq_len)
    for m, sp, sl in zip(mask, start_pos, seq_len):
        for pos, sequence_mask in enumerate(m):
            assert np.all(sequence_mask[sp + pos :] == float("-inf"))


@given(
    start_pos=lists_of_size(start_positions, batch_sizes),
    seq_len=lists_of_size(seq_lens, batch_sizes),
)
def test_causal_mask__does_not_mask_prior_tokens(
    start_pos: list[int], seq_len: list[int]
):
    assert len(start_pos) == len(seq_len)
    mask = causal_attention_mask(start_pos, seq_len)
    for m, sp, sl in zip(mask, start_pos, seq_len):
        for pos, sequence_mask in enumerate(m):
            assert np.all(sequence_mask[: sp + pos] == 0.0)
