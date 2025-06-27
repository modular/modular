# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pytest
from max.interfaces import LogProbabilities
from max.pipelines.lib.log_probabilities import (
    compute_log_probabilities_ragged,
    log_softmax,
)


def _check_log_probabilities_equal(
    actual: LogProbabilities, expected: LogProbabilities
) -> None:
    assert actual.token_log_probabilities == expected.token_log_probabilities
    assert actual.top_log_probabilities == expected.top_log_probabilities


def test_compute_log_probabilities() -> None:
    input_row_offsets = np.array([0, 2], dtype=np.uint32)
    batch_logits = np.array(
        [
            [0.5, 0.25, 0.7, 0.3, 1, 0.05],  # top-3 index = 0, 2, 4
            [0.1, 0.2, 0.9, 0.3, 0.4, 0.14],  # top-3 index = 2, 3, 4
        ]
    )
    batch_tokens = np.array([0, 1, 4])
    get_logits_and_samples = lambda x, y: (batch_logits, batch_tokens)
    batch_echo = [True]  # Value doesn't matter

    log_probs = log_softmax(batch_logits, axis=-1)

    # Check top 3
    output = compute_log_probabilities_ragged(
        input_row_offsets=input_row_offsets,
        logits=batch_logits,
        next_token_logits=batch_logits[-1:],
        tokens=batch_tokens[:-1],
        sampled_tokens=batch_tokens[-1:],
        batch_top_n=[3],
        batch_echo=batch_echo,
    )
    assert len(output) == 1
    expected_log_probs = LogProbabilities(
        token_log_probabilities=[log_probs[0][1], log_probs[1][4]],
        top_log_probabilities=[
            {
                0: log_probs[0][0].item(),
                2: log_probs[0][2].item(),
                4: log_probs[0][4].item(),
                # While not part of the top 3, token 1 was sampled so it gets
                # included in the top log probabilities.
                1: log_probs[0][1].item(),
            },
            {
                2: log_probs[1][2].item(),
                3: log_probs[1][3].item(),
                4: log_probs[1][4].item(),
            },
        ],
    )
    assert isinstance(output[0], LogProbabilities)
    _check_log_probabilities_equal(output[0], expected_log_probs)

    # Check top 1
    output = compute_log_probabilities_ragged(
        input_row_offsets=input_row_offsets,
        logits=batch_logits,
        next_token_logits=batch_logits[-1:],
        tokens=batch_tokens[:-1],
        sampled_tokens=batch_tokens[-1:],
        batch_top_n=[1],
        batch_echo=batch_echo,
    )
    assert len(output) == 1
    expected_log_probs = LogProbabilities(
        token_log_probabilities=[log_probs[0][1], log_probs[1][4]],
        top_log_probabilities=[
            {
                4: log_probs[0][4].item(),
                # While not part of the top 1, token 1 was sampled so it is
                # included in the top log probabilities.
                1: log_probs[0][1].item(),
            },
            {
                2: log_probs[1][2].item(),
                # Same case here, token 4 was sampled so it is included.
                4: log_probs[1][4].item(),
            },
        ],
    )
    assert isinstance(output[0], LogProbabilities)
    _check_log_probabilities_equal(output[0], expected_log_probs)


def test_compute_log_probabilities_batch() -> None:
    input_row_offsets = np.array([0, 2, 3, 4], dtype=np.uint32)
    batch_logits = np.array(
        [
            [0.5, 0.25, 0.7, 0.3, 1, 0.05],  # batch 1 token 1; top index: 4
            [0.1, 0.2, 0.9, 0.3, 0.4, 0.14],  # batch 1 token 2; top index: 2
            [0.4, 0.42, 0.3, 0.89, 0.07, 0.5],  # b. 2; top 5 idx: 0, 1, 2, 3, 5
            [100, 100, 100, 100, 100, 100],  # batch 3
        ]
    )
    batch_next_token_logits = np.array(
        [
            [0.1, 0.2, 0.9, 0.3, 0.4, 0.14],  # batch 1; top index: 2
            [0.4, 0.42, 0.3, 0.89, 0.07, 0.5],  # b. 2; top 5 idx: 0, 1, 2, 3, 5
            [100, 100, 100, 100, 100, 100],  # batch 3
        ]
    )
    # batch 1 = [0, 1]; batch 2 = [0]; batch 3 = [0]
    batch_tokens = np.array([0, 1, 0, 0])
    batch_sampled_tokens = np.array([4, 3, 3])
    batch_top_n = [1, 5, 0]
    batch_echo = [True, False, True]

    log_probs1 = log_softmax(batch_logits[0:2], axis=-1)
    log_probs2 = log_softmax(batch_logits[2:3], axis=-1)

    output = compute_log_probabilities_ragged(
        input_row_offsets=input_row_offsets,
        logits=batch_logits,
        next_token_logits=batch_next_token_logits,
        tokens=batch_tokens,
        sampled_tokens=batch_sampled_tokens,
        batch_top_n=batch_top_n,
        batch_echo=batch_echo,
    )
    assert len(output) == 3
    assert output[2] is None  # Top N was 0, so output[2] should be None.

    assert output[0] is not None
    _check_log_probabilities_equal(
        output[0],
        LogProbabilities(
            token_log_probabilities=[log_probs1[0][1], log_probs1[1][4]],
            top_log_probabilities=[
                {
                    4: log_probs1[0][4].item(),
                    # While not part of the top 1, token 1 was sampled so it is
                    # included in the top log probabilities.
                    1: log_probs1[0][1].item(),
                },
                {
                    2: log_probs1[1][2].item(),
                    # Same case here, token 4 was sampled so it is included.
                    4: log_probs1[1][4].item(),
                },
            ],
        ),
    )
    assert output[1] is not None
    _check_log_probabilities_equal(
        output[1],
        LogProbabilities(
            token_log_probabilities=[log_probs2[0][3]],
            top_log_probabilities=[
                {
                    0: log_probs2[0][0].item(),
                    1: log_probs2[0][1].item(),
                    2: log_probs2[0][2].item(),
                    3: log_probs2[0][3].item(),
                    5: log_probs2[0][5].item(),
                },
            ],
        ),
    )


def test_compute_log_probabilities_ragged() -> None:
    output = compute_log_probabilities_ragged(
        input_row_offsets=np.array([0, 3, 5, 8]),
        logits=np.array(
            [
                [10, 11],  # batch 0 token 0
                [11, 12],  # batch 0 token 1
                [12, 13],  # batch 0 token 2
                [20, 21],  # batch 1 token 0
                [21, 22],  # batch 1 token 1
                [30, 31],  # batch 2 token 0
                [31, 32],  # batch 2 token 1
                [32, 33],  # batch 2 token 2
            ]
        ),
        next_token_logits=np.array(
            [
                [12, 13],  # batch 0 token 2
                [21, 22],  # batch 1 token 1
                [32, 33],  # batch 2 token 2
            ]
        ),
        tokens=np.array([1, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32),
        sampled_tokens=np.array([0, 1, 0], dtype=np.int32),
        batch_top_n=[1, 0, 1],
        batch_echo=[True, False, False],
    )
    assert len(output) == 3
    assert output[0] is not None
    assert len(output[0].token_log_probabilities) == 3
    assert len(output[0].top_log_probabilities) == 3
    assert output[0].top_log_probabilities[0].keys() == {1}
    assert output[0].top_log_probabilities[1].keys() == {0, 1}
    assert output[0].top_log_probabilities[2].keys() == {0, 1}
    assert output[1] is None
    assert output[2] is not None
    assert len(output[2].token_log_probabilities) == 1
    assert len(output[2].top_log_probabilities) == 1
    assert output[2].top_log_probabilities[0].keys() == {0, 1}


@dataclass
class InputBatchItem:
    logits: np.ndarray | None  # shape (seq, vocab_size)
    next_token_logits: np.ndarray  # shape (vocab_size,)
    tokens: np.ndarray  # shape (seq,)
    sampled_token: int
    top_n: int
    echo: bool

    @classmethod
    def random(
        cls, rng: np.random.Generator, *, vocab_size: int, with_logits: bool
    ) -> InputBatchItem:
        seq_len = int(rng.integers(1, 300, endpoint=True))
        want_log_probs = bool(rng.integers(0, 4) == 0)
        fake_quantize = bool(rng.integers(0, 2) == 0)
        if with_logits:
            # This is absolutely the wrong distribution, but it's good enough
            # for our purposes.  That said, if someone felt like figuring out a
            # more representative distribution for logit outputs and could
            # replace this, the test could have more fidelity.
            logits = rng.normal(size=(seq_len, vocab_size)).astype(np.float32)
            if fake_quantize:
                logits = np.round(logits, decimals=1)
            next_token_logits = logits[-1, :]
        else:
            logits = None
            next_token_logits = rng.normal(size=(vocab_size,)).astype(
                np.float32
            )
            if fake_quantize:
                next_token_logits = np.round(next_token_logits, decimals=1)
        return cls(
            logits=logits,
            next_token_logits=next_token_logits,
            tokens=rng.integers(0, vocab_size, size=(seq_len,)),
            sampled_token=int(rng.integers(0, vocab_size)),
            top_n=(
                int(rng.integers(1, min(5, vocab_size), endpoint=True))
                if want_log_probs
                else 0
            ),
            echo=(
                bool(rng.integers(0, 4) == 0)
                if want_log_probs and with_logits
                else False
            ),
        )


def random_batch(rng: np.random.Generator) -> Sequence[InputBatchItem]:
    batch_size = int(rng.integers(1, 64, endpoint=True))
    vocab_size = int(rng.integers(1, 128, endpoint=True))
    with_logits = bool(rng.integers(0, 4) == 0)
    return [
        InputBatchItem.random(
            rng, vocab_size=vocab_size, with_logits=with_logits
        )
        for i in range(batch_size)
    ]


@dataclass
class PackedInput:
    input_row_offsets: np.ndarray
    logits: np.ndarray | None
    next_token_logits: np.ndarray
    tokens: np.ndarray
    sampled_tokens: np.ndarray
    batch_top_n: Sequence[int]
    batch_echo: Sequence[bool]

    @classmethod
    def from_items(cls, batch_items: Sequence[InputBatchItem]) -> PackedInput:
        assert len(batch_items) >= 1
        if batch_items[0].logits is not None:
            logits_parts = []
            for item in batch_items:
                assert item.logits is not None
                logits_parts.append(item.logits)
            logits = np.concatenate(logits_parts, axis=0)
        else:
            assert all(item.logits is None for item in batch_items)
            logits = None
        return cls(
            input_row_offsets=np.concatenate(
                [
                    np.array([0], dtype=np.uint32),
                    np.cumsum(
                        np.array(
                            [item.tokens.shape[0] for item in batch_items],
                            dtype=np.uint32,
                        )
                    ),
                ]
            ),
            logits=logits,
            next_token_logits=np.stack(
                [item.next_token_logits for item in batch_items]
            ),
            tokens=np.concatenate([item.tokens for item in batch_items]),
            sampled_tokens=np.array(
                [item.sampled_token for item in batch_items], dtype=np.uint32
            ),
            batch_top_n=[item.top_n for item in batch_items],
            batch_echo=[item.echo for item in batch_items],
        )


def verify_position(
    alleged_logprob: float,
    top_mapping: Mapping[int, float],
    *,
    top_n: int,
    sampled: int,
    logits: np.ndarray,
) -> None:
    logsoftmaxed_logits = log_softmax(logits)
    threshold = 1e-4
    # Verify logit values -- keys are checked later.
    assert np.isclose(
        alleged_logprob, logsoftmaxed_logits[sampled], rtol=0, atol=threshold
    )
    for token in top_mapping:
        assert np.isclose(
            top_mapping[token],
            logsoftmaxed_logits[token],
            rtol=0,
            atol=threshold,
        )
    # Sampled token must always appear in top_mapping, even if it would cause
    # us to exceed top_n by 1.
    assert sampled in top_mapping
    assert top_n <= len(top_mapping) <= top_n + 1
    if len(top_mapping) == top_n + 1:
        # The only time we should exceed top_n is when the sampled token was
        # not in top_n.  So in this case, the sampled token had better have the
        # minimum logit value.  If there is some other element with a lower
        # logit value, that's a problem.  Relative orderings in top_mapping
        # should be exact, so no usage of tolerance here.
        assert not any(
            value < top_mapping[sampled] for value in top_mapping.values()
        )
    if 1 < len(top_mapping) < len(logits):
        # Sampled token aside, the items in top_mapping had better really be
        # the top_n.  top_n is not necessarily unique (e.g. if there are tokens
        # with identical logits, which does happen in practice), so we're
        # basically saying "everything in top n must be at least as big as
        # everything not in it".  Threshold isn't needed since we're comparing
        # like-for-like -- this test's computed log-softmax rather than
        # top_mapping's values (those we checked separately earlier).
        min_top = min(
            logsoftmaxed_logits[token]
            for token in top_mapping
            if token != sampled
        )
        max_not_in_mapping = max(
            float(value)
            for index, value in enumerate(logsoftmaxed_logits)
            if index not in top_mapping
        )
        assert max_not_in_mapping <= min_top


def verify_output(
    item: InputBatchItem, output: LogProbabilities | None
) -> None:
    if item.top_n == 0:
        assert output is None
        return
    assert output is not None
    if item.echo:
        assert len(output.token_log_probabilities) == item.tokens.shape[0]
        assert len(output.top_log_probabilities) == item.tokens.shape[0]
        assert item.logits is not None
        for seq in range(item.tokens.shape[0] - 1):
            verify_position(
                output.token_log_probabilities[seq],
                output.top_log_probabilities[seq],
                top_n=item.top_n,
                sampled=item.tokens[seq + 1],
                logits=item.logits[seq, :],
            )
    else:
        assert len(output.token_log_probabilities) == 1
        assert len(output.top_log_probabilities) == 1
    verify_position(
        output.token_log_probabilities[-1],
        output.top_log_probabilities[-1],
        top_n=item.top_n,
        sampled=item.sampled_token,
        logits=item.next_token_logits,
    )


# The randomized test provides larger inputs than the manually-written tests
# above, and handles some ambiguous cases for which there are multiple valid
# answers.  (Specifically, ties in the top-n are implementation-dependent, and
# we want to tolerate any tie-breaking strategy.)
@pytest.mark.parametrize("seed", range(50))
def test_log_probabilities_randomized(seed: int) -> None:
    rng = np.random.default_rng(seed)
    batch = random_batch(rng)
    packed = PackedInput.from_items(batch)
    outputs = compute_log_probabilities_ragged(
        input_row_offsets=packed.input_row_offsets,
        logits=packed.logits,
        next_token_logits=packed.next_token_logits,
        tokens=packed.tokens,
        sampled_tokens=packed.sampled_tokens,
        batch_top_n=packed.batch_top_n,
        batch_echo=packed.batch_echo,
    )
    assert len(outputs) == len(batch)
    for item, output in zip(batch, outputs):
        verify_output(item, output)
