# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import numpy as np
from max.pipelines import LogProbabilities
from max.pipelines.nn.compute_log_probabilities import (
    compute_log_probabilities,
    log_softmax,
)


def _check_log_probabilities_equal(
    actual: LogProbabilities, expected: LogProbabilities
):
    assert actual.token_log_probabilities == expected.token_log_probabilities
    assert actual.top_log_probabilities == expected.top_log_probabilities


def test_compute_log_probabilities() -> None:
    batch_logits = np.array(
        [
            [0.5, 0.25, 0.7, 0.3, 1, 0.05],  # top-3 index = 0, 2, 4
            [0.1, 0.2, 0.9, 0.3, 0.4, 0.14],  # top-3 index = 2, 3, 4
        ]
    )
    batch_tokens = np.array([1, 4])
    get_logits_and_samples = lambda x, y: (batch_logits, batch_tokens)
    batch_echo = [True]  # Value doesn't matter

    log_probs = log_softmax(batch_logits, axis=-1)

    # Check top 3
    output = compute_log_probabilities(get_logits_and_samples, [3], batch_echo)
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
    output = compute_log_probabilities(get_logits_and_samples, [1], batch_echo)
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


def test_compute_log_probabilities_batch():
    batch1_logits = np.array(
        [
            [0.5, 0.25, 0.7, 0.3, 1, 0.05],  # top index: 4
            [0.1, 0.2, 0.9, 0.3, 0.4, 0.14],  # top index: 2
        ]
    )
    batch1_tokens = np.array([1, 4])

    batch2_logits = None
    batch2_tokens = None

    batch3_logits = np.array(
        [
            [0.4, 0.42, 0.3, 0.89, 0.07, 0.5],  # top 5 index: 0, 1, 2, 3, 5
        ]
    )
    batch3_tokens = np.array([3])

    batch4_logits = np.array(
        [
            [100, 100, 100, 100, 100, 100],
        ]
    )
    batch4_tokens = np.array([3])

    batch_logits = [batch1_logits, batch2_logits, batch3_logits, batch4_logits]
    batch_tokens = [batch1_tokens, batch2_tokens, batch3_tokens, batch4_tokens]
    batch_top_n = [1, 3, 5, 0]
    batch_echo = [True, True, False, True]

    def get_logits_and_samples(batch_index, echo):
        assert echo == batch_echo[batch_index]
        if batch_logits[batch_index] is None:
            return None
        return batch_logits[batch_index], batch_tokens[batch_index]

    log_probs1 = log_softmax(batch1_logits, axis=-1)
    log_probs3 = log_softmax(batch3_logits, axis=-1)

    output = compute_log_probabilities(
        get_logits_and_samples, batch_top_n, batch_echo
    )
    assert len(output) == 4
    assert output[1] is None  # Batch was None, so output[1] should be None.
    assert output[3] is None  # Top N was 0, so output[3] should be None.

    assert isinstance(output[0], LogProbabilities)
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
    assert isinstance(output[2], LogProbabilities)
    _check_log_probabilities_equal(
        output[2],
        LogProbabilities(
            token_log_probabilities=[log_probs3[0][3]],
            top_log_probabilities=[
                {
                    0: log_probs3[0][0].item(),
                    1: log_probs3[0][1].item(),
                    2: log_probs3[0][2].item(),
                    3: log_probs3[0][3].item(),
                    5: log_probs3[0][5].item(),
                },
            ],
        ),
    )
