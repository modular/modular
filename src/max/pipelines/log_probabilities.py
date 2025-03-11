# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import Callable

import numpy as np
from max.pipelines.interfaces import LogProbabilities


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute the logarithm of the softmax function.

    This implementation uses the identity log(softmax(x)) = x - log(sum(exp(x)))
    with numerical stability improvements to prevent overflow/underflow.

    Args:
        x: Input array
        axis: Axis to compute values along

    Returns:
        Array with same shape as x, representing log(softmax(x))
    """
    # Subtract max value for numerical stability (prevents exp overflow)
    x_max = np.amax(x, axis=axis, keepdims=True)

    # Compute exp(x - x_max) which is now safe from overflow
    shifted_x = x - x_max
    exp_shifted = np.exp(shifted_x)

    # Suppress -inf warnings from log(0)
    # This can happen when input contains extreme negative values (-inf),
    # which become 0 after exp() operation
    with np.errstate(divide="ignore"):
        sum_exp = np.sum(exp_shifted, axis=axis, keepdims=True)
        log_sum_exp = np.log(sum_exp)

    # Final result: x - x_max - log(sum(exp(x - x_max)))
    # This is mathematically equivalent to log(softmax(x))
    return shifted_x - log_sum_exp


def compute_log_probabilities(
    get_logits_and_samples: Callable[
        [int, bool], (tuple[np.ndarray, np.ndarray] | None)
    ],
    batch_top_n: list[int],
    batch_echo: list[bool],
) -> list[LogProbabilities | None]:
    """Computes the log probabilities.

    Args:
        get_logits_and_samples: Callable that takes the batch index and an
        `echo` bool and returns the logits and sampled tokens for that batch.
            Args:
            - batch_index is an int between [0, batch_size)
            - echo is whether that item was requested to echo the input tokens.
            Returns (None if batch item is empty):
            - Logits should have shape = (n_tokens, vocab_size).
            - Sampled tokens should have shape = (n_tokens).
        batch_top_n: Number of top log probabilities to return per input in
            the batch. For any element where `top_n == 0`, the
            LogProbabilities is skipped.
        batch_echo: Whether to include input tokens in the returned log
            probabilities.

    Returns:
        Computed log probabilities for each item in the batch.
    """
    log_probabilities: list[LogProbabilities | None] = []
    for batch, (top_n, echo) in enumerate(zip(batch_top_n, batch_echo)):
        if top_n == 0:
            log_probabilities.append(None)
            continue

        logits_and_samples = get_logits_and_samples(batch, echo)
        if not logits_and_samples:
            log_probabilities.append(None)
            continue

        logits, samples = logits_and_samples
        log_probs = log_softmax(logits, axis=-1)

        # Get top n tokens.
        top_n_indices = np.argpartition(log_probs, -top_n, axis=-1)[
            ..., -top_n:
        ]

        # Get the log probabilities of each sampled token.
        sampled_log_probs = np.take_along_axis(
            log_probs, samples.reshape(-1, 1), axis=1
        ).reshape(-1)

        # Store the stats for each sample token.
        num_tokens = log_probs.shape[0]
        token_log_probabilities = []
        top_log_probabilities = []
        for i in range(num_tokens):
            token_log_probabilities.append(sampled_log_probs[i].item())

            # Compute top n log probs.
            top_tokens = {}
            for n in range(top_n):
                top_token = top_n_indices[i][n]
                top_token_logits = log_probs[i][top_token]
                top_tokens[top_token] = top_token_logits.item()

            # Include sampled token in the top tokens.
            sampled_token = samples[i].item()
            top_tokens[sampled_token] = sampled_log_probs[i].item()

            top_log_probabilities.append(top_tokens)

        log_probabilities.append(
            LogProbabilities(
                token_log_probabilities=token_log_probabilities,
                top_log_probabilities=top_log_probabilities,
            )
        )

    return log_probabilities
