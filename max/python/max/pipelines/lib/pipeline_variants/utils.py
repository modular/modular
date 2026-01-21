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
import numpy as np
import numpy.typing as npt
from max.interfaces import (
    LogProbabilities,
    RequestID,
    TextGenerationContextType,
    TextGenerationOutput,
)


def calculate_num_steps(
    context: TextGenerationContextType,
    num_steps: int,
    max_seq_len: int,
) -> int:
    """Compute the number of generation steps allowed for a context.

    The value is clamped by the remaining capacity with respect to
    the model's configured ``max_seq_len``.

    Args:
        context: The context whose sequence length constraints apply.
        num_steps: Desired number of steps to attempt.
        max_seq_len: The maximum allowed sequence length for the model.

    Returns:
        The number of steps to execute for this context (>= 1).

    Raises:
        ValueError: If the current request length is already >= ``max_seq_len``.
    """
    num_available_steps = context.compute_num_available_steps(max_seq_len)

    if num_available_steps <= 0:
        raise ValueError(
            f"Request {context.request_id} length ({len(context.tokens)}) is larger than or equal to the configured max_length ({max_seq_len})"
        )

    return min(num_available_steps, num_steps)


def update_context_and_prepare_responses(
    generated_tokens_host: npt.NDArray[np.int32],
    flat_batch: list[TextGenerationContextType],
    num_steps: int,
    batch_log_probabilities: list[list[LogProbabilities | None]] | None = None,
    enable_log_probs: bool = False,
) -> dict[RequestID, TextGenerationOutput]:
    """
    Update the context objects and prepare the response objects for each context in the batch after generation.

    Args:
        generated_tokens_host: Array of generated tokens on the host, indexed as [batch, step].
        batch_log_probabilities: List of per-step log probability outputs (or None), each entry is a list per batch for that step.
        flat_batch: List of generation contexts, one per request, matching batch dimension.
        num_steps: Number of generation steps to process for each context.
        enable_log_probs: Whether to include log probability data in outputs.

    Returns:
        A dictionary mapping request IDs to their respective generation outputs.
    """
    res: dict[RequestID, TextGenerationOutput] = {}
    for batch_index, context in enumerate(flat_batch):
        for step in range(num_steps):
            # Convert to a Python scalar to improve serialization performance.
            next_token = int(generated_tokens_host[batch_index, step])

            # Get Log probs if needed.
            log_probs: LogProbabilities | None = None
            if enable_log_probs:
                assert batch_log_probabilities is not None
                if step < len(batch_log_probabilities):
                    log_probs_for_step = batch_log_probabilities[step]
                    if log_probs_for_step and batch_index < len(
                        log_probs_for_step
                    ):
                        log_probs = log_probs_for_step[batch_index]

            context.update(new_token=next_token, log_probabilities=log_probs)

            if context.is_done:
                break

        res[context.request_id] = context.to_generation_output()

    return res
