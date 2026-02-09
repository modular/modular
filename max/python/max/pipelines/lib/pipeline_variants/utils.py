# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from max.interfaces import (
    LogProbabilities,
    RequestID,
    TextGenerationContextType,
    TextGenerationOutput,
)
from transformers import AutoConfig

from ..config_enums import RepoType
from ..hf_utils import download_weight_files

if TYPE_CHECKING:
    from ..model_config import MAXModelConfig

logger = logging.getLogger("max.pipelines")


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
    overwrite_future: bool = False,
) -> dict[RequestID, TextGenerationOutput]:
    """Updates context objects and prepares response objects after generation.

    Args:
        generated_tokens_host: Array of generated tokens on the host, indexed
            as [batch, step].
        flat_batch: List of generation contexts, one per request, matching
            batch dimension.
        num_steps: Number of generation steps to process for each context.
        batch_log_probabilities: List of per-step log probability outputs (or
            None), each entry is a list per batch for that step.
        enable_log_probs: Whether to include log probability data in outputs.
        overwrite_future: Whether to overwrite future tokens in the context.

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

            if overwrite_future:
                # If generated_length is still 0, then there is no placeholder
                # future token. This is possible due to chunked prefill.
                if context.tokens.generated_length:
                    context.realize_future_token(
                        new_token=next_token, log_probabilities=log_probs
                    )
            else:
                context.update(
                    new_token=next_token, log_probabilities=log_probs
                )

            if context.is_done:
                break

        # Only add the output if there are tokens to return.
        # It is possible that there are no generated tokens due to chunked prefill.
        output = context.to_generation_output()
        if output.tokens:
            res[context.request_id] = output

    return res


def get_eos_tokens(hf_config: AutoConfig, eos_token_id: int) -> set[int]:
    """Returns the set of end-of-sequence token IDs from config or fallback.

    Args:
        hf_config: HuggingFace model configuration.
        eos_token_id: Default EOS token id when not present in config.

    Returns:
        Set of EOS token ids to use for generation.
    """
    # Expand eos tokens if more are provided in pipeline_config
    if "eos_token_id" not in hf_config:
        return set([eos_token_id])

    hf_eos_tokens = hf_config.eos_token_id
    if isinstance(hf_eos_tokens, int):
        if hf_eos_tokens != eos_token_id:
            msg = f"eos_token_id provided in huggingface config ({hf_eos_tokens}), does not match provided eos_token_id ({eos_token_id}), using provided eos_token_id"
            logger.warning(msg)
        return set([hf_eos_tokens])
    elif isinstance(hf_eos_tokens, list):
        if eos_token_id in hf_eos_tokens:
            return set(hf_eos_tokens)
        else:
            return set([eos_token_id])
    else:
        msg = f"eos_token_id in huggingface_config is neither int or list: {hf_eos_tokens}"
        logger.warning(msg)
        return set([eos_token_id])


def get_weight_paths(model_config: MAXModelConfig) -> list[Path]:
    """Resolves local paths or downloads weight files for the model config.

    Args:
        model_config: Model configuration containing weight repo and paths.

    Returns:
        List of paths to weight files (local or downloaded).
    """
    weight_repo = model_config.huggingface_weight_repo
    if weight_repo.repo_type == RepoType.online:
        # Download weight files if not existent.
        return download_weight_files(
            huggingface_model_id=weight_repo.repo_id,
            filenames=[str(x) for x in model_config.weight_path],
            revision=model_config.huggingface_weight_revision,
            force_download=model_config.force_download,
        )
    else:
        # Use the resolved repo_id (which points to local cache in offline mode)
        local_path = Path(weight_repo.repo_id)
        return [local_path / x for x in model_config.weight_path]
