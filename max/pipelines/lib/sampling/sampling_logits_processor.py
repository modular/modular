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

"""Fused sampling logits processor used to apply sampling parameters to logits."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
)

import numpy as np
import numpy.typing as npt
from max.driver import Device, Tensor
from max.dtype import DType
from max.engine import Model
from max.interfaces import BatchProcessorInputs, InputContext
from max.profiler import Tracer, traced

if TYPE_CHECKING:
    from ..config import PipelineConfig

logger = logging.getLogger("max.pipelines")


@dataclass(frozen=True)
class FrequencyData:
    """Container for token frequency data in CSR format."""

    data: Tensor
    """data[:, 0]: 1D array of the column indices of the
        non-zero elements in the matrix.
    data[:, 1]: 1D array of the non-zero elements in the
        matrix."""

    offsets: Tensor
    """Row offsets: shape [batch_size + 1] indicating start of each
    sequence's data."""


class FusedSamplingProcessor:
    """Applies sampling parameters to logits and stores the chosen tokens."""

    new_tokens: Tensor | None = None
    """The new tokens that were sampled."""

    generated_tokens: Tensor
    """The generated tokens that have been sampled so far."""

    def __init__(
        self,
        sampler: Model,
        pipeline_config: PipelineConfig,
        context_batch: list[Any],
        num_steps: int,
        device: Device,
        bitmask: npt.NDArray[np.int32] | None = None,
        vocab_size: int | None = None,
    ):
        self.sampler = sampler
        self.batch_size = len(context_batch)
        self.device = device
        self.bitmask = bitmask
        self.vocab_size = vocab_size

        self.generated_tokens = Tensor(
            shape=(len(context_batch), 0),
            dtype=DType.int64,
            device=device,
        )

        self.temperature = Tensor.from_numpy(
            np.array(
                [
                    context.sampling_params.temperature
                    for context in context_batch
                ],
                dtype=np.float32,
            )
        ).to(device)

        top_k_np = np.array(
            [context.sampling_params.top_k for context in context_batch],
            dtype=np.int64,
        )
        self.top_k = Tensor.from_numpy(top_k_np).to(device)
        max_k_np = np.array(np.max(top_k_np), dtype=np.int64)
        self.max_k = Tensor.from_numpy(max_k_np)

        self.top_p = Tensor.from_numpy(
            np.array(
                [context.sampling_params.top_p for context in context_batch],
                dtype=np.float32,
            )
        ).to(device)
        self.seed = Tensor.from_numpy(
            np.array(
                [
                    context.sampling_params.seed + context.current_length
                    for context in context_batch
                ],
                dtype=np.uint64,
            )
        ).to(device)

        self.frequency_data: list[FrequencyData] | None = None
        self.frequency_penalty: Tensor | None = None
        self.presence_penalty: Tensor | None = None
        self.repetition_penalty: Tensor | None = None

        if pipeline_config.sampling_config.do_penalties:
            self.frequency_data = [
                _build_token_frequency_csr(context_batch, num_steps, device),
                _build_token_frequency_csr(
                    context_batch, num_steps, device, include_prompt=True
                ),
            ]

            self.frequency_penalty = Tensor.from_numpy(
                np.array(
                    [
                        context.sampling_params.frequency_penalty
                        for context in context_batch
                    ],
                    dtype=np.float32,
                )
            ).to(device)
            self.presence_penalty = Tensor.from_numpy(
                np.array(
                    [
                        context.sampling_params.presence_penalty
                        for context in context_batch
                    ],
                    dtype=np.float32,
                )
            ).to(device)
            self.repetition_penalty = Tensor.from_numpy(
                np.array(
                    [
                        context.sampling_params.repetition_penalty
                        for context in context_batch
                    ],
                    dtype=np.float32,
                )
            ).to(device)

        else:
            _check_need_penalties(context_batch)

        self.min_tokens_masks = _build_min_tokens_masks(
            context_batch,
            num_steps,
            device,
            pipeline_config.sampling_config.enable_min_tokens,
        )

        self.step_counter = 0

    def __call__(self, inputs: BatchProcessorInputs) -> None:
        logits = inputs.logits
        logit_offsets = inputs.logit_offsets
        tensor_bitmask = None
        if (
            self.bitmask is not None
            and self.bitmask.shape[1] != self.vocab_size
        ):
            assert self.vocab_size is not None
            bits = 2 ** np.arange(32, dtype=np.int32)
            self.bitmask = (self.bitmask[..., np.newaxis] & bits) != 0
            self.bitmask = self.bitmask.reshape(self.batch_size, -1).astype(
                np.bool_
            )

            if logits.shape[1] > self.vocab_size:
                if self.bitmask.shape[1] > logits.shape[1]:
                    self.bitmask = self.bitmask[:, 0 : logits.shape[1]]
                else:
                    self.bitmask = self.bitmask[:, 0 : self.vocab_size]
                    # Pad up to shape[:, logits.shape[1]] with zeros
                    pad_width = logits.shape[1] - self.bitmask.shape[1]
                    if pad_width > 0:
                        self.bitmask = np.pad(
                            self.bitmask,
                            ((0, 0), (0, pad_width)),
                            mode="constant",
                            constant_values=False,
                        )
            else:
                self.bitmask = self.bitmask[:, 0 : self.vocab_size]

            tensor_bitmask = Tensor.from_numpy(self.bitmask).to(self.device)

        new_tokens, new_generated_tokens, new_seed = _sample_logits(
            self.sampler,
            logits,
            self.generated_tokens,
            self.top_k,
            self.max_k,
            self.temperature,
            self.top_p,
            self.seed,
            logit_offsets=logit_offsets,
            bitmask=tensor_bitmask,
            frequency_data=self.frequency_data,
            min_tokens_mask=self.min_tokens_masks[self.step_counter]
            if self.min_tokens_masks
            else None,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            repetition_penalty=self.repetition_penalty,
        )

        assert isinstance(new_tokens, Tensor)
        assert isinstance(new_generated_tokens, Tensor)
        assert isinstance(new_seed, Tensor)

        self.generated_tokens = new_generated_tokens
        self.seed = new_seed
        self.new_tokens = new_tokens

        self.step_counter += 1


T = TypeVar("T", bound=InputContext)


def _check_need_penalties(batch: list[T]) -> None:
    """Check if the batch has penalties, but do_penalties is False."""
    for context in batch:
        if (
            context.sampling_params.frequency_penalty != 0.0
            or context.sampling_params.presence_penalty != 0.0
            or context.sampling_params.repetition_penalty != 1.0
        ):
            logger.warning(
                "penalties are provided in the request, but the model was not configured with do_penalties=True, ignoring"
            )
            return


@traced
def _build_token_frequency_csr(
    batch: list[T],
    padding_size: int,
    device: Device,
    include_prompt: bool = False,
) -> FrequencyData:
    """Build a CSR matrix of token frequency in the batch.
    The original matrix is (batch_size, vocab_size), where each element is
    the number of times a token appears in the batch.

    Returns:
        FrequencyData containing the CSR representation with:
        - data: 2D array where each row is [token_id, count]
        - row_offsets: 1D array of the starting index of each sequence's data
    """
    tracer: Tracer = Tracer("build_token_frequency_csr")

    PADDING_TOKEN = -1

    frequency_row_offsets = np.zeros(len(batch) + 1, dtype=np.uint32)
    # Calculate max size needed for token frequency pairs
    if include_prompt:
        total_tokens = sum(
            context.current_length + padding_size for context in batch
        )
    else:
        total_tokens = sum(
            len(context.generated_tokens) + padding_size for context in batch
        )
    token_frequency_pairs = np.zeros((total_tokens, 2), dtype=np.int32)

    tracer.next("build_token_frequency_csr_loop")
    for i, context in enumerate(batch):
        unique_tokens, counts = np.unique(
            context.all_tokens if include_prompt else context.generated_tokens,
            return_counts=True,
        )
        # Pad the tokens and counts to reserve space for new tokens
        unique_tokens = np.pad(
            unique_tokens,
            (0, padding_size),
            mode="constant",
            constant_values=PADDING_TOKEN,
        )
        counts = np.pad(
            counts, (0, padding_size), mode="constant", constant_values=0
        )
        frequency_row_offsets[i + 1] = frequency_row_offsets[i] + len(
            unique_tokens
        )
        token_frequency_pairs[
            frequency_row_offsets[i] : frequency_row_offsets[i + 1], 0
        ] = unique_tokens
        token_frequency_pairs[
            frequency_row_offsets[i] : frequency_row_offsets[i + 1], 1
        ] = counts

    token_frequency_pairs = token_frequency_pairs[
        : frequency_row_offsets[-1], :
    ]

    return FrequencyData(
        data=Tensor.from_dlpack(token_frequency_pairs).to(device),
        offsets=Tensor.from_dlpack(frequency_row_offsets).to(device),
    )


@traced
def _build_min_tokens_masks(
    batch: list[T],
    num_steps: int,
    device: Device,
    enable_min_tokens: bool,
) -> list[Tensor] | None:
    """Build a mask of the min tokens for the batch."""
    if not enable_min_tokens:
        for context in batch:
            if context.min_tokens > 0:
                logger.warning(
                    "min_tokens is provided in the request, but the model was not configured with enable_min_tokens=True, ignoring"
                )
        return None

    min_tokens_masks: list[npt.NDArray[np.int32]] = []
    min_tokens_masks = batch[0].get_min_token_logit_mask(num_steps)

    for bs in range(1, len(batch)):
        new_min_tokens_masks = batch[bs].get_min_token_logit_mask(num_steps)
        for i in range(num_steps):
            new_min_tokens_masks[i][:, 0] += bs
            min_tokens_masks[i] = np.concatenate(
                (min_tokens_masks[i], new_min_tokens_masks[i])
            )

    min_tokens_masks_max = [
        Tensor.from_dlpack(mask).to(device) for mask in min_tokens_masks
    ]
    return min_tokens_masks_max


@traced
def _sample_logits(
    sampler: Model,
    logits: Tensor,
    prev_tokens: Tensor,
    top_k: Tensor,
    max_k: Tensor,
    temperature: Tensor,
    top_p: Tensor,
    seed: Tensor,
    *,
    logit_offsets: Tensor | None = None,
    bitmask: Tensor | None = None,
    frequency_data: Sequence[FrequencyData] | None = None,
    min_tokens_mask: Tensor | None = None,
    frequency_penalty: Tensor | None = None,
    presence_penalty: Tensor | None = None,
    repetition_penalty: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    opt_inputs = [logit_offsets, bitmask]

    base_inputs = [
        logits,
        prev_tokens,
        top_k,
        max_k,
        temperature,
        top_p,
        seed,
    ]

    # Add frequency data if provided
    if frequency_data:
        for freq_data in frequency_data:
            opt_inputs.extend([freq_data.data, freq_data.offsets])
        assert frequency_penalty is not None
        assert presence_penalty is not None
        assert repetition_penalty is not None
        opt_inputs.extend(
            [frequency_penalty, presence_penalty, repetition_penalty]
        )

    if min_tokens_mask:
        opt_inputs.append(min_tokens_mask)

    graph_inputs = base_inputs + [
        tensor for tensor in opt_inputs if tensor is not None
    ]

    sampler_output = sampler(*graph_inputs)
    tokens, generated_tokens = sampler_output[:2]
    new_seed = sampler_output[-1]
    assert isinstance(tokens, Tensor)
    assert isinstance(generated_tokens, Tensor)
    assert isinstance(new_seed, Tensor)
    return (tokens, generated_tokens, new_seed)
