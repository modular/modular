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

"""Fused sampling logits processor used to apply sampling parameters to logits."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import Model
from max.interfaces import BatchProcessorInputs, TextGenerationContextType
from max.profiler import Tracer, traced

if TYPE_CHECKING:
    from ..config import PipelineConfig

logger = logging.getLogger("max.pipelines")


@dataclass(frozen=True)
class FrequencyData:
    """Container for token frequency data in CSR format."""

    data: Buffer
    """data[:, 0]: 1D array of the column indices of the
        non-zero elements in the matrix.
    data[:, 1]: 1D array of the non-zero elements in the
        matrix."""

    offsets: Buffer
    """Row offsets: shape [batch_size + 1] indicating start of each
    sequence's data."""


def to_pinned_host_buffer(
    values: Sequence[Any], device: Device, dtype: DType = DType.float32
) -> Buffer:
    """Copies a sequence of values into a buffer, pinned when device is not host.

    Args:
        values: Values to copy into the buffer.
        device: Target device (buffer is pinned if this is a non-host device).
        dtype: Buffer dtype. Defaults to float32.

    Returns:
        A buffer containing the values, with pinning if applicable.
    """
    pinned = not device.is_host

    buffer = Buffer(
        shape=(len(values),),
        dtype=dtype,
        device=device,
        pinned=pinned,
    )

    if pinned:
        buffer.disable_auto_sync()

    buffer_np = buffer.to_numpy()
    buffer_np[:] = values
    return buffer


class FusedSamplingProcessor:
    """Applies sampling parameters to logits and stores the chosen tokens."""

    new_tokens: Buffer | None = None
    """The new tokens that were sampled."""

    generated_tokens: Buffer
    """The generated tokens that have been sampled so far."""

    @traced
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

        # If a structured decoding bitmask was provided, unpack packed-int masks once.
        self.tensor_bitmask: Buffer | None
        if (
            self.bitmask is not None
            and self.vocab_size is not None
            and self.bitmask.shape[1] != self.vocab_size
        ):
            # TODO: migrate bitmask to pinned memory
            bits = 2 ** np.arange(32, dtype=np.int32)
            self.bitmask = (self.bitmask[..., np.newaxis] & bits) != 0
            self.bitmask = self.bitmask.reshape(self.batch_size, -1).astype(
                np.bool_
            )
            self.tensor_bitmask = Buffer.from_numpy(self.bitmask).to(
                self.device
            )
        else:
            self.tensor_bitmask = None

        batch_size = len(context_batch)

        # Allocate input and output tensors on pinned memory for faster async
        # h2d and d2h transfer speeds. If model is on host, then fall back to
        # normal pageable memory.
        pinned = not device.is_host

        # generated_tokens is a tensor of shape (batch_size, 0)
        self.generated_tokens = Buffer(
            shape=(batch_size, 0),
            dtype=DType.int64,
            device=device,
            pinned=pinned,
        )
        if pinned:
            self.generated_tokens.disable_auto_sync()

        temperature_host = to_pinned_host_buffer(
            [context.sampling_params.temperature for context in context_batch],
            device=device,
        )
        self.temperature = temperature_host.to(device)

        # top_k is a tensor of shape (batch_size,)
        top_k_host = to_pinned_host_buffer(
            [context.sampling_params.top_k for context in context_batch],
            device=device,
            dtype=DType.int64,
        )
        self.top_k = top_k_host.to(device)

        # max_k is a scalar 0-d tensor. It does not need to be pinned since it
        # is not copied to the device.
        max_k_np = np.array(np.max(top_k_host.to_numpy()), dtype=np.int64)
        self.max_k = Buffer.from_numpy(max_k_np)

        # top_p is a tensor of shape (batch_size,)
        top_p_host = to_pinned_host_buffer(
            [context.sampling_params.top_p for context in context_batch],
            device=device,
        )
        self.top_p = top_p_host.to(device)

        # min_top_p is a scalar 0-d tensor. It does not need to be pinned since it
        # is not copied to the device.
        min_top_p_np = np.array(np.min(top_p_host.to_numpy()), dtype=np.float32)
        self.min_top_p = Buffer.from_numpy(min_top_p_np)

        # seed is a tensor of shape (batch_size,)
        seed_host = to_pinned_host_buffer(
            [
                context.sampling_params.seed + len(context.tokens)
                for context in context_batch
            ],
            device=device,
            dtype=DType.uint64,
        )
        self.seed = seed_host.to(device)

        self.frequency_data: list[FrequencyData] | None = None
        self.frequency_penalty: Buffer | None = None
        self.presence_penalty: Buffer | None = None
        self.repetition_penalty: Buffer | None = None

        if pipeline_config.sampling.enable_penalties:
            # TODO: migrate penalties to pinned memory
            self.frequency_data = [
                _build_token_frequency_csr(context_batch, num_steps, device),
                _build_token_frequency_csr(
                    context_batch, num_steps, device, include_prompt=True
                ),
            ]

            self.frequency_penalty = Buffer.from_numpy(
                np.array(
                    [
                        context.sampling_params.frequency_penalty
                        for context in context_batch
                    ],
                    dtype=np.float32,
                )
            ).to(device)
            self.presence_penalty = Buffer.from_numpy(
                np.array(
                    [
                        context.sampling_params.presence_penalty
                        for context in context_batch
                    ],
                    dtype=np.float32,
                )
            ).to(device)
            self.repetition_penalty = Buffer.from_numpy(
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
            pipeline_config.sampling.enable_min_tokens,
        )

        self.step_counter = 0

    def __call__(self, inputs: BatchProcessorInputs) -> None:
        """Processes the batch logits and updates generated tokens and seed."""
        logits = inputs.logits
        logit_offsets = inputs.logit_offsets

        new_tokens, new_generated_tokens, new_seed = _sample_logits(
            self.sampler,
            logits,
            self.generated_tokens,
            self.top_k,
            self.max_k,
            self.temperature,
            self.top_p,
            self.min_top_p,
            self.seed,
            logit_offsets=logit_offsets,
            bitmask=self.tensor_bitmask,
            frequency_data=self.frequency_data,
            min_tokens_mask=self.min_tokens_masks[self.step_counter]
            if self.min_tokens_masks
            else None,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            repetition_penalty=self.repetition_penalty,
        )

        assert isinstance(new_tokens, Buffer)
        assert isinstance(new_generated_tokens, Buffer)
        assert isinstance(new_seed, Buffer)

        self.generated_tokens = new_generated_tokens
        self.seed = new_seed
        self.new_tokens = new_tokens

        self.step_counter += 1


def _check_need_penalties(batch: list[TextGenerationContextType]) -> None:
    """Check if the batch has penalties, but enable_penalties is False."""
    for context in batch:
        if (
            context.sampling_params.frequency_penalty != 0.0
            or context.sampling_params.presence_penalty != 0.0
            or context.sampling_params.repetition_penalty != 1.0
        ):
            logger.warning(
                "penalties are provided in the request, but the model was not configured with enable_penalties=True, ignoring"
            )
            return


@traced
def _build_token_frequency_csr(
    batch: list[TextGenerationContextType],
    padding_size: int,
    device: Device,
    include_prompt: bool = False,
) -> FrequencyData:
    """Builds a CSR matrix of token frequency in the batch.

    The original matrix is (batch_size, vocab_size), where each element is
    the number of times a token appears in the batch.

    Returns:
        FrequencyData containing the CSR representation with:
        - data: 2D array where each row is [token_id, count]
        - row_offsets: 1D array of the starting index of each sequence's data
    """
    tracer: Tracer = Tracer("build_token_frequency_csr")

    PADDING_TOKEN = -1
    batch_size = len(batch)

    # Pre-allocate row offsets
    frequency_row_offsets = np.zeros(batch_size + 1, dtype=np.uint32)

    # Calculate max size needed for token frequency pairs (upper bound)
    if include_prompt:
        total_tokens = sum(
            len(context.tokens) + padding_size for context in batch
        )
    else:
        total_tokens = sum(
            len(context.tokens.generated) + padding_size for context in batch
        )
    token_frequency_pairs = np.empty((total_tokens, 2), dtype=np.int32)

    tracer.next("build_token_frequency_csr_loop")

    current_offset = 0
    for i, context in enumerate(batch):
        tokens = (
            context.tokens.all if include_prompt else context.tokens.generated
        )
        unique_tokens, counts = np.unique(tokens, return_counts=True)

        num_unique = len(unique_tokens)
        padded_size = num_unique + padding_size

        # Record start offset for this context
        frequency_row_offsets[i] = current_offset

        # Write unique tokens and their counts directly (no np.pad)
        end_unique = current_offset + num_unique
        token_frequency_pairs[current_offset:end_unique, 0] = unique_tokens
        token_frequency_pairs[current_offset:end_unique, 1] = counts

        # Write padding directly
        end_padded = current_offset + padded_size
        token_frequency_pairs[end_unique:end_padded, 0] = PADDING_TOKEN
        token_frequency_pairs[end_unique:end_padded, 1] = 0

        current_offset = end_padded

    # Record final offset
    frequency_row_offsets[batch_size] = current_offset

    # Trim to actual size used
    token_frequency_pairs = token_frequency_pairs[:current_offset]

    return FrequencyData(
        data=Buffer.from_dlpack(token_frequency_pairs).to(device),
        offsets=Buffer.from_dlpack(frequency_row_offsets).to(device),
    )


@traced
def _build_min_tokens_masks(
    batch: list[TextGenerationContextType],
    num_steps: int,
    device: Device,
    enable_min_tokens: bool,
) -> list[Buffer] | None:
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
        Buffer.from_dlpack(mask).to(device) for mask in min_tokens_masks
    ]
    return min_tokens_masks_max


@traced
def _sample_logits(
    sampler: Model,
    logits: Buffer,
    prev_tokens: Buffer,
    top_k: Buffer,
    max_k: Buffer,
    temperature: Buffer,
    top_p: Buffer,
    min_top_p: Buffer,
    seed: Buffer,
    *,
    logit_offsets: Buffer | None = None,
    bitmask: Buffer | None = None,
    frequency_data: Sequence[FrequencyData] | None = None,
    min_tokens_mask: Buffer | None = None,
    frequency_penalty: Buffer | None = None,
    presence_penalty: Buffer | None = None,
    repetition_penalty: Buffer | None = None,
) -> tuple[Buffer, Buffer, Buffer]:
    opt_inputs = [logit_offsets, bitmask]

    base_inputs = [
        logits,
        prev_tokens,
        top_k,
        max_k,
        temperature,
        top_p,
        min_top_p,
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
    assert isinstance(tokens, Buffer)
    assert isinstance(generated_tokens, Buffer)
    assert isinstance(new_seed, Buffer)
    return (tokens, generated_tokens, new_seed)
