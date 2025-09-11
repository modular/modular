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

from typing import TypeVar

from max.driver import CPU, Tensor
from max.interfaces.context import InputContext
from max.interfaces.logit_processors_type import (
    BatchLogitsProcessor,
    BatchProcessorInputs,
    ProcessorInputs,
)

T = TypeVar("T", bound=InputContext)


def apply_logits_processors(
    context_batch: list[T],
    batch_logits: Tensor,
    batch_logit_offsets: Tensor | None,
    batch_processors: list[BatchLogitsProcessor] | None = None,
) -> None:
    """Applies logits processors to a batch of logits.

    Args:
        context_batch: The batch of contexts containing the inputs to the model.
        batch_logits: The model logits, a float32 tensor with shape `(N_batch, vocab_size)`.
        batch_logit_offsets: If the model returns multiple logits, this is a tensor with
            shape `(batch_size + 1, 1)` that contains the offsets of each sequence in
            the batch. Otherwise, this is `None`.
        logits_processors: List of logits processors to apply to the logits for
            each context in the batch. The length of this list must match the
            number of contexts in the batch.
        batch_processors: List of batch processors to apply to the batch logits.
            These are applied in order after the individual context-level
            processors.
    """
    if batch_logit_offsets and not batch_logit_offsets.device.is_host:
        batch_logit_offsets = batch_logit_offsets.to(CPU())
    for i, context in enumerate(context_batch):
        processors = context.sampling_params.logits_processors
        if processors is None:
            continue
        for processor in processors:
            start_idx: int | None = None
            end_idx: int | None = None

            if batch_logit_offsets is not None:
                start_idx = batch_logit_offsets[i].item()
                end_idx = batch_logit_offsets[i + 1].item()
            else:
                start_idx = i
                end_idx = i + 1
            logits = batch_logits[start_idx:end_idx, :]
            inputs = ProcessorInputs(
                logits=logits,
                context=context,
            )
            processor(inputs)

    if batch_processors is not None:
        for batch_processor in batch_processors:
            batch_processor(
                BatchProcessorInputs(
                    logits=batch_logits,
                    logit_offsets=batch_logit_offsets,
                    context_batch=context_batch,
                )
            )
