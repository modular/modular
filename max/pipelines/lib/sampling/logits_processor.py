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

from max.driver import Tensor
from max.interfaces.context import InputContext
from max.interfaces.logit_processors_type import ProcessorInputs

T = TypeVar("T", bound=InputContext)


def apply_logits_processors(
    context_batch: list[T],
    batch_logits: Tensor,
    batch_logit_offsets: Tensor | None,
) -> Tensor:
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
    return batch_logits
