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
"""Input batching for GLM-5.3-Flash."""

from __future__ import annotations

from max import tree
from max.driver import Buffer
from max.nn.kv_cache import KVCacheInputs
from max.pipelines.architectures.deepseekV3.batch_processor import (
    DeepseekV3BatchProcessorBase,
)
from max.pipelines.architectures.deepseekV3.model import DeepseekV3Inputs

__all__ = ["Glm5NextBatchProcessor", "Glm5NextInputs"]


class Glm5NextInputs(DeepseekV3Inputs):
    """DeepSeek-V3 inputs with the vision-merge buffers after ``tokens``.

    ``vision_embeddings`` and ``vision_scatter_indices`` are set by the
    pipeline's vision seam on every prepared batch, empties included, so the
    tuple always matches :meth:`.glm5_next.Glm5Next.input_types`.
    """

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        return (
            self.tokens,
            *self.vision_embeddings,
            *self.vision_scatter_indices,
            self.input_row_offsets,
            self.host_input_row_offsets,
            self.return_n_logits,
            self.data_parallel_splits,
            *self.signal_buffers,
            *(
                tree.leaves(self.kv_cache_inputs)
                if self.kv_cache_inputs
                else ()
            ),
            *self.batch_context_lengths,
            *self.ep_inputs,
        )


class Glm5NextBatchProcessor(DeepseekV3BatchProcessorBase[Glm5NextInputs]):
    """DeepSeek-V3 batching that builds :class:`Glm5NextInputs`."""

    def _make_mla_inputs(
        self,
        *,
        tokens: Buffer,
        input_row_offsets: Buffer,
        host_input_row_offsets: Buffer,
        batch_context_lengths: list[Buffer],
        signal_buffers: list[Buffer],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None,
        return_n_logits: Buffer,
        data_parallel_splits: Buffer,
        ep_inputs: tuple[Buffer, ...],
    ) -> Glm5NextInputs:
        return Glm5NextInputs(
            tokens=tokens,
            input_row_offsets=input_row_offsets,
            host_input_row_offsets=host_input_row_offsets,
            batch_context_lengths=batch_context_lengths,
            signal_buffers=signal_buffers,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=return_n_logits,
            data_parallel_splits=data_parallel_splits,
            ep_inputs=ep_inputs,
        )
