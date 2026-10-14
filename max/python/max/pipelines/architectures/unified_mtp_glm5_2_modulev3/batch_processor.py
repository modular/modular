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
"""Inputs and batching for the ModuleV3 GLM-5.x MTP graph."""

from __future__ import annotations

from max import tree
from max.driver import Buffer
from max.nn.kv_cache import KVCacheInputs

from ..unified_mtp_glm5_2.batch_processor import UnifiedMTPGlm5_2BatchProcessor
from ..unified_mtp_glm5_2.model import UnifiedMTPGlm5_2Inputs


class UnifiedMTPGlm5_2ModuleV3Inputs(UnifiedMTPGlm5_2Inputs):
    """The graph-API MTP inputs, ordered for a ``Module.compile`` graph.

    A compiled graph appends its signal buffers after every declared input,
    so they move from the middle of the target prefix to the end.
    """

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        return (
            self.tokens,
            self.input_row_offsets,
            self.host_input_row_offsets,
            self.return_n_logits,
            self.data_parallel_splits,
            *(
                tree.leaves(self.kv_cache_inputs)
                if self.kv_cache_inputs
                else ()
            ),
            *self.batch_context_lengths,
            *self.ep_inputs,
            *self._spec_decode_tail_buffers(include_in_thinking_phase=True),
            *self.signal_buffers,
        )


class UnifiedMTPGlm5_2ModuleV3BatchProcessor(UnifiedMTPGlm5_2BatchProcessor):
    """Batches as the graph-API MTP model does, into the ModuleV3 order."""

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
    ) -> UnifiedMTPGlm5_2ModuleV3Inputs:
        return UnifiedMTPGlm5_2ModuleV3Inputs(
            tokens=tokens,
            input_row_offsets=input_row_offsets,
            host_input_row_offsets=host_input_row_offsets,
            batch_context_lengths=batch_context_lengths,
            signal_buffers=signal_buffers,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=return_n_logits,
            data_parallel_splits=data_parallel_splits,
            ep_inputs=ep_inputs,
            draft_tokens=None,
            structured_output=self.runtime.pipeline_config.needs_bitmask_constraints,
        )
