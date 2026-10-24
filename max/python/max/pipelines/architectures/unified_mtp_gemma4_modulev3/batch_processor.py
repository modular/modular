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

"""Inputs and batching for the ModuleV3 Gemma4 MTP graph."""

from __future__ import annotations

from collections.abc import Sequence

from max import tree
from max.driver import Buffer
from max.nn.kv_cache import KVCacheInputs

from ..gemma4.context import Gemma4Context
from ..unified_mtp_gemma4.batch_processor import (
    UnifiedMTPGemma4BatchProcessor,
)
from ..unified_mtp_gemma4.model import UnifiedMTPGemma4Inputs


class UnifiedMTPGemma4ModuleV3Inputs(UnifiedMTPGemma4Inputs):
    """The graph-API MTP inputs, ordered for a ``Module.compile`` graph.

    A compiled graph appends its signal buffers after every declared
    input, so they move from the middle of the signature (where the
    graph-API declares them) to the end.
    """

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        assert self.kv_cache_inputs is not None
        return (
            self.tokens,
            # Vision embeds + scatter indices follow tokens, matching
            # build_spec_decode_input_types(enable_vision=True).
            *self.vision_embeddings,
            *self.vision_scatter_indices,
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
            *self._spec_decode_tail_buffers(include_in_thinking_phase=True),
            *self.signal_buffers,
        )


class UnifiedMTPGemma4ModuleV3BatchProcessor(UnifiedMTPGemma4BatchProcessor):
    """Batches as the graph-API MTP model does, into the ModuleV3 order."""

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[Gemma4Context]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> UnifiedMTPGemma4ModuleV3Inputs:
        """Prepare batch inputs for a UnifiedMTPGemma4ModuleV3 forward pass.

        Builds the same buffers as the graph-API processor (pinned staging
        tokens/offsets, host offsets mirror, per-device context lengths,
        signals), but constructs the ModuleV3 inputs dataclass whose
        ``buffers`` order matches the compiled graph's signature.
        """
        # Reuse the parent's entire body, then rebuild only the dataclass.
        inputs = super().prepare_initial_token_inputs(
            replica_batches,
            kv_cache_inputs,
            return_n_logits,
        )
        return UnifiedMTPGemma4ModuleV3Inputs(
            tokens=inputs.tokens,
            input_row_offsets=inputs.input_row_offsets,
            host_input_row_offsets=inputs.host_input_row_offsets,
            return_n_logits=inputs.return_n_logits,
            data_parallel_splits=inputs.data_parallel_splits,
            signal_buffers=inputs.signal_buffers,
            kv_cache_inputs=inputs.kv_cache_inputs,
            batch_context_lengths=inputs.batch_context_lengths,
            draft_tokens=inputs.draft_tokens,
            structured_output=inputs.structured_output,
        )
