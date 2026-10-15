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
"""Inputs and batching for the fused MiMo-V2 DFlash pipeline model."""

from __future__ import annotations

from dataclasses import dataclass

from max import tree
from max.driver import Buffer
from max.nn.kv_cache import KVCacheInputs
from max.pipelines.lib import UnifiedSpecDecodeInputs
from max.pipelines.lib.interfaces.batch_processor import (
    UnifiedSpecDecodeBatchProcessor,
)


@dataclass
class UnifiedDflashMiMoV2Inputs(UnifiedSpecDecodeInputs):
    """The fused graph's inputs: tokens, one row-offsets tensor, the logit
    count, a signal buffer per device, the ``{target, draft}`` KV tree, then
    the spec-decode tail."""

    tokens: Buffer
    input_row_offsets: Buffer
    return_n_logits: Buffer
    signal_buffers: list[Buffer]

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        buffers = (
            self.tokens,
            self.input_row_offsets,
            self.return_n_logits,
            *self.signal_buffers,
            *(
                tree.leaves(self.kv_cache_inputs)
                if self.kv_cache_inputs
                else ()
            ),
        )
        return buffers + self._spec_decode_tail_buffers(
            include_in_thinking_phase=False
        )


class UnifiedDflashMiMoV2BatchProcessor(
    UnifiedSpecDecodeBatchProcessor[UnifiedDflashMiMoV2Inputs]
):
    """Ragged batching for the fused graph, with the per-device signal
    buffers its collectives need."""

    def _make_inputs(
        self,
        *,
        tokens: Buffer,
        input_row_offsets: Buffer,
        return_n_logits: Buffer,
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None,
        seed: Buffer,
        structured_output: bool,
    ) -> UnifiedDflashMiMoV2Inputs:
        return UnifiedDflashMiMoV2Inputs(
            tokens=tokens,
            input_row_offsets=input_row_offsets,
            return_n_logits=return_n_logits,
            signal_buffers=list(self.runtime.signal_buffers),
            kv_cache_inputs=kv_cache_inputs,
            seed=seed,
            structured_output=structured_output,
        )
