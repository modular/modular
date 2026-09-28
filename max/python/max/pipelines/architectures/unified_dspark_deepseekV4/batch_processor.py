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
"""Input batching for the unified DSpark DeepSeek-V4 pipeline model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from max import tree
from max.driver import Buffer
from max.nn.kv_cache import KVCacheInputs
from max.pipelines.context import TextContext
from max.pipelines.lib import UnifiedSpecDecodeInputs
from max.pipelines.lib.interfaces.batch_processor import (
    UnifiedSpecDecodeBatchProcessor,
)
from typing_extensions import override

from ..deepseekV4.layers.ragged import window_count
from ..deepseekV4.model_config import DeepseekV4Config

__all__ = [
    "UnifiedDSparkDeepseekV4BatchProcessor",
    "UnifiedDSparkDeepseekV4Inputs",
]


@dataclass
class UnifiedDSparkDeepseekV4Inputs(UnifiedSpecDecodeInputs):
    """The ragged prefix, the signal buffers (more than one device only), the
    V4 leaves, the spec tail, then the verify's window counts."""

    tokens: Buffer
    input_row_offsets: Buffer
    return_n_logits: Buffer
    signal_buffers: list[Buffer]
    window_rows: list[Buffer] = field(default_factory=list)
    """Host, one per compression ratio; only the length is read, the verify
    forward's window count."""

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        assert self.kv_cache_inputs is not None
        return (
            self.tokens,
            self.input_row_offsets,
            self.return_n_logits,
            *self.signal_buffers,
            *tree.leaves(self.kv_cache_inputs),
        ) + (
            *self._spec_decode_tail_buffers(include_in_thinking_phase=False),
            *self.window_rows,
        )


class UnifiedDSparkDeepseekV4BatchProcessor(
    UnifiedSpecDecodeBatchProcessor[UnifiedDSparkDeepseekV4Inputs]
):
    """Ragged batching with the spec-decode seed."""

    @override
    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> UnifiedDSparkDeepseekV4Inputs:
        inputs = super().prepare_initial_token_inputs(
            replica_batches, kv_cache_inputs, return_n_logits
        )
        # The verify forward runs over each request's tokens and the drafts
        # it verifies, which the pipeline stages on the context before this.
        lengths = [
            ctx.tokens.active_length
            + len(ctx.spec_decoding_state.draft_tokens_to_verify)
            for batch in replica_batches
            for ctx in batch
        ]
        assert isinstance(self.config, DeepseekV4Config)
        inputs.window_rows = [
            Buffer.from_numpy(np.zeros(window_count(lengths, r), np.uint8))
            for r in self.config.window_ratios
        ]
        return inputs

    def _make_inputs(
        self,
        *,
        tokens: Buffer,
        input_row_offsets: Buffer,
        return_n_logits: Buffer,
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None,
        seed: Buffer,
        structured_output: bool,
    ) -> UnifiedDSparkDeepseekV4Inputs:
        return UnifiedDSparkDeepseekV4Inputs(
            tokens=tokens,
            input_row_offsets=input_row_offsets,
            return_n_logits=return_n_logits,
            # Empty on one device, as the graph declares none there.
            signal_buffers=list(self.runtime.signal_buffers),
            kv_cache_inputs=kv_cache_inputs,
            seed=seed,
            structured_output=structured_output,
        )
