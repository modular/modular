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
"""The fused Qwen3.5 MTP graph's inputs, shared by its model and batch processor."""

from __future__ import annotations

from dataclasses import dataclass, field

from max import tree
from max.driver import Buffer
from max.pipelines.lib import UnifiedSpecDecodeInputs

from ..qwen3_5.model import Qwen3_5Inputs


@dataclass(kw_only=True)
class UnifiedMTPQwen3_5Inputs(UnifiedSpecDecodeInputs, Qwen3_5Inputs):
    """Inputs for the fused Qwen3.5 MTP graph.

    The prefix and the spec-decode tail follow the canonical unified ordering;
    everything after the bitmask triple is this architecture's state-pool tail,
    which no other unified MTP graph has. Also a :class:`Qwen3_5Inputs`, so
    the Qwen3.5 model and batch processor overrides keep their return types.
    """

    tokens: Buffer
    input_row_offsets: Buffer
    host_input_row_offsets: Buffer
    return_n_logits: Buffer
    data_parallel_splits: Buffer
    signal_buffers: list[Buffer]
    batch_context_lengths: list[Buffer]
    live_conv_pools: list[Buffer]
    live_recurrent_pools: list[Buffer]
    live_conv_row_ids: list[Buffer]
    live_recurrent_row_ids: list[Buffer]
    #: Empty on the ring rollback.
    shadow_recurrent_pools: list[Buffer]
    #: The ring pool per device. Empty on the snapshot rollback.
    ring_pools: list[Buffer] = field(default_factory=list)
    #: The ring's ``[num_layers, batch_size]`` rows per device.
    ring_row_ids: list[Buffer] = field(default_factory=list)
    #: ``[3, merged_total_seq_len]`` M-RoPE positions for the merged
    #: ``[real, draft_1..draft_k]`` window. ``None`` on a text-only graph,
    #: whose rotary stays on the static cache-derived table.
    position_ids: Buffer | None = None

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        assert self.kv_cache_inputs is not None
        prefix = (
            self.tokens,
            self.input_row_offsets,
            self.host_input_row_offsets,
            self.return_n_logits,
            self.data_parallel_splits,
            *self.signal_buffers,
            *tree.leaves(self.kv_cache_inputs),
            *self.batch_context_lengths,
        )
        return (
            prefix
            + self._spec_decode_tail_buffers(include_in_thinking_phase=True)
            + (
                *self.live_conv_pools,
                *self.live_recurrent_pools,
                *self.ring_pools,
                *self.live_conv_row_ids,
                *self.live_recurrent_row_ids,
                *self.ring_row_ids,
                *self.shadow_recurrent_pools,
            )
            + (() if self.position_ids is None else (self.position_ids,))
        )
