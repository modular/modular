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
"""Model inputs for the DeepseekV3 ModuleV3 pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

from max.driver import Buffer

from ..deepseekV2_modulev3.inputs import DeepseekV2Inputs


@dataclass
class DeepseekV3Inputs(DeepseekV2Inputs):
    batch_context_lengths: list[Buffer] = field(kw_only=True)
    """Host (CPU) page-aligned KV context length, one per DP replica.

    Substituted for the planner's device-resident ``buffer_lengths`` so the
    per-layer ``.to(CPU())`` stays host-to-host and the graph is capturable.
    """

    data_parallel_splits: Buffer | None = field(default=None, kw_only=True)
    input_row_offsets_i64: Buffer | None = field(default=None, kw_only=True)
    ep_inputs: tuple[Buffer, ...] = field(default=(), kw_only=True)

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        """Flat graph inputs in compile ABI order."""
        dp_inputs: tuple[Buffer, ...] = ()
        if self.data_parallel_splits is not None:
            assert self.input_row_offsets_i64 is not None
            dp_inputs = (self.data_parallel_splits, self.input_row_offsets_i64)
        return (
            self.tokens,
            self.return_n_logits,
            self.input_row_offsets,
            *self.batch_context_lengths,
            *dp_inputs,
            *(self.kv_cache_inputs.flatten() if self.kv_cache_inputs else ()),
            *self.ep_inputs,
        )
