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
"""Batching for Kimi K3 (ModuleV3): the generic Llama3 processor, plus the
EP exchange's real buffers, bound once after the model loads."""

from __future__ import annotations

from max.driver import Buffer
from max.nn.kv_cache import KVCacheInputs
from max.pipelines.architectures.llama3_modulev3.batch_processor import (
    Llama3ModuleV3BatchProcessor,
)

from .inputs import KimiK3Inputs


class KimiK3ModuleV3BatchProcessor(Llama3ModuleV3BatchProcessor):
    """Ragged batching for Kimi K3, plus the EP exchange's real buffers."""

    _ep_buffers: tuple[Buffer, ...] = ()

    def bind_ep_buffers(self, ep_buffers: tuple[Buffer, ...]) -> None:
        """Wires the EP buffers ``KimiK3PipelineModel`` allocates once."""
        self._ep_buffers = ep_buffers

    def _make_inputs(
        self,
        *,
        tokens: Buffer,
        input_row_offsets: Buffer,
        return_n_logits: Buffer,
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer],
    ) -> KimiK3Inputs:
        return KimiK3Inputs(
            tokens=tokens,
            input_row_offsets=input_row_offsets,
            return_n_logits=return_n_logits,
            kv_cache_inputs=kv_cache_inputs,
            ep_inputs=self._ep_buffers,
        )
