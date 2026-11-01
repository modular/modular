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
"""Batching for Kimi K3, on top of the generic Llama3 processor.

The EP exchange's real buffers are bound once, after the model loads.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from max.driver import Buffer
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.nn.legacy.kv_cache import KVCacheInputs, MLAKVCacheParams
from max.nn.legacy.kv_cache.cache_params import KVCacheParamInterface
from max.pipelines.architectures.llama3_modulev3.batch_processor import (
    Llama3ModuleV3BatchProcessor,
)
from max.pipelines.context import TextContext
from max.pipelines.lib.interfaces.batch_processor import BatchProcessorRuntime

from .inputs import KimiK3Inputs
from .model_config import KimiK3Config
from .state_cache import attn_cache


def page_aligned_batch_context_length(
    positions: Iterable[int], page_size: int
) -> int:
    """Returns the MLA plan's sum of individually page-aligned context lengths."""
    return sum(
        (position + page_size - 1) // page_size * page_size
        for position in positions
    )


class KimiK3ModuleV3BatchProcessor(Llama3ModuleV3BatchProcessor):
    """Ragged batching with a host MLA prefill length and EP buffers."""

    _ep_buffers: tuple[Buffer, ...] = ()

    def __init__(
        self, config: KimiK3Config, runtime: BatchProcessorRuntime
    ) -> None:
        super().__init__(config, runtime)
        self._page_size = config.kv_params.page_size
        self._has_mla_prefill = isinstance(
            attn_cache(config.kv_params), MLAKVCacheParams
        )
        self._batch_context_length = Buffer.zeros(shape=[1], dtype=DType.int32)

    def get_symbolic_inputs(
        self,
        *,
        kv_params: KVCacheParamInterface,
        device_refs: list[DeviceRef],
    ) -> list[TensorType | BufferType]:
        """Appends the host prefill length after the KV inputs."""
        inputs = super().get_symbolic_inputs(
            kv_params=kv_params, device_refs=device_refs
        )
        if self._has_mla_prefill:
            inputs.append(TensorType(DType.int32, [1], device=DeviceRef.CPU()))
        return inputs

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> KimiK3Inputs:
        """Computes the prefill extent without reading the device plan back."""
        self._batch_context_length.to_numpy()[0] = (
            page_aligned_batch_context_length(
                (
                    ctx.tokens.current_position
                    for batch in replica_batches
                    for ctx in batch
                ),
                self._page_size,
            )
        )
        inputs = super().prepare_initial_token_inputs(
            replica_batches, kv_cache_inputs, return_n_logits
        )
        assert isinstance(inputs, KimiK3Inputs)
        return inputs

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
            batch_context_length=self._batch_context_length
            if self._has_mla_prefill
            else None,
            ep_inputs=self._ep_buffers,
        )
