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
"""Input batching for the Hy4 pipeline."""

from __future__ import annotations

from collections.abc import Sequence

from max.driver import Buffer
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.nn.kv_cache import KVCacheInputs, MultiKVCacheParams
from max.nn.kv_cache.cache_params import KVCacheParamInterface
from max.pipelines.context import TextContext
from max.pipelines.lib.interfaces.batch_processor import (
    BatchProcessorRuntime,
    ModuleV3SingleReplicaBatchProcessor,
)

from .inputs import HYV4Inputs
from .model_config import HYV4Config


def prefill_buffer_length(
    contexts: Sequence[TextContext], page_size: int
) -> int:
    """Returns the MLA prefill plan's ``buffer_lengths[0]`` for a batch: each
    request's cache plus new tokens, ``current_position``, rounded up to a
    page and summed."""
    return sum(
        (ctx.tokens.current_position + page_size - 1) // page_size * page_size
        for ctx in contexts
    )


class HYV4BatchProcessor(
    ModuleV3SingleReplicaBatchProcessor[TextContext, HYV4Inputs]
):
    """Ragged single-replica batching plus the host copy of the MLA prefill
    buffer length, so the graph never reads the plan's back (a host sync that
    device graph capture rejects)."""

    def __init__(
        self, config: HYV4Config, runtime: BatchProcessorRuntime
    ) -> None:
        super().__init__(config, runtime)
        assert isinstance(config.kv_params, MultiKVCacheParams)
        self._page_size = config.kv_params.page_size
        self._batch_context_length = Buffer.zeros(shape=[1], dtype=DType.int32)

    def get_symbolic_inputs(
        self,
        *,
        kv_params: KVCacheParamInterface,
        device_refs: list[DeviceRef],
    ) -> list[TensorType | BufferType]:
        """Returns the ModuleV3 inputs with the host length after the row
        offsets, as ``HYV4.forward`` takes them."""
        inputs = super().get_symbolic_inputs(
            kv_params=kv_params, device_refs=device_refs
        )
        inputs.insert(3, TensorType(DType.int32, [1], device=DeviceRef.CPU()))
        return inputs

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> HYV4Inputs:
        self._batch_context_length.to_numpy()[0] = prefill_buffer_length(
            [ctx for batch in replica_batches for ctx in batch],
            self._page_size,
        )
        return super().prepare_initial_token_inputs(
            replica_batches, kv_cache_inputs, return_n_logits
        )

    def _make_inputs(
        self,
        *,
        tokens: Buffer,
        input_row_offsets: Buffer,
        return_n_logits: Buffer,
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer],
    ) -> HYV4Inputs:
        return HYV4Inputs(
            tokens=tokens,
            input_row_offsets=input_row_offsets,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=return_n_logits,
            batch_context_length=self._batch_context_length,
        )
