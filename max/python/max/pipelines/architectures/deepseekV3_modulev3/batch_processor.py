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
"""Input batching for DeepseekV3 ModuleV3 pipeline models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.nn.comm.ep import EPCommInitializer
from max.nn.kv_cache import KVCacheInputsInterface
from max.nn.kv_cache.cache_params import KVCacheParamInterface
from max.pipelines.context import TextContext
from max.pipelines.lib.interfaces.arch_config import ArchConfig
from max.pipelines.lib.interfaces.batch_processor import (
    BatchProcessorRuntime,
    build_single_replica_ragged_token_arrays,
    modulev3_ragged_kv_symbolic_inputs,
)

from ..deepseekV2_modulev3.batch_processor import (
    DeepseekV2ModuleV3BatchProcessor,
)

if TYPE_CHECKING:
    from .model import DeepseekV3Inputs


class DeepseekV3ModuleV3BatchProcessor(DeepseekV2ModuleV3BatchProcessor):
    """Ragged batching and DP/EP support for DeepseekV3 ModuleV3 models."""

    def __init__(
        self, config: ArchConfig, runtime: BatchProcessorRuntime
    ) -> None:
        super().__init__(config, runtime)
        self._ep_comm_initializer: EPCommInitializer | None = None
        self._batch_context_length = Buffer.zeros(shape=[1], dtype=DType.int32)
        self._kv_cache_page_size = (
            runtime.pipeline_config.model.kv_cache.kv_cache_page_size
        )

    def bind_ep_comm_initializer(
        self, initializer: EPCommInitializer | None
    ) -> None:
        """Wires EP buffers created during model ``load_model``."""
        self._ep_comm_initializer = initializer

    def _ep_inputs(self) -> tuple[Buffer, ...]:
        if self._ep_comm_initializer is None:
            return ()
        return tuple(self._ep_comm_initializer.model_inputs())

    def _update_batch_context_length(
        self, context_batch: Sequence[TextContext]
    ) -> Buffer:
        """Writes the page-aligned total KV context length into the CPU buffer.

        Mirrors what the MLA prefill planner computes for ``buffer_lengths``.
        """
        page_size = self._kv_cache_page_size

        def align_length(length: int) -> int:
            return (length + page_size - 1) // page_size * page_size

        total = sum(
            align_length(ctx.tokens.current_position) for ctx in context_batch
        )
        self._batch_context_length[0] = total
        return self._batch_context_length

    def get_symbolic_inputs(
        self,
        *,
        kv_params: KVCacheParamInterface,
        device_refs: list[DeviceRef],
        extra_input_types: Sequence[TensorType | BufferType] = (),
    ) -> list[TensorType | BufferType]:
        """Returns ModuleV3 symbolic inputs plus DP splits and EP buffer types."""
        inputs: list[TensorType | BufferType] = list(
            modulev3_ragged_kv_symbolic_inputs(
                kv_params=kv_params,
                device_refs=device_refs,
            )
        )
        # Host batch context length, inserted right after input_row_offsets to
        # match DeepseekV3.forward's positional signature (and DeepseekV3Inputs
        # .buffers). Stays on CPU so the MLA plan's buffer_length copy is a
        # host-to-host no-op and the graph remains capturable.
        inputs.insert(
            3,
            TensorType(DType.int32, shape=[1], device=DeviceRef.CPU()),
        )
        return inputs

    def _make_inputs(
        self,
        *,
        tokens: Buffer,
        input_row_offsets: Buffer,
        return_n_logits: Buffer,
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer],
    ) -> DeepseekV3Inputs:
        from .model import DeepseekV3Inputs

        return DeepseekV3Inputs(
            tokens=tokens,
            input_row_offsets=input_row_offsets,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=return_n_logits,
            batch_context_length=self._batch_context_length,
            ep_inputs=self._ep_inputs(),
        )

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> DeepseekV3Inputs:
        """Packs the ragged batch."""
        from .model import DeepseekV3Inputs

        assert kv_cache_inputs is not None
        device0 = self.runtime.devices[0]
        context_batch = [ctx for batch in replica_batches for ctx in batch]
        if context_batch:
            tokens_np, offsets_np = build_single_replica_ragged_token_arrays(
                context_batch
            )
        else:
            tokens_np = np.empty(0, dtype=np.int64)
            offsets_np = np.zeros(1, dtype=np.uint32)
        return DeepseekV3Inputs(
            tokens=Buffer.from_numpy(tokens_np.astype(np.int64)).to(device0),
            input_row_offsets=Buffer.from_numpy(
                offsets_np.astype(np.uint32)
            ).to(device0),
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            batch_context_length=self._update_batch_context_length(
                context_batch
            ),
            ep_inputs=self._ep_inputs(),
        )
