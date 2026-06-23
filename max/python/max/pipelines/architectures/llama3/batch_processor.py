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
"""Input batching for Llama 3 pipeline models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from max.driver import Buffer, DevicePinnedBuffer
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.nn.comm.ep import EPCommInitializer
from max.nn.kv_cache import KVCacheInputsInterface
from max.nn.kv_cache.cache_params import KVCacheParamInterface
from max.pipelines.context import TextContext
from max.pipelines.lib.interfaces.arch_config import ArchConfig
from max.pipelines.lib.interfaces.batch_processor import (
    BatchProcessorRuntime,
    RaggedBatchProcessor,
    process_ragged_kv_outputs,
    ragged_kv_symbolic_inputs,
)
from max.pipelines.lib.interfaces.pipeline_model import ModelOutputs
from max.pipelines.lib.utils import compute_data_parallel_splits
from max.pipelines.lora import LoRAInputs
from max.support.algorithm import flatten2d

if TYPE_CHECKING:
    from .model import Llama3Inputs


class Llama3BatchProcessor(RaggedBatchProcessor[TextContext, "Llama3Inputs"]):
    """Ragged batching with pinned host buffers and optional DP / LoRA."""

    def __init__(
        self,
        config: ArchConfig,
        runtime: BatchProcessorRuntime,
    ) -> None:
        super().__init__(config, runtime)
        self._execution_input_buffers: dict[
            tuple[int, int], tuple[Buffer, Buffer, Buffer, Buffer]
        ] = {}

    def get_symbolic_inputs(
        self,
        *,
        kv_params: KVCacheParamInterface,
        device_refs: list[DeviceRef],
    ) -> list[TensorType | BufferType]:
        return ragged_kv_symbolic_inputs(
            kv_params=kv_params,
            device_refs=device_refs,
            include_signal_buffers=len(device_refs) > 1,
        )

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> Llama3Inputs:
        from .model import Llama3Inputs

        dp = self.runtime.pipeline_config.model.data_parallel_degree
        if len(replica_batches) != dp:
            raise ValueError(
                "Number of replica batches must match data parallel degree"
            )

        context_batch = flatten2d(replica_batches)
        device0 = self.runtime.devices[0]
        pinned = not device0.is_host

        batch_size = len(context_batch)
        total_seq_len = sum(ctx.tokens.active_length for ctx in context_batch)
        buffer_key = (batch_size, total_seq_len)
        buffers = self._execution_input_buffers.get(buffer_key)
        if buffers is None:
            if pinned:
                host_tokens: Buffer = DevicePinnedBuffer(
                    dtype=DType.int64,
                    shape=(total_seq_len,),
                    device=device0,
                )
                host_row_offsets: Buffer = DevicePinnedBuffer(
                    dtype=DType.uint32,
                    shape=(batch_size + 1,),
                    device=device0,
                )
            else:
                host_tokens = Buffer(
                    shape=(total_seq_len,),
                    dtype=DType.int64,
                    device=device0,
                )
                host_row_offsets = Buffer(
                    shape=(batch_size + 1,),
                    dtype=DType.uint32,
                    device=device0,
                )
            device_tokens = host_tokens.to(device0)
            device_row_offsets = host_row_offsets.to(device0)
            buffers = (
                host_tokens,
                host_row_offsets,
                device_tokens,
                device_row_offsets,
            )
            self._execution_input_buffers[buffer_key] = buffers
        (
            host_tokens,
            host_row_offsets,
            device_tokens,
            device_row_offsets,
        ) = buffers

        input_row_offsets_np = host_row_offsets.to_numpy()
        np.cumsum(
            [0] + [ctx.tokens.active_length for ctx in context_batch],
            dtype=np.uint32,
            out=input_row_offsets_np,
        )

        return_n_logits_tensor = Buffer.from_numpy(
            np.array([return_n_logits], dtype=np.int64)
        )

        tokens_np = host_tokens.to_numpy()
        if context_batch:
            np.concatenate(
                [ctx.tokens.active for ctx in context_batch],
                out=tokens_np,
            )
        device_tokens.inplace_copy_from(host_tokens)
        device_row_offsets.inplace_copy_from(host_row_offsets)

        if dp > 1:
            data_parallel_splits = Buffer.from_numpy(
                compute_data_parallel_splits(replica_batches)
            )
        else:
            data_parallel_splits = None

        inputs = Llama3Inputs(
            tokens=device_tokens,
            input_row_offsets=device_row_offsets,
            return_n_logits=return_n_logits_tensor,
            signal_buffers=list(self.runtime.signal_buffers),
            kv_cache_inputs=kv_cache_inputs,
            data_parallel_splits=data_parallel_splits,
        )

        lora_manager = self.runtime.lora_manager
        if lora_manager is not None:
            inputs.lora = LoRAInputs(
                *lora_manager.get_lora_graph_inputs(
                    context_batch, input_row_offsets_np, device0
                )
            )

        return inputs

    def process_outputs(
        self, outputs: Sequence[Buffer | object]
    ) -> ModelOutputs:
        return process_ragged_kv_outputs(
            outputs,
            return_logits=self.runtime.return_logits,
            return_hidden_states=self.runtime.return_hidden_states,
        )


class Llama3EpBatchProcessor(Llama3BatchProcessor):
    """Llama3 batching extended with EP MoE communication buffers."""

    def __init__(
        self,
        config: ArchConfig,
        runtime: BatchProcessorRuntime,
    ) -> None:
        super().__init__(config, runtime)
        self._ep_comm_initializer: EPCommInitializer | None = None

    def bind_ep_comm_initializer(
        self, initializer: EPCommInitializer | None
    ) -> None:
        """Wires EP buffers created during model ``load_model``."""
        self._ep_comm_initializer = initializer

    def _ep_inputs(self) -> tuple[Buffer, ...]:
        if self._ep_comm_initializer is None:
            return ()
        return tuple(self._ep_comm_initializer.model_inputs())

    def _host_input_row_offsets_for_dp(
        self, host_row_offsets: Buffer, dp: int
    ) -> Buffer | None:
        return host_row_offsets if dp > 1 else None

    def _prepare_ep_moe_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        return_n_logits: int,
    ) -> tuple[
        Buffer,
        Buffer,
        Buffer,
        Buffer | None,
        tuple[Buffer, ...],
        Buffer | None,
    ]:
        dp = self.runtime.pipeline_config.model.data_parallel_degree
        if len(replica_batches) != dp:
            raise ValueError(
                "Number of replica batches must match data parallel degree"
            )

        context_batch = flatten2d(replica_batches)
        device0 = self.runtime.devices[0]
        pinned = not device0.is_host

        batch_size = len(context_batch)
        total_seq_len = sum(ctx.tokens.active_length for ctx in context_batch)
        buffer_key = (batch_size, total_seq_len)
        buffers = self._execution_input_buffers.get(buffer_key)
        if buffers is None:
            if pinned:
                host_tokens: Buffer = DevicePinnedBuffer(
                    dtype=DType.int64,
                    shape=(total_seq_len,),
                    device=device0,
                )
                host_row_offsets: Buffer = DevicePinnedBuffer(
                    dtype=DType.uint32,
                    shape=(batch_size + 1,),
                    device=device0,
                )
            else:
                host_tokens = Buffer(
                    shape=(total_seq_len,),
                    dtype=DType.int64,
                    device=device0,
                )
                host_row_offsets = Buffer(
                    shape=(batch_size + 1,),
                    dtype=DType.uint32,
                    device=device0,
                )
            device_tokens = host_tokens.to(device0)
            device_row_offsets = host_row_offsets.to(device0)
            buffers = (
                host_tokens,
                host_row_offsets,
                device_tokens,
                device_row_offsets,
            )
            self._execution_input_buffers[buffer_key] = buffers
        (
            host_tokens,
            host_row_offsets,
            device_tokens,
            device_row_offsets,
        ) = buffers

        np.cumsum(
            [0] + [ctx.tokens.active_length for ctx in context_batch],
            dtype=np.uint32,
            out=host_row_offsets.to_numpy(),
        )

        return_n_logits_tensor = Buffer.from_numpy(
            np.array([return_n_logits], dtype=np.int64)
        )

        tokens_np = host_tokens.to_numpy()
        if context_batch:
            np.concatenate(
                [ctx.tokens.active for ctx in context_batch],
                out=tokens_np,
            )
        device_tokens.inplace_copy_from(host_tokens)
        device_row_offsets.inplace_copy_from(host_row_offsets)

        if dp > 1:
            data_parallel_splits = Buffer.from_numpy(
                compute_data_parallel_splits(replica_batches)
            )
        else:
            data_parallel_splits = None

        return (
            device_tokens,
            device_row_offsets,
            return_n_logits_tensor,
            data_parallel_splits,
            self._ep_inputs(),
            self._host_input_row_offsets_for_dp(host_row_offsets, dp),
        )
