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
"""Batch processor base types and shared helpers for MAX pipeline models."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Generic,
    Protocol,
    TypeVar,
    cast,
    runtime_checkable,
)

import numpy as np
from max.driver import Buffer, Device, is_virtual_device_mode
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.nn.comm import Signals
from max.nn.kv_cache import KVCacheInputsInterface
from max.nn.kv_cache.cache_params import KVCacheParamInterface
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.context import BaseContext
from max.pipelines.context.tokens import TokenBuffer
from max.pipelines.lib.interfaces.arch_config import ArchConfig
from max.pipelines.lib.interfaces.pipeline_model import (
    ModelInputs,
    ModelOutputs,
)

if TYPE_CHECKING:
    from max.pipelines.lib import PipelineConfig
    from max.pipelines.lora import LoRAManager

logger = logging.getLogger("max.pipelines")

ContextT = TypeVar("ContextT", bound=BaseContext)
InputsT = TypeVar("InputsT", bound=ModelInputs)


@runtime_checkable
class RaggableContext(Protocol):
    """Context protocol for ragged token batching helpers."""

    tokens: TokenBuffer


@dataclass
class BatchProcessorRuntime:
    """Runtime dependencies shared by batch processors."""

    pipeline_config: PipelineConfig
    devices: list[Device]
    return_logits: ReturnLogits
    return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE
    signal_buffers: Sequence[Buffer] = ()
    lora_manager: LoRAManager | None = None
    pad_token_id: int = 0
    max_batch_size: int | None = None


class BatchProcessor(ABC, Generic[ContextT, InputsT]):
    """Batches pipeline contexts into model inputs and parses execution outputs."""

    def __init__(
        self,
        config: ArchConfig,
        runtime: BatchProcessorRuntime,
    ) -> None:
        self.config = config
        self.runtime = runtime

    @abstractmethod
    def get_symbolic_inputs(
        self,
        *,
        kv_params: KVCacheParamInterface,
        device_refs: list[DeviceRef],
    ) -> list[TensorType | BufferType]:
        """Returns non-KV graph input types in execution order."""

    @abstractmethod
    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[ContextT]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> InputsT:
        """Prepares inputs for the first execution step of a batch."""

    @abstractmethod
    def process_outputs(
        self, outputs: Sequence[Buffer | object]
    ) -> ModelOutputs:
        """Maps raw ``Model.execute`` buffers to :class:`ModelOutputs`."""


class RaggedBatchProcessor(BatchProcessor[ContextT, InputsT]):
    """Base for ragged KV text batching."""

    def __init__(
        self,
        config: ArchConfig,
        runtime: BatchProcessorRuntime,
    ) -> None:
        super().__init__(config, runtime)
        # Pre-allocate row offsets for multistep decode to avoid materializing
        # and copying a buffer on each step. Skip in virtual device mode
        # (warm-cache/cross-compilation) since VirtualDeviceContext does not
        # support memAlloc.
        assert runtime.max_batch_size, "Expected max_batch_size to be set"
        self._input_row_offsets_prealloc: Buffer | None = None
        if not is_virtual_device_mode() and runtime.devices:
            self._input_row_offsets_prealloc = Buffer.from_numpy(
                np.arange(runtime.max_batch_size + 1, dtype=np.uint32),
            ).to(runtime.devices[0])


def single_replica_context_batch(
    replica_batches: Sequence[Sequence[ContextT]],
    *,
    processor_name: str,
) -> Sequence[ContextT]:
    """Returns the sole replica batch or raises when DP is unsupported."""
    if len(replica_batches) > 1:
        raise ValueError(f"{processor_name} does not support DP>1")
    return replica_batches[0]


def build_single_replica_ragged_token_arrays(
    context_batch: Sequence[RaggableContext],
) -> tuple[np.ndarray, np.ndarray]:
    """Builds concatenated token and row-offset arrays for a ragged batch."""
    input_row_offsets = np.cumsum(
        [0] + [ctx.tokens.active_length for ctx in context_batch],
        dtype=np.uint32,
    )
    tokens = np.concatenate([ctx.tokens.active for ctx in context_batch])
    return tokens, input_row_offsets


class SingleReplicaRaggedBatchProcessor(
    RaggedBatchProcessor[ContextT, InputsT]
):
    """Single-replica ragged KV batching for Graph-path models (no DP / LoRA).

    Subclasses implement :meth:`_make_inputs` to construct their architecture-
    specific :class:`~max.pipelines.lib.interfaces.pipeline_model.ModelInputs`
    type. Override :attr:`_include_signal_buffers` or :meth:`get_symbolic_inputs`
    when signal-buffer wiring differs.
    """

    _include_signal_buffers: ClassVar[bool] = False

    def get_symbolic_inputs(
        self,
        *,
        kv_params: KVCacheParamInterface,
        device_refs: list[DeviceRef],
    ) -> list[TensorType | BufferType]:
        """Returns symbolic graph inputs for single-replica ragged KV models."""
        return ragged_kv_symbolic_inputs(
            kv_params=kv_params,
            device_refs=device_refs,
            include_signal_buffers=self._include_signal_buffers,
        )

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[ContextT]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> InputsT:
        """Prepares ragged token inputs for a single-replica Graph-path batch."""
        context_batch = single_replica_context_batch(
            replica_batches,
            processor_name=type(self).__qualname__,
        )
        device0 = self.runtime.devices[0]
        tokens_np, offsets_np = build_single_replica_ragged_token_arrays(
            cast(Sequence[RaggableContext], context_batch)
        )
        return self._make_inputs(
            tokens=Buffer.from_numpy(tokens_np).to(device0),
            input_row_offsets=Buffer.from_numpy(offsets_np).to(device0),
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            kv_cache_inputs=kv_cache_inputs,
            signal_buffers=list(self.runtime.signal_buffers),
        )

    def _make_inputs(
        self,
        *,
        tokens: Buffer,
        input_row_offsets: Buffer,
        return_n_logits: Buffer,
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None,
        signal_buffers: list[Buffer],
    ) -> InputsT:
        """Constructs architecture-specific model inputs."""
        raise NotImplementedError(
            f"{type(self).__qualname__} must implement _make_inputs"
        )

    def process_outputs(
        self, outputs: Sequence[Buffer | object]
    ) -> ModelOutputs:
        """Maps raw execution buffers to :class:`ModelOutputs`."""
        return process_ragged_kv_outputs(
            outputs,
            return_logits=self.runtime.return_logits,
            return_hidden_states=self.runtime.return_hidden_states,
        )


class ModuleV3SingleReplicaBatchProcessor(BatchProcessor[ContextT, InputsT]):
    """Single-replica ragged KV batching for ModuleV3 compile-path models."""

    def get_symbolic_inputs(
        self,
        *,
        kv_params: KVCacheParamInterface,
        device_refs: list[DeviceRef],
    ) -> list[TensorType | BufferType]:
        """Returns symbolic graph inputs for single-replica ModuleV3 models."""
        return modulev3_ragged_kv_symbolic_inputs(
            kv_params=kv_params,
            device_refs=device_refs,
        )

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[ContextT]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> InputsT:
        """Prepares ragged token inputs for a single-replica ModuleV3 batch."""
        context_batch = single_replica_context_batch(
            replica_batches,
            processor_name=type(self).__qualname__,
        )
        assert kv_cache_inputs is not None
        tokens_np, offsets_np = build_single_replica_ragged_token_arrays(
            cast(Sequence[RaggableContext], context_batch)
        )
        device0 = self.runtime.devices[0]
        return self._make_inputs(
            tokens=Buffer.from_numpy(tokens_np).to(device0),
            input_row_offsets=Buffer.from_numpy(offsets_np).to(device0),
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            kv_cache_inputs=kv_cache_inputs,
        )

    def _make_inputs(
        self,
        *,
        tokens: Buffer,
        input_row_offsets: Buffer,
        return_n_logits: Buffer,
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer],
    ) -> InputsT:
        """Constructs architecture-specific model inputs."""
        raise NotImplementedError(
            f"{type(self).__qualname__} must implement _make_inputs"
        )

    def process_outputs(
        self, outputs: Sequence[Buffer | object]
    ) -> ModelOutputs:
        """Maps raw execution buffers to :class:`ModelOutputs`."""
        return process_ragged_kv_outputs(
            outputs,
            return_logits=self.runtime.return_logits,
            return_hidden_states=self.runtime.return_hidden_states,
        )


def process_ragged_kv_outputs(
    outputs: Sequence[Buffer | object],
    *,
    return_logits: ReturnLogits,
    return_hidden_states: ReturnHiddenStates,
) -> ModelOutputs:
    """Maps standard ragged+KV logits buffers to :class:`ModelOutputs`."""
    has_offsets = return_logits in (ReturnLogits.VARIABLE, ReturnLogits.ALL)
    has_hidden_states = return_hidden_states != ReturnHiddenStates.NONE

    assert isinstance(outputs[0], Buffer)
    if has_offsets and has_hidden_states:
        assert len(outputs) == 4
        assert isinstance(outputs[1], Buffer)
        assert isinstance(outputs[2], Buffer)
        assert isinstance(outputs[3], Buffer)
        return ModelOutputs(
            logits=outputs[1],
            next_token_logits=outputs[0],
            logit_offsets=outputs[2],
            hidden_states=outputs[3],
        )
    if has_offsets:
        assert len(outputs) == 3
        assert isinstance(outputs[1], Buffer)
        assert isinstance(outputs[2], Buffer)
        return ModelOutputs(
            logits=outputs[1],
            next_token_logits=outputs[0],
            logit_offsets=outputs[2],
        )
    if has_hidden_states:
        assert len(outputs) == 2
        assert isinstance(outputs[1], Buffer)
        return ModelOutputs(
            logits=outputs[0],
            next_token_logits=outputs[0],
            hidden_states=outputs[1],
        )
    assert len(outputs) == 1
    return ModelOutputs(
        logits=outputs[0],
        next_token_logits=outputs[0],
    )


def ragged_kv_symbolic_inputs(
    *,
    kv_params: KVCacheParamInterface,
    device_refs: list[DeviceRef],
    include_signal_buffers: bool,
) -> list[TensorType | BufferType]:
    """Returns symbolic graph inputs for a standard ragged KV text model."""
    device_ref = device_refs[0]
    return_n_logits_type = TensorType(
        DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
    )
    tokens_type = TensorType(
        DType.int64, shape=["total_seq_len"], device=device_ref
    )
    input_row_offsets_type = TensorType(
        DType.uint32, shape=["input_row_offsets_len"], device=device_ref
    )
    kv_inputs = kv_params.get_symbolic_inputs().flatten()
    if include_signal_buffers:
        signals = Signals(devices=device_refs)
        return [
            tokens_type,
            input_row_offsets_type,
            return_n_logits_type,
            *signals.input_types(),
            *kv_inputs,
        ]
    return [
        tokens_type,
        input_row_offsets_type,
        return_n_logits_type,
        *kv_inputs,
    ]


def modulev3_ragged_kv_symbolic_inputs(
    *,
    kv_params: KVCacheParamInterface,
    device_refs: list[DeviceRef],
) -> list[TensorType | BufferType]:
    """Symbolic compile inputs for ModuleV3 ragged KV models.

    ModuleV3 ``forward`` expects ``(tokens, return_n_logits, input_row_offsets,
    *kv)`` — a different argument order than the Graph-path
    :func:`ragged_kv_symbolic_inputs`.
    """
    device_ref = device_refs[0]
    return_n_logits_type = TensorType(
        DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
    )
    tokens_type = TensorType(
        DType.int64, shape=["total_seq_len"], device=device_ref
    )
    input_row_offsets_type = TensorType(
        DType.uint32, shape=["input_row_offsets_len"], device=device_ref
    )
    kv_inputs = kv_params.get_symbolic_inputs().flatten()
    return [
        tokens_type,
        return_n_logits_type,
        input_row_offsets_type,
        *kv_inputs,
    ]
