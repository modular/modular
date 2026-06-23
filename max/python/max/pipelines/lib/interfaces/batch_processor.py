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
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
from max.driver import Buffer, Device, is_virtual_device_mode
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.nn.comm import Signals
from max.nn.kv_cache import KVCacheInputsInterface
from max.nn.kv_cache.cache_params import KVCacheParamInterface
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.context import BaseContext
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
