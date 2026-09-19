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

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Protocol,
    TypeVar,
    cast,
    runtime_checkable,
)

import numpy as np
from max.driver import (
    Buffer,
    Device,
    DevicePinnedBuffer,
    is_virtual_device_mode,
)
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.nn.comm import Signals
from max.nn.kv_cache import KVCacheInputs
from max.nn.kv_cache.cache_params import KVCacheParamInterface
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.context import BaseContext, TextContext
from max.pipelines.context.tokens import TokenBuffer
from max.pipelines.lib.interfaces.arch_config import ArchConfig
from max.pipelines.lib.interfaces.pipeline_model import (
    ModelInputs,
    ModelOutputs,
    UnifiedSpecDecodeInputs,
)
from max.pipelines.lora import LoRAManagerV3
from max.pipelines.modeling.dataprocessing import collate_batch

if TYPE_CHECKING:
    from max.pipelines.lib import PipelineConfig

ContextT = TypeVar("ContextT", bound=BaseContext)
InputsT = TypeVar("InputsT", bound=ModelInputs)
SpecDecodeInputsT = TypeVar("SpecDecodeInputsT", bound=UnifiedSpecDecodeInputs)


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

    max_batch_size: int
    """Most contexts the scheduler puts in one replica's batch."""

    max_seq_len: int
    """Longest a single sequence can grow, after memory planning clamped it."""

    max_batch_input_tokens: int
    """The scheduler's per-step budget for active (unencoded) tokens."""

    enable_chunked_prefill: bool = True
    """Whether the scheduler splits a long prompt to stay inside that budget.

    Mirrors ``pipeline_config.runtime.enable_chunked_prefill``, which the
    scheduler reads as ``target_tokens_per_batch_ce``'s chunking policy.
    """

    data_parallel_degree: int = 1
    """Replicas the scheduler fills, each to its own batch size and budget.

    A processor stages every replica's contexts into one ragged stream, so
    the whole product bounds the buffers it stages into.
    """

    return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE
    signal_buffers: Sequence[Buffer] = ()
    lora_manager: LoRAManagerV3 | None = None
    pad_token_id: int = 0

    @property
    def max_global_batch_size(self) -> int:
        """Contexts in the largest batch a processor stages.

        :attr:`max_batch_size` bounds one replica; this bounds the batch a
        processor actually sees, which is every replica's contexts staged
        into one ragged stream.
        """
        return self.max_batch_size * self.data_parallel_degree

    @property
    def max_batch_active_tokens(self) -> int:
        """Most active tokens one batch's ragged token stream can carry.

        Every context contributes its active window, and chunked prefill
        holds a replica's sum under :attr:`max_batch_input_tokens` by
        splitting a prompt that would overflow it. With chunked prefill off
        that budget is only a soft limit -- the scheduler admits an
        oversized request whole rather than dropping it -- so one
        full-length sequence can land on top of an already-full budget.
        Neither case can exceed a full batch of full-length sequences, and
        every batch carries at least one token per context. Each replica
        gets its own budget, and they are staged into one stream.
        """
        budget = self.max_batch_input_tokens
        if not self.enable_chunked_prefill:
            budget += self.max_seq_len
        per_replica = max(
            self.max_batch_size,
            min(budget, self.max_batch_size * self.max_seq_len),
        )
        return per_replica * self.data_parallel_degree


RAGGED_INPUT_TOKENS = "ragged_input_tokens"
"""Device input holding a batch's concatenated active tokens."""

RAGGED_INPUT_ROW_OFFSETS = "ragged_input_row_offsets"
"""Device input holding each context's start offset into that token stream."""


class ModelInputBuffers:
    """Device input buffers sized to the largest batch the pipeline allows.

    Each logical input is declared once, up front, with the exact shape its
    batching dimensions let it reach, so its backing allocation is that bound
    and no more. A step then asks for the prefix it needs, and that view is
    cached per shape: graph replay skips the input copy only when capture and
    replay are handed the same ``Buffer`` object, so recreating an equal view
    would copy.

    A name's buffer is per device and never shared with another name, and a
    step writing under a name overwrites what the previous step left there.
    Callers own copying host data in.

    A step that outgrows its declaration raises: the declared maximum is
    meant to be the real bound, so overrunning it means the dimensions the
    input was sized from are wrong, not that this batch is unusual.

    Never keep pinned host staging here: that must be allocated fresh every
    step so the next overlap step's host writes can't clobber an in-flight
    H2D copy.
    """

    def __init__(self) -> None:
        self._declared: dict[str, tuple[DType, tuple[int, ...]]] = {}
        self._backings: dict[tuple[str, int], Buffer] = {}
        self._views: dict[tuple[str, int, tuple[int, ...]], Buffer] = {}

    def declare(
        self, *, name: str, dtype: DType, max_shape: tuple[int, ...]
    ) -> None:
        """Registers an input and the largest shape it can ever take.

        Declaring allocates nothing: the backing buffer is created on the
        first :meth:`view` for a device, which keeps virtual device mode
        (warm-cache and cross-compilation, where ``VirtualDeviceContext``
        cannot ``memAlloc``) from allocating for batches it never runs.

        Args:
            name: Stable identifier for this logical input.
            dtype: Element type, fixed for the life of the input.
            max_shape: The shape at the pipeline's batching limits.

        Raises:
            ValueError: If ``name`` was already declared differently.
        """
        max_shape = tuple(max_shape)
        declared = self._declared.setdefault(name, (dtype, max_shape))
        if declared != (dtype, max_shape):
            raise ValueError(
                f"Model input {name!r} is already declared as "
                f"{declared[0]}{list(declared[1])}, cannot redeclare it as "
                f"{dtype}{list(max_shape)}"
            )

    def view(
        self, *, name: str, shape: tuple[int, ...], device: Device
    ) -> Buffer:
        """Returns this step's buffer for ``name``: a prefix of its backing.

        Args:
            name: A name passed to :meth:`declare`.
            shape: This step's shape, within the declared maximum.
            device: Device to hold the buffer; each device gets its own.

        Raises:
            KeyError: If ``name`` was never declared.
            RuntimeError: If ``shape`` outgrows the declared maximum, which
                means the batching dimensions the input was sized from do
                not bound it.
        """
        shape = tuple(shape)
        key = (name, id(device), shape)
        view = self._views.get(key)
        if view is not None:
            return view

        dtype, max_shape = self._declared[name]
        num_elements = math.prod(shape)
        capacity = math.prod(max_shape)
        if num_elements > capacity:
            raise RuntimeError(
                f"Model input {name!r} needs {num_elements} elements "
                f"(shape={list(shape)}) on {device}, beyond the {capacity} "
                f"it was sized for (max_shape={list(max_shape)}). The "
                f"batching dimensions this input was declared from do not "
                f"bound it."
            )

        backing_key = (name, id(device))
        backing = self._backings.get(backing_key)
        if backing is None:
            backing = Buffer(shape=(capacity,), dtype=dtype, device=device)
            self._backings[backing_key] = backing
        view = backing[:num_elements].view(dtype, shape)
        self._views[key] = view
        return view


class BatchProcessor(ABC, Generic[ContextT, InputsT]):
    """Batches pipeline contexts into model inputs and parses execution outputs."""

    def __init__(
        self,
        config: ArchConfig,
        runtime: BatchProcessorRuntime,
    ) -> None:
        self.config = config
        self.runtime = runtime
        # Holds the reused non-pinned device input buffers so captured graphs
        # replay in place. Pinned host staging never belongs here.
        self._device_inputs = ModelInputBuffers()

    def _declare_ragged_token_inputs(self) -> None:
        """Declares the two device inputs every ragged batch stages.

        Sizes them from the batching dimensions rather than a shared
        capacity: the token stream holds one batch's active windows, and the
        row offsets hold one entry per context plus the trailing total.
        """
        self._device_inputs.declare(
            name=RAGGED_INPUT_TOKENS,
            dtype=DType.int64,
            max_shape=(self.runtime.max_batch_active_tokens,),
        )
        self._device_inputs.declare(
            name=RAGGED_INPUT_ROW_OFFSETS,
            dtype=DType.uint32,
            max_shape=(self.runtime.max_global_batch_size + 1,),
        )

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
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
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
        self._declare_ragged_token_inputs()
        # Pre-allocate row offsets for multistep decode to avoid materializing
        # and copying a buffer on each step. Skip in virtual device mode
        # (warm-cache/cross-compilation) since VirtualDeviceContext does not
        # support memAlloc.
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
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
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
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None,
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
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> InputsT:
        """Prepares ragged token inputs for a single-replica ModuleV3 batch.

        Appends the per-call ModuleV3 LoRA buffers when the runtime's manager
        is a :class:`LoRAManagerV3`, reusing the row offsets already built here.
        """
        context_batch = single_replica_context_batch(
            replica_batches,
            processor_name=type(self).__qualname__,
        )
        assert kv_cache_inputs is not None
        tokens_np, offsets_np = build_single_replica_ragged_token_arrays(
            cast(Sequence[RaggableContext], context_batch)
        )
        device0 = self.runtime.devices[0]
        inputs = self._make_inputs(
            tokens=Buffer.from_numpy(tokens_np).to(device0),
            input_row_offsets=Buffer.from_numpy(offsets_np).to(device0),
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            kv_cache_inputs=kv_cache_inputs,
        )
        if isinstance(self.runtime.lora_manager, LoRAManagerV3):
            inputs.lora_buffers = self.runtime.lora_manager.input_buffers(
                context_batch, offsets_np, device0
            )
        return inputs

    def _make_inputs(
        self,
        *,
        tokens: Buffer,
        input_row_offsets: Buffer,
        return_n_logits: Buffer,
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer],
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
    kv_inputs = kv_params.flattened_kv_inputs()
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
    kv_inputs = kv_params.flattened_kv_inputs()
    return [
        tokens_type,
        return_n_logits_type,
        input_row_offsets_type,
        *kv_inputs,
    ]


class UnifiedSpecDecodeBatchProcessor(
    BatchProcessor[TextContext, SpecDecodeInputsT], Generic[SpecDecodeInputsT]
):
    """Ragged batching with persistent buffers and seed for unified spec-decode graphs."""

    def __init__(
        self,
        config: Any,
        runtime: BatchProcessorRuntime,
    ) -> None:
        super().__init__(config, runtime)
        self._declare_ragged_token_inputs()
        self._seed_counter = 0

    def _next_seed(self, device0: Device) -> Buffer:
        self._seed_counter += 1
        return Buffer.from_numpy(
            np.array([self._seed_counter], dtype=np.uint64)
        ).to(device0)

    def get_symbolic_inputs(
        self,
        *,
        kv_params: KVCacheParamInterface,
        device_refs: list[DeviceRef],
    ) -> list[TensorType | BufferType]:
        """Returns symbolic graph inputs for unified spec-decode ragged KV models."""
        return ragged_kv_symbolic_inputs(
            kv_params=kv_params,
            device_refs=device_refs,
            include_signal_buffers=False,
        )

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> SpecDecodeInputsT:
        """Prepares ragged token inputs with persistent buffers and a per-step seed."""
        context_batch = [ctx for batch in replica_batches for ctx in batch]
        device0 = self.runtime.devices[0]
        buffer_type = Buffer if device0.is_host else DevicePinnedBuffer

        total_seq_len = sum(ctx.tokens.active_length for ctx in context_batch)
        batch_size = len(context_batch)

        persistent_tokens = self._device_inputs.view(
            name=RAGGED_INPUT_TOKENS,
            shape=(total_seq_len,),
            device=device0,
        )
        persistent_input_row_offsets = self._device_inputs.view(
            name=RAGGED_INPUT_ROW_OFFSETS,
            shape=(batch_size + 1,),
            device=device0,
        )

        tokens_host = buffer_type(
            dtype=DType.int64,
            shape=(total_seq_len,),
            device=device0,
        )
        offsets_host = buffer_type(
            dtype=DType.uint32,
            shape=(batch_size + 1,),
            device=device0,
        )

        np.concatenate(
            [ctx.tokens.active for ctx in context_batch],
            out=tokens_host.to_numpy(),
        )
        persistent_tokens.inplace_copy_from(tokens_host)
        np.cumsum(
            [0] + [ctx.tokens.active_length for ctx in context_batch],
            dtype=np.uint32,
            out=offsets_host.to_numpy(),
        )
        persistent_input_row_offsets.inplace_copy_from(offsets_host)

        return_n_logits_buf = Buffer.from_numpy(
            np.array([return_n_logits], dtype=np.int64)
        )

        return self._make_inputs(
            tokens=persistent_tokens,
            input_row_offsets=persistent_input_row_offsets,
            return_n_logits=return_n_logits_buf,
            kv_cache_inputs=kv_cache_inputs,
            seed=self._next_seed(device0),
            structured_output=self.runtime.pipeline_config.needs_bitmask_constraints,
        )

    def _make_inputs(
        self,
        *,
        tokens: Buffer,
        input_row_offsets: Buffer,
        return_n_logits: Buffer,
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None,
        seed: Buffer,
        structured_output: bool,
    ) -> SpecDecodeInputsT:
        """Constructs architecture-specific unified spec-decode model inputs."""
        raise NotImplementedError(
            f"{type(self).__qualname__} must implement _make_inputs"
        )

    def process_outputs(
        self, outputs: Sequence[Buffer | object]
    ) -> ModelOutputs:
        """Maps raw execution buffers to :class:`ModelOutputs`."""
        assert isinstance(outputs[0], Buffer)
        return ModelOutputs(logits=outputs[0])


def embedding_ragged_symbolic_inputs(
    *,
    device_refs: list[DeviceRef],
) -> list[TensorType | BufferType]:
    """Symbolic compile inputs for single-replica ragged embedding models."""
    device_ref = device_refs[0]
    return [
        TensorType(DType.int64, shape=["total_seq_len"], device=device_ref),
        TensorType(
            DType.uint32,
            shape=["input_row_offsets_len"],
            device=device_ref,
        ),
        TensorType(
            DType.uint32,
            shape=["return_n_logits"],
            device=DeviceRef.CPU(),
        ),
    ]


class SingleReplicaEmbeddingBatchProcessor(
    BatchProcessor[TextContext, InputsT], Generic[InputsT]
):
    """Single-replica ragged batching for encoder embedding models (no KV cache)."""

    def get_symbolic_inputs(
        self,
        *,
        kv_params: KVCacheParamInterface,
        device_refs: list[DeviceRef],
    ) -> list[TensorType | BufferType]:
        """Returns symbolic graph inputs for ragged embedding models."""
        del kv_params
        return embedding_ragged_symbolic_inputs(device_refs=device_refs)

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> InputsT:
        """Prepares ragged token inputs for a single-replica embedding batch."""
        del kv_cache_inputs
        context_batch = single_replica_context_batch(
            replica_batches,
            processor_name=type(self).__qualname__,
        )
        device = self.runtime.devices[0]

        all_tokens: list[int] = []
        row_offsets = [0]
        for ctx in context_batch:
            all_tokens.extend(ctx.tokens.active)
            row_offsets.append(len(all_tokens))

        return self._make_inputs(
            tokens=Buffer.from_numpy(np.array(all_tokens, dtype=np.uint32)).to(
                device
            ),
            input_row_offsets=Buffer.from_numpy(
                np.array(row_offsets, dtype=np.uint32)
            ),
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.uint32)
            ),
        )

    def _make_inputs(
        self,
        *,
        tokens: Buffer,
        input_row_offsets: Buffer,
        return_n_logits: Buffer,
    ) -> InputsT:
        """Constructs architecture-specific embedding model inputs."""
        raise NotImplementedError(
            f"{type(self).__qualname__} must implement _make_inputs"
        )

    def process_outputs(
        self, outputs: Sequence[Buffer | object]
    ) -> ModelOutputs:
        """Maps raw execution buffers to :class:`ModelOutputs`."""
        assert isinstance(outputs[0], Buffer)
        return ModelOutputs(logits=outputs[0])


def padded_encoder_symbolic_inputs(
    *,
    device_refs: list[DeviceRef],
) -> list[TensorType | BufferType]:
    """Symbolic compile inputs for fixed-shape padded encoder models."""
    device_ref = device_refs[0]
    return [
        TensorType(
            DType.int64,
            shape=["batch_size", "seq_len"],
            device=device_ref,
        ),
        TensorType(
            DType.float32,
            shape=["batch_size", "seq_len"],
            device=device_ref,
        ),
    ]


class PaddedEncoderBatchProcessor(
    BatchProcessor[TextContext, InputsT], Generic[InputsT]
):
    """Fixed-shape padded batching for encoder-only BERT-style models."""

    def get_symbolic_inputs(
        self,
        *,
        kv_params: KVCacheParamInterface,
        device_refs: list[DeviceRef],
    ) -> list[TensorType | BufferType]:
        """Returns symbolic graph inputs for padded encoder models."""
        del kv_params
        return padded_encoder_symbolic_inputs(device_refs=device_refs)

    def _pad_token_id(self) -> int:
        """Returns the pad token id used for ``collate_batch``."""
        return self.runtime.pad_token_id

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> InputsT:
        """Prepares padded token and attention-mask inputs for one replica."""
        del kv_cache_inputs, return_n_logits
        context_batch = single_replica_context_batch(
            replica_batches,
            processor_name=type(self).__qualname__,
        )
        device0 = self.runtime.devices[0]
        tokens = [ctx.tokens.active for ctx in context_batch]
        pad_value = self._pad_token_id()
        next_tokens_batch, _ = collate_batch(
            tokens,
            pad_value=pad_value,
            batch_size=len(tokens),
        )
        attention_mask = (next_tokens_batch != pad_value).astype(np.float32)
        return self._make_inputs(
            next_tokens_batch=Buffer.from_numpy(next_tokens_batch).to(device0),
            attention_mask=Buffer.from_numpy(attention_mask).to(device0),
        )

    def _make_inputs(
        self,
        *,
        next_tokens_batch: Buffer,
        attention_mask: Buffer,
    ) -> InputsT:
        """Constructs architecture-specific padded encoder model inputs."""
        raise NotImplementedError(
            f"{type(self).__qualname__} must implement _make_inputs"
        )

    def process_outputs(
        self, outputs: Sequence[Buffer | object]
    ) -> ModelOutputs:
        """Maps raw execution buffers to :class:`ModelOutputs`."""
        assert isinstance(outputs[0], Buffer)
        return ModelOutputs(logits=outputs[0])


def modulev3_gemma_multimodal_language_symbolic_inputs(
    *,
    kv_params: KVCacheParamInterface,
    device_ref: DeviceRef,
    hidden_size: int,
    embedding_dtype: DType = DType.bfloat16,
) -> list[TensorType | BufferType]:
    """Symbolic language-model inputs for Gemma3 multimodal ModuleV3 compile."""
    tokens_type = TensorType(
        DType.int64, shape=["total_seq_len"], device=device_ref
    )
    image_embeddings_type = TensorType(
        embedding_dtype,
        shape=["num_image_tokens", hidden_size],
        device=device_ref,
    )
    image_token_indices_type = TensorType(
        DType.int32,
        shape=["total_image_tokens"],
        device=device_ref,
    )
    input_row_offsets_type = TensorType(
        DType.uint32,
        shape=["input_row_offsets_len"],
        device=device_ref,
    )
    return_n_logits_type = TensorType(
        DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
    )
    kv_inputs = kv_params.flattened_kv_inputs()
    return [
        tokens_type,
        return_n_logits_type,
        input_row_offsets_type,
        image_embeddings_type,
        image_token_indices_type,
        *kv_inputs,
    ]


def modulev3_idefics3_language_symbolic_inputs(
    *,
    kv_params: KVCacheParamInterface,
    device_ref: DeviceRef,
    hidden_size: int,
    embedding_dtype: DType,
) -> list[TensorType | BufferType]:
    """Symbolic language-model inputs for Idefics3 ModuleV3 compile."""
    tokens_type = TensorType(
        DType.int64, shape=["total_seq_len"], device=device_ref
    )
    input_row_offsets_type = TensorType(
        DType.uint32, shape=["input_row_offsets_len"], device=device_ref
    )
    return_n_logits_type = TensorType(
        DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
    )
    image_embeddings_type = TensorType(
        embedding_dtype,
        shape=["num_image_tokens", hidden_size],
        device=device_ref,
    )
    image_token_indices_type = TensorType(
        DType.int32,
        shape=["total_image_tokens"],
        device=device_ref,
    )
    kv_inputs = kv_params.flattened_kv_inputs()
    return [
        tokens_type,
        input_row_offsets_type,
        return_n_logits_type,
        image_embeddings_type,
        image_token_indices_type,
        *kv_inputs,
    ]
