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

import re
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Generic,
    Protocol,
    cast,
    runtime_checkable,
)

import numpy as np
from max import tree
from max.driver import Buffer, Device, is_virtual_device_mode
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.nn.comm import Signals
from max.nn.graph_groups import (
    RaggedBatch,
    TPSignals,
    VisionMerge,
)
from max.nn.kv_cache import KVCacheInputs
from max.nn.kv_cache.cache_params import (
    KVCacheParamInterface,
    max_pages_per_request,
)
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.context import BaseContext, TextContext
from max.pipelines.context.tokens import TokenBuffer
from max.pipelines.graph_input_stager import (
    GraphInputStager,
    InputDescriptor,
)
from max.pipelines.lib.interfaces.arch_config import ArchConfig
from max.pipelines.lib.interfaces.pipeline_model import (
    ModelInputs,
    ModelOutputs,
    UnifiedSpecDecodeInputs,
)
from max.pipelines.lora import LoRAManagerV3
from max.pipelines.modeling.dataprocessing import collate_batch
from typing_extensions import TypeVar

if TYPE_CHECKING:
    from max.pipelines.lib import PipelineConfig

ContextT = TypeVar("ContextT", bound=BaseContext)
InputsT = TypeVar("InputsT", bound=ModelInputs)
SpecDecodeInputsT = TypeVar("SpecDecodeInputsT", bound=UnifiedSpecDecodeInputs)
ArchConfigT_co = TypeVar(
    "ArchConfigT_co", bound=ArchConfig, default=ArchConfig, covariant=True
)


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

    def max_dims(
        self, dims: Iterable[str], kv_params: KVCacheParamInterface
    ) -> dict[str, int]:
        """Returns the largest size each per-step input dim can take.

        A serving engine allocates each per-step input once at these sizes and
        slices it per batch, so each bound comes from the code that sizes the
        same input for MAX's own staging.

        Args:
            dims: Symbolic dims of a graph's per-step inputs.
            kv_params: The KV cache parameters, which bound a lookup table's
                pages per context.

        Returns:
            Each dim's maximum, keyed by name.

        Raises:
            ValueError: If a dim is not one this runtime bounds, or is an image
                row dim while the vision cache is off.
        """
        tokens = self.max_batch_active_tokens
        exact = {
            "total_seq_len": tokens,
            "input_row_offsets_len": self.max_global_batch_size + 1,
            "return_n_logits": 1,
        }
        image_rows = {
            "num_image_tokens",
            "total_image_tokens",
            "vision_merged_seq_len",
        }
        bounds: dict[str, int] = {}
        for dim in dims:
            if dim in exact:
                bounds[dim] = exact[dim]
            elif dim in image_rows:
                # The vision encoder cache stages only the rows inside each
                # context's active window, which replace tokens of the step.
                # Without it, a context that needs encoding stages every row
                # of every image, which the step's tokens do not bound.
                if self.pipeline_config.runtime.vision_cache_utilization == 0:
                    raise ValueError(
                        f"input dim {dim!r} has no per-step maximum with the"
                        " vision cache off: enable it with"
                        " --vision-cache-utilization"
                    )
                bounds[dim] = tokens
            elif re.fullmatch(r"replica_\d+_batch_size", dim):
                bounds[dim] = self.max_batch_size
            elif re.fullmatch(r"replica_\d+_\w*max_num_pages", dim):
                bounds[dim] = max_pages_per_request(kv_params, self.max_seq_len)
            else:
                raise ValueError(f"no maximum is known for input dim {dim!r}")
        return bounds

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


def ragged_token_descriptors(
    runtime: BatchProcessorRuntime,
) -> list[InputDescriptor]:
    """The two device inputs every ragged batch stages.

    Sized from the batching dimensions rather than a shared capacity: the
    token stream holds one batch's active windows, and the row offsets hold
    one entry per context plus the trailing total. Described in one place so
    the architectures that stage a ragged batch cannot drift apart on the
    sizes.
    """
    staging_device = [runtime.devices[0]]
    return [
        InputDescriptor(
            name=RAGGED_INPUT_TOKENS,
            dtype=DType.int64,
            max_shape=(runtime.max_batch_active_tokens,),
            destinations=staging_device,
        ),
        InputDescriptor(
            name=RAGGED_INPUT_ROW_OFFSETS,
            dtype=DType.uint32,
            max_shape=(runtime.max_global_batch_size + 1,),
            destinations=staging_device,
        ),
    ]


class BatchProcessor(ABC, Generic[ContextT, InputsT, ArchConfigT_co]):
    """Batches pipeline contexts into model inputs and parses execution outputs."""

    input_tree: OrderedDict[str, object] | None = None
    """The graph's named inputs, or ``None`` until the model binds them."""

    def __init__(
        self,
        config: ArchConfigT_co,
        runtime: BatchProcessorRuntime,
        inputs: Sequence[InputDescriptor] = (),
    ) -> None:
        self.config = config
        self.runtime = runtime
        # Holds the reused non-pinned device input buffers so captured graphs
        # replay in place. Pinned host staging never belongs here.
        self._stager = GraphInputStager(inputs)

    @abstractmethod
    def get_symbolic_inputs(
        self,
        *,
        kv_params: KVCacheParamInterface,
        device_refs: list[DeviceRef],
    ) -> list[TensorType | BufferType]:
        """Returns non-KV graph input types in execution order."""

    def bind_input_tree(self, input_tree: OrderedDict[str, object]) -> None:
        """Receives the named inputs of the graph this processor feeds.

        The model owns the tree and builds its graph from it, so the tree
        cannot describe a different graph than the one the processor feeds.

        Args:
            input_tree: The graph's inputs keyed by name, whose leaves are the
                graph's input types in declaration order.
        """
        self.input_tree = input_tree

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


class RaggedBatchProcessor(BatchProcessor[ContextT, InputsT, ArchConfigT_co]):
    """Base for ragged KV text batching."""

    def __init__(
        self,
        config: ArchConfigT_co,
        runtime: BatchProcessorRuntime,
        inputs: Sequence[InputDescriptor] = (),
    ) -> None:
        super().__init__(
            config, runtime, [*ragged_token_descriptors(runtime), *inputs]
        )
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
    RaggedBatchProcessor[ContextT, InputsT, ArchConfigT_co]
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


def ragged_batch_inputs(
    device_ref: DeviceRef,
    fields: Sequence[str] = ("tokens", "row_offsets", "return_n_logits"),
    *,
    data_parallel_degree: int = 1,
    hidden_states: tuple[DType, int] | None = None,
    row_offsets_devices: Sequence[DeviceRef] | None = None,
) -> RaggedBatch:
    """Returns the named per-batch graph inputs, in the order ``fields`` gives.

    Args:
        device_ref: The device that holds the token and row-offset tensors.
        fields: The inputs to include, in graph order. Any of ``tokens``,
            ``row_offsets``, ``host_row_offsets``, ``return_n_logits``,
            ``data_parallel_splits`` and ``hidden_states``.
        data_parallel_degree: The number of replicas, which sizes
            ``data_parallel_splits``.
        hidden_states: The ``(dtype, hidden_size)`` of the EAGLE draft's
            hidden-state input. Required when ``fields`` names
            ``hidden_states``.
        row_offsets_devices: The devices that each take their own
            ``row_offsets``, for a graph that does not broadcast them.
            ``row_offsets`` is then a tuple with one type per device.

    Returns:
        The input types keyed by field name.
    """
    available: dict[str, TensorType | tuple[TensorType, ...]] = {
        "tokens": TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        ),
        "row_offsets": TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        ),
        "host_row_offsets": TensorType(
            DType.uint32,
            shape=["input_row_offsets_len"],
            device=DeviceRef.CPU(),
        ),
        "return_n_logits": TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        ),
        "data_parallel_splits": TensorType(
            DType.int64,
            shape=[data_parallel_degree + 1],
            device=DeviceRef.CPU(),
        ),
    }
    if row_offsets_devices is not None:
        available["row_offsets"] = tuple(
            TensorType(
                DType.uint32, shape=["input_row_offsets_len"], device=device
            )
            for device in row_offsets_devices
        )
    if hidden_states is not None:
        dtype, hidden_size = hidden_states
        available["hidden_states"] = TensorType(
            dtype, shape=["total_seq_len", hidden_size], device=device_ref
        )
    return RaggedBatch((name, available[name]) for name in fields)


def signal_inputs(
    device_refs: list[DeviceRef],
) -> TPSignals:
    """Returns the named per-device allreduce signal buffers.

    Args:
        device_refs: The devices that take part in the allreduce.

    Returns:
        The signal buffer types, one per device, under ``signal``.
    """
    return TPSignals(signal=tuple(Signals(devices=device_refs).input_types()))


def vision_merge_inputs(
    device_refs: Sequence[DeviceRef],
    dtype: DType,
    hidden_size: int,
    *,
    rows_dim: str,
) -> VisionMerge:
    """Returns the per-device vision embeddings and where they merge.

    Args:
        device_refs: The devices that each take a copy of the embeddings.
        dtype: The dtype of the embeddings.
        hidden_size: The width of each embedding row.
        rows_dim: The name of the embeddings' row dimension, which each
            architecture spells its own way.

    Returns:
        The embedding types under ``image_embeddings`` and the positions in the
        token stream that their rows replace under ``image_token_indices``,
        one of each per device.
    """
    return VisionMerge(
        image_embeddings=tuple(
            TensorType(dtype, shape=[rows_dim, hidden_size], device=device)
            for device in device_refs
        ),
        image_token_indices=tuple(
            TensorType(DType.int32, shape=["total_image_tokens"], device=device)
            for device in device_refs
        ),
    )


def ragged_kv_symbolic_input_tree(
    *,
    kv_params: KVCacheParamInterface,
    device_refs: list[DeviceRef],
    include_signal_buffers: bool,
    batch_order: Sequence[str] = ("tokens", "row_offsets", "return_n_logits"),
    hidden_states: tuple[DType, int] | None = None,
) -> OrderedDict[str, object]:
    """Returns the named graph input types of a standard ragged KV text model.

    The tree's leaves are the inputs in execution order (``batch_order``
    fields, signal buffers, KV leaves), and its paths name each one, so a
    consumer that needs to know what a graph slot holds reads the tree rather
    than re-deriving the order.

    Args:
        kv_params: The KV cache parameters that supply the ``kv`` subtree.
        device_refs: The devices the model runs on. The first holds the batch
            tensors.
        include_signal_buffers: Whether to add the allreduce ``signals``.
        batch_order: The per-batch inputs to include, in graph order. See
            :func:`ragged_batch_inputs`.
        hidden_states: The ``(dtype, hidden_size)`` of the EAGLE draft's
            hidden-state input, appended to ``batch_order`` when given.

    Returns:
        The tree of input types, keyed ``batch``, ``signals`` and ``kv``.
    """
    inputs: OrderedDict[str, object] = OrderedDict(
        batch=ragged_batch_inputs(
            device_refs[0],
            (*batch_order, "hidden_states") if hidden_states else batch_order,
            hidden_states=hidden_states,
        )
    )
    if include_signal_buffers:
        inputs["signals"] = signal_inputs(device_refs)
    inputs["kv"] = kv_params.get_symbolic_inputs()
    return inputs


def ragged_kv_symbolic_inputs(
    *,
    kv_params: KVCacheParamInterface,
    device_refs: list[DeviceRef],
    include_signal_buffers: bool,
) -> list[TensorType | BufferType]:
    """Returns symbolic graph inputs for a standard ragged KV text model."""
    return tree.leaves(
        ragged_kv_symbolic_input_tree(
            kv_params=kv_params,
            device_refs=device_refs,
            include_signal_buffers=include_signal_buffers,
        )
    )


def modulev3_ragged_kv_symbolic_input_tree(
    *,
    kv_params: KVCacheParamInterface,
    device_refs: list[DeviceRef],
) -> OrderedDict[str, object]:
    """Returns the named inputs of a ModuleV3 ragged KV model.

    ModuleV3 ``forward`` expects ``(tokens, return_n_logits,
    input_row_offsets, *kv)``, a different order than
    :func:`ragged_kv_symbolic_input_tree`.

    Args:
        kv_params: The KV cache parameters that supply the ``kv`` subtree.
        device_refs: The devices the model runs on.

    Returns:
        The tree of input types, keyed ``batch`` and ``kv``.
    """
    return ragged_kv_symbolic_input_tree(
        kv_params=kv_params,
        device_refs=device_refs,
        include_signal_buffers=False,
        batch_order=("tokens", "return_n_logits", "row_offsets"),
    )


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
    return tree.leaves(
        modulev3_ragged_kv_symbolic_input_tree(
            kv_params=kv_params, device_refs=device_refs
        )
    )


class UnifiedSpecDecodeBatchProcessor(
    BatchProcessor[TextContext, SpecDecodeInputsT, ArchConfigT_co],
    Generic[SpecDecodeInputsT, ArchConfigT_co],
):
    """Ragged batching with persistent buffers and seed for unified spec-decode graphs."""

    def __init__(
        self,
        config: ArchConfigT_co,
        runtime: BatchProcessorRuntime,
    ) -> None:
        super().__init__(config, runtime, ragged_token_descriptors(runtime))
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

        total_seq_len = sum(ctx.tokens.active_length for ctx in context_batch)
        batch_size = len(context_batch)

        with self._stager.stage() as staging:
            tokens_host, (persistent_tokens,) = staging.get(
                RAGGED_INPUT_TOKENS, (total_seq_len,)
            )
            offsets_host, (persistent_input_row_offsets,) = staging.get(
                RAGGED_INPUT_ROW_OFFSETS, (batch_size + 1,)
            )
            np.concatenate(
                [ctx.tokens.active for ctx in context_batch],
                out=tokens_host.to_numpy(),
            )
            np.cumsum(
                [0] + [ctx.tokens.active_length for ctx in context_batch],
                dtype=np.uint32,
                out=offsets_host.to_numpy(),
            )

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
    # The dtypes and devices SingleReplicaEmbeddingBatchProcessor builds.
    return [
        TensorType(DType.uint32, shape=["total_seq_len"], device=device_ref),
        TensorType(
            DType.uint32,
            shape=["input_row_offsets_len"],
            device=DeviceRef.CPU(),
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
