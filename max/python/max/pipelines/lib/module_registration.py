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
"""Serves architectures that register their modules by role.

An architecture that sets
:attr:`~max.pipelines.lib.arch_lookup.SupportedArchitecture.modules` writes
no pipeline model. :func:`module_pipeline_model` builds one from the
registration, and it owns what a per-architecture pipeline model used to:
finalizing the config, constructing each module as ``module_cls(config)``,
the graph signature, batching and execution.

A registered module class can also set:

- ``graph_name``, the name its graph compiles under (:func:`graph_name`);
- ``input_layout``, the role field names in a graph's existing input order.
  Naming ``signals`` places the allreduce signal buffers there, on one device
  too, as the graph-API graph that keeps the layout takes them;
- ``device_context_length_dtype``, when its graph takes
  :attr:`~max.pipelines.lib.role_contracts.DecoderInputs.batch_context_lengths`
  as a graph-API graph does. Setting it changes both their count and their
  dtype: one per device, each replica's on every device of the replica, of
  that dtype, rather than one ``int32`` per replica;
- ``lora_targets``, the projections it serves LoRA adapters on
  (:func:`lora_targets`);
- ``supports_data_parallel``, when it runs one replica per device
  (:func:`supports_data_parallel`);
- ``supports_multi_device_replicas``, when each of its data-parallel
  replicas may also span several devices
  (:func:`supports_multi_device_replicas`);
- ``consumes_vision``, when it reads
  :attr:`~max.pipelines.lib.role_contracts.DecoderInputs.vision`
  (:func:`consumes_vision`);
- ``supports_pipeline_stages``, when its config builds one pipeline
  stage's layers (:func:`supports_pipeline_stages`);
- ``scratch_types(config, *, max_batch_size, devices)``, a classmethod that
  returns the buffer types of the workspace its kernels write each step (see
  :mod:`~max.pipelines.lib.decoder_scratch`). Its batch processor must be a
  :class:`DecoderBatchProcessor`, which passes the buffers;
- ``state_store(config, *, max_batch_size, devices)``, a classmethod that
  returns the store of the per-request state it carries between steps (see
  :mod:`~max.pipelines.lib.request_state`). Its batch processor must be a
  :class:`DecoderBatchProcessor`, which feeds the state.

A registration can add a ``VISION`` module beside the ``DECODER``. The
pipeline model then compiles the vision graph with the decoder's, and the
pipeline encodes each batch's new images through its vision cache and passes
the embeddings to the decoder's
:attr:`~max.pipelines.lib.role_contracts.DecoderInputs.vision` input. The
registered batch processor packs the images and runs the vision graph, since
pixel layouts differ by architecture: it implements
:class:`VisionBatchProcessor`. A registered ``VISION`` class constructs from
the config like any role, and also has ``input_types()``, the vision graph's
inputs, and ``embedding_dtype`` and ``embedding_dim``, the type of each
embedding row it produces. Registration checks the combination up front: a
``VISION`` module needs a decoder that consumes the embeddings and a batch
processor that prepares them, and a decoder that consumes them needs a
``VISION`` module.

:mod:`~max.pipelines.lib.registration_runtime` lists what a registered
config can declare about its deployment.
"""

from __future__ import annotations

import copy
import copyreg
import dataclasses
import itertools
import re
from abc import ABCMeta
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (
    Any,
    ClassVar,
    Protocol,
    TypeAlias,
    cast,
    runtime_checkable,
)

import numpy as np
from max import functional as F
from max import tree
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, DeviceRef, TensorType, Type
from max.graph.weights import WeightsAdapter
from max.nn import Module
from max.nn.common_layers.mesh_axis import DP, TP
from max.nn.kv_cache import KVCacheInputs, KVCacheParamInterface
from max.nn.legacy.comm.ep import EPInit
from max.nn.legacy.comm.ep.ep_manager import _ep_counter_words
from max.nn.legacy.graph_groups import (
    DecoderGraphInputs,
    HiddenStates,
    Logits,
    VisionMerge,
)
from max.nn.legacy.graph_groups import DecoderOutputs as GroupedDecoderOutputs
from max.nn.outputs import ReturnHiddenStates
from max.pipelines.context import TextContext
from max.pipelines.lora import LoRAManagerV3, LoRATargetModule
from max.pipelines.modeling.types import RequestID
from max.tensor import Tensor, default_device, default_dtype
from max.tensor.compilation import CompiledCallable
from max.tensor.sharding import (
    DeviceMapping,
    DeviceMesh,
    Replicated,
    TensorLayout,
)
from transformers import AutoConfig

from .decoder_scratch import DecoderScratch, build_decoder_scratch
from .graph_specs import GraphRole, GraphSpec
from .input_binding import _flatten_role_inputs
from .interfaces.arch_config import ArchConfig, ModelOutputConfig
from .interfaces.batch_processor import (
    BatchProcessor,
    RaggableContext,
    build_single_replica_ragged_token_arrays,
    process_ragged_kv_outputs,
    ragged_batch_inputs,
    single_replica_context_batch,
)
from .interfaces.pipeline_model import (
    ModelInputs,
    ModelLoadContext,
    ModelOutputs,
    ModuleV3PipelineModelWithKVCache,
    _cast_to_parameter_dtypes,
)
from .log_probabilities import LogProbabilitiesMixin
from .pipeline_parallel import (
    PipelinedModel,
    raise_stage_traced,
    stage_input_types,
)
from .registration_runtime import (
    BatchContextLengths,
    ConstantInputs,
    DeploymentConfig,
    ExpertParallelDeploymentConfig,
    ExpertParallelExchange,
    ModuleRuntime,
    batch_context_length_types,
    ep_input_tree,
    initialize_ep_comm_buffers,
    prepare_weights,
    role_weights,
    uses_expert_parallel_exchange,
)
from .request_state import (
    RequestStateGroup,
    RequestStateMixin,
    RequestStateStore,
    build_state_store,
)
from .role_contracts import DecoderInputs, DecoderOutputs, VisionInputs
from .utils import (
    compute_data_parallel_splits,
    parse_state_dict_from_weights,
)
from .vision_encoder_cache import (
    SupportsPooledVisionMetrics,
    SupportsVisionEncoding,
    VideoEncoderMetrics,
    VisionEncoderMetrics,
)
from .weight_transform import (
    CheckpointRole,
    WeightEntry,
    WeightRenaming,
    WeightTransform,
)

__all__ = [
    "DecoderBatchProcessor",
    "DecoderModelInputs",
    "ModuleV3DecoderPipelineModel",
    "RegisteredDecoderBuilder",
    "VisionBatchProcessor",
    "bind_decoder_scratch",
    "bind_state_store",
    "construct_module",
    "decoder_input_types",
    "decoder_vision_input_types",
    "device_mesh",
    "graph_name",
    "module_pipeline_model",
    "supports_pipeline_stages",
]

# Roles that the framework builds, never an architecture.
_FRAMEWORK_ROLES = frozenset(
    {
        GraphRole.SAMPLER,
        GraphRole.SAMPLER_BITMASK,
        GraphRole.EP_INIT,
        GraphRole.SPEC_DECODER,
    }
)

# The roles a registration can fill.
_SERVED_ROLES = frozenset({GraphRole.DECODER, GraphRole.VISION})

# The ragged batch inputs each DecoderInputs field takes its type from.
_BATCH_FIELDS = {
    "tokens": "tokens",
    "input_row_offsets": "row_offsets",
    "host_input_row_offsets": "host_row_offsets",
    "return_n_logits": "return_n_logits",
    "data_parallel_splits": "data_parallel_splits",
}


_FIELD_ORDER = tuple(field.name for field in dataclasses.fields(DecoderInputs))


def graph_name(module_cls: type[Module[..., Any]]) -> str:
    """Returns the name a registered module's graph compiles under.

    The name is part of the precompiled MEF key, so a module whose graph
    predates registration sets ``graph_name`` to keep it.
    """
    return getattr(module_cls, "graph_name", module_cls.__qualname__)


def device_mesh(
    devices: Sequence[Device], data_parallel_degree: int
) -> DeviceMesh:
    """Returns the mesh registered modules are constructed on.

    It is a ``dp`` axis under data parallelism, else a ``tp`` axis.
    """
    axis = DP if data_parallel_degree > 1 else TP
    return DeviceMesh(tuple(devices), (len(devices),), (axis,))


def construct_module(
    module_cls: type[Module[..., Any]], config: object, mesh: DeviceMesh
) -> Module[..., Any]:
    """Constructs a registered module as ``module_cls(config)`` on ``mesh``.

    The caller enters ``F.lazy()``, so the module holds no weights until it
    compiles.
    """
    # A registered module takes its config, which type[Module] can't say.
    constructor: Any = module_cls
    with default_device(mesh):
        module = constructor(config)
    assert isinstance(module, Module)
    return module


def lora_targets(
    module_cls: type[Module[..., Any]],
) -> tuple[LoRATargetModule, ...]:
    """Returns the projections a registered module serves LoRA adapters on.

    A module that sets no ``lora_targets`` can't serve LoRA.
    """
    return tuple(getattr(module_cls, "lora_targets", ()))


def supports_data_parallel(module_cls: type[Module[..., Any]]) -> bool:
    """Returns whether a registered module splits a batch across replicas.

    A module that sets ``supports_data_parallel`` reads
    :attr:`~max.pipelines.lib.role_contracts.DecoderInputs.data_parallel_splits`
    and runs one replica per device of its mesh's ``dp`` axis.
    """
    return bool(getattr(module_cls, "supports_data_parallel", False))


def supports_multi_device_replicas(module_cls: type[Module[..., Any]]) -> bool:
    """Returns whether a registered decoder's replicas may span devices.

    A decoder that sets ``supports_multi_device_replicas`` as well as
    ``supports_data_parallel`` serves fewer replicas than devices, each on
    an equal share of them: it is constructed on a ``dp`` by ``tp`` mesh.
    """
    return bool(getattr(module_cls, "supports_multi_device_replicas", False))


def consumes_vision(module_cls: type[Module[..., Any]]) -> bool:
    """Returns whether a registered decoder reads image embeddings.

    A decoder that sets ``consumes_vision`` reads
    :attr:`~max.pipelines.lib.role_contracts.DecoderInputs.vision`, so it
    registers beside a ``VISION`` module.
    """
    return bool(getattr(module_cls, "consumes_vision", False))


def supports_pipeline_stages(module_cls: type[Module[..., Any]]) -> bool:
    """Returns whether a registered decoder builds as pipeline stages.

    A decoder that sets ``supports_pipeline_stages`` builds the layers of the
    stage its config carries. On a stage after the first it starts from
    :attr:`~max.pipelines.lib.role_contracts.DecoderInputs.stage_boundary`,
    and on a stage before the last it returns the tree it hands the next
    stage in place of its outputs.
    """
    return bool(getattr(module_cls, "supports_pipeline_stages", False))


def decoder_input_types(
    kv_params: KVCacheParamInterface,
    device_ref: DeviceRef,
    *,
    data_parallel_degree: int = 1,
    vision: VisionInputs[TensorLayout] | VisionInputs[TensorType] | None = None,
    token_positions: TensorType | None = None,
    batch_context_lengths: Sequence[TensorType] | None = None,
    ep: object | None = None,
    lora: Sequence[TensorType] = (),
    constants: object | None = None,
    state: Sequence[Any] = (),
    scratch: Sequence[BufferType] = (),
    input_layout: Sequence[str] | None = None,
) -> DecoderGraphInputs:
    """Returns a ``DECODER`` graph's input types, keyed by field.

    The keys are the :class:`~max.pipelines.lib.role_contracts.DecoderInputs`
    fields the deployment sets, in its order, and the leaves are the graph's
    inputs. The graph takes the tree as its one argument, so an exported
    graph names its inputs. It takes ``vision`` as a
    :class:`~max.nn.legacy.graph_groups.VisionMerge` group, the kind a
    serving engine feeds image embeddings by.

    Args:
        kv_params: The KV cache parameters that supply the ``kv`` subtree.
        device_ref: The device that holds the token and row-offset tensors.
        data_parallel_degree: The number of replicas. Above one, the graph
            also takes ``host_input_row_offsets`` and
            ``data_parallel_splits``.
        vision: The image embeddings' input types, when the graph is built
            with a ``VISION`` module. A graph built with them takes them on
            every batch, empty when the batch has no images.
        token_positions: The absolute position of each ragged token.
        batch_context_lengths: Each replica's context-length input type,
            when the decoder reads them.
        ep: The expert-parallel input types, when serving with expert
            parallelism, in the layout of the config's
            :attr:`~max.pipelines.lib.registration_runtime.ExpertParallelDeploymentConfig.expert_parallel_inputs`.
        lora: The LoRA input types, when serving LoRA adapters.
        constants: The checkpoint-backed deployment input types.
        state: The input types of the decoder's per-request state.
        scratch: The decoder's workspace buffer types.
        input_layout: The semantic fields in an existing graph's input
            order. Defaults to the role's canonical order.
    """
    data_parallel = data_parallel_degree > 1
    fields = [
        name
        for name in _BATCH_FIELDS
        if data_parallel
        or name not in ("host_input_row_offsets", "data_parallel_splits")
    ]
    batch = ragged_batch_inputs(
        device_ref,
        [_BATCH_FIELDS[name] for name in fields],
        data_parallel_degree=data_parallel_degree,
    )
    types: OrderedDict[str, object] = OrderedDict(
        (name, batch[_BATCH_FIELDS[name]]) for name in fields
    )
    types["kv"] = kv_params.get_symbolic_inputs()
    if token_positions is not None:
        types["token_positions"] = token_positions
    if vision is not None:
        types["vision"] = VisionMerge(
            image_embeddings=vision.embeddings,
            image_token_indices=vision.scatter_indices,
        )
    if batch_context_lengths is not None:
        types["batch_context_lengths"] = tuple(batch_context_lengths)
    if ep is not None:
        types["ep"] = ep
    if lora:
        types["lora"] = tuple(lora)
    if constants is not None:
        types["constants"] = constants
    if state:
        types["state"] = tuple(state)
    if scratch:
        types["scratch"] = tuple(scratch)
    order = _FIELD_ORDER if input_layout is None else tuple(input_layout)
    if input_layout is not None:
        duplicates = sorted({name for name in order if order.count(name) > 1})
        unknown = sorted(set(order) - set(_FIELD_ORDER))
        missing = sorted(set(types) - set(order))
        if duplicates or unknown or missing:
            raise ValueError(
                f"invalid decoder input_layout: duplicate fields {duplicates}, "
                f"unknown fields {unknown}, missing fields {missing}"
            )
    return DecoderGraphInputs(
        (name, types[name]) for name in order if name in types
    )


def _decoder_fields(inputs: Mapping[str, Any]) -> OrderedDict[str, Any]:
    """Returns a decoder graph's named inputs as its role's fields.

    The graph takes ``vision`` as a ``vision_merge`` group, which the decoder
    reads as :class:`~max.pipelines.lib.role_contracts.VisionInputs`.
    """
    fields = OrderedDict(inputs)
    if (vision := fields.get("vision")) is not None:
        fields["vision"] = VisionInputs(
            embeddings=vision["image_embeddings"],
            scatter_indices=vision["image_token_indices"],
        )
    return fields


def _signal_buffer_index(
    input_types: Mapping[str, object], input_layout: Sequence[str] | None
) -> int | None:
    """Returns where a ModuleV3 decoder's graph takes its signal buffers.

    A decoder whose ``input_layout`` names ``signals`` takes them there:
    after the graph inputs of every field before it, a distributed field
    taking one input per device. Otherwise its compiled graph adds them after
    the last input, and only on several devices.

    Args:
        input_types: The graph's input types, keyed by field.
        input_layout: The decoder's ``input_layout``, if it declares one.
    """
    if input_layout is None or "signals" not in input_layout:
        return None
    before = input_layout[: list(input_layout).index("signals")]
    return sum(
        leaf.mesh.num_devices if isinstance(leaf, TensorLayout) else 1
        for name in before
        if name in input_types
        for leaf in tree.leaves(input_types[name])
    )


def decoder_vision_input_types(
    mesh: DeviceMesh, dtype: DType, hidden_size: int
) -> VisionInputs[TensorLayout]:
    """Returns the decoder's ``vision`` input types, replicated on ``mesh``."""
    replicated = DeviceMapping(mesh, (Replicated(),) * len(mesh.mesh_shape))
    return VisionInputs(
        embeddings=TensorLayout(
            dtype, ["vision_merged_seq_len", hidden_size], replicated
        ),
        scatter_indices=TensorLayout(
            DType.int32, ["total_image_tokens"], replicated
        ),
    )


@runtime_checkable
class VisionBatchProcessor(SupportsVisionEncoding[Any], Protocol):
    """The batch processor of a registration with a ``VISION`` module.

    It packs and encodes the images, through the vision graph the pipeline
    model compiles and binds to it.
    """

    def bind_vision_encoder(
        self,
        *,
        vision_model: CompiledCallable[Any, Any],
        session: InferenceSession,
    ) -> None:
        """Binds the compiled vision graph and the session that runs it."""


@dataclass(kw_only=True)
class DecoderModelInputs(ModelInputs):
    """The prepared inputs of a registered ``DECODER``."""

    tokens: Buffer
    """The batch's ragged tokens."""

    input_row_offsets: Buffer
    """Each sequence's start in :attr:`tokens`."""

    return_n_logits: Buffer
    """How many logits per sequence to return."""

    token_positions: Buffer | None = None
    """Each ragged token's absolute sequence position, when declared."""

    host_input_row_offsets: Buffer | None = None
    """A CPU copy of :attr:`input_row_offsets`, under data parallelism."""

    data_parallel_splits: Buffer | None = None
    """Each replica's first sequence, under data parallelism."""

    batch_context_lengths: list[Buffer] | None = None
    """Each replica's page-aligned KV context length, when the graph takes
    them."""

    ep_inputs: tuple[Buffer, ...] = ()
    """The expert-parallel communication buffers, under expert parallelism."""

    signal_buffers: tuple[Buffer, ...] = ()
    """The graph-API decoder's allreduce signal buffers, when required."""

    constants: tree.Tree[Buffer] | None = None
    """The checkpoint-backed inputs, retained across steps."""

    state: tuple[Buffer, ...] = ()
    """The decoder's per-request state inputs, when it declares any."""

    scratch: tuple[Buffer, ...] = ()
    """The decoder's workspace buffers, when it declares any."""
    stage_boundary: tree.Tree[Buffer] | None = None
    """The preceding stage's forwarded tensors, in its declared tree."""

    request_ids: tuple[RequestID, ...] = ()
    """The step's requests in batch order, when the decoder keeps state for
    them."""

    input_tree_def: tree.TreeDef | None = None
    """The graph's semantic input order, bound by its batch processor."""

    def input_fields(self) -> DecoderInputs[tree.Tree[Buffer], Buffer]:
        """Returns the step's buffers as the shared role tree dataclass."""
        return DecoderInputs[tree.Tree[Buffer], Buffer](
            tokens=self.tokens,
            vision=VisionInputs[tree.Tree[Buffer]](
                embeddings=tuple(self.vision_embeddings),
                scatter_indices=tuple(self.vision_scatter_indices),
            ),
            input_row_offsets=self.input_row_offsets,
            token_positions=self.token_positions,
            host_input_row_offsets=self.host_input_row_offsets,
            return_n_logits=self.return_n_logits,
            data_parallel_splits=self.data_parallel_splits,
            signals=self.signal_buffers,
            kv=self.kv_cache_inputs,
            batch_context_lengths=(
                tuple(self.batch_context_lengths)
                if self.batch_context_lengths is not None
                else None
            ),
            ep=self.ep_inputs,
            lora=tuple(self.lora_buffers),
            constants=self.constants,
            state=self.state,
            scratch=self.scratch,
            stage_boundary=self.stage_boundary,
        )

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        """The step's buffers, in the graph's input order."""
        fields = self.input_fields()
        structure = self.input_tree_def
        if structure is None:
            _, structure = tree.flatten(fields)
        return cast(
            tuple[Buffer, ...],
            _flatten_role_inputs(structure, fields, flat_fields=("ep",)),
        )


class DecoderBatchProcessor(BatchProcessor[TextContext, DecoderModelInputs]):
    """Batches ragged text for a registered ``DECODER``.

    Under data parallelism it concatenates every replica's batch, in replica
    order, and records where each replica's sequences start. It also fills
    the optional inputs the bound graph takes: each replica's context length
    and the expert-parallel buffers.
    """

    _batch_context_lengths: BatchContextLengths | None = None
    _ep_comm_initializer: ExpertParallelExchange | None = None
    _input_tree_def: tree.TreeDef | None = None

    def bind_input_tree(self, input_tree: OrderedDict[str, object]) -> None:
        """Binds the graph's inputs, and sizes the ones this fills."""
        super().bind_input_tree(input_tree)
        _, self._input_tree_def = tree.flatten(_decoder_fields(input_tree))
        # One per replica, or one per device of each replica, as declared.
        lengths = tree.leaves(input_tree.get("batch_context_lengths", ()))
        self._batch_context_lengths = None
        if lengths:
            assert isinstance(lengths[0], TensorType)
            replicas = self.runtime.data_parallel_degree
            self._batch_context_lengths = BatchContextLengths(
                replicas,
                self.runtime.pipeline_config.model.kv_cache.kv_cache_page_size,
                devices_per_replica=len(lengths) // max(1, replicas),
                dtype=lengths[0].dtype,
            )

    def bind_ep_comm_initializer(
        self, initializer: ExpertParallelExchange | None
    ) -> None:
        """Binds the expert-parallel buffers each batch passes the graph."""
        self._ep_comm_initializer = initializer

    constant_inputs: ConstantInputs | None = None
    """The deployment's constant input trees, owned for its lifetime."""

    state_store: RequestStateStore | None = None
    """The decoder's per-request state store, which fills the batch's state
    inputs. The pipeline model sets it once it builds the store."""

    scratch: DecoderScratch | None = None
    """The decoder's workspace, which every batch passes. The pipeline model
    sets it once it allocates the buffers."""

    def max_dims(
        self, dims: Iterable[str], kv_params: KVCacheParamInterface
    ) -> dict[str, int]:
        """Returns the largest size each per-step input dim can take.

        The state store bounds the dims its inputs introduce, and the runtime
        the rest.

        Raises:
            ValueError: If neither bounds a dim.
        """
        dims = list(dims)
        state_bounds = (
            self.state_store.max_dims() if self.state_store is not None else {}
        )
        bounds = self.runtime.max_dims(
            [dim for dim in dims if dim not in state_bounds], kv_params
        )
        return {
            dim: state_bounds[dim] if dim in state_bounds else bounds[dim]
            for dim in dims
        }

    def get_symbolic_inputs(
        self,
        *,
        kv_params: KVCacheParamInterface,
        device_refs: list[DeviceRef],
    ) -> list[Any]:
        """Returns the bound graph's inputs, or its defaults before binding."""
        if self.input_tree is not None:
            return tree.leaves(self.input_tree)
        return tree.leaves(
            decoder_input_types(
                kv_params,
                device_refs[0],
                data_parallel_degree=self.runtime.data_parallel_degree,
                state=(
                    self.state_store.input_types()
                    if self.state_store is not None
                    else ()
                ),
                scratch=(
                    self.scratch.input_types if self.scratch is not None else ()
                ),
            )
        )

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> DecoderModelInputs:
        """Packs the replicas' contexts into one ragged batch.

        Raises:
            ValueError: If there is a batch for other than one replica per
                data-parallel rank.
        """
        assert kv_cache_inputs is not None
        data_parallel_degree = self.runtime.data_parallel_degree
        context_batch: Sequence[TextContext]
        if data_parallel_degree == 1:
            context_batch = single_replica_context_batch(
                replica_batches, processor_name=type(self).__qualname__
            )
        elif len(replica_batches) != data_parallel_degree:
            raise ValueError(
                f"data parallelism expects {data_parallel_degree} replica "
                f"batches, got {len(replica_batches)}"
            )
        else:
            context_batch = [ctx for batch in replica_batches for ctx in batch]
        if context_batch:
            tokens, input_row_offsets = (
                build_single_replica_ragged_token_arrays(
                    cast(Sequence[RaggableContext], context_batch)
                )
            )
        else:
            # Under expert parallelism a replica with no requests still runs,
            # since the others' experts wait on it.
            tokens = np.empty(0, dtype=np.int64)
            input_row_offsets = np.zeros(1, dtype=np.uint32)
        device = self.runtime.devices[0]
        inputs = DecoderModelInputs(
            tokens=Buffer.from_numpy(tokens).to(device),
            input_row_offsets=Buffer.from_numpy(input_row_offsets).to(device),
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            kv_cache_inputs=kv_cache_inputs,
            input_tree_def=self._input_tree_def,
        )
        if self.input_tree is not None and "token_positions" in self.input_tree:
            position_type = self.input_tree["token_positions"]
            assert isinstance(position_type, TensorType)
            positions = (
                np.concatenate(
                    [
                        np.arange(
                            ctx.tokens.current_position
                            - ctx.tokens.active_length,
                            ctx.tokens.current_position,
                            dtype=position_type.dtype.to_numpy(),
                        )
                        for ctx in context_batch
                    ]
                )
                if context_batch
                else np.empty(0, dtype=position_type.dtype.to_numpy())
            )
            inputs.token_positions = Buffer.from_numpy(positions).to(
                position_type.device.to_device()
            )
        if data_parallel_degree > 1:
            inputs.host_input_row_offsets = Buffer.from_numpy(input_row_offsets)
            inputs.data_parallel_splits = Buffer.from_numpy(
                compute_data_parallel_splits(replica_batches)
            )
        if self._batch_context_lengths is not None:
            inputs.batch_context_lengths = self._batch_context_lengths.update(
                replica_batches
            )
        if self._ep_comm_initializer is not None:
            inputs.ep_inputs = tuple(self._ep_comm_initializer.model_inputs())
        if isinstance(self.runtime.lora_manager, LoRAManagerV3):
            inputs.lora_buffers = self.runtime.lora_manager.input_buffers(
                context_batch, input_row_offsets, device
            )
        if self.constant_inputs is not None:
            inputs.constants = self.constant_inputs.buffers
        if self.state_store is not None:
            inputs.state = self.state_store.prepare(context_batch)
            inputs.request_ids = tuple(ctx.request_id for ctx in context_batch)
        if self.scratch is not None:
            inputs.scratch = self.scratch.buffers
        return inputs

    def process_outputs(
        self, outputs: Sequence[Buffer | object]
    ) -> ModelOutputs:
        """Maps raw execution buffers to :class:`ModelOutputs`."""
        return process_ragged_kv_outputs(
            outputs,
            return_logits=self.runtime.return_logits,
            return_hidden_states=self.runtime.return_hidden_states,
        )


_GroupedOutputs: TypeAlias = (
    GroupedDecoderOutputs[Tensor]
    | tuple[GroupedDecoderOutputs[Tensor], RequestStateGroup]
)
"""A decoder graph's outputs, named by group as :func:`_grouped_outputs`
names them."""


class _DecoderGraph(
    Module[[DecoderGraphInputs], _GroupedOutputs | DecoderOutputs]
):
    """Calls a decoder with :class:`DecoderInputs` built from its graph inputs.

    The graph takes the :func:`decoder_input_types` tree as its one argument
    and returns the decoder's outputs named by group, so an exported graph
    names both. A pipeline stage before the last returns what it hands the
    next stage, in the tree the decoder returns it. The graph binds the LoRA
    inputs into the decoder's LoRA layers before calling it. It is
    name-transparent to the decoder, so the weight names are the decoder's
    own.
    """

    def __init__(
        self,
        decoder: Module[[DecoderInputs], DecoderOutputs],
        *,
        lora_manager: LoRAManagerV3 | None = None,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
        last_stage: bool = True,
    ) -> None:
        super().__init__()
        # The LoRA weight adapter nests every checkpoint key under `model.`.
        decoder.name_transparent = lora_manager is None
        self.model = decoder
        self.lora_manager = lora_manager
        self.return_hidden_states = return_hidden_states
        self.last_stage = last_stage

    def forward(
        self, inputs: DecoderGraphInputs
    ) -> _GroupedOutputs | DecoderOutputs:
        """Groups the graph's inputs into fields and runs the decoder."""
        fields = DecoderInputs(**_decoder_fields(inputs))
        if fields.lora is not None:
            assert self.lora_manager is not None
            self.lora_manager.bind_inputs(self.model, fields.lora)
        outputs = self.model(fields)
        if not self.last_stage:
            return outputs
        return _grouped_outputs(outputs, self.return_hidden_states)


def _grouped_outputs(
    outputs: DecoderOutputs, return_hidden_states: ReturnHiddenStates
) -> _GroupedOutputs:
    """Names a decoder's outputs by group, as a graph-API decoder's are.

    The role's fields go in a ``logits`` group, then a ``hidden_states`` group
    keyed by ``return_hidden_states`` when there are any, and its per-request
    ``state`` follows in a
    :class:`~max.pipelines.lib.request_state.RequestStateGroup`. The leaves
    keep the role's order, so the graph's outputs do too.
    """
    grouped: GroupedDecoderOutputs[Tensor] = GroupedDecoderOutputs(
        logits=Logits(
            next_token_logits=outputs.next_token_logits,
            logits=outputs.logits,
            logit_offsets=outputs.logit_offsets,
        )
    )
    if outputs.hidden_states is not None:
        grouped["hidden_states"] = HiddenStates(
            {return_hidden_states.value: (outputs.hidden_states,)}
        )
    if not outputs.state:
        return grouped
    return grouped, RequestStateGroup(state=tuple(outputs.state))


def _role_outputs(outputs: _GroupedOutputs) -> DecoderOutputs:
    """Returns what :func:`_grouped_outputs` named as the role's outputs."""
    state: tuple[Tensor, ...] | None = None
    if isinstance(outputs, tuple):
        outputs, state_group = outputs
        state = state_group["state"]
    logits = outputs.logits
    hidden_states: Tensor | None = None
    if "hidden_states" in outputs:
        (hidden_states,) = outputs.hidden_states.tensors()
    return DecoderOutputs(
        next_token_logits=logits["next_token_logits"],
        logits=logits.get("logits"),
        logit_offsets=logits.get("logit_offsets"),
        hidden_states=hidden_states,
        state=state,
    )


def _returned_types(spec: GraphSpec) -> tree.Tree[Type[Any]]:
    """Returns the types of what a ModuleV3 graph returns, in its tree.

    Raises:
        ValueError: If it returns a tensor on several devices, which one
            tensor type can't describe.
    """
    staged, _ = spec._staged
    # The graph outputs every shard of a distributed tensor.
    types = staged.graph.output_types
    if len(types) != len(staged.out_layouts):
        raise ValueError(
            f"{spec.name} returns a tensor on several devices, which can't"
            " cross a pipeline cut"
        )
    return tree.unflatten(staged.out_tree, types)


def finalize_config(
    config: ArchConfig,
    huggingface_config: AutoConfig,
    state_dict: Mapping[str, Any],
    outputs: ModelOutputConfig,
) -> ArchConfig:
    """Returns ``config`` finalized against the checkpoint.

    A config without ``finalize`` needs nothing from the checkpoint. A config
    whose ``finalize`` completes it in place and returns ``None`` finalizes a
    copy, so ``config`` itself never changes.

    Raises:
        TypeError: If ``finalize`` returns another type of config.
    """
    finalized_copy = copy.copy(config)
    finalize = getattr(finalized_copy, "finalize", None)
    if finalize is None:
        return config
    finalized = finalize(huggingface_config, state_dict, outputs)
    if finalized is None:
        return finalized_copy
    if not isinstance(finalized, type(config)):
        raise TypeError(
            f"{type(config).__qualname__}.finalize() returned "
            f"{type(finalized).__qualname__}, not the config"
        )
    return finalized


def _check_data_parallel(
    decoder_cls: type[Module[..., Any]],
    data_parallel_degree: int,
    num_devices: int,
) -> None:
    """Raises if a decoder can't run ``data_parallel_degree`` replicas.

    Raises:
        ValueError: If data parallelism is on and the decoder doesn't
            support it, or the replicas don't split the devices evenly into
            shares the decoder supports.
    """
    if data_parallel_degree <= 1:
        return
    if not supports_data_parallel(decoder_cls):
        raise ValueError(
            f"{decoder_cls.__qualname__} doesn't support data parallelism"
        )
    if data_parallel_degree != num_devices and not (
        supports_multi_device_replicas(decoder_cls)
        and num_devices % data_parallel_degree == 0
    ):
        raise ValueError(
            f"data parallelism needs one replica per device, got "
            f"{data_parallel_degree} replicas over {num_devices} devices"
        )


def _decoder_mesh(
    decoder_cls: type[Module[..., Any]],
    devices: Sequence[Device],
    data_parallel_degree: int,
) -> DeviceMesh:
    """Returns the mesh a registered decoder is constructed on.

    It splits each replica's share of the devices along a ``tp`` axis when
    the decoder's replicas span several devices.
    """
    replicas, count = data_parallel_degree, len(devices)
    if (
        1 < replicas < count
        and count % replicas == 0
        and supports_multi_device_replicas(decoder_cls)
    ):
        return DeviceMesh(
            tuple(devices), (replicas, count // replicas), (DP, TP)
        )
    return device_mesh(devices, replicas)


class _ModulePipelineModelMeta(ABCMeta):
    """Lets a pipeline model built from a registration be pickled.

    Serving pickles the pipeline model class into the model worker. A class
    :func:`module_pipeline_model` builds has no importable name, so it
    pickles as the registration that rebuilds it.
    """


def _reduce_module_pipeline_model(
    cls: _ModulePipelineModelMeta,
) -> str | tuple[Any, ...]:
    registration = cls.__dict__.get("_registration")
    if registration is None:
        return cls.__qualname__
    return module_pipeline_model, registration


copyreg.pickle(_ModulePipelineModelMeta, _reduce_module_pipeline_model)


class ModuleV3DecoderPipelineModel(
    LogProbabilitiesMixin,
    ModuleV3PipelineModelWithKVCache[TextContext, Any],
    metaclass=_ModulePipelineModelMeta,
):
    """Serves a registered text-generation ``DECODER``, and its ``VISION``.

    :func:`module_pipeline_model` subclasses it for each registration. The
    decoder is constructed as ``decoder_cls(config)`` on a mesh over the
    pipeline's devices: a ``dp`` axis under data parallelism, else a ``tp``
    axis. A ``VISION`` module runs on the first device; the vision cache
    copies its embeddings to every device.
    """

    decoder_cls: ClassVar[type[Module[..., Any]]]
    """The registered ``DECODER`` class."""

    vision_cls: ClassVar[type[Module[..., Any]] | None] = None
    """The registered ``VISION`` class, when the registration has one."""

    ep_comm_initializer: ExpertParallelExchange | None = None
    """The expert-parallel buffers the decoder takes, under expert
    parallelism."""

    export_records: Mapping[str, object] = MappingProxyType({})
    """What the completed config records for exporters, by name
    (:meth:`~max.pipelines.lib.registration_runtime.DeploymentConfig.export_records`)."""

    _session: InferenceSession
    _runtime: ModuleRuntime
    _vision_module: Any = None

    batch_processor_cls: ClassVar[type[BatchProcessor[Any, Any]] | None] = (
        DecoderBatchProcessor
    )

    _state_store: RequestStateStore | None = None

    @property
    def _data_parallel_degree(self) -> int:
        return self.pipeline_config.model.data_parallel_degree

    def _device_mesh(self) -> DeviceMesh:
        """Returns the mesh the modules are constructed on."""
        return _decoder_mesh(
            type(self).decoder_cls, self.devices, self._data_parallel_degree
        )

    @property
    def vision_encoding(self) -> VisionBatchProcessor | None:
        """What encodes the batch's images, when ``VISION`` is registered.

        It is the registered batch processor.

        Raises:
            TypeError: If the registered batch processor can't encode images.
        """
        if type(self).vision_cls is None:
            return None
        batching = self._batch_processor
        if not isinstance(batching, VisionBatchProcessor):
            raise TypeError(
                f"{type(batching).__qualname__} must encode images: "
                "implement VisionBatchProcessor"
            )
        return batching

    def pop_vision_metrics(self) -> VisionEncoderMetrics | None:
        """Returns and clears the batch processor's image encoder metrics."""
        batching = self._batch_processor
        if isinstance(batching, SupportsPooledVisionMetrics):
            return batching.pop_vision_metrics()
        return None

    def pop_video_metrics(self) -> VideoEncoderMetrics | None:
        """Returns and clears the batch processor's video encoder metrics.

        The pipeline reads video metrics only from the pipeline model, and
        a registered batch processor that encodes videos counts them.
        """
        batching = self._batch_processor
        if isinstance(batching, SupportsPooledVisionMetrics):
            return batching.pop_video_metrics()
        return None

    def load_graphs(self, session: InferenceSession) -> dict[str, Model]:
        """Compiles the graphs and binds the vision one to the batching.

        It keeps ``session`` for expert parallelism, which initializes while
        the config is completed.
        """
        self._session = session
        models = super().load_graphs(session)
        # A pipeline stage after the first compiles no VISION graph.
        vision_model = self.compiled_modules.get(GraphRole.VISION)
        if (
            vision_model is not None
            and (vision := self.vision_encoding) is not None
        ):
            vision.bind_vision_encoder(
                vision_model=vision_model, session=session
            )
        return models

    def _load_state_dict(self) -> dict[str, Any]:
        """Loads raw keys when config finalization precedes weight transforms.

        A checkpoint that loads through rules names each module's weights.
        """
        if getattr(
            type(self).model_config_cls, "preserve_checkpoint_keys", False
        ):
            self._checkpoint = {
                name: weight.data() for name, weight in self.weights.items()
            }
            return self._checkpoint
        if naming := _registered_weight_naming(type(self), self.adapter):
            self.weight_naming = naming
        return super()._load_state_dict()

    def _create_model_config(self, state_dict: dict[str, Any]) -> Any:
        """Finalizes the config, then completes it with the deployment.

        The expert-parallel buffers initialize in between, since they depend
        on the finalized config and the modules read their node ID. What the
        completed config records for exporters becomes
        :attr:`export_records`.

        Raises:
            ValueError: If the deployment asks for expert parallelism and the
                config doesn't support it.
        """
        config = finalize_config(
            self.arch_config,
            self.huggingface_config,
            state_dict,
            ModelOutputConfig(
                return_logits=self.return_logits,
                return_hidden_states=self.return_hidden_states,
            ),
        )
        settings = self.pipeline_config.runtime
        runtime = ModuleRuntime(
            mesh=self._device_mesh(),
            data_parallel_degree=self._data_parallel_degree,
            max_batch_input_tokens=settings.max_batch_input_tokens,
            max_batch_total_tokens=self.planned_max_batch_total_tokens,
            ep_size=settings.ep_size,
            ep_use_allreduce=settings.ep_use_allreduce,
            max_batch_size=self.max_batch_size,
        )
        if runtime.ep_size > 1 and uses_expert_parallel_exchange(config):
            if not isinstance(config, ExpertParallelDeploymentConfig):
                raise ValueError(
                    f"{type(config).__qualname__} doesn't support expert "
                    "parallelism"
                )
            runtime, self.ep_comm_initializer = initialize_ep_comm_buffers(
                config, runtime, self._session
            )
        self._runtime = runtime
        if isinstance(config, DeploymentConfig):
            config = config.apply_runtime(runtime)
            self.export_records = config.export_records(runtime)
        return config

    def _decoder_ep_inputs(self, model_config: Any) -> object | None:
        """Returns the decoder's ``ep`` input types, under expert parallelism."""
        if self._runtime.ep is None:
            return None
        assert isinstance(model_config, ExpertParallelDeploymentConfig)
        return ep_input_tree(model_config, self._runtime.ep)

    def _batch_context_length_types(
        self, model_config: Any
    ) -> tuple[TensorType, ...] | None:
        """Returns the decoder's ``batch_context_lengths`` input types.

        One ``int32`` per replica, or for a decoder that sets
        ``device_context_length_dtype``, one of that dtype per device. Replicas
        that span several devices of a ``(dp, tp)`` mesh also take one
        ``int32`` per device, each device its replica's.
        ``None`` when its config doesn't take them.
        """
        if not (
            isinstance(model_config, DeploymentConfig)
            and model_config.takes_batch_context_lengths
        ):
            return None
        replicas = self._data_parallel_degree
        dtype = getattr(
            type(self).decoder_cls, "device_context_length_dtype", None
        )
        if dtype is None:
            mesh = self._device_mesh()
            if not (DP in mesh.axis_names and TP in mesh.axis_names):
                return batch_context_length_types(replicas)
            dtype = DType.int32
        return batch_context_length_types(
            replicas,
            devices_per_replica=len(self.devices) // max(1, replicas),
            dtype=dtype,
        )

    def _decoder_vision_inputs(
        self, model_config: Any
    ) -> VisionInputs[TensorLayout] | VisionInputs[TensorType] | None:
        """Returns the decoder's ``vision`` input types, with ``VISION``."""
        declared_types = getattr(
            type(self).decoder_cls, "vision_input_types", None
        )
        if declared_types is not None:
            return declared_types(model_config, self._device_mesh())
        if self._vision_module is None:
            return None
        return decoder_vision_input_types(
            self._device_mesh(),
            self._vision_module.embedding_dtype,
            self._vision_module.embedding_dim,
        )

    def _instantiate_module(self, model_config: Any) -> Any:
        """Constructs the registered decoder on the pipeline's mesh.

        Raises:
            ValueError: If data parallelism is on and the decoder doesn't
                support it, or the replicas don't split the devices evenly
                into shares the decoder supports.
        """
        decoder_cls = type(self).decoder_cls
        _check_data_parallel(
            decoder_cls, self._data_parallel_degree, len(self.devices)
        )
        return construct_module(decoder_cls, model_config, self._device_mesh())

    def _vision_graph_spec(
        self, model_config: Any, checkpoint: Mapping[str, Any] | None = None
    ) -> GraphSpec | None:
        """Constructs the ``VISION`` module, and returns its graph.

        Only the first pipeline stage embeds tokens, so only it encodes
        images.
        """
        vision_cls = type(self).vision_cls
        stage = self.pipeline_config.model.pipeline_stage
        if vision_cls is None or (stage is not None and not stage.is_first):
            return None
        if not _vision_enabled(vision_cls, model_config):
            self._vision_module = None
            return None
        spec = _vision_spec(
            vision_cls,
            model_config,
            self._checkpoint if checkpoint is None else checkpoint,
            self.devices[0],
        )
        self._vision_module = spec.module
        return spec

    def graph_specs(self, model_config: Any) -> dict[str, GraphSpec]:
        """Returns the decoder graph, and the vision graph when registered.

        Under a pipeline stage the decoder graph is the stage's: after the
        first it also takes what the previous stage returned, and it binds
        only its own layers' weights.

        Raises:
            NotImplementedError: If it serves LoRA adapters under data
                parallelism.
            ValueError: If the config carries a pipeline stage the decoder
                doesn't build, or a stage that only traces hands the next a
                tensor on several devices.
            StageTraced: On a stage that only traces, carrying the types of
                what it hands the next stage, in the tree it returns them.
        """
        stage = self.pipeline_config.model.pipeline_stage
        decoder_cls = type(self).decoder_cls
        if stage is not None and not supports_pipeline_stages(decoder_cls):
            raise ValueError(
                f"{decoder_cls.__qualname__} doesn't build pipeline stages"
            )
        lora_manager = (
            self._lora_manager
            if isinstance(self._lora_manager, LoRAManagerV3)
            else None
        )
        if lora_manager is not None and self._data_parallel_degree > 1:
            raise NotImplementedError(
                "LoRA adapters can't be served under data parallelism yet"
            )
        processor = self.batch_processor
        if processor is not None:
            processor.config = model_config
        checkpoint = prepare_weights(self._checkpoint, model_config)
        vision_spec = self._vision_graph_spec(model_config, checkpoint)
        state_dict = role_weights(
            self._prepare_state_dict(checkpoint, model_config),
            model_config,
            GraphRole.DECODER,
        )
        self._init_distributed_runtime(model_config)
        constant_provider = getattr(decoder_cls, "constant_inputs", None)
        constants = (
            constant_provider(model_config, devices=self.devices)
            if constant_provider is not None
            else None
        )
        if constants is not None:
            assert isinstance(constants, ConstantInputs)
            assert isinstance(processor, DecoderBatchProcessor)
            processor.constant_inputs = constants
        self._state_store = bind_state_store(
            type(self).decoder_cls,
            model_config,
            self.batch_processor,
            max_batch_size=self.max_batch_size,
            devices=self.devices,
        )
        scratch = bind_decoder_scratch(
            type(self).decoder_cls,
            model_config,
            self.batch_processor,
            max_batch_size=self.max_batch_size,
            devices=self.devices,
        )
        module_default_dtype = self._module_default_dtype(
            state_dict, model_config
        )
        lora_types: list[TensorType] = []
        input_layout = getattr(decoder_cls, "input_layout", None)
        with F.lazy(), default_dtype(module_default_dtype):
            decoder = self._instantiate_module(model_config)
            if lora_manager is not None:
                # wrap() puts the adapters on the decoder's projections. The
                # graph binds the LoRA inputs itself, so the wrapper wrap()
                # returns isn't the graph's root.
                lora_manager.wrap(decoder)
                lora_types = lora_manager.symbolic_inputs(self.device_refs[0])
        decoder_inputs = decoder_input_types(
            self.kv_params,
            self.device_refs[0],
            data_parallel_degree=self._data_parallel_degree,
            vision=self._decoder_vision_inputs(model_config),
            token_positions=(
                TensorType(
                    decoder_cls.token_position_dtype,
                    ["total_seq_len"],
                    device=self.device_refs[0],
                )
                if hasattr(decoder_cls, "token_position_dtype")
                else None
            ),
            batch_context_lengths=self._batch_context_length_types(
                model_config
            ),
            ep=self._decoder_ep_inputs(model_config),
            lora=lora_types,
            constants=constants.input_types if constants is not None else None,
            state=(
                self._state_store.input_types()
                if self._state_store is not None
                else ()
            ),
            scratch=scratch.input_types if scratch is not None else (),
            input_layout=input_layout,
        )
        # A stage after the first takes the previous stage's return last.
        if boundary := stage_input_types(stage, ()):
            (decoder_inputs["stage_boundary"],) = boundary
        self._bind_graph_inputs(decoder_inputs)
        graph = _DecoderGraph(
            decoder,
            lora_manager=lora_manager,
            return_hidden_states=self.return_hidden_states,
            last_stage=stage is None or stage.is_last,
        )
        if stage is not None:
            # A stage holds only its own layers.
            declared = dict(graph.parameters)
            state_dict = {
                name: value
                for name, value in state_dict.items()
                if name in declared
            }
        spec = GraphSpec(
            graph,
            graph_name(decoder_cls),
            (decoder_inputs,),
            weights=_cast_to_parameter_dtypes(graph, state_dict),
            signal_buffer_index=_signal_buffer_index(
                decoder_inputs, input_layout
            ),
        )
        if stage is not None and not stage.is_last and stage.trace_only:
            raise_stage_traced(_returned_types(spec))
        specs: dict[str, GraphSpec] = {GraphRole.DECODER: spec}
        if vision_spec is not None:
            specs[GraphRole.VISION] = vision_spec
        if (
            isinstance(model_config, DeploymentConfig)
            and model_config.export_ep_init
            and self._runtime.ep is not None
        ):
            ep = self._runtime.ep
            init = EPInit(
                ep,
                [
                    _ep_counter_words(
                        ep.n_experts,
                        ep.n_gpus_per_node,
                        ep.use_allreduce,
                        ep.moe_dim,
                    )
                ],
            )
            specs[GraphRole.EP_INIT] = GraphSpec(
                init, "ep_init", init.input_types(), weights={}
            )
        return specs

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Runs the decoder on the prepared batch.

        The batch holds one buffer per device for a distributed input, such
        as the replicated image embeddings, which is how a captured device
        graph replays it too, so the decoder runs on the raw buffers.
        """
        assert isinstance(model_inputs, DecoderModelInputs)
        compiled = self.model
        assert isinstance(compiled, CompiledCallable)
        raw_outputs = iter(compiled.execute_raw(*model_inputs.buffers))
        outputs = _role_outputs(
            tree.unflatten(
                compiled.out_tree,
                [
                    Tensor._from_shards(
                        tuple(
                            itertools.islice(
                                raw_outputs, layout.mesh.num_devices
                            )
                        ),
                        layout.mapping,
                    )
                    for layout in compiled.out_layouts
                ],
            )
        )
        if self._state_store is not None:
            self._state_store.update(
                model_inputs.request_ids,
                [state.driver_tensor for state in outputs.state or ()],
            )
        return self._to_model_outputs(outputs)


def _vision_enabled(
    vision_cls: type[Module[..., Any]], model_config: object
) -> bool:
    """Returns whether a registered ``VISION`` module serves ``model_config``.

    It does unless its ``is_enabled`` turns it off.
    """
    enabled = getattr(vision_cls, "is_enabled", None)
    return enabled is None or bool(enabled(model_config))


def _vision_spec(
    vision_cls: type[Module[..., Any]],
    model_config: Any,
    checkpoint: Mapping[str, Any],
    device: Device,
) -> GraphSpec:
    """Returns a registered ``VISION`` module's graph, built on ``device``."""
    # A registered module takes its config, which type[Module] can't say.
    constructor: Any = vision_cls
    with F.lazy():
        vision = constructor(model_config)
        vision.to(device)
    return GraphSpec(
        vision,
        graph_name(vision_cls),
        tuple(vision.input_types()),
        weights=_cast_to_parameter_dtypes(
            vision, role_weights(checkpoint, model_config, GraphRole.VISION)
        ),
    )


def _registered_weight_naming(
    pipeline_model: type[ModuleV3DecoderPipelineModel],
    adapter: WeightsAdapter | None,
) -> dict[str, tuple[WeightEntry, ...]]:
    """Returns the entry that names each registered module's weights, by role.

    Empty unless the checkpoint loads through rules into the modules' names:
    a config that keeps the checkpoint's keys or prepares the weights
    transforms them itself. A module that takes its weights from under a
    prefix gets them with the prefix stripped.
    """
    config_cls = pipeline_model.model_config_cls
    if (
        not isinstance(adapter, WeightTransform)
        or getattr(config_cls, "preserve_checkpoint_keys", False)
        or getattr(
            config_cls, "prepare_weights", DeploymentConfig.prepare_weights
        )
        is not DeploymentConfig.prepare_weights
    ):
        return {}
    roles = [GraphRole.DECODER]
    if pipeline_model.vision_cls is not None:
        roles.append(GraphRole.VISION)
    prefixes = getattr(config_cls, "weight_prefixes", {})
    naming: dict[str, tuple[WeightEntry, ...]] = {}
    for role in roles:
        rules = list(adapter.rules)
        if prefix := prefixes.get(role):
            rules.append(WeightRenaming(f"^{re.escape(prefix)}", ""))
        naming[role] = (
            WeightEntry(CheckpointRole.TARGET, WeightTransform(rules)),
        )
    return naming


@dataclass
class RegisteredDecoderBuilder:
    """Constructs a registered ``DECODER`` for a caller that composes modules.

    A speculative pairing builds its target this way and compiles it beside
    its draft as one module, so the decoder holds no weights: it compiles
    with those :meth:`checkpoint_weights` returns. The caller passes the
    decoder its image embeddings, which the registration's ``VISION``
    module, built by :meth:`build_vision`, encodes.
    """

    pipeline_model: type[ModuleV3DecoderPipelineModel]
    """The pipeline model that serves the registration alone."""
    _weights: dict[str, Any] = field(default_factory=dict, init=False)

    def build_module(
        self, context: ModelLoadContext
    ) -> tuple[Module[..., Any], Any]:
        """Constructs the decoder from the checkpoint ``context`` names.

        The config is finalized and completed with the deployment as serving
        completes it, on the caller's KV params, and expert parallelism
        starts through ``context``, which the caller may share with other
        modules.

        Raises:
            NotImplementedError: If the decoder keeps per-request state,
                which a composed module doesn't pass it.
            ValueError: If the deployment asks for expert or data
                parallelism the decoder doesn't support.
        """
        decoder_cls = self.pipeline_model.decoder_cls
        state_requires_host = hasattr(
            decoder_cls, "state_store"
        ) and not getattr(decoder_cls, "composes_request_state", False)
        if state_requires_host:
            raise NotImplementedError(
                f"{decoder_cls.__qualname__} keeps per-request state, which "
                "a composed module doesn't pass it"
            )
        assert context.kv_params is not None
        pipeline_config = context.pipeline_config
        config_cls: Any = self.pipeline_model.model_config_cls
        state_dict = self._state_dict(context)
        config: Any = finalize_config(
            dataclasses.replace(
                config_cls.initialize(
                    pipeline_config, max_seq_len=context.max_seq_len
                ),
                kv_params=context.kv_params,
            ),
            context.huggingface_config,
            state_dict,
            ModelOutputConfig(
                return_logits=context.return_logits,
                return_hidden_states=context.return_hidden_states,
                target_layer_ids=context.target_layer_ids or None,
            ),
        )
        if isinstance(config, DeploymentConfig):
            config = config.apply_model_load_context(context)
        devices = [ref.to_device() for ref in context.kv_params.devices]
        data_parallel_degree = pipeline_config.model.data_parallel_degree
        _check_data_parallel(decoder_cls, data_parallel_degree, len(devices))
        settings = pipeline_config.runtime
        runtime = ModuleRuntime(
            mesh=_decoder_mesh(decoder_cls, devices, data_parallel_degree),
            data_parallel_degree=data_parallel_degree,
            max_batch_input_tokens=settings.max_batch_input_tokens,
            max_batch_total_tokens=context.max_batch_total_tokens,
            ep_size=settings.ep_size,
            ep_use_allreduce=settings.ep_use_allreduce,
            max_batch_size=context.memory_plan.planned_max_batch_size,
        )
        if runtime.ep_size > 1 and uses_expert_parallel_exchange(config):
            if not isinstance(config, ExpertParallelDeploymentConfig):
                raise ValueError(
                    f"{type(config).__qualname__} doesn't support expert "
                    "parallelism"
                )
            ep = config.expert_parallel_config(runtime)
            context.start_ep(ep)
            runtime = dataclasses.replace(runtime, ep=ep)
        if isinstance(config, DeploymentConfig):
            config = config.apply_runtime(runtime)
        with F.lazy(), default_dtype(config.dtype):
            module = construct_module(decoder_cls, config, runtime.mesh)
        self._weights = _cast_to_parameter_dtypes(
            module,
            role_weights(
                prepare_weights(state_dict, config), config, GraphRole.DECODER
            ),
        )
        return module, config

    def build_vision(
        self, context: ModelLoadContext, config: object, device: Device
    ) -> GraphSpec | None:
        """Constructs the registration's ``VISION`` module on ``device``.

        It is built as serving builds it, from ``config``, the one
        :meth:`build_module` returned, and the checkpoint ``context`` names.

        Returns:
            The ``VISION`` graph, or ``None`` when the registration has no
            ``VISION`` module or its ``is_enabled`` turns it off.
        """
        vision_cls = self.pipeline_model.vision_cls
        if vision_cls is None or not _vision_enabled(vision_cls, config):
            return None
        checkpoint = prepare_weights(self._state_dict(context), config)
        return _vision_spec(vision_cls, config, checkpoint, device)

    def checkpoint_weights(
        self, context: ModelLoadContext, config: object
    ) -> dict[str, Any]:
        """Returns the weights the built decoder compiles with.

        They are the checkpoint's ``DECODER`` weights, named as the module
        names them.
        """
        del context, config
        weights, self._weights = self._weights, {}
        return weights

    def weight_entries(
        self, context: ModelLoadContext, config: object
    ) -> list[WeightEntry]:
        """Returns the entry that names the built decoder's weights.

        Empty when the checkpoint doesn't load through rules into the
        decoder's names; the decoder then compiles with
        :meth:`checkpoint_weights`.
        """
        del config
        naming = _registered_weight_naming(self.pipeline_model, context.adapter)
        return list(naming.get(GraphRole.DECODER, ()))

    def vision_weight_entries(
        self, context: ModelLoadContext
    ) -> list[WeightEntry]:
        """Returns the entry that names the ``VISION`` module's weights.

        Empty when the registration has no ``VISION`` module, or the
        checkpoint doesn't load through rules into its names.
        """
        naming = _registered_weight_naming(self.pipeline_model, context.adapter)
        return list(naming.get(GraphRole.VISION, ()))

    def _state_dict(self, context: ModelLoadContext) -> dict[str, Any]:
        """Returns the checkpoint ``context`` names, as serving loads it."""
        if getattr(
            self.pipeline_model.model_config_cls,
            "preserve_checkpoint_keys",
            False,
        ):
            return {
                name: weight.data() for name, weight in context.weights.items()
            }
        return parse_state_dict_from_weights(
            context.pipeline_config, context.weights, context.adapter
        )


def bind_decoder_scratch(
    decoder_cls: type[Any],
    model_config: Any,
    batch_processor: object,
    *,
    max_batch_size: int,
    devices: Sequence[Device],
) -> DecoderScratch | None:
    """Allocates a registered decoder's workspace and gives it to the batcher.

    Args:
        decoder_cls: The registered ``DECODER`` class.
        model_config: The finalized config the decoder is built from.
        batch_processor: The pipeline model's batch processor.
        max_batch_size: The most requests a step holds.
        devices: The devices the decoder runs on.

    Returns:
        The workspace, or ``None`` when the decoder declares none.

    Raises:
        TypeError: If the decoder declares workspace and the batch processor
            isn't a :class:`DecoderBatchProcessor`, which passes it.
    """
    scratch = build_decoder_scratch(
        decoder_cls,
        model_config,
        max_batch_size=max_batch_size,
        devices=devices,
    )
    if scratch is None:
        return None
    if not isinstance(batch_processor, DecoderBatchProcessor):
        raise TypeError(
            f"{decoder_cls.__qualname__} declares workspace, which only a "
            f"DecoderBatchProcessor passes, not "
            f"{type(batch_processor).__qualname__}"
        )
    batch_processor.scratch = scratch
    return scratch


def bind_state_store(
    decoder_cls: type[Any],
    model_config: Any,
    batch_processor: object,
    *,
    max_batch_size: int,
    devices: Sequence[Device],
) -> RequestStateStore | None:
    """Builds a registered decoder's state store and gives it to the batcher.

    Args:
        decoder_cls: The registered ``DECODER`` class.
        model_config: The finalized config the decoder is built from.
        batch_processor: The pipeline model's batch processor.
        max_batch_size: The most requests a step holds.
        devices: The devices the decoder runs on.

    Returns:
        The store, or ``None`` when the decoder declares no state.

    Raises:
        TypeError: If the decoder declares state and the batch processor
            isn't a :class:`DecoderBatchProcessor`, which feeds it.
    """
    store = build_state_store(
        decoder_cls,
        model_config,
        max_batch_size=max_batch_size,
        devices=devices,
    )
    if store is None:
        return None
    if not isinstance(batch_processor, DecoderBatchProcessor):
        raise TypeError(
            f"{decoder_cls.__qualname__} declares per-request state, which "
            f"only a DecoderBatchProcessor feeds, not "
            f"{type(batch_processor).__qualname__}"
        )
    batch_processor.state_store = store
    return store


_PIPELINE_MODELS: dict[
    tuple[
        type[ArchConfig], tuple[tuple[str, type[Module[..., Any]]], ...], Any
    ],
    type[ModuleV3DecoderPipelineModel],
] = {}


def _check_vision_registration(
    decoder_cls: type[Module[..., Any]],
    vision_cls: type[Module[..., Any]] | None,
    batching: type[Any] | None,
) -> None:
    """Checks that a registration's image path is complete, or absent.

    Raises:
        TypeError: If a ``VISION`` module lacks ``input_types()``, its
            decoder doesn't consume image embeddings, or its batch processor
            doesn't implement :class:`VisionBatchProcessor`; or if the
            decoder consumes image embeddings with no ``VISION`` module.
    """
    if vision_cls is None:
        if consumes_vision(decoder_cls):
            raise TypeError(
                f"{decoder_cls.__qualname__} consumes image embeddings, so "
                "its registration needs a VISION module"
            )
        return
    if not callable(getattr(vision_cls, "input_types", None)):
        raise TypeError(
            f"the VISION module {vision_cls.__qualname__} needs input_types()"
        )
    if not consumes_vision(decoder_cls):
        raise TypeError(
            f"{decoder_cls.__qualname__} doesn't consume image embeddings, "
            "so it can't register beside a VISION module"
        )
    if batching is None or not issubclass(batching, VisionBatchProcessor):
        raise TypeError(
            "a registration with a VISION module needs a batch processor "
            "that implements VisionBatchProcessor"
        )


def module_pipeline_model(
    config: type[ArchConfig],
    modules: tuple[tuple[str, type[Module[..., Any]]], ...],
    batching: type[Any] | None,
) -> type[ModuleV3DecoderPipelineModel]:
    """Returns the pipeline model that serves a module registration.

    One registration always returns the same class.

    Args:
        config: The registered config class.
        modules: The registered module class of each role, as items.
        batching: The registered batch processor, or ``None`` for
            :class:`DecoderBatchProcessor`.

    Raises:
        ValueError: If the registration names a framework-owned role, has no
            ``DECODER``.
        NotImplementedError: If it registers a role other than ``DECODER``
            and ``VISION``.
        TypeError: If a registered class isn't a :class:`~max.nn.Module`, or
            the registration's ``VISION`` module, decoder and batch processor
            don't fit together.
    """
    registration = (config, modules, batching)
    if (cached := _PIPELINE_MODELS.get(registration)) is not None:
        return cached
    by_role = dict(modules)
    for role, module_cls in by_role.items():
        if role in _FRAMEWORK_ROLES:
            raise ValueError(f"the framework builds the {role} role")
        if not isinstance(module_cls, type) or not issubclass(
            module_cls, Module
        ):
            raise TypeError(f"the {role} module must be a max.nn.Module class")
    if GraphRole.DECODER not in by_role:
        raise ValueError("a module registration needs a DECODER module")
    if unsupported := sorted(set(by_role) - _SERVED_ROLES):
        raise NotImplementedError(
            f"registered modules can't fill {unsupported} yet"
        )
    decoder_cls = by_role[GraphRole.DECODER]
    _check_vision_registration(
        decoder_cls, by_role.get(GraphRole.VISION), batching
    )
    name = f"{decoder_cls.__name__}PipelineModel"
    # Exporters read how tokens reach the expert dispatch off the pipeline
    # model class, to pick the matching expert-parallel backend.
    layout = (
        config.ep_dispatch_token_layout
        if issubclass(config, ExpertParallelDeploymentConfig)
        else None
    )
    exporter_attrs = (
        {} if layout is None else {"ep_dispatch_token_layout": layout}
    )
    # Only a decoder with state gets the release hooks, which serving
    # detects by their presence.
    bases: tuple[type, ...] = (ModuleV3DecoderPipelineModel,)
    if hasattr(decoder_cls, "state_store"):
        bases = (RequestStateMixin, *bases)
    # Exporters build the stages of a pipelined model only.
    if supports_pipeline_stages(decoder_cls):
        bases = (*bases, PipelinedModel)
    model = _ModulePipelineModelMeta(
        name,
        bases,
        {
            "__module__": __name__,
            "__qualname__": name,
            "_registration": registration,
            "decoder_cls": decoder_cls,
            "vision_cls": by_role.get(GraphRole.VISION),
            **exporter_attrs,
            "model_config_cls": config,
            "batch_processor_cls": batching or DecoderBatchProcessor,
            "lora_modulev3": bool(lora_targets(decoder_cls)),
            "lora_targets": lora_targets(decoder_cls),
            "checkpoints_recurrent_state": bool(
                getattr(decoder_cls, "checkpoints_recurrent_state", False)
            ),
        },
    )
    assert issubclass(model, ModuleV3DecoderPipelineModel)
    _PIPELINE_MODELS[registration] = model
    return model
