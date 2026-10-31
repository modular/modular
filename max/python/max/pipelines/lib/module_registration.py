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
- ``lora_targets``, the projections it serves LoRA adapters on
  (:func:`lora_targets`);
- ``supports_data_parallel``, when it runs one replica per device
  (:func:`supports_data_parallel`);
- ``consumes_vision``, when it reads
  :attr:`~max.pipelines.lib.role_contracts.DecoderInputs.vision`
  (:func:`consumes_vision`).

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
from abc import ABCMeta
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, cast, runtime_checkable

import numpy as np
from max import functional as F
from max import tree
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, TensorType
from max.nn import Module
from max.nn.common_layers.mesh_axis import DP, TP
from max.nn.legacy.comm.ep import EPCommInitializer
from max.nn.legacy.graph_groups import EPPayload
from max.nn.legacy.kv_cache import KVCacheInputs, KVCacheParamInterface
from max.pipelines.context import TextContext
from max.pipelines.lora import LoRAManagerV3, LoRATargetModule
from max.tensor import Tensor, default_device, default_dtype
from max.tensor.compilation import CompiledCallable
from max.tensor.sharding import (
    DeviceMapping,
    DeviceMesh,
    Replicated,
    TensorLayout,
)
from transformers import AutoConfig

from .graph_specs import GraphRole, GraphSpec
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
    ModelOutputs,
    ModuleV3PipelineModelWithKVCache,
    _cast_to_parameter_dtypes,
)
from .log_probabilities import LogProbabilitiesMixin
from .registration_runtime import (
    BatchContextLengths,
    DeploymentConfig,
    ExpertParallelDeploymentConfig,
    ModuleRuntime,
    batch_context_length_types,
    ep_input_tree,
    initialize_ep_comm_buffers,
    role_weights,
)
from .role_contracts import DecoderInputs, DecoderOutputs, VisionInputs
from .utils import compute_data_parallel_splits
from .vision_encoder_cache import SupportsVisionEncoding

__all__ = [
    "DecoderBatchProcessor",
    "DecoderModelInputs",
    "ModuleV3DecoderPipelineModel",
    "VisionBatchProcessor",
    "decoder_input_types",
    "decoder_vision_input_types",
    "graph_name",
    "module_pipeline_model",
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


def consumes_vision(module_cls: type[Module[..., Any]]) -> bool:
    """Returns whether a registered decoder reads image embeddings.

    A decoder that sets ``consumes_vision`` reads
    :attr:`~max.pipelines.lib.role_contracts.DecoderInputs.vision`, so it
    registers beside a ``VISION`` module.
    """
    return bool(getattr(module_cls, "consumes_vision", False))


def decoder_input_types(
    kv_params: KVCacheParamInterface,
    device_ref: DeviceRef,
    *,
    data_parallel_degree: int = 1,
    vision: VisionInputs[TensorLayout] | None = None,
    batch_context_lengths: Sequence[TensorType] | None = None,
    ep: EPPayload | None = None,
    lora: Sequence[TensorType] = (),
) -> OrderedDict[str, object]:
    """Returns a ``DECODER`` graph's input types, keyed by field.

    The keys are the :class:`~max.pipelines.lib.role_contracts.DecoderInputs`
    fields the deployment sets, in its order, and the leaves are the graph's
    inputs.

    Args:
        kv_params: The KV cache parameters that supply the ``kv`` subtree.
        device_ref: The device that holds the token and row-offset tensors.
        data_parallel_degree: The number of replicas. Above one, the graph
            also takes ``host_input_row_offsets`` and
            ``data_parallel_splits``.
        vision: The image embeddings' input types, when the graph is built
            with a ``VISION`` module. A graph built with them takes them on
            every batch, empty when the batch has no images.
        batch_context_lengths: Each replica's context-length input type,
            when the decoder reads them.
        ep: The expert-parallel input types, when serving with expert
            parallelism.
        lora: The LoRA input types, when serving LoRA adapters.
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
    if vision is not None:
        types["vision"] = vision
    if batch_context_lengths is not None:
        types["batch_context_lengths"] = tuple(batch_context_lengths)
    if ep is not None:
        types["ep"] = ep
    if lora:
        types["lora"] = tuple(lora)
    return OrderedDict(
        (name, types[name]) for name in _FIELD_ORDER if name in types
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

    host_input_row_offsets: Buffer | None = None
    """A CPU copy of :attr:`input_row_offsets`, under data parallelism."""

    data_parallel_splits: Buffer | None = None
    """Each replica's first sequence, under data parallelism."""

    batch_context_lengths: list[Buffer] | None = None
    """Each replica's page-aligned KV context length, when the graph takes
    them."""

    ep_inputs: tuple[Buffer, ...] = ()
    """The expert-parallel communication buffers, under expert parallelism."""

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        """The step's buffers, in the graph's input order."""
        data_parallel = (
            ()
            if self.data_parallel_splits is None
            else (self.data_parallel_splits,)
        )
        host_input_row_offsets = (
            ()
            if self.host_input_row_offsets is None
            else (self.host_input_row_offsets,)
        )
        return (
            self.tokens,
            *self.vision_embeddings,
            *self.vision_scatter_indices,
            self.input_row_offsets,
            *host_input_row_offsets,
            self.return_n_logits,
            *data_parallel,
            *(
                tree.leaves(self.kv_cache_inputs)
                if self.kv_cache_inputs is not None
                else ()
            ),
            *(self.batch_context_lengths or ()),
            *self.ep_inputs,
            *self.lora_buffers,
        )


class DecoderBatchProcessor(BatchProcessor[TextContext, DecoderModelInputs]):
    """Batches ragged text for a registered ``DECODER``.

    Under data parallelism it concatenates every replica's batch, in replica
    order, and records where each replica's sequences start. It also fills
    the optional inputs the bound graph takes: each replica's context length
    and the expert-parallel buffers.
    """

    _batch_context_lengths: BatchContextLengths | None = None
    _ep_comm_initializer: EPCommInitializer | None = None

    def bind_input_tree(self, input_tree: OrderedDict[str, object]) -> None:
        """Binds the graph's inputs, and sizes the ones this fills."""
        super().bind_input_tree(input_tree)
        self._batch_context_lengths = (
            BatchContextLengths(
                self.runtime.data_parallel_degree,
                self.runtime.pipeline_config.model.kv_cache.kv_cache_page_size,
            )
            if "batch_context_lengths" in input_tree
            else None
        )

    def bind_ep_comm_initializer(
        self, initializer: EPCommInitializer | None
    ) -> None:
        """Binds the expert-parallel buffers each batch passes the graph."""
        self._ep_comm_initializer = initializer

    def get_symbolic_inputs(
        self,
        *,
        kv_params: KVCacheParamInterface,
        device_refs: list[DeviceRef],
    ) -> list[Any]:
        """Returns the graph's input types without LoRA's, in order."""
        return tree.leaves(
            decoder_input_types(
                kv_params,
                device_refs[0],
                data_parallel_degree=self.runtime.data_parallel_degree,
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


class _DecoderGraph(Module[..., DecoderOutputs]):
    """Calls a decoder with :class:`DecoderInputs` built from its graph inputs.

    The graph takes the leaves of :func:`decoder_input_types` in order, and
    binds the LoRA inputs into the decoder's LoRA layers before calling it.
    It is name-transparent to the decoder, so the weight names are the
    decoder's own.
    """

    def __init__(
        self,
        decoder: Module[[DecoderInputs], DecoderOutputs],
        input_types: Mapping[str, object],
        lora_manager: LoRAManagerV3 | None = None,
    ) -> None:
        super().__init__()
        # The LoRA weight adapter nests every checkpoint key under `model.`.
        decoder.name_transparent = lora_manager is None
        self.model = decoder
        # A plain dict flattens in sorted key order; the graph takes its
        # inputs in field order.
        _, self.input_tree = tree.flatten(OrderedDict(input_types))
        self.lora_manager = lora_manager

    def forward(self, *inputs: Tensor) -> DecoderOutputs:
        """Groups the graph's inputs into fields and runs the decoder."""
        fields = DecoderInputs(**tree.unflatten(self.input_tree, inputs))
        if fields.lora is not None:
            assert self.lora_manager is not None
            self.lora_manager.bind_inputs(self.model, fields.lora)
        return self.model(fields)


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

    ep_comm_initializer: EPCommInitializer | None = None
    """The expert-parallel buffers' owner, under expert parallelism."""

    _session: InferenceSession
    _runtime: ModuleRuntime
    _vision_module: Any = None

    batch_processor_cls: ClassVar[type[BatchProcessor[Any, Any]] | None] = (
        DecoderBatchProcessor
    )

    @property
    def _data_parallel_degree(self) -> int:
        return self.pipeline_config.model.data_parallel_degree

    def _device_mesh(self) -> DeviceMesh:
        """Returns the mesh the modules are constructed on."""
        axis = DP if self._data_parallel_degree > 1 else TP
        return DeviceMesh(tuple(self.devices), (len(self.devices),), (axis,))

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

    def load_graphs(self, session: InferenceSession) -> dict[str, Model]:
        """Compiles the graphs and binds the vision one to the batching.

        It keeps ``session`` for expert parallelism, which initializes while
        the config is completed.
        """
        self._session = session
        models = super().load_graphs(session)
        if (vision := self.vision_encoding) is not None:
            vision.bind_vision_encoder(
                vision_model=self.compiled_modules[GraphRole.VISION],
                session=session,
            )
        return models

    def _create_model_config(self, state_dict: dict[str, Any]) -> Any:
        """Finalizes the config, then completes it with the deployment.

        The expert-parallel buffers initialize in between, since they depend
        on the finalized config and the modules read their node ID.

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
        )
        if runtime.ep_size > 1:
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
            return config.apply_runtime(runtime)
        return config

    def _decoder_vision_inputs(self) -> VisionInputs[TensorLayout] | None:
        """Returns the decoder's ``vision`` input types, with ``VISION``."""
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
                support it, or doesn't give each device its own replica.
        """
        decoder_cls = type(self).decoder_cls
        data_parallel_degree = self._data_parallel_degree
        if data_parallel_degree > 1:
            if not supports_data_parallel(decoder_cls):
                raise ValueError(
                    f"{decoder_cls.__qualname__} doesn't support data "
                    "parallelism"
                )
            if data_parallel_degree != len(self.devices):
                raise ValueError(
                    f"data parallelism needs one replica per device, got "
                    f"{data_parallel_degree} replicas over "
                    f"{len(self.devices)} devices"
                )
        mesh = self._device_mesh()
        # A registered module takes its config, which type[Module] can't say.
        constructor: Any = decoder_cls
        with default_device(mesh):
            return constructor(model_config)

    def _vision_graph_spec(self, model_config: Any) -> GraphSpec | None:
        """Constructs the ``VISION`` module, and returns its graph."""
        vision_cls = type(self).vision_cls
        if vision_cls is None:
            return None
        # A registered module takes its config, which type[Module] can't say.
        constructor: Any = vision_cls
        with F.lazy():
            vision = constructor(model_config)
            vision.to(self.devices[0])
        self._vision_module = vision
        return GraphSpec(
            vision,
            graph_name(vision_cls),
            tuple(vision.input_types()),
            weights=_cast_to_parameter_dtypes(
                vision,
                role_weights(self._checkpoint, model_config, GraphRole.VISION),
            ),
        )

    def graph_specs(self, model_config: Any) -> dict[str, GraphSpec]:
        """Returns the decoder graph, and the vision graph when registered.

        Raises:
            NotImplementedError: If it serves LoRA adapters under data
                parallelism.
        """
        lora_manager = (
            self._lora_manager
            if isinstance(self._lora_manager, LoRAManagerV3)
            else None
        )
        if lora_manager is not None and self._data_parallel_degree > 1:
            raise NotImplementedError(
                "LoRA adapters can't be served under data parallelism yet"
            )
        vision_spec = self._vision_graph_spec(model_config)
        state_dict = role_weights(
            self._prepare_state_dict(self._checkpoint, model_config),
            model_config,
            GraphRole.DECODER,
        )
        self._init_distributed_runtime(model_config)
        module_default_dtype = self._module_default_dtype(
            state_dict, model_config
        )
        lora_types: list[TensorType] = []
        with F.lazy(), default_dtype(module_default_dtype):
            decoder = self._instantiate_module(model_config)
            if lora_manager is not None:
                # wrap() puts the adapters on the decoder's projections. The
                # graph binds the LoRA inputs itself, so the wrapper wrap()
                # returns isn't the graph's root.
                lora_manager.wrap(decoder)
                lora_types = lora_manager.symbolic_inputs(self.device_refs[0])
        input_types = self._bind_graph_inputs(
            decoder_input_types(
                self.kv_params,
                self.device_refs[0],
                data_parallel_degree=self._data_parallel_degree,
                vision=self._decoder_vision_inputs(),
                batch_context_lengths=(
                    batch_context_length_types(self._data_parallel_degree)
                    if isinstance(model_config, DeploymentConfig)
                    and model_config.takes_batch_context_lengths
                    else None
                ),
                ep=(
                    None
                    if self._runtime.ep is None
                    else ep_input_tree(self._runtime.ep)
                ),
                lora=lora_types,
            )
        )
        graph = _DecoderGraph(decoder, input_types, lora_manager)
        specs: dict[str, GraphSpec] = {
            GraphRole.DECODER: GraphSpec(
                graph,
                graph_name(type(self).decoder_cls),
                tuple(tree.leaves(input_types)),
                weights=_cast_to_parameter_dtypes(graph, state_dict),
            )
        }
        if vision_spec is not None:
            specs[GraphRole.VISION] = vision_spec
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
        outputs = iter(compiled.execute_raw(*model_inputs.buffers))
        return self._to_model_outputs(
            tree.unflatten(
                compiled.out_tree,
                [
                    Tensor._from_shards(
                        tuple(
                            itertools.islice(outputs, layout.mesh.num_devices)
                        ),
                        layout.mapping,
                    )
                    for layout in compiled.out_layouts
                ],
            )
        )


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
    model = _ModulePipelineModelMeta(
        name,
        (ModuleV3DecoderPipelineModel,),
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
        },
    )
    assert issubclass(model, ModuleV3DecoderPipelineModel)
    _PIPELINE_MODELS[registration] = model
    return model
