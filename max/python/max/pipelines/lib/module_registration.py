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
  (:func:`supports_data_parallel`).
"""

from __future__ import annotations

import copy
import copyreg
from abc import ABCMeta
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, cast

import numpy as np
from max import functional as F
from max import tree
from max.driver import Buffer
from max.graph import DeviceRef
from max.nn import Module
from max.nn.common_layers.mesh_axis import DP, TP
from max.nn.legacy.kv_cache import KVCacheInputs, KVCacheParamInterface
from max.pipelines.context import TextContext
from max.pipelines.lora import LoRAManagerV3, LoRATargetModule
from max.tensor import Tensor, default_device, default_dtype
from max.tensor.sharding import DeviceMesh
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
from .role_contracts import DecoderInputs, DecoderOutputs
from .utils import compute_data_parallel_splits

__all__ = [
    "DecoderBatchProcessor",
    "DecoderModelInputs",
    "ModuleV3DecoderPipelineModel",
    "decoder_input_types",
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

# The ragged batch inputs each DecoderInputs field takes its type from.
_BATCH_FIELDS = {
    "tokens": "tokens",
    "input_row_offsets": "row_offsets",
    "host_input_row_offsets": "host_row_offsets",
    "return_n_logits": "return_n_logits",
    "data_parallel_splits": "data_parallel_splits",
}


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


def decoder_input_types(
    kv_params: KVCacheParamInterface,
    device_ref: DeviceRef,
    *,
    data_parallel_degree: int = 1,
    lora: Sequence[Any] = (),
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
    if lora:
        types["lora"] = tuple(lora)
    return types


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
            self.input_row_offsets,
            *host_input_row_offsets,
            self.return_n_logits,
            *data_parallel,
            *(
                tree.leaves(self.kv_cache_inputs)
                if self.kv_cache_inputs is not None
                else ()
            ),
            *self.lora_buffers,
        )


class DecoderBatchProcessor(BatchProcessor[TextContext, DecoderModelInputs]):
    """Batches ragged text for a registered ``DECODER``.

    Under data parallelism it concatenates every replica's batch, in replica
    order, and records where each replica's sequences start.
    """

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
        tokens, input_row_offsets = build_single_replica_ragged_token_arrays(
            cast(Sequence[RaggableContext], context_batch)
        )
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
    """Serves a registered ``DECODER`` module.

    :func:`module_pipeline_model` subclasses it for each registration. The
    decoder is constructed as ``decoder_cls(config)`` on a mesh over the
    pipeline's devices: a ``dp`` axis under data parallelism, else a ``tp``
    axis.
    """

    decoder_cls: ClassVar[type[Module[..., Any]]]
    """The registered ``DECODER`` class."""

    batch_processor_cls: ClassVar[type[BatchProcessor[Any, Any]] | None] = (
        DecoderBatchProcessor
    )

    @property
    def _data_parallel_degree(self) -> int:
        return self.pipeline_config.model.data_parallel_degree

    def _create_model_config(self, state_dict: dict[str, Any]) -> Any:
        return finalize_config(
            self.arch_config,
            self.huggingface_config,
            state_dict,
            ModelOutputConfig(
                return_logits=self.return_logits,
                return_hidden_states=self.return_hidden_states,
            ),
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
        axis = DP if data_parallel_degree > 1 else TP
        mesh = DeviceMesh(tuple(self.devices), (len(self.devices),), (axis,))
        # A registered module takes its config, which type[Module] can't say.
        constructor: Any = decoder_cls
        with default_device(mesh):
            return constructor(model_config)

    def graph_specs(self, model_config: Any) -> dict[str, GraphSpec]:
        """Returns the decoder graph, built from the checkpoint weights.

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
        state_dict = self._prepare_state_dict(self._checkpoint, model_config)
        self._init_distributed_runtime(model_config)
        module_default_dtype = self._module_default_dtype(
            state_dict, model_config
        )
        lora_types: list[Any] = []
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
                lora=lora_types,
            )
        )
        graph = _DecoderGraph(decoder, input_types, lora_manager)
        return {
            GraphRole.DECODER: GraphSpec(
                graph,
                graph_name(type(self).decoder_cls),
                tuple(tree.leaves(input_types)),
                weights=_cast_to_parameter_dtypes(graph, state_dict),
            )
        }

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Runs the decoder on the prepared batch."""
        assert isinstance(model_inputs, DecoderModelInputs)
        return self._to_model_outputs(self.model(*model_inputs.buffers))


_PIPELINE_MODELS: dict[
    tuple[
        type[ArchConfig], tuple[tuple[str, type[Module[..., Any]]], ...], Any
    ],
    type[ModuleV3DecoderPipelineModel],
] = {}


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
        NotImplementedError: If it registers a role other than ``DECODER``,
            which the framework doesn't serve yet.
        TypeError: If a registered class isn't a :class:`~max.nn.Module`.
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
    if unsupported := sorted(set(by_role) - {GraphRole.DECODER}):
        raise NotImplementedError(
            f"registered modules can't fill {unsupported} yet"
        )
    decoder_cls = by_role[GraphRole.DECODER]
    name = f"{decoder_cls.__name__}PipelineModel"
    model = _ModulePipelineModelMeta(
        name,
        (ModuleV3DecoderPipelineModel,),
        {
            "__module__": __name__,
            "__qualname__": name,
            "_registration": registration,
            "decoder_cls": decoder_cls,
            "model_config_cls": config,
            "batch_processor_cls": batching or DecoderBatchProcessor,
            "lora_modulev3": bool(lora_targets(decoder_cls)),
            "lora_targets": lora_targets(decoder_cls),
        },
    )
    assert issubclass(model, ModuleV3DecoderPipelineModel)
    _PIPELINE_MODELS[registration] = model
    return model
