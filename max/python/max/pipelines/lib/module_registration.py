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
"""

from __future__ import annotations

import copyreg
from abc import ABCMeta
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from max import tree
from max.driver import Buffer
from max.nn import Module
from max.nn.legacy.kv_cache import KVCacheInputs
from max.pipelines.context import TextContext
from max.tensor import Tensor, default_device
from max.tensor.sharding import DeviceMesh
from transformers import AutoConfig

from .graph_specs import GraphRole
from .interfaces.arch_config import ArchConfig, ModelOutputConfig
from .interfaces.batch_processor import (
    BatchProcessor,
    ModuleV3SingleReplicaBatchProcessor,
    ragged_batch_inputs,
)
from .interfaces.pipeline_model import (
    ModelInputs,
    ModelOutputs,
    ModuleV3PipelineModelWithKVCache,
)
from .log_probabilities import LogProbabilitiesMixin
from .role_contracts import DecoderInputs, DecoderOutputs

__all__ = [
    "DecoderBatchProcessor",
    "DecoderModelInputs",
    "ModuleV3DecoderPipelineModel",
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


def graph_name(module_cls: type[Module[..., Any]]) -> str:
    """Returns the name a registered module's graph compiles under.

    The name is part of the precompiled MEF key, so a module whose graph
    predates registration sets ``graph_name`` to keep it.
    """
    return getattr(module_cls, "graph_name", module_cls.__qualname__)


@dataclass(kw_only=True)
class DecoderModelInputs(ModelInputs):
    """The prepared inputs of a registered ``DECODER``."""

    tokens: Buffer
    """The batch's ragged tokens."""

    input_row_offsets: Buffer
    """Each sequence's start in :attr:`tokens`."""

    return_n_logits: Buffer
    """How many logits per sequence to return."""


class DecoderBatchProcessor(
    ModuleV3SingleReplicaBatchProcessor[TextContext, DecoderModelInputs]
):
    """Batches ragged text for a registered ``DECODER`` on one replica."""

    def _make_inputs(
        self,
        *,
        tokens: Buffer,
        input_row_offsets: Buffer,
        return_n_logits: Buffer,
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer],
    ) -> DecoderModelInputs:
        return DecoderModelInputs(
            tokens=tokens,
            input_row_offsets=input_row_offsets,
            return_n_logits=return_n_logits,
            kv_cache_inputs=kv_cache_inputs,
        )


class _DecoderGraph(Module[..., DecoderOutputs]):
    """Calls a decoder with :class:`DecoderInputs` built from its graph inputs.

    The graph takes the fields in declaration order, with the KV tree
    flattened in place. It is name-transparent to the decoder, so the weight
    names are the decoder's own.
    """

    def __init__(
        self,
        decoder: Module[[DecoderInputs], DecoderOutputs],
        kv_tree: tree.TreeDef,
    ) -> None:
        super().__init__()
        decoder.name_transparent = True
        self.decoder = decoder
        self.kv_tree = kv_tree

    def forward(
        self,
        tokens: Tensor,
        input_row_offsets: Tensor,
        return_n_logits: Tensor,
        *kv: Tensor,
    ) -> DecoderOutputs:
        """Groups the graph's inputs into fields and runs the decoder."""
        return self.decoder(
            DecoderInputs(
                tokens=tokens,
                input_row_offsets=input_row_offsets,
                return_n_logits=return_n_logits,
                kv=tree.unflatten(self.kv_tree, kv),
            )
        )


def finalize_config(
    config: ArchConfig,
    huggingface_config: AutoConfig,
    state_dict: Mapping[str, Any],
    outputs: ModelOutputConfig,
) -> ArchConfig:
    """Returns ``config`` finalized against the checkpoint.

    A config without ``finalize`` needs nothing from the checkpoint.

    Raises:
        TypeError: If ``finalize`` returns another type of config.
    """
    finalize = getattr(config, "finalize", None)
    if finalize is None:
        return config
    finalized = finalize(huggingface_config, state_dict, outputs)
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
    decoder is constructed as ``decoder_cls(config)`` on a tensor-parallel
    mesh over the pipeline's devices.
    """

    decoder_cls: ClassVar[type[Module[..., Any]]]
    """The registered ``DECODER`` class."""

    batch_processor_cls: ClassVar[type[BatchProcessor[Any, Any]] | None] = (
        DecoderBatchProcessor
    )

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
        decoder_cls = type(self).decoder_cls
        mesh = DeviceMesh(tuple(self.devices), (len(self.devices),), ("tp",))
        # A registered module takes its config, which type[Module] can't say.
        constructor: Any = decoder_cls
        with default_device(mesh):
            decoder = constructor(model_config)
        _, kv_tree = tree.flatten(self.kv_params.get_symbolic_inputs())
        return _DecoderGraph(decoder, kv_tree)

    def _graph_name(self, nn_model: Module[..., Any]) -> str:
        return graph_name(type(self).decoder_cls)

    def _get_compile_input_types(self, model_config: Any) -> tuple[Any, ...]:
        del model_config
        batch = ragged_batch_inputs(self.device_refs[0])
        return (
            batch["tokens"],
            batch["row_offsets"],
            batch["return_n_logits"],
            *tree.leaves(self.kv_params.get_symbolic_inputs()),
        )

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Runs the decoder on the prepared batch."""
        assert isinstance(model_inputs, DecoderModelInputs)
        buffers = (
            model_inputs.tokens,
            model_inputs.input_row_offsets,
            model_inputs.return_n_logits,
            *tree.leaves(model_inputs.kv_cache_inputs),
        )
        return self._to_model_outputs(self.model(*buffers))


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
        },
    )
    assert issubclass(model, ModuleV3DecoderPipelineModel)
    _PIPELINE_MODELS[registration] = model
    return model
