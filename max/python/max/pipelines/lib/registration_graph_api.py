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
"""Serves architectures that register a graph-API module by role.

A graph-API (:class:`~max.nn.legacy.Module`) ``DECODER`` registers like a
ModuleV3 one: :func:`registered_pipeline_model` builds the pipeline model,
which finalizes the config, constructs the decoder as ``decoder_cls(config)``,
loads the checkpoint with :meth:`~max.nn.legacy.Module.load_state_dict`,
builds the graph from :func:`graph_api_decoder_input_types`, and batches and
executes it. The decoder's ``__call__`` takes
:class:`~max.pipelines.lib.role_contracts.DecoderInputs` of graph values and
returns :class:`~max.pipelines.lib.role_contracts.DecoderOutputs`.

A registered graph-API decoder class can also set:

- ``graph_name``, the name its graph compiles under (:func:`graph_name`);
- ``always_signal_buffers``, when it takes allreduce signal buffers even on
  one device;
- ``strict_state_dict_loading``, ``False`` when the checkpoint may hold
  weights the decoder doesn't read;
- ``override_quantization_encoding``, when the checkpoint's quantization
  encoding overrides the one a weight declares.
"""

from __future__ import annotations

import copyreg
from abc import ABCMeta
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from max import tree
from max.driver import Buffer, Device
from max.engine import InferenceSession, Model
from max.graph import (
    BufferType,
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    Type,
    Value,
)
from max.graph import Module as GraphModule
from max.graph.weights import Weights, WeightsAdapter
from max.nn.legacy.graph_groups import DecoderOutputs as GroupedDecoderOutputs
from max.nn.legacy.kv_cache import KVCacheInputs, KVCacheParamInterface
from max.nn.legacy.layer import Module as GraphApiModule
from max.nn.legacy.transformer import (
    ReturnHiddenStates,
    ReturnLogits,
    logits_outputs,
)
from max.pipelines.context import TextContext
from max.pipelines.kv_cache.config import KVCacheConfig
from max.pipelines.modeling.types import PipelineTask
from transformers import AutoConfig

from .graph_specs import GraphRole, GraphSpec
from .interfaces.arch_config import ArchConfig, ModelOutputConfig
from .interfaces.batch_processor import (
    BatchProcessor,
    ragged_batch_inputs,
    signal_inputs,
)
from .interfaces.pipeline_model import (
    AlwaysSignalBuffersMixin,
    ModelInputs,
    ModelOutputs,
    PipelineModelWithKVCache,
)
from .log_probabilities import LogProbabilitiesMixin
from .module_registration import (
    DecoderBatchProcessor,
    DecoderModelInputs,
    finalize_config,
    module_pipeline_model,
)
from .role_contracts import DecoderInputs, DecoderOutputs

if TYPE_CHECKING:
    from .config import PipelineConfig

    # TODO(MXF-517): Remove the interfaces/memory_estimation import cycle.
    from .memory_estimation import MemoryPlan

__all__ = [
    "GraphApiDecoderBatchProcessor",
    "GraphApiDecoderModelInputs",
    "GraphApiDecoderPipelineModel",
    "decoder_outputs",
    "graph_api_decoder_input_types",
    "graph_api_pipeline_model",
    "is_graph_api_module",
    "registered_pipeline_model",
]

# The DecoderInputs field each input of the ``batch`` group fills.
_BATCH_FIELDS = {
    "tokens": "tokens",
    "row_offsets": "input_row_offsets",
    "return_n_logits": "return_n_logits",
}


def is_graph_api_module(module_cls: object) -> bool:
    """Returns whether ``module_cls`` is a graph-API module class."""
    return isinstance(module_cls, type) and issubclass(
        module_cls, GraphApiModule
    )


def graph_name(module_cls: type[GraphApiModule]) -> str:
    """Returns the name a registered graph-API module's graph compiles under.

    The name is part of the precompiled MEF key, so a module whose graph
    predates registration sets ``graph_name`` to keep it.
    """
    return getattr(module_cls, "graph_name", module_cls.__qualname__)


def _takes_signals(
    module_cls: type[GraphApiModule], devices: Sequence[object]
) -> bool:
    return len(devices) > 1 or bool(
        getattr(module_cls, "always_signal_buffers", False)
    )


_DecoderInputTypes = (
    tree.Tree[Type[Any]] | KVCacheInputs[TensorType, BufferType]
)


def graph_api_decoder_input_types(
    kv_params: KVCacheParamInterface,
    device_refs: Sequence[DeviceRef],
    *,
    signals: bool = False,
) -> OrderedDict[str, _DecoderInputTypes]:
    """Returns a graph-API ``DECODER`` graph's input types.

    A serving engine places a named decoder input by the kind of the group
    that holds it, so the inputs sit in kinded groups: ``batch`` (tokens, row
    offsets and the logit count), then ``signals``, then ``kv``. Their leaves
    are the graph's inputs, in the canonical order of
    :class:`~max.pipelines.lib.role_contracts.DecoderInputs`. The row offsets
    are one input on the first device, as for a ModuleV3 decoder; a decoder
    that needs them on every device copies them inside its graph.

    Args:
        kv_params: The KV cache parameters that supply the ``kv`` subtree.
        device_refs: The devices the decoder runs on; the first holds the
            tokens.
        signals: Whether the graph takes one allreduce signal buffer per
            device.
    """
    types: OrderedDict[str, _DecoderInputTypes] = OrderedDict(
        batch=ragged_batch_inputs(device_refs[0], tuple(_BATCH_FIELDS))
    )
    if signals:
        types["signals"] = signal_inputs(list(device_refs))
    types["kv"] = kv_params.get_symbolic_inputs()
    return types


def decoder_outputs(
    outputs: GroupedDecoderOutputs,
) -> DecoderOutputs[TensorValue]:
    """Returns a graph-API text model's grouped outputs as the role's.

    A graph-API text model names its outputs by group, as
    :func:`~max.nn.legacy.transformer.logits_outputs` does; a registered
    decoder that delegates to one returns them through this.

    Raises:
        ValueError: If the outputs hold more than one hidden state.
    """
    logits = outputs.logits
    hidden_states = None
    if "hidden_states" in outputs:
        states = outputs.hidden_states.tensors()
        if len(states) != 1:
            raise ValueError(
                f"a decoder returns one hidden state, not {len(states)}"
            )
        (hidden_states,) = states
    return DecoderOutputs(
        next_token_logits=logits["next_token_logits"],
        logits=logits.get("logits"),
        logit_offsets=logits.get("logit_offsets"),
        hidden_states=hidden_states,
    )


def _graph_outputs(
    outputs: DecoderOutputs[TensorValue],
    return_hidden_states: ReturnHiddenStates,
) -> GroupedDecoderOutputs:
    """Names a decoder's outputs as an exported graph's outputs.

    An exported graph names its outputs by group, so the role's fields go in
    a ``logits`` group, then a ``hidden_states`` group when there are any.
    Their leaves keep the role's order.

    Raises:
        TypeError: If the decoder returned something other than
            :class:`DecoderOutputs`.
    """
    if not isinstance(outputs, DecoderOutputs):
        raise TypeError(
            "a registered decoder returns DecoderOutputs, not "
            f"{type(outputs).__qualname__}"
        )
    return logits_outputs(
        outputs.next_token_logits,
        outputs.logits,
        outputs.logit_offsets,
        return_hidden_states,
        () if outputs.hidden_states is None else (outputs.hidden_states,),
    )


class _DecoderGraph(GraphApiModule):
    """Traces a decoder with :class:`DecoderInputs` built from graph inputs.

    The graph takes the leaves of :func:`graph_api_decoder_input_types` in
    order. Its weights keep the names the decoder's own
    :meth:`~max.nn.legacy.Module.load_state_dict` gave them, so tracing
    through this module doesn't nest them.
    """

    def __init__(
        self,
        decoder: GraphApiModule,
        input_types: Mapping[str, object],
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self._field_types = OrderedDict(input_types)
        self._return_hidden_states = return_hidden_states

    def _unflatten_inputs(
        self, values: Sequence[Value[Any]]
    ) -> DecoderInputs[Any]:
        """Unflattens the graph's flat inputs into the decoder's fields.

        Raises:
            ValueError: If the input types hold a group the contract doesn't
                name.
        """
        flat = iter(values)
        fields: dict[str, Any] = {}
        for name, group in self._field_types.items():
            leaves, treedef = tree.flatten(group)
            taken = [next(flat) for _ in leaves]
            if name == "kv":
                fields["kv"] = tree.unflatten(treedef, taken)
            elif name == "signals":
                fields["signals"] = tuple(v.buffer for v in taken)
            elif name == "batch":
                assert isinstance(group, Mapping)
                for key, value in zip(group, taken, strict=True):
                    fields[_BATCH_FIELDS[key]] = value.tensor
            else:
                raise ValueError(f"a decoder takes no {name!r} inputs")
        return DecoderInputs(**fields)

    def __call__(self, *values: Value[Any]) -> GroupedDecoderOutputs:
        """Runs the decoder on the graph's flat inputs.

        Returns:
            The decoder's outputs, named by group.
        """
        return _graph_outputs(
            self.decoder(self._unflatten_inputs(values)),
            self._return_hidden_states,
        )

    def trace(
        self,
        *input_types: Type[Any],
        name: str | None = None,
        module: GraphModule | None = None,
        custom_extensions: Iterable[Path] = (),
        is_device_graph: bool = False,
    ) -> Graph:
        """Traces the decoder over the graph's flat inputs.

        It overrides :meth:`~max.nn.legacy.Module.trace` only to name the
        graph: the base names it after this wrapper class, while a registered
        decoder's graph compiles under the decoder's :func:`graph_name`,
        which keys its precompiled MEFs.

        Args:
            *input_types: The leaves of the decoder's input types.
            name: The graph's name, by default the decoder's
                :func:`graph_name`.
            module: The MLIR module to add the graph to.
            custom_extensions: Paths to custom kernel libraries.
            is_device_graph: Whether to build a device graph.

        Returns:
            The traced graph.
        """
        return Graph(
            graph_name(type(self.decoder)) if name is None else name,
            self,
            input_types=input_types,
            custom_extensions=list(custom_extensions),
            module=module,
            is_device_graph=is_device_graph,
        )


@dataclass(kw_only=True)
class GraphApiDecoderModelInputs(DecoderModelInputs):
    """The prepared inputs of a registered graph-API ``DECODER``."""

    signal_buffers: tuple[Buffer, ...] = ()
    """The allreduce signal buffers, when the graph takes them."""

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        """The step's buffers, in the graph's input order."""
        return (
            self.tokens,
            self.input_row_offsets,
            self.return_n_logits,
            *self.signal_buffers,
            *(
                tree.leaves(self.kv_cache_inputs)
                if self.kv_cache_inputs is not None
                else ()
            ),
        )


class GraphApiDecoderBatchProcessor(DecoderBatchProcessor):
    """Batches ragged text for a registered graph-API ``DECODER``.

    It feeds the fields of the input tree the pipeline model binds, so it
    adds signal buffers only when the graph takes them.
    """

    def get_symbolic_inputs(
        self,
        *,
        kv_params: KVCacheParamInterface,
        device_refs: list[DeviceRef],
    ) -> list[Any]:
        """Returns the bound graph's input types, in order."""
        del kv_params, device_refs
        assert self.input_tree is not None, "bind the graph's input tree first"
        return tree.leaves(self.input_tree)

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> GraphApiDecoderModelInputs:
        """Packs the contexts into one ragged batch.

        Raises:
            NotImplementedError: Under data parallelism.
        """
        if self.runtime.data_parallel_degree > 1:
            raise NotImplementedError(
                "graph-API decoders don't support data parallelism yet"
            )
        inputs = super().prepare_initial_token_inputs(
            replica_batches, kv_cache_inputs, return_n_logits
        )
        input_tree = self.input_tree
        assert input_tree is not None, "bind the graph's input tree first"
        signals: tuple[Buffer, ...] = ()
        if "signals" in input_tree:
            signals = tuple(self.runtime.signal_buffers)
        return GraphApiDecoderModelInputs(
            tokens=inputs.tokens,
            input_row_offsets=inputs.input_row_offsets,
            return_n_logits=inputs.return_n_logits,
            signal_buffers=signals,
            kv_cache_inputs=inputs.kv_cache_inputs,
        )


class _GraphApiPipelineModelMeta(ABCMeta):
    """Lets a pipeline model built from a registration be pickled.

    Serving pickles the pipeline model class into the model worker. A class
    :func:`graph_api_pipeline_model` builds has no importable name, so it
    pickles as the registration that rebuilds it.
    """


def _reduce_graph_api_pipeline_model(
    cls: _GraphApiPipelineModelMeta,
) -> str | tuple[Any, ...]:
    registration = cls.__dict__.get("_registration")
    if registration is None:
        return cls.__qualname__
    return graph_api_pipeline_model, registration


copyreg.pickle(_GraphApiPipelineModelMeta, _reduce_graph_api_pipeline_model)


class GraphApiDecoderPipelineModel(
    LogProbabilitiesMixin,
    PipelineModelWithKVCache[TextContext, Any, Model],
    metaclass=_GraphApiPipelineModelMeta,
):
    """Serves a registered graph-API ``DECODER`` module.

    :func:`graph_api_pipeline_model` subclasses it for each registration.
    The decoder is constructed as ``decoder_cls(config)`` from the finalized
    config, which holds the devices it runs on.
    """

    decoder_cls: ClassVar[type[GraphApiModule]]
    """The registered ``DECODER`` class."""

    batch_processor_cls: ClassVar[type[BatchProcessor[Any, Any]] | None] = (
        GraphApiDecoderBatchProcessor
    )

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
        *,
        memory_plan: MemoryPlan,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
        max_batch_size: int = 1,
    ) -> None:
        """Builds, compiles and initializes the decoder graph.

        Raises:
            NotImplementedError: If data parallelism is on.
        """
        if pipeline_config.model.data_parallel_degree > 1:
            raise NotImplementedError(
                f"{type(self).decoder_cls.__qualname__} doesn't support data "
                "parallelism"
            )
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits,
            memory_plan=memory_plan,
            return_hidden_states=return_hidden_states,
            max_batch_size=max_batch_size,
        )
        self._checkpoint: dict[str, Any] = {}
        self._models = self.load_graphs(session)

    @property
    def model(self) -> Model:
        """The compiled and initialized decoder."""
        return self._models[GraphRole.DECODER]

    def _hf_config_for_weights(self) -> AutoConfig | None:
        return self.huggingface_config

    def _create_model_config(self, state_dict: dict[str, Any]) -> Any:
        # graph_specs loads these into the decoder it constructs.
        self._checkpoint = state_dict
        return finalize_config(
            self.arch_config,
            self.huggingface_config,
            state_dict,
            ModelOutputConfig(
                return_logits=self.return_logits,
                return_hidden_states=self.return_hidden_states,
            ),
        )

    def graph_specs(self, model_config: Any) -> dict[str, GraphSpec]:
        """Returns the decoder graph, with the checkpoint loaded into it."""
        decoder_cls = type(self).decoder_cls
        # A registered module takes its config, which type[Module] can't say.
        constructor: Any = decoder_cls
        decoder = constructor(model_config)
        decoder.load_state_dict(
            self._checkpoint,
            override_quantization_encoding=bool(
                getattr(decoder_cls, "override_quantization_encoding", False)
            ),
            weight_alignment=1,
            strict=getattr(decoder_cls, "strict_state_dict_loading", True),
        )
        # The graph's weights registry holds what it binds; drop the rest.
        self._checkpoint = {}
        input_types = self._bind_graph_inputs(
            OrderedDict[str, object](
                graph_api_decoder_input_types(
                    self.kv_params,
                    self.device_refs,
                    signals=_takes_signals(decoder_cls, self.device_refs),
                )
            )
        )
        return {
            GraphRole.DECODER: GraphSpec(
                _DecoderGraph(decoder, input_types, self.return_hidden_states),
                graph_name(decoder_cls),
                input_types,
                weights=decoder.state_dict(auto_initialize=False),
            )
        }

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Runs the decoder on the prepared batch."""
        assert isinstance(model_inputs, GraphApiDecoderModelInputs)
        assert self.batch_processor is not None
        return self.batch_processor.process_outputs(
            self.model.execute(*model_inputs.buffers)
        )


_PIPELINE_MODELS: dict[
    tuple[type[ArchConfig], tuple[tuple[str, type[Any]], ...], Any],
    type[GraphApiDecoderPipelineModel],
] = {}


def graph_api_pipeline_model(
    config: type[ArchConfig],
    modules: tuple[tuple[str, type[Any]], ...],
    batching: type[Any] | None,
) -> type[GraphApiDecoderPipelineModel]:
    """Returns the pipeline model that serves a graph-API registration.

    One registration always returns the same class.

    Args:
        config: The registered config class.
        modules: The registered module class of each role, as items.
        batching: The registered batch processor, or ``None`` for
            :class:`GraphApiDecoderBatchProcessor`.

    Raises:
        ValueError: If the registration has no ``DECODER``.
        NotImplementedError: If it registers a role other than ``DECODER``.
        TypeError: If the ``DECODER`` isn't a graph-API module class.
    """
    registration = (config, modules, batching)
    if (cached := _PIPELINE_MODELS.get(registration)) is not None:
        return cached
    by_role = dict(modules)
    if GraphRole.DECODER not in by_role:
        raise ValueError("a module registration needs a DECODER module")
    if unsupported := sorted(set(by_role) - {GraphRole.DECODER}):
        raise NotImplementedError(
            f"registered graph-API modules can't fill {unsupported} yet"
        )
    decoder_cls = by_role[GraphRole.DECODER]
    if not is_graph_api_module(decoder_cls):
        raise TypeError(
            "the DECODER module must be a max.nn.legacy.Module class"
        )
    bases: tuple[type, ...] = (GraphApiDecoderPipelineModel,)
    if getattr(decoder_cls, "always_signal_buffers", False):
        bases = (AlwaysSignalBuffersMixin, *bases)
    name = f"{decoder_cls.__name__}PipelineModel"
    model = _GraphApiPipelineModelMeta(
        name,
        bases,
        {
            "__module__": __name__,
            "__qualname__": name,
            "_registration": registration,
            "decoder_cls": decoder_cls,
            "model_config_cls": config,
            "batch_processor_cls": batching or GraphApiDecoderBatchProcessor,
        },
    )
    assert issubclass(model, GraphApiDecoderPipelineModel)
    _PIPELINE_MODELS[registration] = model
    return model


def registered_pipeline_model(
    task: PipelineTask,
    config: type[ArchConfig],
    modules: tuple[tuple[str, type[Any]], ...],
    batching: type[Any] | None,
) -> type[GraphApiDecoderPipelineModel | Any]:
    """Returns the pipeline model that serves a module registration.

    The task picks the pipeline model, since each task prepares its inputs
    differently; every registration of a task shares it. A text-generation
    registration whose ``DECODER`` is a graph-API module gets
    :func:`graph_api_pipeline_model`'s; any other gets
    :func:`~max.pipelines.lib.module_registration.module_pipeline_model`'s.

    Raises:
        NotImplementedError: If the task isn't text generation, which is all
            registered modules serve so far.
    """
    if task != PipelineTask.TEXT_GENERATION:
        raise NotImplementedError(
            f"registered modules can't serve {task.value} yet"
        )
    decoder_cls = dict(modules).get(GraphRole.DECODER)
    if is_graph_api_module(decoder_cls):
        return graph_api_pipeline_model(config, modules, batching)
    module_items: Any = modules
    return module_pipeline_model(config, module_items, batching)
