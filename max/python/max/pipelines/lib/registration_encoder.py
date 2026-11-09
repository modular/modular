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
"""Serves embedding architectures that register an ``ENCODER``.

An encoder-only architecture registers its encoder module for the
``ENCODER`` role, and :func:`encoder_pipeline_model` builds the pipeline
model: it finalizes the config, constructs the encoder as
``encoder_cls(config)``, loads the checkpoint, builds the graph from
:func:`encoder_input_types`, and batches and executes it. The encoder takes
:class:`EncoderInputs` and returns the embeddings: a graph-API
(:class:`max.nn.legacy.Module`) encoder in its ``__call__``, over graph
values, and a ModuleV3 (:class:`max.nn.Module`) encoder in its
``forward``, over tensors.

An encoder takes its batch in one of two layouts, which its class picks:

- padded (the default): ``tokens`` and an ``attention_mask``, both
  ``[batch_size, seq_len]``;
- ragged, when it sets ``ragged_batch = True``: ``tokens``,
  ``input_row_offsets`` and ``return_n_logits``, as a ragged decoder takes
  them but with no KV cache.

A registered encoder class can also set ``graph_name``, the name its graph
compiles under. A graph-API encoder can set ``strict_state_dict_loading``,
``override_quantization_encoding`` and ``weight_alignment``, the options it
loads the checkpoint with, which an instance may also set from its config.
A graph-API encoder whose graph predates registration with input types
other than its layout's defines ``legacy_input_types()``, which returns
them in field order, so its precompiled MEFs keep their key.

A ModuleV3 encoder is built under the pipeline's dtype and moved to its
device, and the checkpoint's ``float32`` and ``bfloat16`` weights are cast
to the dtypes its parameters declare.
"""

from __future__ import annotations

import copyreg
from abc import ABCMeta
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Generic

from max import functional as F
from max import tree
from max.driver import Buffer, Device
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, TensorValue, Type, Value
from max.graph import Module as GraphModule
from max.graph.weights import Weights, WeightsAdapter
from max.nn import Module
from max.nn.legacy.layer import Module as GraphApiModule
from max.nn.outputs import ReturnLogits
from max.pipelines.context import TextContext
from max.pipelines.kv_cache.config import KVCacheConfig
from max.tensor import Tensor, default_dtype
from typing_extensions import TypeVar

from .graph_specs import GraphRole, GraphSpec
from .interfaces.arch_config import ArchConfig, ModelOutputConfig
from .interfaces.batch_processor import (
    BatchProcessor,
    PaddedEncoderBatchProcessor,
    SingleReplicaEmbeddingBatchProcessor,
    embedding_ragged_symbolic_inputs,
    padded_encoder_symbolic_inputs,
)
from .interfaces.pipeline_model import (
    ModelInputs,
    ModelOutputs,
    PipelineModel,
    _cast_to_parameter_dtypes,
)
from .module_registration import finalize_config
from .utils import parse_state_dict_from_weights

if TYPE_CHECKING:
    from .config import PipelineConfig

    # TODO(MXF-517): Remove the interfaces/memory_estimation import cycle.
    from .memory_estimation import MemoryPlan

__all__ = [
    "EncoderInputs",
    "EncoderPipelineModel",
    "GraphApiEncoderPipelineModel",
    "ModuleV3EncoderPipelineModel",
    "PaddedEncoderInputs",
    "PaddedEncoderProcessor",
    "RaggedEncoderInputs",
    "RaggedEncoderProcessor",
    "encoder_input_types",
    "encoder_pipeline_model",
]

TensorT = TypeVar("TensorT", default=Tensor)

_PADDED = ("tokens", "attention_mask")
_RAGGED = ("tokens", "input_row_offsets", "return_n_logits")


@tree.dataclass(frozen=True, kw_only=True)
class EncoderInputs(Generic[TensorT]):
    """The inputs of an ``ENCODER`` module's ``__call__``.

    A field that is ``None`` is not a graph input: a padded encoder takes
    :attr:`attention_mask`, and a ragged one :attr:`input_row_offsets` and
    :attr:`return_n_logits`.
    """

    tokens: TensorT
    """The token IDs: ``[batch_size, seq_len]`` padded, or
    ``[total_seq_len]`` ragged."""

    attention_mask: TensorT | None = None
    """``1.0`` for a real token and ``0.0`` for padding,
    ``[batch_size, seq_len]``, for a padded encoder."""

    input_row_offsets: TensorT | None = None
    """Each sequence's start in :attr:`tokens`, for a ragged encoder."""

    return_n_logits: TensorT | None = None
    """The ragged batch's logit count, which an encoder ignores; the graph
    takes it so a ragged encoder's inputs match a decoder's."""


def is_ragged(encoder_cls: type[Any]) -> bool:
    """Returns whether a registered encoder takes a ragged batch."""
    return bool(getattr(encoder_cls, "ragged_batch", False))


def encoder_input_types(
    encoder_cls: type[Any], device_refs: list[DeviceRef]
) -> dict[str, TensorType]:
    """Returns an ``ENCODER`` graph's input types, keyed by field, in order.

    Args:
        encoder_cls: The registered ``ENCODER`` class, which picks the
            layout.
        device_refs: The devices the encoder runs on; the first holds the
            batch.
    """
    names: tuple[str, ...] = _PADDED
    types = padded_encoder_symbolic_inputs(device_refs=device_refs)
    if is_ragged(encoder_cls):
        names = _RAGGED
        types = embedding_ragged_symbolic_inputs(device_refs=device_refs)
    input_types: dict[str, TensorType] = {}
    for name, input_type in zip(names, types, strict=True):
        assert isinstance(input_type, TensorType)
        input_types[name] = input_type
    return input_types


class _EncoderGraph(GraphApiModule):
    """Traces an encoder with :class:`EncoderInputs` built from graph inputs.

    The graph's weights keep the names the encoder's own
    :meth:`~max.nn.legacy.Module.load_state_dict` gave them.
    """

    def __init__(self, encoder: GraphApiModule, fields: Sequence[str]) -> None:
        super().__init__()
        self.encoder = encoder
        self._fields = tuple(fields)

    def __call__(self, *values: Value[Any]) -> TensorValue:
        """Runs the encoder on the graph's flat inputs.

        Returns:
            The embeddings.
        """
        inputs = EncoderInputs(
            **{
                name: value.tensor
                for name, value in zip(self._fields, values, strict=True)
            }
        )
        return self.encoder(inputs)

    def trace(
        self,
        *input_types: Type[Any],
        name: str | None = None,
        module: GraphModule | None = None,
        custom_extensions: Iterable[Path] = (),
        is_device_graph: bool = False,
    ) -> Graph:
        """Traces the encoder over the graph's flat inputs.

        Args:
            *input_types: The encoder's input types, in field order.
            name: The graph's name.
            module: The MLIR module to add the graph to.
            custom_extensions: Paths to custom kernel libraries.
            is_device_graph: Whether to build a device graph.

        Returns:
            The traced graph.
        """
        return Graph(
            _graph_name(type(self.encoder)) if name is None else name,
            self,
            input_types=input_types,
            custom_extensions=list(custom_extensions),
            module=module,
            is_device_graph=is_device_graph,
        )


def _graph_name(encoder_cls: type[Any]) -> str:
    return getattr(encoder_cls, "graph_name", encoder_cls.__qualname__)


class _ModuleV3EncoderGraph(Module[..., Tensor]):
    """Calls a ModuleV3 encoder with :class:`EncoderInputs` of its inputs.

    It is name-transparent to the encoder, so the graph's weight names are
    the encoder's own.
    """

    def __init__(
        self, encoder: Module[..., Tensor], fields: Sequence[str]
    ) -> None:
        super().__init__()
        encoder.name_transparent = True
        self.encoder = encoder
        self._fields = tuple(fields)

    def forward(self, *inputs: Tensor) -> Tensor:
        """Groups the graph's inputs into fields and runs the encoder."""
        return self.encoder(
            EncoderInputs(**dict(zip(self._fields, inputs, strict=True)))
        )


@dataclass
class PaddedEncoderInputs(ModelInputs):
    """The prepared inputs of a padded ``ENCODER``."""

    tokens: Buffer
    attention_mask: Buffer

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        """The step's buffers, in the graph's input order."""
        return (self.tokens, self.attention_mask)


@dataclass
class RaggedEncoderInputs(ModelInputs):
    """The prepared inputs of a ragged ``ENCODER``."""

    tokens: Buffer
    input_row_offsets: Buffer
    return_n_logits: Buffer

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        """The step's buffers, in the graph's input order."""
        return (self.tokens, self.input_row_offsets, self.return_n_logits)


class PaddedEncoderProcessor(PaddedEncoderBatchProcessor[PaddedEncoderInputs]):
    """Pads a batch of text for a registered padded ``ENCODER``."""

    def _make_inputs(
        self, *, next_tokens_batch: Buffer, attention_mask: Buffer
    ) -> PaddedEncoderInputs:
        return PaddedEncoderInputs(
            tokens=next_tokens_batch, attention_mask=attention_mask
        )


class RaggedEncoderProcessor(
    SingleReplicaEmbeddingBatchProcessor[RaggedEncoderInputs]
):
    """Batches ragged text for a registered ragged ``ENCODER``."""

    def _make_inputs(
        self,
        *,
        tokens: Buffer,
        input_row_offsets: Buffer,
        return_n_logits: Buffer,
    ) -> RaggedEncoderInputs:
        return RaggedEncoderInputs(
            tokens=tokens,
            input_row_offsets=input_row_offsets,
            return_n_logits=return_n_logits,
        )


class _EncoderPipelineModelMeta(ABCMeta):
    """Lets a pipeline model built from a registration be pickled.

    Serving pickles the pipeline model class into the model worker. A class
    :func:`encoder_pipeline_model` builds has no importable name, so it
    pickles as the registration that rebuilds it.
    """


def _reduce_encoder_pipeline_model(
    cls: _EncoderPipelineModelMeta,
) -> str | tuple[Any, ...]:
    registration = cls.__dict__.get("_registration")
    if registration is None:
        return cls.__qualname__
    return encoder_pipeline_model, registration


copyreg.pickle(_EncoderPipelineModelMeta, _reduce_encoder_pipeline_model)


class EncoderPipelineModel(
    PipelineModel[TextContext, Any, Model],
    metaclass=_EncoderPipelineModelMeta,
):
    """Serves a registered ``ENCODER`` module.

    :func:`encoder_pipeline_model` subclasses
    :class:`GraphApiEncoderPipelineModel` or
    :class:`ModuleV3EncoderPipelineModel` for each registration. It keeps no
    KV cache, and returns the embeddings as the outputs' logits.
    """

    encoder_cls: ClassVar[type[Any]]
    """The registered ``ENCODER`` class."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.ALL,
        *,
        memory_plan: MemoryPlan,
        max_batch_size: int = 1,
    ) -> None:
        """Builds, compiles and initializes the encoder graph."""
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits,
            memory_plan=memory_plan,
            max_batch_size=max_batch_size,
        )
        self._checkpoint: dict[str, Any] = {}
        self._models = self.load_graphs(session)

    @property
    def model(self) -> Model:
        """The compiled and initialized encoder."""
        return self._models[GraphRole.ENCODER]

    def _load_state_dict(self) -> dict[str, Any]:
        # Passes the checkpoint and pipeline configs to the weight adapter,
        # as a decoder's pipeline model does.
        return parse_state_dict_from_weights(
            self.pipeline_config, self.weights, self.adapter
        )

    def _create_model_config(self, state_dict: dict[str, Any]) -> Any:
        # graph_specs builds the encoder's weights from these.
        self._checkpoint = state_dict
        return finalize_config(
            self.arch_config,
            self.huggingface_config,
            state_dict,
            ModelOutputConfig(return_logits=self.return_logits),
        )

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Runs the encoder on the prepared batch."""
        assert isinstance(
            model_inputs, (PaddedEncoderInputs, RaggedEncoderInputs)
        )
        assert self.batch_processor is not None
        return self.batch_processor.process_outputs(
            self.model.execute(*model_inputs.buffers)
        )


class GraphApiEncoderPipelineModel(EncoderPipelineModel):
    """Serves a registered graph-API ``ENCODER`` module."""

    encoder_cls: ClassVar[type[GraphApiModule]]

    def graph_specs(self, model_config: Any) -> dict[str, GraphSpec]:
        """Returns the encoder graph, with the checkpoint loaded into it."""
        encoder_cls = type(self).encoder_cls
        # A registered module takes its config, which type[Module] can't say.
        constructor: Any = encoder_cls
        encoder = constructor(model_config)
        encoder.load_state_dict(
            self._checkpoint,
            override_quantization_encoding=bool(
                getattr(encoder, "override_quantization_encoding", False)
            ),
            # The alignment is part of the graph, so it keys the MEF.
            weight_alignment=getattr(encoder, "weight_alignment", None),
            strict=bool(getattr(encoder, "strict_state_dict_loading", True)),
        )
        # The graph's weights registry holds what it binds; drop the rest.
        self._checkpoint = {}
        input_types = encoder_input_types(encoder_cls, self.device_refs)
        legacy_input_types = getattr(encoder, "legacy_input_types", None)
        if legacy_input_types is not None:
            input_types = dict(
                zip(input_types, legacy_input_types(), strict=True)
            )
        return {
            GraphRole.ENCODER: GraphSpec(
                _EncoderGraph(encoder, list(input_types)),
                _graph_name(encoder_cls),
                tuple(input_types.values()),
                weights=encoder.state_dict(auto_initialize=False),
            )
        }


class ModuleV3EncoderPipelineModel(EncoderPipelineModel):
    """Serves a registered ModuleV3 ``ENCODER`` module."""

    encoder_cls: ClassVar[type[Module[..., Any]]]

    def graph_specs(self, model_config: Any) -> dict[str, GraphSpec]:
        """Returns the encoder graph, built from the checkpoint weights."""
        encoder_cls = type(self).encoder_cls
        # A registered module takes its config, which type[Module] can't say.
        constructor: Any = encoder_cls
        with F.lazy(), default_dtype(self.dtype):
            encoder = constructor(model_config)
            encoder.to(self.devices[0])
        input_types = encoder_input_types(encoder_cls, self.device_refs)
        graph = _ModuleV3EncoderGraph(encoder, list(input_types))
        weights = _cast_to_parameter_dtypes(graph, self._checkpoint)
        self._checkpoint = {}
        return {
            GraphRole.ENCODER: GraphSpec(
                graph,
                _graph_name(encoder_cls),
                tuple(input_types.values()),
                weights=weights,
            )
        }


_PIPELINE_MODELS: dict[
    tuple[type[ArchConfig], tuple[tuple[str, type[Any]], ...], Any],
    type[EncoderPipelineModel],
] = {}


def encoder_pipeline_model(
    config: type[ArchConfig],
    modules: tuple[tuple[str, type[Any]], ...],
    batching: type[Any] | None,
) -> type[EncoderPipelineModel]:
    """Returns the pipeline model that serves an ``ENCODER`` registration.

    One registration always returns the same class.

    Args:
        config: The registered config class.
        modules: The registered module class of each role, as items.
        batching: The registered batch processor, or ``None`` for the one
            the encoder's layout takes.

    Raises:
        NotImplementedError: If it registers a role other than ``ENCODER``.
        TypeError: If the ``ENCODER`` isn't a graph-API or ModuleV3 module
            class.
    """
    registration = (config, modules, batching)
    if (cached := _PIPELINE_MODELS.get(registration)) is not None:
        return cached
    by_role = dict(modules)
    if unsupported := sorted(set(by_role) - {GraphRole.ENCODER}):
        raise NotImplementedError(
            f"an encoder registration can't fill {unsupported}"
        )
    encoder_cls = by_role[GraphRole.ENCODER]
    base: type[EncoderPipelineModel]
    if isinstance(encoder_cls, type) and issubclass(encoder_cls, Module):
        base = ModuleV3EncoderPipelineModel
    elif isinstance(encoder_cls, type) and issubclass(
        encoder_cls, GraphApiModule
    ):
        base = GraphApiEncoderPipelineModel
    else:
        raise TypeError(
            "the ENCODER module must be a max.nn.Module or "
            "max.nn.legacy.Module class"
        )
    default_batching: type[BatchProcessor[Any, Any]] = (
        RaggedEncoderProcessor
        if is_ragged(encoder_cls)
        else PaddedEncoderProcessor
    )
    name = f"{encoder_cls.__name__}PipelineModel"
    model = _EncoderPipelineModelMeta(
        name,
        (base,),
        {
            "__module__": __name__,
            "__qualname__": name,
            "_registration": registration,
            "encoder_cls": encoder_cls,
            "model_config_cls": config,
            "batch_processor_cls": batching or default_batching,
        },
    )
    assert issubclass(model, EncoderPipelineModel)
    _PIPELINE_MODELS[registration] = model
    return model
