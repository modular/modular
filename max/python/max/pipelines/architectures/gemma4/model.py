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

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, cast

import numpy as np
import numpy.typing as npt
from max import tree
from max.driver import Buffer, Device, DLPackArray
from max.engine import InferenceSession, Model
from max.graph import (
    DeviceRef,
    Graph,
    Module,
)
from max.graph.weights import WeightData, Weights, WeightsAdapter
from max.nn.kv_cache import MultiKVCacheParams
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.context import ImageMetadata
from max.pipelines.lib import (
    AlwaysSignalBuffersMixin,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModelWithKVCache,
)
from max.pipelines.lib.graph_specs import GraphRole, GraphSpec
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from max.pipelines.lib.memory_estimation import MemoryPlan
from max.pipelines.lib.pipeline_parallel import (
    PipelinedModel,
    PipelineStage,
    raise_stage_traced,
    stage_input_types,
)
from max.pipelines.lib.vision_encoder_cache import VisionEncodeResult
from max.profiler import traced

from .batch_processor import Gemma4BatchProcessor
from .batch_vision_inputs import VisionRawInputs
from .context import Gemma4Context
from .gemma4 import Gemma4TextModel, language_input_tree
from .model_config import Gemma4ForConditionalGenerationConfig
from .vision_model.vision_model import Gemma4VisionModel
from .weight_adapters import (
    convert_safetensor_language_state_dict,
    convert_safetensor_vision_state_dict,
)

logger = logging.getLogger("max.pipelines")

# The language and vision graphs compile into one artifact.
_COMPILE_GROUP = "gemma4"


def _declared_weights(
    language_model: Gemma4TextModel, state_dict: Mapping[str, WeightData]
) -> dict[str, WeightData]:
    """Returns the weights ``language_model`` declares.

    A pipeline stage declares only its own layers, so it loads only their
    weights, strictly, and a misnamed one still fails.
    """
    declared = language_model.raw_state_dict()
    return {
        name: weight for name, weight in state_dict.items() if name in declared
    }


def language_spec(
    config: Gemma4ForConditionalGenerationConfig,
    state_dict: Mapping[str, WeightData],
    *,
    stage: PipelineStage | None = None,
    strict: bool = True,
    is_device_graph: bool = False,
) -> GraphSpec:
    """Returns the language graph, or one pipeline stage's part of it.

    Args:
        config: The model config, a stage's for a pipeline stage.
        state_dict: The adapted language weights of the whole model.
        stage: The pipeline stage to build, or ``None`` for the whole model.
        strict: Whether every declared weight must load.
        is_device_graph: Whether to synthesize a device graph.

    Raises:
        StageTraced: On a stage that only traces, carrying the types of the
            hidden states it hands the next stage.
    """
    language_model = Gemma4TextModel(config)
    if stage is not None:
        state_dict = _declared_weights(language_model, state_dict)
    language_model.load_state_dict(
        state_dict, weight_alignment=1, strict=strict
    )
    spec = GraphSpec(
        language_model,
        "gemma4_language",
        stage_input_types(stage, [language_model.input_tree()]),
        weights=language_model.state_dict(),
        group=_COMPILE_GROUP,
        is_device_graph=is_device_graph,
    )
    if stage is not None and stage.trace_only:
        raise_stage_traced(spec.trace().output_tree)
    return spec


@dataclass
class Gemma3MultiModalModelInputs(ModelInputs):
    """A class representing inputs for the Gemma3 multi modal model.

    This class encapsulates the input tensors required for the Gemma3 multi
    modal model, for text and vision processing.

    Args:
        tokens: Input token IDs.
        input_row_offsets: Input row offsets (ragged tensors).
        return_n_logits: Number of logits to return.
        signal_buffers: Device buffers for distributed communication.
        kv_cache_inputs: Combined KV cache inputs (sliding-window + global).

    Image (and video-frame) embeddings come from the pipeline-driven encoder
    cache on the base ``vision_embeddings`` / ``vision_scatter_indices`` fields.
    """

    tokens: npt.NDArray[np.integer[Any]] | Buffer
    input_row_offsets: npt.NDArray[np.integer[Any]] | Buffer
    signal_buffers: list[Buffer]
    return_n_logits: Buffer

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        """Positional Buffer inputs for the language-model ABI."""
        assert self.kv_cache_inputs is not None
        return (
            self.tokens,
            self.return_n_logits,
            self.input_row_offsets,
            *self.vision_embeddings,
            *self.vision_scatter_indices,
            *self.signal_buffers,
            *tree.leaves(self.kv_cache_inputs),
        )


class Gemma3_MultiModalModel(
    AlwaysSignalBuffersMixin,
    PipelinedModel[Gemma4Context],
    PipelineModelWithKVCache[
        Gemma4Context, Gemma4ForConditionalGenerationConfig, Model | None
    ],
):
    """Gemma 3 multimodal pipeline model for text generation.

    This class integrates the Gemma 3 multimodal architecture with the MAX
    pipeline infrastructure, handling model loading, KV cache management, and
    input preparation for inference.

    Args:
        pipeline_config: The configuration settings for the entire pipeline.
        session: The MAX inference session managing the runtime.
        huggingface_config: The configuration loaded from HuggingFace
            (:obj:`transformers.AutoConfig`).
        devices: A list of MAX devices (:obj:`max.driver.Device`) to
            run the model on.
        kv_cache_config: Configuration settings for the Key-Value cache
            (:obj:`max.pipelines.max_config.KVCacheConfig`).
        weights: The model weights (:obj:`max.graph.weights.Weights`).
        adapter: An optional adapter to modify weights before loading
            (:obj:`max.graph.weights.WeightsAdapter`).
        return_logits: The number of top logits to return from the model
            execution.
    """

    model_config_cls: ClassVar[type[Any]] = Gemma4ForConditionalGenerationConfig
    batch_processor_cls: ClassVar[type[Gemma4BatchProcessor]] = (
        Gemma4BatchProcessor
    )

    _vision_weights_dict: dict[str, Any]
    _language_weights_dict: dict[str, Any]

    # The vision and text towers are in the same weights file, but are in
    # separate models, so load_state_dict will naturally be loading subsets in
    # each case.
    _strict_state_dict_loading = True

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        *,
        memory_plan: MemoryPlan,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
        max_batch_size: int = 1,
    ) -> None:
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter=adapter,
            return_logits=return_logits,
            memory_plan=memory_plan,
            max_batch_size=max_batch_size,
        )

        # signal_buffers are provided by AlwaysSignalBuffersMixin as a cached_property
        # to avoid GPU memory allocation during compile-only mode (cross-compilation).
        # Force initialization here to ensure buffers are ready before model execution,
        # preventing potential race conditions in multi-GPU scenarios.
        _ = self.signal_buffers

        self._models = self.load_graphs(session)

        assert isinstance(self.kv_params, MultiKVCacheParams)

        if self._batch_processor is not None:
            assert isinstance(self._batch_processor, Gemma4BatchProcessor)
            self._batch_processor.bind_model_state(config=self.config)
            if self.vision_model is not None:
                self._batch_processor.bind_vision_encoder(
                    vision_model=self.vision_model, session=session
                )

    @property
    def language_model(self) -> Model:
        """The compiled and initialized language model."""
        return self._models[GraphRole.DECODER]

    @property
    def vision_model(self) -> Model | None:
        """The compiled vision model, or None for text-only
        ("gemma4_unified") checkpoints whose vision embedder is not
        implemented yet."""
        return self._models.get(GraphRole.VISION)

    @property
    def model(self) -> Model:
        """Expose language model for graph capture/replay.

        Only the language model is captured since vision runs
        during prefill only.
        """
        return self.language_model

    @property
    def _batching(self) -> Gemma4BatchProcessor:
        assert isinstance(self._batch_processor, Gemma4BatchProcessor)
        return self._batch_processor

    def pack_vision_inputs(
        self,
        selection: Sequence[tuple[Gemma4Context, Sequence[ImageMetadata]]],
        devices: list[Device],
    ) -> VisionRawInputs | None:
        """Packs the selected images, as the batch processor does."""
        return self._batching.pack_vision_inputs(selection, devices)

    def vision_execute(
        self,
        selection: Sequence[tuple[Gemma4Context, Sequence[ImageMetadata]]],
        devices: list[Device],
        packed: VisionRawInputs | None,
    ) -> VisionEncodeResult:
        """Encodes the selected images, as the batch processor does."""
        return self._batching.vision_execute(selection, devices, packed)

    def empty_vision_embeddings(self, devices: list[Device]) -> list[Buffer]:
        """Returns the batch processor's embeddings for a batch with no new
        images."""
        return self._batching.empty_vision_embeddings(devices)

    def graph_specs(
        self, model_config: Gemma4ForConditionalGenerationConfig
    ) -> dict[str, GraphSpec]:
        """Returns the decoder, and the vision encoder unless the checkpoint
        is text-only, in one compile group."""
        specs: dict[str, GraphSpec] = {
            GraphRole.DECODER: self._language_spec(
                model_config, self._language_weights_dict
            )
        }
        if self._include_vision_graph(model_config):
            specs[GraphRole.VISION] = self._vision_spec(
                model_config, self._vision_weights_dict
            )
        return specs

    def _load_state_dict(self) -> dict[str, Any]:
        assert self.max_batch_size, "Expected max_batch_size to be set"

        # Get processed state dict for language and vision models
        weights_dict = dict(self.weights.items())
        self._language_weights_dict = convert_safetensor_language_state_dict(
            weights_dict
        )
        self._vision_weights_dict = convert_safetensor_vision_state_dict(
            weights_dict
        )
        return {k: v.data() for k, v in weights_dict.items()}

    @classmethod
    def build_module(
        cls, context: ModelLoadContext
    ) -> tuple[Gemma4TextModel, Gemma4ForConditionalGenerationConfig]:
        """Builds the language model, without its vision tower.

        The vision tower builds in its own graph, which only serving runs.
        """
        weights = dict(context.weights.items())
        config = cls._finalized_config(
            context, {name: value.data() for name, value in weights.items()}
        )
        language_weights = convert_safetensor_language_state_dict(weights)
        return cls._language_module(config, language_weights), config

    @classmethod
    def build_vision_graph(
        cls, context: ModelLoadContext
    ) -> tuple[Graph, dict[str, DLPackArray]]:
        """Builds the vision tower's graph.

        Raises:
            ValueError: If the checkpoint has no vision tower.
        """
        weights = dict(context.weights.items())
        config = cls._finalized_config(
            context, {name: value.data() for name, value in weights.items()}
        )
        if config.vision_config is None:
            raise ValueError("This Gemma 4 checkpoint has no vision tower.")
        spec = cls._vision_spec(
            config, convert_safetensor_vision_state_dict(weights)
        )
        return spec.trace(), dict(spec.weights_registry())

    @classmethod
    def _finalized_config(
        cls, context: ModelLoadContext, state_dict: dict[str, Any]
    ) -> Gemma4ForConditionalGenerationConfig:
        # Quantization is declared per checkpoint name, so ``state_dict`` is
        # the unconverted checkpoint rather than the language weights.
        config = Gemma4ForConditionalGenerationConfig.initialize(
            context.pipeline_config, max_seq_len=context.max_seq_len
        )
        config.finalize(
            huggingface_config=context.huggingface_config,
            state_dict=state_dict,
            return_logits=context.return_logits,
        )
        if context.return_hidden_states != ReturnHiddenStates.NONE:
            config.text_config.return_hidden_states = (
                context.return_hidden_states
            )
        if context.target_layer_ids:
            config.text_config.target_layer_ids = list(context.target_layer_ids)
        return config

    @classmethod
    def _language_module(
        cls,
        config: Gemma4ForConditionalGenerationConfig,
        state_dict: dict[str, WeightData],
    ) -> Gemma4TextModel:
        language_model = Gemma4TextModel(config)
        if config.layer_ids is not None:
            state_dict = _declared_weights(language_model, state_dict)
        language_model.load_state_dict(
            state_dict,
            weight_alignment=1,
            strict=cls._strict_state_dict_loading,
        )
        return language_model

    def _create_model_config(
        self, state_dict: dict[str, Any]
    ) -> Gemma4ForConditionalGenerationConfig:
        model_config = self._finalized_config(
            ModelLoadContext.from_pipeline_model(self), state_dict
        )
        self.config = model_config
        return model_config

    def _include_vision_graph(
        self, model_config: Gemma4ForConditionalGenerationConfig
    ) -> bool:
        return model_config.vision_config is not None

    def _language_model_input_tree(
        self, config: Gemma4ForConditionalGenerationConfig
    ) -> OrderedDict[str, object]:
        """Names the language graph's inputs; its leaves are its inputs."""
        return language_input_tree(
            [DeviceRef.from_device(d) for d in self.devices],
            config.unquantized_dtype,
            config.text_config.hidden_size,
            self.kv_params,
        )

    def _language_spec(
        self,
        config: Gemma4ForConditionalGenerationConfig,
        state_dict: dict[str, WeightData],
    ) -> GraphSpec:
        spec = language_spec(
            config,
            state_dict,
            stage=self.pipeline_config.model.pipeline_stage,
            strict=self._strict_state_dict_loading,
            is_device_graph=(
                self.pipeline_config.runtime.experimental_device_graph_synthesis
            ),
        )
        assert isinstance(spec.module, Gemma4TextModel)
        self._bind_graph_inputs(spec.module.input_tree())
        return spec

    @classmethod
    def _vision_spec(
        cls,
        config: Gemma4ForConditionalGenerationConfig,
        state_dict: dict[str, WeightData],
    ) -> GraphSpec:
        vision_model = Gemma4VisionModel(config, device=config.devices[0])
        vision_model.load_state_dict(
            state_dict=state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
            strict=cls._strict_state_dict_loading,
        )
        return GraphSpec(
            vision_model,
            "gemma4_vision",
            vision_model.input_types(),
            weights=vision_model.state_dict(),
            group=_COMPILE_GROUP,
        )

    def _build_vision_graph(
        self,
        config: Gemma4ForConditionalGenerationConfig,
        state_dict: dict[str, WeightData],
        module: Module | None = None,
    ) -> tuple[Graph, Mapping[str, DLPackArray]]:
        """Build the vision model with our input types and graph"""
        spec = self._vision_spec(config, state_dict)
        return spec.trace(module), spec.weights_registry()

    @traced
    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Execute the language model over the pipeline-finalized inputs."""
        model_inputs = cast(Gemma3MultiModalModelInputs, model_inputs)
        model_outputs = self.language_model.execute(*model_inputs.buffers)

        if len(model_outputs) == 3:
            assert isinstance(model_outputs[0], Buffer)
            assert isinstance(model_outputs[1], Buffer)
            assert isinstance(model_outputs[2], Buffer)
            return ModelOutputs(
                logits=model_outputs[1],
                next_token_logits=model_outputs[0],
                logit_offsets=model_outputs[2],
            )
        else:
            assert isinstance(model_outputs[0], Buffer)
            return ModelOutputs(
                logits=model_outputs[0],
                next_token_logits=model_outputs[0],
            )
