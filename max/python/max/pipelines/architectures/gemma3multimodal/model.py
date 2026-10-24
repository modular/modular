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
from dataclasses import dataclass
from typing import Any, ClassVar, cast

import numpy as np
import numpy.typing as npt
from max import tree
from max.driver import Buffer, Device
from max.engine import InferenceSession, Model
from max.graph import DeviceRef
from max.graph.weights import WeightData, Weights, WeightsAdapter
from max.nn.transformer import ReturnLogits
from max.pipelines.context import TextAndVisionContext
from max.pipelines.lib import (
    AlwaysSignalBuffersMixin,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModelWithKVCache,
)
from max.pipelines.lib.graph_specs import GraphRole, GraphSpec
from max.pipelines.lib.log_probabilities import LogProbabilitiesMixin
from max.pipelines.lib.memory_estimation import MemoryPlan
from transformers import AutoConfig

from .batch_processor import Gemma3MultiModalBatchProcessor
from .model_config import Gemma3ForConditionalGenerationConfig
from .vision_model.gemma3multimodal import (
    Gemma3LanguageModel,
    Gemma3VisionModel,
)
from .weight_adapters import (
    convert_safetensor_language_state_dict,
    convert_safetensor_vision_state_dict,
)

logger = logging.getLogger("max.pipelines")

# The vision and language graphs compile into one artifact.
_COMPILE_GROUP = "gemma3multimodal"


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
        kv_cache_inputs: Inputs for the KV cache.
        pixel_values: Raw pixel values for vision inputs. Defaults to ``None``.
        image_token_indices: Pre-computed indices of image tokens. Defaults to
            ``None``.
    """

    tokens: npt.NDArray[np.integer[Any]] | Buffer
    """Tensor containing the input token IDs."""

    input_row_offsets: npt.NDArray[np.integer[Any]] | list[Buffer]
    """Tensor containing the offsets for each row in the ragged input sequence,
    or the attention mask for the padded input sequence. For distributed execution,
    this can be a list of tensors, one per device."""

    signal_buffers: list[Buffer]
    """Device buffers used for synchronization in communication collectives."""

    return_n_logits: Buffer
    """Number of logits to return, used by speculative decoding for example."""

    pixel_values: list[Buffer] | None = None
    """Raw pixel values for vision inputs: [batch, channels, height, width]."""

    image_token_indices: list[Buffer] | None = None
    """Pre-computed indices of image tokens in the input sequence."""

    @property
    def has_vision_inputs(self) -> bool:
        """Check if this input contains vision data."""
        return self.pixel_values is not None


class Gemma3_MultiModalModel(
    LogProbabilitiesMixin,
    AlwaysSignalBuffersMixin,
    PipelineModelWithKVCache[
        TextAndVisionContext, Gemma3ForConditionalGenerationConfig, Model
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

    model_config_cls: ClassVar[type[Any]] = Gemma3ForConditionalGenerationConfig
    batch_processor_cls: ClassVar[type[Gemma3MultiModalBatchProcessor]] = (
        Gemma3MultiModalBatchProcessor
    )

    _vision_weights_dict: dict[str, WeightData]
    _language_weights_dict: dict[str, WeightData]

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

    @property
    def language_model(self) -> Model:
        """The compiled and initialized MAX Engine model ready for inference."""
        return self._models[GraphRole.DECODER]

    @property
    def vision_model(self) -> Model:
        """The compiled and initialized MAX Engine vision model ready for
        inference."""
        return self._models[GraphRole.VISION]

    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        """Gets the number of hidden layers from the HuggingFace configuration."""
        return Gemma3ForConditionalGenerationConfig.get_num_layers(
            huggingface_config
        )

    def _load_state_dict(self) -> dict[str, Any]:
        assert self.max_batch_size, "Expected max_batch_size to be set"

        weights_dict = dict(self.weights.items())
        self._language_weights_dict = convert_safetensor_language_state_dict(
            weights_dict
        )
        self._vision_weights_dict = convert_safetensor_vision_state_dict(
            weights_dict
        )
        return {k: v.data() for k, v in weights_dict.items()}

    def _create_model_config(
        self, state_dict: dict[str, Any]
    ) -> Gemma3ForConditionalGenerationConfig:
        model_config = Gemma3ForConditionalGenerationConfig.initialize(
            self.pipeline_config, max_seq_len=self.max_seq_len
        )
        model_config.finalize(
            huggingface_config=self.huggingface_config,
            state_dict=state_dict,
            return_logits=self.return_logits,
        )
        self.config = model_config
        return model_config

    def graph_specs(
        self, model_config: Gemma3ForConditionalGenerationConfig
    ) -> dict[str, GraphSpec]:
        """Returns the vision encoder and the decoder, in one compile group."""
        return {
            GraphRole.VISION: self._vision_spec(
                model_config, self._vision_weights_dict
            ),
            GraphRole.DECODER: self._language_spec(
                model_config, self._language_weights_dict
            ),
        }

    def _language_spec(
        self,
        config: Gemma3ForConditionalGenerationConfig,
        state_dict: dict[str, WeightData],
    ) -> GraphSpec:
        language_model = Gemma3LanguageModel(config)
        language_model.load_state_dict(
            state_dict,
            weight_alignment=1,
            strict=self._strict_state_dict_loading,
        )
        return GraphSpec(
            language_model,
            getattr(self.huggingface_config, "model_type", "Gemma3"),
            language_model.input_tree(self.kv_params),
            weights=language_model.state_dict(),
            group=_COMPILE_GROUP,
        )

    def _vision_spec(
        self,
        config: Gemma3ForConditionalGenerationConfig,
        state_dict: dict[str, WeightData],
    ) -> GraphSpec:
        vision_model = Gemma3VisionModel(
            config,
            device=DeviceRef.from_device(self.devices[0]),
        )
        vision_model.load_state_dict(
            state_dict=state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
            strict=self._strict_state_dict_loading,
        )
        return GraphSpec(
            vision_model,
            getattr(self.huggingface_config, "model_type", "Gemma3")
            + "_vision",
            vision_model.input_types(),
            weights=vision_model.state_dict(),
            group=_COMPILE_GROUP,
        )

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """If required, execute the vision model, then continue to execute the
        language model.  Either pass through image embeddings or create an empty
        placeholder."""
        model_inputs = cast(Gemma3MultiModalModelInputs, model_inputs)

        input_row_offsets = model_inputs.input_row_offsets

        image_embeddings: list[Buffer]
        image_token_indices: list[Buffer]
        if model_inputs.has_vision_inputs:
            assert self.vision_model is not None
            assert model_inputs.pixel_values is not None

            # Execute vision model: patched pixel_values -> image_embeddings.
            vision_outputs = self.vision_model.execute(
                *model_inputs.pixel_values, *model_inputs.signal_buffers
            )
            assert len(vision_outputs) == len(self.devices)

            image_embeddings = [
                output
                for output in vision_outputs
                if isinstance(output, Buffer)
            ]
            assert model_inputs.image_token_indices is not None
            image_token_indices = model_inputs.image_token_indices
        else:
            assert isinstance(
                self._batch_processor, Gemma3MultiModalBatchProcessor
            )
            image_embeddings = self._batch_processor.empty_image_embeddings()
            image_token_indices = (
                self._batch_processor.empty_image_token_indices()
            )

        assert model_inputs.kv_cache_inputs

        model_outputs = self.language_model.execute(
            model_inputs.tokens,
            model_inputs.return_n_logits,
            *input_row_offsets,
            *image_embeddings,
            *image_token_indices,
            *model_inputs.signal_buffers,
            *tree.leaves(model_inputs.kv_cache_inputs),
        )

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
