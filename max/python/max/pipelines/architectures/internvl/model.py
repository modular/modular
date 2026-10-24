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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from max import tree
from max.driver import Buffer, Device
from max.engine import InferenceSession, Model
from max.graph.weights import (
    SafetensorWeights,
    WeightData,
    Weights,
    WeightsAdapter,
)
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

from .batch_processor import InternVLBatchProcessor
from .internvl import InternVLLanguageModel, InternVLVisionModel
from .model_config import InternVLConfig
from .tokenizer import _get_image_context_token_id
from .weight_adapters import (
    convert_internvl_language_model_state_dict,
    convert_internvl_vision_model_state_dict,
)

logger = logging.getLogger("max.pipelines")

# The vision and language graphs compile into one artifact.
_COMPILE_GROUP = "internvl"


@dataclass
class InternVLInputs(ModelInputs):
    """A class representing inputs for the InternVL model."""

    tokens: Buffer
    """Tensor containing the input token IDs."""

    input_row_offsets: list[Buffer]
    """Per-device tensors containing the offsets for each row in the ragged
    input sequence.
    """

    signal_buffers: list[Buffer]
    """Device buffers used for synchronization in communication collectives."""

    return_n_logits: Buffer
    """Number of logits to return, used by speculative decoding for example."""

    # Vision inputs.
    pixel_values: list[Buffer] | None = None
    """Pixel values for vision inputs."""

    image_token_indices: list[Buffer] | None = None
    """Per-device pre-computed indices of image tokens in the input sequence."""

    @property
    def has_vision_inputs(self) -> bool:
        """Check if this input contains vision data."""
        return self.pixel_values is not None


def assert_image_embeddings_invariant(
    image_embeddings: Sequence[Buffer], image_token_indices: Sequence[Buffer]
) -> None:
    # Check for shape mismatch that causes scatter_nd OOB access.
    for i, (embed, indices) in enumerate(
        zip(image_embeddings, image_token_indices, strict=True)
    ):
        embed_count = embed.shape[0]
        indices_count = indices.shape[0]
        if embed_count != indices_count:
            logger.error(
                f"[CRITICAL] Device {i}: Vision embedding count ({embed_count}) "
                f"!= image token indices count ({indices_count})."
            )
        assert embed_count == indices_count, (
            f"Vision embedding shape mismatch on device {i}: {embed_count} embeddings "
            f"but {indices_count} indices."
        )


class InternVLModel(
    LogProbabilitiesMixin,
    AlwaysSignalBuffersMixin,
    PipelineModelWithKVCache[TextAndVisionContext, InternVLConfig, Model],
):
    """An InternVL pipeline model for multimodal text generation."""

    model_config_cls: ClassVar[type[Any]] = InternVLConfig
    batch_processor_cls: ClassVar[type[InternVLBatchProcessor]] = (
        InternVLBatchProcessor
    )

    _vision_weights_dict: dict[str, WeightData]
    _language_weights_dict: dict[str, WeightData]

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
            max_batch_size=max_batch_size,
            memory_plan=memory_plan,
        )

        self._models = self.load_graphs(session)

    @property
    def vision_model(self) -> Model:
        """The compiled vision model for processing images."""
        return self._models[GraphRole.VISION]

    @property
    def language_model(self) -> Model:
        """The compiled language model for text generation."""
        return self._models[GraphRole.DECODER]

    def _load_state_dict(self) -> dict[str, Any]:
        if not isinstance(self.weights, SafetensorWeights):
            raise ValueError(
                "InternVL currently only supports safetensors weights"
            )

        # Get processed state dict for language and vision models.
        # NOTE: use weights_dict to mean WeightData, and state dict to mean
        # DLPack arrays, since state dict is overloaded.
        weights_dict = dict(self.weights.items())
        self._language_weights_dict = (
            convert_internvl_language_model_state_dict(weights_dict)
        )
        self._vision_weights_dict = convert_internvl_vision_model_state_dict(
            weights_dict
        )
        return {}

    def _create_model_config(
        self, state_dict: dict[str, Any]
    ) -> InternVLConfig:
        del state_dict

        internvl_config = InternVLConfig.initialize(
            self.pipeline_config, max_seq_len=self.max_seq_len
        )
        internvl_config.finalize(
            huggingface_config=self.huggingface_config,
            llm_state_dict=self._language_weights_dict,
            vision_state_dict=self._vision_weights_dict,
            dtype=self.dtype,
            return_logits=self.return_logits,
        )
        return internvl_config

    def graph_specs(self, model_config: InternVLConfig) -> dict[str, GraphSpec]:
        """Returns the vision encoder and the decoder, in one compile group."""
        return {
            GraphRole.VISION: self._vision_spec(
                model_config, self._vision_weights_dict
            ),
            GraphRole.DECODER: self._language_spec(
                model_config, self._language_weights_dict
            ),
        }

    def _vision_spec(
        self, config: InternVLConfig, state_dict: dict[str, WeightData]
    ) -> GraphSpec:
        vision_model = InternVLVisionModel(config)
        vision_model.load_state_dict(
            state_dict=state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
            strict=True,
        )
        return GraphSpec(
            vision_model,
            "internvl_vision",
            vision_model.input_types(),
            weights=vision_model.state_dict(),
            group=_COMPILE_GROUP,
        )

    def _language_spec(
        self, config: InternVLConfig, state_dict: dict[str, WeightData]
    ) -> GraphSpec:
        language_model = InternVLLanguageModel(
            config, _get_image_context_token_id(self.huggingface_config)
        )
        language_model.load_state_dict(
            state_dict=state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
        )
        return GraphSpec(
            language_model,
            "internvl_language",
            language_model.input_tree(self.kv_params),
            weights=language_model.state_dict(),
            group=_COMPILE_GROUP,
        )

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Executes the InternVL model with the prepared inputs."""
        assert model_inputs.kv_cache_inputs is not None, (
            "InternVL requires KV cache inputs"
        )
        assert isinstance(model_inputs, InternVLInputs)

        # Process vision inputs if present.
        image_embeddings: list[Buffer]
        image_token_indices: list[Buffer]
        if model_inputs.has_vision_inputs:
            assert self.vision_model is not None
            assert model_inputs.pixel_values is not None
            assert model_inputs.image_token_indices is not None

            # Execute vision model: pixel_values -> image_embeddings.
            vision_outputs = self.vision_model.execute(
                *model_inputs.pixel_values, *model_inputs.signal_buffers
            )
            assert len(vision_outputs) == len(self.devices)

            image_embeddings = [
                output
                for output in vision_outputs
                if isinstance(output, Buffer)
            ]
            image_token_indices = model_inputs.image_token_indices

            assert_image_embeddings_invariant(
                image_embeddings, image_token_indices
            )
        else:
            # Initialize empty tensors for text-only mode.
            assert isinstance(self.batch_processor, InternVLBatchProcessor)
            image_embeddings = self.batch_processor.empty_image_embeddings()
            image_token_indices = (
                self.batch_processor.empty_image_token_indices()
            )

        # Prepare KV cache inputs as list of tensors
        assert model_inputs.kv_cache_inputs

        # Execute language model with text and image embeddings
        language_outputs = self.language_model.execute(
            model_inputs.tokens,
            model_inputs.return_n_logits,
            *model_inputs.input_row_offsets,
            *image_embeddings,
            *image_token_indices,
            *model_inputs.signal_buffers,
            *tree.leaves(model_inputs.kv_cache_inputs),
        )

        # Return model outputs based on what the language model returns
        if len(language_outputs) == 3:
            assert isinstance(language_outputs[0], Buffer)
            assert isinstance(language_outputs[1], Buffer)
            assert isinstance(language_outputs[2], Buffer)
            return ModelOutputs(
                next_token_logits=language_outputs[0],
                logits=language_outputs[1],
                logit_offsets=language_outputs[2],
            )
        else:
            assert isinstance(language_outputs[0], Buffer)
            return ModelOutputs(
                next_token_logits=language_outputs[0],
                logits=language_outputs[0],
            )
