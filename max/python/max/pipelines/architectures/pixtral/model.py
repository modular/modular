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
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, DeviceRef, TensorType
from max.graph.weights import (
    SafetensorWeights,
    WeightData,
    Weights,
    WeightsAdapter,
)
from max.nn.transformer import ReturnLogits
from max.pipelines.context import TextAndVisionContext
from max.pipelines.lib import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModelWithKVCache,
)
from max.pipelines.lib.graph_specs import GraphRole, GraphSpec
from max.pipelines.lib.memory_estimation import MemoryPlan
from max.pipelines.lib.utils import parse_state_dict_from_weights

from .batch_processor import PixtralBatchProcessor
from .model_config import PixtralConfig
from .pixtral import PixtralLanguage, PixtralVision

logger = logging.getLogger("max.pipelines")

# The vision and language graphs compile into one artifact.
_COMPILE_GROUP = "pixtral"


@dataclass
class PixtralInputs(ModelInputs):
    """Holds inputs for the Pixtral model."""

    tokens: Buffer
    input_row_offsets: Buffer
    return_n_logits: Buffer

    # Vision inputs — ragged tensor of pre-extracted patches from all images.
    pixel_patches: Buffer | None = None
    vision_attention_mask: Buffer | None = None
    vision_position_ids: Buffer | None = None
    image_token_indices: Buffer | None = None

    @property
    def has_vision_inputs(self) -> bool:
        return self.pixel_patches is not None


class PixtralModel(
    PipelineModelWithKVCache[TextAndVisionContext, PixtralConfig, Model]
):
    """Pixtral pipeline model with separate vision and language graphs."""

    model_config_cls: ClassVar[type[Any]] = PixtralConfig
    batch_processor_cls: ClassVar[type[PixtralBatchProcessor]] = (
        PixtralBatchProcessor
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
            memory_plan=memory_plan,
            max_batch_size=max_batch_size,
        )

        if self.pipeline_config.model.enable_echo:
            raise ValueError(
                "Pixtral model does not currently implement enable echo."
            )

        assert self.max_batch_size, "Expected max_batch_size to be set"

        if len(self.devices) > 1:
            raise NotImplementedError(
                "Pixtral does not support distributed inference"
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

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        assert isinstance(model_inputs, PixtralInputs)
        assert model_inputs.kv_cache_inputs is not None, (
            "Pixtral requires KV cache inputs"
        )

        # Process vision inputs if present.
        if model_inputs.has_vision_inputs:
            assert self.vision_model is not None
            assert model_inputs.pixel_patches is not None
            assert model_inputs.vision_attention_mask is not None
            assert model_inputs.vision_position_ids is not None
            assert model_inputs.image_token_indices is not None

            vision_outputs = self.vision_model.execute(
                model_inputs.pixel_patches,
                model_inputs.vision_attention_mask,
                model_inputs.vision_position_ids,
            )
            assert isinstance(vision_outputs[0], Buffer)
            image_embeddings = vision_outputs[0]
            image_token_indices = model_inputs.image_token_indices
        else:
            assert isinstance(self.batch_processor, PixtralBatchProcessor)
            image_embeddings = self.batch_processor.empty_image_embeddings()
            image_token_indices = (
                self.batch_processor.empty_image_token_indices()
            )

        # Execute language model with text and image embeddings.
        language_outputs = self.language_model.execute(
            model_inputs.tokens,
            model_inputs.input_row_offsets,
            model_inputs.return_n_logits,
            image_embeddings,
            image_token_indices,
            *tree.leaves(model_inputs.kv_cache_inputs),
        )

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

    def _vision_graph_input_types(
        self, patch_dim: int
    ) -> Sequence[TensorType | BufferType]:
        return (
            TensorType(
                DType.float32,
                shape=["total_patches", patch_dim],
                device=DeviceRef.GPU(),
            ),
            TensorType(
                DType.float32,
                shape=[1, 1, "total_patches", "total_patches"],
                device=DeviceRef.GPU(),
            ),
            TensorType(
                DType.int64,
                shape=["total_patches"],
                device=DeviceRef.GPU(),
            ),
        )

    def graph_specs(self, model_config: PixtralConfig) -> dict[str, GraphSpec]:
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
        self, config: PixtralConfig, state_dict: dict[str, WeightData]
    ) -> GraphSpec:
        vision_nn = PixtralVision(config)
        vision_nn.load_state_dict(state_dict, weight_alignment=1, strict=True)
        return GraphSpec(
            vision_nn,
            "pixtral_vision",
            self._vision_graph_input_types(self._patch_dim),
            weights=vision_nn.state_dict(),
            group=_COMPILE_GROUP,
        )

    def _language_spec(
        self, config: PixtralConfig, state_dict: dict[str, WeightData]
    ) -> GraphSpec:
        language_nn = PixtralLanguage(config)
        language_nn.load_state_dict(
            state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
            strict=True,
        )
        return GraphSpec(
            language_nn,
            "pixtral_language",
            language_nn.input_tree(self.kv_params),
            weights=language_nn.state_dict(),
            group=_COMPILE_GROUP,
        )

    def _load_state_dict(self) -> dict[str, Any]:
        if not isinstance(self.weights, SafetensorWeights):
            raise ValueError(
                "Only safetensors weights are currently supported in Pixtral."
            )
        state_dict = parse_state_dict_from_weights(
            self.pipeline_config, self.weights, self.adapter
        )

        vision_state_dict: dict[str, WeightData] = {}
        language_state_dict: dict[str, WeightData] = {}
        for k, v in state_dict.items():
            if k.startswith(("vision_encoder.", "multi_modal_projector.")):
                if k.startswith("vision_encoder."):
                    new_key = k.replace("vision_encoder.", "", 1)
                    vision_state_dict[new_key] = v
                else:
                    vision_state_dict[k] = v
            elif k.startswith("language_model."):
                language_state_dict[k] = v

        self._vision_weights_dict = vision_state_dict
        self._language_weights_dict = language_state_dict
        return state_dict

    def _create_model_config(
        self, state_dict: dict[str, WeightData]
    ) -> PixtralConfig:
        del state_dict
        vision_config = self.huggingface_config.vision_config
        self._patch_dim = (
            vision_config.num_channels
            * vision_config.patch_size
            * vision_config.patch_size
        )

        model_config = PixtralConfig.initialize(
            self.pipeline_config, max_seq_len=self.max_seq_len
        )
        model_config.return_logits = self.return_logits
        return model_config
