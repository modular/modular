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
from dataclasses import dataclass, field
from typing import Any, ClassVar

from max import tree
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph.buffer_utils import cast_tensors_to
from max.graph.weights import (
    SafetensorWeights,
    WeightData,
    Weights,
    WeightsAdapter,
)
from max.nn.kv_cache import KVCacheInputs
from max.nn.layer import Module
from max.nn.transformer import ReturnLogits
from max.pipelines.lib import (
    AlwaysSignalBuffersMixin,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModelWithKVCache,
)
from max.pipelines.lib.graph_specs import GraphRole, GraphSpec
from max.pipelines.lib.memory_estimation import MemoryPlan

from .batch_processor import Qwen3VLMoeBatchProcessor
from .context import Qwen3VLTextAndVisionContext
from .model_config import Qwen3VLConfig
from .nn.decoder import Qwen3VLMoEDecoder
from .qwen3vl import Qwen3VL
from .weight_adapters import convert_qwen3vl_model_state_dict

logger = logging.getLogger("max.pipelines")

# The vision and language graphs compile into one artifact.
_COMPILE_GROUP = "qwen3vl_moe"


@dataclass
class Qwen3VLInputs(ModelInputs):
    """A class representing inputs for the Qwen3VL model.

    This class encapsulates the input tensors required for the Qwen3VL model execution,
    including both text and vision inputs. Vision inputs are optional and can be None
    for text-only processing.
    """

    tokens: Buffer
    """Tensor containing the input token IDs."""

    input_row_offsets: list[Buffer]
    """Per-device tensors containing the offsets for each row in the ragged input sequence."""

    signal_buffers: list[Buffer]
    """Device buffers used for synchronization in communication collectives."""

    decoder_position_ids: Buffer
    """3D RoPE position IDs for the decoder."""

    return_n_logits: Buffer
    """Number of logits to return, used by speculative decoding for example."""

    kv_cache_inputs: KVCacheInputs[Buffer, Buffer] = field(kw_only=True)
    """KV cache inputs for the model."""

    image_token_indices: list[Buffer] | None = None
    """Per-device pre-computed multimodal merge indices for the image embeddings.

    These are the locations of the image_token_id in the inputs fed to the model.

    Some indices may be negative, which means that they are ignored by the multimodal merge."""

    # Vision inputs.
    pixel_values: list[Buffer] | None = None
    """Pixel values for vision inputs."""

    vision_position_ids: list[Buffer] | None = None
    """Vision rotary position IDs per device."""

    weights: list[Buffer] | None = None
    """Bilinear interpolation weights for vision position embeddings per device."""

    indices: list[Buffer] | None = None
    """Bilinear interpolation indices for vision position embeddings per device."""

    max_grid_size: list[Buffer] | None = None
    """Maximum grid size for vision inputs per device."""

    cu_seqlens: list[Buffer] | None = None
    """Cumulative sequence lengths for full attention per device."""

    max_seqlen: list[Buffer] | None = None
    """Maximum sequence length for full attention for vision inputs per device."""

    grid_thw: list[Buffer] | None = None
    """Grid dimensions (temporal, height, width) for each image/video, shape (n_images, 3) per device."""

    @property
    def has_vision_inputs(self) -> bool:
        """Check if this input contains vision data."""
        return self.pixel_values is not None


class Qwen3VLModel(
    AlwaysSignalBuffersMixin,
    PipelineModelWithKVCache[Qwen3VLTextAndVisionContext, Qwen3VLConfig, Model],
):
    """A Qwen3VL pipeline model for multimodal text generation."""

    model_config_cls: ClassVar[type[Any]] = Qwen3VLConfig
    batch_processor_cls: ClassVar[type[Qwen3VLMoeBatchProcessor]] = (
        Qwen3VLMoeBatchProcessor
    )

    _vision_weights_dict: dict[str, WeightData]
    _language_weights_dict: dict[str, WeightData]

    model_config: Qwen3VLConfig | None
    """The Qwen3VL model configuration."""

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

        self.model_config = None
        self._session = session  # reuse for on-device casts

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
        # Validate SafetensorWeights requirement
        if not isinstance(self.weights, SafetensorWeights):
            raise ValueError(
                "Qwen3VL currently only supports safetensors weights"
            )

        # Get processed state dict
        if self.adapter:
            model_state_dict = self.adapter(
                dict(self.weights.items()),
            )
        else:
            # Use the weight adapter to convert Qwen3VL checkpoint format
            model_state_dict = convert_qwen3vl_model_state_dict(
                dict(self.weights.items())
            )

        vision_state_dict: dict[str, WeightData] = {}
        llm_state_dict: dict[str, WeightData] = {}
        for key, value in model_state_dict.items():
            if key.startswith("vision_encoder."):
                vision_state_dict[key] = value
            elif key.startswith("language_model."):
                llm_state_dict[key] = value
            else:
                raise ValueError(
                    f"Key: {key} is not part of the vision or language model"
                )

        self._vision_weights_dict = vision_state_dict
        self._language_weights_dict = llm_state_dict
        return model_state_dict

    def _create_model_config(
        self, model_state_dict: dict[str, WeightData]
    ) -> Qwen3VLConfig:
        # Generate Qwen3VL config from HuggingFace config
        qwen3vl_config = Qwen3VLConfig.initialize_from_config(
            pipeline_config=self.pipeline_config,
            huggingface_config=self.huggingface_config,
            max_seq_len=self.max_seq_len,
        )
        qwen3vl_config.finalize(
            huggingface_config=self.huggingface_config,
            llm_state_dict=self._language_weights_dict,
            vision_state_dict=self._vision_weights_dict,
            return_logits=self.return_logits,
        )
        self.model_config = qwen3vl_config

        # Use the local non-optional variable to satisfy typing.
        self.model: Module = Qwen3VL(qwen3vl_config)
        self.model.load_state_dict(
            model_state_dict, weight_alignment=1, strict=True
        )
        return qwen3vl_config

    def graph_specs(self, model_config: Qwen3VLConfig) -> dict[str, GraphSpec]:
        """Returns the vision encoder and the decoder, in one compile group.

        Both trace towers of the ``Qwen3VL`` module that
        :meth:`_create_model_config` builds and loads, and bind the weights
        :meth:`_load_state_dict` split between them.
        """
        del model_config
        assert isinstance(self.model, Qwen3VL)
        vision_encoder = self.model.vision_encoder
        language_model = self.model.language_model
        assert isinstance(language_model, Qwen3VLMoEDecoder)
        return {
            GraphRole.VISION: GraphSpec(
                vision_encoder,
                "qwen3vl_vision",
                vision_encoder.input_tree(),
                weights=self._vision_weights_dict,
                group=_COMPILE_GROUP,
            ),
            GraphRole.DECODER: GraphSpec(
                language_model,
                "qwen3vl_moe_language",
                language_model.input_tree(self.kv_params),
                weights=self._language_weights_dict,
                group=_COMPILE_GROUP,
            ),
        }

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Executes the Qwen3VL model with the prepared inputs."""
        assert isinstance(model_inputs, Qwen3VLInputs)
        assert model_inputs.kv_cache_inputs is not None, (
            "Qwen3VL requires KV cache inputs"
        )

        # Process vision inputs if present
        image_embeddings: list[Buffer]
        deepstack_image_embeddings: list[Buffer]
        if model_inputs.has_vision_inputs:
            assert self.vision_model is not None
            assert model_inputs.image_token_indices is not None
            assert model_inputs.pixel_values is not None
            assert model_inputs.vision_position_ids is not None
            assert model_inputs.weights is not None
            assert model_inputs.indices is not None
            assert model_inputs.max_grid_size is not None
            assert model_inputs.cu_seqlens is not None
            assert model_inputs.max_seqlen is not None
            assert model_inputs.grid_thw is not None

            assert self.model_config is not None, (
                "Model config must be initialized"
            )
            n_deepstack_layers = len(
                self.model_config.vision_config.deepstack_visual_indexes
            )
            # Execute vision model: pixel_values -> image_embeddings
            vision_outputs = self.vision_model.execute(
                *model_inputs.pixel_values,
                *model_inputs.weights,
                *model_inputs.indices,
                *model_inputs.vision_position_ids,
                *model_inputs.max_grid_size,
                *model_inputs.grid_thw,
                *model_inputs.cu_seqlens,
                *model_inputs.max_seqlen,
                *model_inputs.signal_buffers,
            )
            assert len(vision_outputs) == len(self.devices) * (
                1 + n_deepstack_layers
            )
            # assert image embeddings and deepstack image embeddings have the same hidden size
            for output in vision_outputs:
                assert isinstance(output, Buffer)
                assert (
                    output.shape[1]
                    == self.huggingface_config.text_config.hidden_size
                )

            # Extract image embeddings (first len(self.devices) outputs)
            n_devices = len(self.devices)
            image_embeddings = [
                output
                for output in vision_outputs[:n_devices]
                if isinstance(output, Buffer)
            ]

            deepstack_image_embeddings = [
                output
                for output in vision_outputs[n_devices:]
                if isinstance(output, Buffer)
            ]

            image_embeddings = cast_tensors_to(
                image_embeddings, self.dtype, self._session
            )
            deepstack_image_embeddings = cast_tensors_to(
                deepstack_image_embeddings, self.dtype, self._session
            )

            image_token_indices = model_inputs.image_token_indices

            # The size of scatter indices must match the number of image embeddings.
            assert (
                image_token_indices[0].shape[0] == image_embeddings[0].shape[0]
            )

            # Normalize index dtypes to match the language graph contract.
            image_token_indices = cast_tensors_to(
                image_token_indices, DType.int32, self._session
            )
        else:
            assert isinstance(self._batch_processor, Qwen3VLMoeBatchProcessor)
            image_embeddings, deepstack_image_embeddings = (
                self._batch_processor.empty_image_embeddings()
            )
            image_token_indices = (
                self._batch_processor.empty_image_token_indices()
            )

        # Prepare KV cache inputs as list of tensors
        assert model_inputs.kv_cache_inputs
        kv_cache_inputs_list = list(tree.leaves(model_inputs.kv_cache_inputs))

        # Execute language model with text and image embeddings and deepstack features
        # deepstack_image_embeddings Structure: [layer0_device0, layer0_device1, ..., layer1_device0, layer1_device1, ...]

        language_outputs = self.language_model.execute(
            model_inputs.tokens,
            model_inputs.return_n_logits,
            *model_inputs.input_row_offsets,
            *image_embeddings,
            *deepstack_image_embeddings,
            *image_token_indices,
            model_inputs.decoder_position_ids,
            *model_inputs.signal_buffers,
            *kv_cache_inputs_list,
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
