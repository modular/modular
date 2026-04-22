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

import numpy as np
from max.driver import Buffer, Device, DLPackArray
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, DeviceRef, Graph, TensorType
from max.graph.weights import (
    SafetensorWeights,
    WeightData,
    Weights,
    WeightsAdapter,
)
from max.nn.kv_cache import KVCacheInputs, KVCacheParams
from max.nn.transformer import ReturnLogits
from max.pipelines.core import TextAndVisionContext
from max.pipelines.lib import (
    CompilationTimer,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModelWithKVCache,
    upper_bounded_default,
)
from max.pipelines.lib.utils import parse_state_dict_from_weights
from max.profiler import traced
from transformers import AutoConfig

from .model_config import PixtralConfig
from .pixtral import PixtralLanguage, PixtralVision

logger = logging.getLogger("max.pipelines")


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


class PixtralModel(PipelineModelWithKVCache[TextAndVisionContext]):
    """Pixtral pipeline model with separate vision and language graphs."""

    vision_model: Model
    language_model: Model

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
    ) -> None:
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits,
        )

        self.vision_model, self.language_model = self._load_models(session)

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        assert isinstance(model_inputs, PixtralInputs)
        assert model_inputs.kv_cache_inputs is not None, (
            "Pixtral requires KV cache inputs"
        )

        # Process vision inputs if present.
        if model_inputs.has_vision_inputs:
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
            image_embeddings = self._create_empty_image_embeddings()
            image_token_indices = self._create_empty_indices()

        # Execute language model with text and image embeddings.
        language_outputs = self.language_model.execute(
            model_inputs.tokens,
            model_inputs.input_row_offsets,
            model_inputs.return_n_logits,
            image_embeddings,
            image_token_indices,
            *model_inputs.kv_cache_inputs.flatten(),
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

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextAndVisionContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> PixtralInputs:
        if len(replica_batches) > 1:
            raise ValueError("Model does not support DP>1")

        context_batch = replica_batches[0]

        # Input row offsets.
        input_row_offsets = Buffer.from_numpy(
            np.cumsum(
                [0] + [ctx.tokens.active_length for ctx in context_batch],
                dtype=np.uint32,
            )
        ).to(self.devices[0])

        # Ragged token vector.
        tokens = np.ascontiguousarray(
            np.concatenate([ctx.tokens.active for ctx in context_batch])
        )
        input_ids = Buffer.from_numpy(tokens).to(self.devices[0])

        # Pre-extract patches from all images and build ragged vision inputs.
        patch_size = self.huggingface_config.vision_config.patch_size
        image_token_index = self.huggingface_config.image_token_index
        max_patches_per_side = (
            self.huggingface_config.vision_config.image_size // patch_size
        )

        all_patches: list[np.ndarray] = []
        all_position_ids: list[np.ndarray] = []
        patch_counts: list[int] = []
        indices_parts: list[np.ndarray] = []
        batch_offset = 0

        for ctx in context_batch:
            if ctx.needs_vision_encoding:
                for img_data in ctx.next_images:
                    image = np.ascontiguousarray(img_data.pixel_values)
                    C, H, W = image.shape
                    n_h = H // patch_size
                    n_w = W // patch_size
                    n_patches = n_h * n_w

                    # Extract patches: [C, H, W] -> [n_patches, C*p*p]
                    patches = image.reshape(C, n_h, patch_size, n_w, patch_size)
                    patches = patches.transpose(1, 3, 0, 2, 4)
                    patches = patches.reshape(
                        n_patches, C * patch_size * patch_size
                    )
                    all_patches.append(patches.astype(np.float32))

                    # Position IDs for 2D RoPE.
                    row_ids = np.repeat(np.arange(n_h), n_w)
                    col_ids = np.tile(np.arange(n_w), n_h)
                    pos_ids = row_ids * max_patches_per_side + col_ids
                    all_position_ids.append(pos_ids.astype(np.int64))
                    patch_counts.append(n_patches)

            # Find image token positions in this context's active tokens.
            active_tokens = ctx.tokens.active
            image_positions = np.where(active_tokens == image_token_index)[0]
            if len(image_positions) > 0:
                indices_parts.append(
                    (image_positions + batch_offset).astype(np.int32)
                )
            batch_offset += ctx.tokens.active_length

        pixel_patches: Buffer | None = None
        vision_attention_mask: Buffer | None = None
        vision_position_ids: Buffer | None = None
        image_token_indices: Buffer | None = None

        if all_patches:
            pixel_patches = Buffer.from_numpy(np.concatenate(all_patches)).to(
                self.devices[0]
            )

            vision_position_ids = Buffer.from_numpy(
                np.concatenate(all_position_ids)
            ).to(self.devices[0])

            # Block-diagonal attention mask.
            # NOTE: This is a dense N x N mask where N = total patches across
            # all images. For multi-image batches this scales O(n^2) in memory
            # and should be replaced with a sparse or per-image scheme.
            total_patches = sum(patch_counts)
            # TODO(KERN-782): fill_val should be -inf but softmax saturates.
            fill_val = -10000.0
            mask = np.full(
                (1, 1, total_patches, total_patches),
                fill_val,
                dtype=np.float32,
            )
            offset = 0
            for count in patch_counts:
                mask[0, 0, offset : offset + count, offset : offset + count] = (
                    0.0
                )
                offset += count
            vision_attention_mask = Buffer.from_numpy(mask).to(self.devices[0])

        if indices_parts:
            image_token_indices = Buffer.from_numpy(
                np.concatenate(indices_parts)
            ).to(self.devices[0])

        return PixtralInputs(
            tokens=input_ids,
            input_row_offsets=input_row_offsets,
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            pixel_patches=pixel_patches,
            vision_attention_mask=vision_attention_mask,
            vision_position_ids=vision_position_ids,
            image_token_indices=image_token_indices,
            kv_cache_inputs=kv_cache_inputs,
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Buffer,
        prev_model_inputs: ModelInputs,
    ) -> PixtralInputs:
        assert isinstance(prev_model_inputs, PixtralInputs)

        old_row_offsets = prev_model_inputs.input_row_offsets
        row_offsets_size = old_row_offsets.shape[0]
        next_row_offsets = self._input_row_offsets_prealloc[:row_offsets_size]

        # Next-token steps have no vision inputs.
        return PixtralInputs(
            tokens=next_tokens,
            input_row_offsets=next_row_offsets,
            return_n_logits=prev_model_inputs.return_n_logits,
            kv_cache_inputs=prev_model_inputs.kv_cache_inputs,
        )

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return PixtralConfig.construct_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

    @classmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        try:
            return upper_bounded_default(
                upper_bound=huggingface_config.text_config.max_position_embeddings,
                default=pipeline_config.model.max_length,
            )
        except ValueError as e:
            raise ValueError(
                "Unable to infer max_length for Pixtral, the provided "
                f"max_length ({pipeline_config.model.max_length}) exceeds the "
                f"model's max_position_embeddings "
                f"({huggingface_config.text_config.max_position_embeddings})."
            ) from e

    def _create_empty_image_embeddings(self) -> Buffer:
        return Buffer.zeros(
            shape=[0, self.huggingface_config.text_config.hidden_size],
            dtype=self.dtype,
        ).to(self.devices[0])

    def _create_empty_indices(self) -> Buffer:
        return Buffer.zeros(shape=[0], dtype=DType.int32).to(self.devices[0])

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

    def _language_graph_input_types(self) -> Sequence[TensorType | BufferType]:
        device_ref = DeviceRef.from_device(self.devices[0])
        return (
            TensorType(DType.int64, shape=["total_seq_len"], device=device_ref),
            TensorType(
                DType.uint32,
                shape=["input_row_offsets_len"],
                device=device_ref,
            ),
            TensorType(
                DType.int64,
                shape=["return_n_logits"],
                device=DeviceRef.CPU(),
            ),
            TensorType(
                self.dtype,
                shape=[
                    "num_image_tokens",
                    self.huggingface_config.text_config.hidden_size,
                ],
                device=device_ref,
            ),
            TensorType(
                DType.int32,
                shape=["total_image_tokens"],
                device=device_ref,
            ),
            *self.kv_params.get_symbolic_inputs().flatten(),
        )

    @traced
    def _build_vision_graph(
        self,
        config: PixtralConfig,
        state_dict: dict[str, WeightData],
        patch_dim: int,
    ) -> tuple[Graph, dict[str, DLPackArray]]:
        with Graph(
            "pixtral_vision",
            input_types=self._vision_graph_input_types(patch_dim),
        ) as graph:
            vision_nn = PixtralVision(config)
            vision_nn.load_state_dict(
                state_dict, weight_alignment=1, strict=True
            )

            pixel_patches, attention_mask, position_ids = graph.inputs
            output = vision_nn(
                pixel_patches.tensor,
                attention_mask.tensor,
                position_ids.tensor,
            )
            graph.output(output)
            return graph, vision_nn.state_dict()

    @traced
    def _build_language_graph(
        self,
        config: PixtralConfig,
        state_dict: dict[str, WeightData],
    ) -> tuple[Graph, dict[str, DLPackArray]]:
        with Graph(
            "pixtral_language",
            input_types=self._language_graph_input_types(),
        ) as graph:
            language_nn = PixtralLanguage(config)
            language_nn.load_state_dict(
                state_dict,
                override_quantization_encoding=True,
                weight_alignment=1,
                strict=True,
            )

            (
                tokens,
                input_row_offsets,
                return_n_logits,
                image_embeddings,
                image_token_indices,
                *kv_cache_inputs,
            ) = graph.inputs

            kv_collections = self._unflatten_kv_inputs(kv_cache_inputs)
            outputs = language_nn(
                tokens=tokens.tensor,
                kv_collection=kv_collections[0],
                return_n_logits=return_n_logits.tensor,
                input_row_offsets=input_row_offsets.tensor,
                image_embeddings=image_embeddings.tensor,
                image_token_indices=image_token_indices.tensor,
            )
            graph.output(*outputs)
            return graph, language_nn.state_dict()

    @traced
    def _load_models(self, session: InferenceSession) -> tuple[Model, Model]:
        if self.pipeline_config.model.enable_echo:
            raise ValueError(
                "Pixtral model does not currently implement enable echo."
            )

        assert self.pipeline_config.runtime.max_batch_size, (
            "Expected max_batch_size to be set"
        )
        self._input_row_offsets_prealloc = Buffer.from_numpy(
            np.arange(
                self.pipeline_config.runtime.max_batch_size + 1,
                dtype=np.uint32,
            )
        ).to(self.devices[0])

        if not isinstance(self.weights, SafetensorWeights):
            raise ValueError(
                "Only safetensors weights are currently supported in Pixtral."
            )

        if len(self.devices) > 1:
            raise NotImplementedError(
                "Pixtral does not support distributed inference"
            )

        # Split full state dict into vision and language parts.
        state_dict = parse_state_dict_from_weights(
            self.pipeline_config, self.weights, self.adapter
        )

        vision_config = self.huggingface_config.vision_config
        patch_dim = (
            vision_config.num_channels
            * vision_config.patch_size
            * vision_config.patch_size
        )

        vision_state_dict: dict[str, WeightData] = {}
        language_state_dict: dict[str, WeightData] = {}
        for k, v in state_dict.items():
            if k.startswith("vision_encoder.") or k.startswith(
                "multi_modal_projector."
            ):
                if k.startswith("vision_encoder."):
                    new_key = k.replace("vision_encoder.", "", 1)
                    vision_state_dict[new_key] = v
                else:
                    vision_state_dict[k] = v
            elif k.startswith("language_model."):
                language_state_dict[k] = v

        model_config = PixtralConfig.initialize(self.pipeline_config)
        model_config.return_logits = self.return_logits

        # Build and compile vision model.
        with CompilationTimer("vision model") as timer:
            vision_graph, vision_weights = self._build_vision_graph(
                model_config, vision_state_dict, patch_dim
            )
            timer.mark_build_complete()
            vision_model = session.load(
                vision_graph, weights_registry=vision_weights
            )

        # Build and compile language model.
        with CompilationTimer("language model") as timer:
            language_graph, language_weights = self._build_language_graph(
                model_config, language_state_dict
            )
            timer.mark_build_complete()
            language_model = session.load(
                language_graph, weights_registry=language_weights
            )

        return vision_model, language_model
