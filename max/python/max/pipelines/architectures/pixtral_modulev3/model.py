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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental import functional as F
from max.experimental.tensor import default_dtype
from max.graph import DeviceRef, TensorType
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

    # Vision inputs - ragged tensor of pre-extracted patches from all images
    pixel_patches: Buffer | None = None
    vision_attention_mask: Buffer | None = None
    vision_position_ids: Buffer | None = None
    image_token_indices: Buffer | None = None

    @property
    def has_vision_inputs(self) -> bool:
        """Returns true iff this includes vision model inputs."""
        return self.pixel_patches is not None


class PixtralModel(PipelineModelWithKVCache[TextAndVisionContext]):
    """The overall interface to the Pixtral model."""

    vision_model: Callable[..., Any]
    """Compiled vision model (encoder + projector) for a ragged batch of images."""

    language_model: Callable[..., Any]
    """Compiled language model with multimodal embedding merge."""

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

        self.vision_model, self.language_model = self._load_models()

    def execute(
        self,
        model_inputs: ModelInputs,
    ) -> ModelOutputs:
        assert isinstance(model_inputs, PixtralInputs)
        assert model_inputs.kv_cache_inputs is not None, (
            "Pixtral has KV cache inputs, but none were provided"
        )

        # Process vision inputs: single call for all images in the batch
        if model_inputs.has_vision_inputs:
            vision_output = self.vision_model(
                model_inputs.pixel_patches,
                model_inputs.vision_attention_mask,
                model_inputs.vision_position_ids,
            )
            image_embeddings = cast(Buffer, vision_output[0].driver_tensor)
            image_token_indices = model_inputs.image_token_indices
        else:
            image_embeddings = self._create_empty_image_embeddings()
            image_token_indices = self._create_empty_indices()

        model_outputs = self.language_model(
            model_inputs.tokens,
            model_inputs.input_row_offsets,
            model_inputs.return_n_logits,
            image_embeddings,
            image_token_indices,
            *model_inputs.kv_cache_inputs.flatten(),
        )

        if len(model_outputs) == 3:
            return ModelOutputs(
                next_token_logits=cast(Buffer, model_outputs[0].driver_tensor),
                logits=cast(Buffer, model_outputs[1].driver_tensor),
                logit_offsets=cast(Buffer, model_outputs[2].driver_tensor),
            )
        else:
            return ModelOutputs(
                next_token_logits=cast(Buffer, model_outputs[0].driver_tensor),
                logits=cast(Buffer, model_outputs[0].driver_tensor),
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

        # Input row offsets: ["input_row_offsets_len"], UInt32
        input_row_offsets = Buffer.from_numpy(
            np.cumsum(
                [0] + [ctx.tokens.active_length for ctx in context_batch],
                dtype=np.uint32,
            )
        ).to(self.devices[0])

        # Input IDs: ["total_seq_len"], Int64 - ragged token vector
        tokens = np.ascontiguousarray(
            np.concatenate([ctx.tokens.active for ctx in context_batch])
        )
        input_ids = Buffer.from_numpy(tokens).to(self.devices[0])

        # Pre-extract patches from all images and build ragged vision inputs
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
                    patches = patches.transpose(
                        1, 3, 0, 2, 4
                    )  # [n_h, n_w, C, p, p]
                    patches = patches.reshape(
                        n_patches, C * patch_size * patch_size
                    )
                    all_patches.append(patches.astype(np.float32))

                    # Position IDs: row * max_patches_per_side + col
                    row_ids = np.repeat(np.arange(n_h), n_w)
                    col_ids = np.tile(np.arange(n_w), n_h)
                    pos_ids = row_ids * max_patches_per_side + col_ids
                    all_position_ids.append(pos_ids.astype(np.int64))
                    patch_counts.append(n_patches)

            # Find image token positions in this context's active tokens
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
            # Concatenate all patches into a ragged tensor
            pixel_patches = Buffer.from_numpy(np.concatenate(all_patches)).to(
                self.devices[0]
            )

            # Position IDs for 2D RoPE
            vision_position_ids = Buffer.from_numpy(
                np.concatenate(all_position_ids)
            ).to(self.devices[0])

            # Block-diagonal attention mask: patches from different images
            # cannot attend to each other.
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

        # Next-token steps have no vision inputs
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

    @traced
    def _load_models(self) -> tuple[Callable[..., Any], Callable[..., Any]]:
        if self.pipeline_config.model.enable_echo:
            raise ValueError(
                "Pixtral model does not currently implement enable echo."
            )

        # Pre-allocate a buffer for input_row_offsets in multistep execution.
        assert self.pipeline_config.runtime.max_batch_size, (
            "Expected max_batch_size to be set"
        )
        self._input_row_offsets_prealloc = Buffer.from_numpy(
            np.arange(
                self.pipeline_config.runtime.max_batch_size + 1, dtype=np.uint32
            )
        ).to(self.devices[0])

        if not isinstance(self.weights, SafetensorWeights):
            raise ValueError(
                "only safetensors weights are currently supported in Pixtral models."
            )

        if len(self.devices) > 1:
            raise NotImplementedError(
                "Pixtral does not support distributed inference"
            )

        # Prepare full state dict then split for vision and language models
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
                # Remap vision_encoder.X -> X since PixtralVision owns
                # these components directly (no VisionEncoder sub-module).
                if k.startswith("vision_encoder."):
                    new_key = k.replace("vision_encoder.", "", 1)
                    # patch_conv is a Tensor param, not a Linear submodule,
                    # so strip the .weight suffix from the key.
                    if new_key == "patch_conv.weight":
                        new_key = "patch_conv"
                    vision_state_dict[new_key] = v
                else:
                    # multi_modal_projector.* stays as-is
                    vision_state_dict[k] = v
            elif k.startswith("language_model."):
                language_state_dict[k] = v

        # Validate that expected vision weight keys are present after remapping.
        expected_vision_prefixes = {
            "patch_conv",
            "layer_norm.",
            "patch_positional_embedding.",
            "transformer.",
            "multi_modal_projector.",
        }
        for key in vision_state_dict:
            if not any(
                key == prefix or key.startswith(prefix)
                for prefix in expected_vision_prefixes
            ):
                logger.warning(
                    "Unexpected vision weight key after remapping: %s", key
                )

        model_config = PixtralConfig.initialize(self.pipeline_config)
        model_config.return_logits = self.return_logits
        device_ref = DeviceRef.from_device(self.devices[0])

        # ---- Build and compile vision model ----
        with F.lazy(), default_dtype(model_config.dtype):
            vision_nn = PixtralVision(model_config)
            vision_nn.to(self.devices[0])

        pixel_patches_type = TensorType(
            DType.float32,
            shape=["total_patches", patch_dim],
            device=DeviceRef.GPU(),
        )
        attention_mask_type = TensorType(
            DType.float32,
            shape=[1, 1, "total_patches", "total_patches"],
            device=DeviceRef.GPU(),
        )
        position_ids_type = TensorType(
            DType.int64,
            shape=["total_patches"],
            device=DeviceRef.GPU(),
        )

        compiled_vision = vision_nn.compile(
            pixel_patches_type,
            attention_mask_type,
            position_ids_type,
            weights=vision_state_dict,
        )

        # ---- Build and compile language model ----
        with F.lazy(), default_dtype(model_config.dtype):
            language_nn = PixtralLanguage(model_config)
            language_nn.kv_params = self.kv_params
            language_nn.to(self.devices[0])

        input_ids_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=DeviceRef.GPU()
        )
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        )
        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )
        image_embeddings_type = TensorType(
            model_config.dtype,
            shape=["total_image_tokens", model_config.hidden_size],
            device=DeviceRef.GPU(),
        )
        image_token_indices_type = TensorType(
            DType.int32,
            shape=["num_image_token_indices"],
            device=device_ref,
        )

        kv_inputs = self.kv_params.get_symbolic_inputs().flatten()

        compiled_language = language_nn.compile(
            input_ids_type,
            input_row_offsets_type,
            return_n_logits_type,
            image_embeddings_type,
            image_token_indices_type,
            *kv_inputs,
            weights=language_state_dict,
        )

        return compiled_vision, compiled_language
