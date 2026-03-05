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
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheParams,
)
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
from max.profiler import traced
from transformers import AutoConfig

from .model_config import PixtralConfig
from .pixtral import Pixtral
from .vision_encoder.attention_utils import causal_attention_mask_2d_from_imgs

logger = logging.getLogger("max.pipelines")


@dataclass
class PixtralInputs(ModelInputs):
    """Holds inputs for the Pixtral model."""

    input_ids: Buffer
    input_row_offsets: Buffer
    return_n_logits: Buffer

    # Image inputs
    pixel_values: Buffer
    attention_mask: Buffer

    @property
    def has_vision_inputs(self) -> bool:
        """Returns true iff this includes vision model inputs."""
        return self.pixel_values is not None


class PixtralModel(PipelineModelWithKVCache[TextAndVisionContext]):
    """The overall interface to the Pixtral model."""

    model: Callable[..., Any]
    """Compiled and initialized model ready for inference."""

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

        self.model = self.load_model()

    def execute(
        self,
        model_inputs: ModelInputs,
    ) -> ModelOutputs:
        assert isinstance(model_inputs, PixtralInputs)

        curr_kv_cache_inputs = model_inputs.kv_cache_inputs or ()

        model_inputs = cast(PixtralInputs, model_inputs)
        assert model_inputs.kv_cache_inputs is not None, (
            "Pixtral has KV cache inputs, but none were provided"
        )
        model_outputs = self.model(
            model_inputs.input_ids,
            model_inputs.pixel_values,
            model_inputs.attention_mask,
            model_inputs.input_row_offsets,
            model_inputs.return_n_logits,
            *curr_kv_cache_inputs,
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
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> PixtralInputs:
        if len(replica_batches) > 1:
            raise ValueError("Model does not support DP>1")

        context_batch = replica_batches[0]

        # Input row offset type: ["input_row_offsets_len"], UInt32
        input_row_offsets = Buffer.from_numpy(
            np.cumsum(
                [0] + [ctx.tokens.active_length for ctx in context_batch],
                dtype=np.uint32,
            )
        ).to(self.devices[0])

        # Input Ids: ["total_seq_len"], Int64
        # Create a ragged token vector of length: sum(len(t) for t in tokens).
        tokens = np.ascontiguousarray(
            np.concatenate([ctx.tokens.active for ctx in context_batch])
        )
        input_ids = Buffer.from_numpy(tokens).to(self.devices[0])

        num_images = sum(len(ctx.next_images) for ctx in context_batch)

        # TODO(MODELS-810): Support multiple images per batch
        if num_images > 1:
            raise ValueError(
                "The pixtral implementation currently supports only one image per batch"
            )

        # TODO: change this to work with all contexts in the batch.
        # check if the request has pixel_values
        if context_batch[0].needs_vision_encoding:
            # Get first image in first batch. Pixtral processor returns CHW images.
            next_images = context_batch[0].next_images
            if len(next_images) != 1:
                raise ValueError("Pixtral only supports one image per request")
            image = np.ascontiguousarray(next_images[0].pixel_values)
            pixel_values = Buffer.from_numpy(image).to(self.devices[0])
            # TODO(KERN-782): This should be -inf but softmax saturates with NaNs.
            fill_val = -10000.0
            attention_mask = causal_attention_mask_2d_from_imgs(
                [image],
                self.huggingface_config.vision_config.patch_size,
                1,
                fill_val,
            )
            attention_mask_tensor = Buffer.from_numpy(attention_mask).to(
                self.devices[0]
            )
            return PixtralInputs(
                input_ids=input_ids,
                input_row_offsets=input_row_offsets,
                pixel_values=pixel_values,
                attention_mask=attention_mask_tensor,
                return_n_logits=Buffer.from_numpy(
                    np.array([return_n_logits], dtype=np.int64)
                ),
                kv_cache_inputs=kv_cache_inputs,
            )
        # TODO: return empty tensors for pixel_values and attention_mask
        return PixtralInputs(
            input_ids=input_ids,
            input_row_offsets=input_row_offsets,
            pixel_values=Buffer.zeros(shape=(0, 0, 0), dtype=DType.float32).to(
                self.devices[0]
            ),
            attention_mask=Buffer.zeros(
                shape=(0, 1, 0, 0), dtype=DType.float32
            ).to(self.devices[0]),
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Buffer,
        prev_model_inputs: ModelInputs,
    ) -> PixtralInputs:
        assert isinstance(prev_model_inputs, PixtralInputs)

        # input_ids, old_row_offsets, Optional: [pixel_values, attention_mask]
        old_row_offsets = prev_model_inputs.input_row_offsets

        row_offsets_size = old_row_offsets.shape[0]
        next_row_offsets = self._input_row_offsets_prealloc[:row_offsets_size]
        # In multi-step execution, don't re-pass the pixel_values and attention_mask.
        # TODO: return empty tensors for pixel_values and attention_mask
        return PixtralInputs(
            input_ids=next_tokens,
            input_row_offsets=next_row_offsets,
            pixel_values=Buffer.zeros(shape=(0, 0, 0), dtype=DType.float32).to(
                self.devices[0]
            ),
            attention_mask=Buffer.zeros(
                shape=(0, 1, 0, 0), dtype=DType.float32
            ).to(self.devices[0]),
            kv_cache_inputs=prev_model_inputs.kv_cache_inputs,
            return_n_logits=prev_model_inputs.return_n_logits,
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

    def _get_state_dict(
        self,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
    ) -> dict[str, WeightData]:
        if adapter:
            state_dict = adapter(
                dict(weights.items()),
                huggingface_config=self.huggingface_config,
                pipeline_config=self.pipeline_config,
            )
        else:
            state_dict = {key: value.data() for key, value in weights.items()}
        return state_dict

    @traced
    def load_model(self) -> Callable[..., Any]:
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

        timer = CompilationTimer("model")

        # Prepare state dict
        state_dict = self._get_state_dict(self.weights, self.adapter)

        # Prepare model config
        model_config = PixtralConfig.initialize(self.pipeline_config)
        model_config.return_logits = self.return_logits

        # Generate DeviceRef and input types
        device_ref = DeviceRef.from_device(self.devices[0])

        input_ids_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=DeviceRef.GPU()
        )
        pixel_values_type = TensorType(
            DType.float32,
            shape=["num_channels", "image_height", "image_width"],
            device=DeviceRef.GPU(),
        )
        attention_mask_type = TensorType(
            DType.float32,
            shape=["n_images", 1, "num_patches", "num_patches"],
            device=DeviceRef.GPU(),
        )
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        )
        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )

        kv_inputs = self.kv_params.get_symbolic_inputs()
        flattened_kv_types = [
            kv_type for sublist in kv_inputs for kv_type in sublist
        ]

        # Build model with F.lazy() context
        if len(self.devices) > 1:
            raise NotImplementedError(
                "Pixtral does not support distributed inference"
            )

        with F.lazy(), default_dtype(model_config.dtype):
            nn_model = Pixtral(model_config)
            nn_model.kv_params = self.kv_params
            nn_model.to(self.devices[0])

        timer.mark_build_complete()
        compiled_model = nn_model.compile(
            input_ids_type,
            pixel_values_type,
            attention_mask_type,
            input_row_offsets_type,
            return_n_logits_type,
            *flattened_kv_types,
            weights=state_dict,
        )
        timer.done()

        return compiled_model
