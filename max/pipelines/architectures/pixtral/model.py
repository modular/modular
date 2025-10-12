# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
import math
import time
from collections.abc import Sequence
from typing import cast

import numpy as np
from max.driver import Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType
from max.graph.weights import (
    SafetensorWeights,
    WeightData,
    Weights,
    WeightsAdapter,
)
from max.nn import Module, ReturnLogits
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheParams,
    PagedCacheValues,
    PagedKVCacheManager,
    estimate_kv_cache_size,
    load_kv_manager,
)
from max.pipelines.core import TextAndVisionContext
from max.pipelines.lib import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
    SupportedEncoding,
    upper_bounded_default,
)
from max.profiler import traced
from transformers import AutoConfig

from .model_config import PixtralConfig
from .pixtral import Pixtral
from .vision_encoder.attention_utils import causal_attention_mask_2d_from_imgs

logger = logging.getLogger("max.pipelines")


# TODO(GEX-2071): Re-enable when parallel compilation works.
_DO_PARALLEL_COMPILATION = False


class PixtralInputs(ModelInputs):
    """Holds inputs for the Pixtral model."""

    input_ids: Tensor
    input_row_offsets: Tensor
    return_n_logits: Tensor

    # Image inputs
    _pixel_values: Tensor
    _attention_mask: Tensor

    def __init__(
        self,
        input_ids: Tensor,
        input_row_offsets: Tensor,
        return_n_logits: Tensor,
        pixel_values: Tensor,
        attention_mask: Tensor,
        kv_cache_inputs: KVCacheInputs | None = None,
    ) -> None:
        self.input_ids = input_ids
        self.input_row_offsets = input_row_offsets
        self.return_n_logits = return_n_logits
        self._pixel_values = pixel_values
        self._attention_mask = attention_mask
        self.kv_cache_inputs = kv_cache_inputs

    @property
    def has_vision_inputs(self) -> bool:
        """Returns true iff this includes vision model inputs."""
        return self._pixel_values is not None

    @property
    def pixel_values(self) -> Tensor:
        return self._pixel_values

    @property
    def attention_mask(self) -> Tensor:
        return self._attention_mask


class PixtralModel(PipelineModel[TextAndVisionContext]):
    """The overall interface to the Pixtral model."""

    model: Model
    """Compiled and initialized model ready for inference."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        huggingface_config: AutoConfig,
        encoding: SupportedEncoding,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
    ) -> None:
        super().__init__(
            pipeline_config,
            session,
            huggingface_config,
            encoding,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits,
        )

        self.model = self.load_model(session)

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
        model_outputs = self.model.execute(
            model_inputs.input_ids,
            model_inputs.pixel_values,
            model_inputs.attention_mask,
            model_inputs.input_row_offsets,
            model_inputs.return_n_logits,
            *curr_kv_cache_inputs,
        )
        if len(model_outputs) == 3:
            assert isinstance(model_outputs[0], Tensor)
            assert isinstance(model_outputs[1], Tensor)
            assert isinstance(model_outputs[2], Tensor)
            return ModelOutputs(
                next_token_logits=model_outputs[0],
                logits=model_outputs[1],
                logit_offsets=model_outputs[2],
            )
        else:
            assert isinstance(model_outputs[0], Tensor)
            return ModelOutputs(
                next_token_logits=model_outputs[0],
                logits=model_outputs[0],
            )

    def prepare_initial_token_inputs(
        self,
        context_batch: Sequence[TextAndVisionContext],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> PixtralInputs:
        # Input row offset type: ["input_row_offsets_len"], UInt32
        input_row_offsets = Tensor.from_numpy(
            np.cumsum(
                [0] + [ctx.active_length for ctx in context_batch],
                dtype=np.uint32,
            )
        ).to(self.devices[0])

        # Input Ids: ["total_seq_len"], Int64
        # Create a ragged token vector of length: sum(len(t) for t in tokens).
        tokens = np.ascontiguousarray(
            np.concatenate([ctx.next_tokens for ctx in context_batch])
        )
        input_ids = Tensor.from_numpy(tokens).to(self.devices[0])

        # TODO: change this to work with all contexts in the batch.
        # check if the request has pixel_values
        if context_batch[0].needs_vision_encoding:
            # Get first image in first batch. Pixtral processor returns CHW images.
            next_images = context_batch[0].next_images
            if len(next_images) != 1:
                raise ValueError("Pixtral only supports one image per request")
            image = np.ascontiguousarray(next_images[0].pixel_values)
            pixel_values = Tensor.from_numpy(image).to(self.devices[0])
            # TODO(KERN-782): This should be -inf but softmax saturates with NaNs.
            fill_val = -10000.0
            attention_mask = causal_attention_mask_2d_from_imgs(
                [image],
                self.huggingface_config.vision_config.patch_size,
                1,
                fill_val,
            )
            attention_mask_tensor = Tensor.from_numpy(attention_mask).to(
                self.devices[0]
            )
            return PixtralInputs(
                input_ids=input_ids,
                input_row_offsets=input_row_offsets,
                pixel_values=pixel_values,
                attention_mask=attention_mask_tensor,
                return_n_logits=Tensor.from_numpy(
                    np.array([return_n_logits], dtype=np.int64)
                ),
                kv_cache_inputs=kv_cache_inputs,
            )
        # TODO: return empty tensors for pixel_values and attention_mask
        return PixtralInputs(
            input_ids=input_ids,
            input_row_offsets=input_row_offsets,
            pixel_values=Tensor.zeros(shape=(0, 0, 0), dtype=DType.float32).to(
                self.devices[0]
            ),
            attention_mask=Tensor.zeros(
                shape=(0, 1, 0, 0), dtype=DType.float32
            ).to(self.devices[0]),
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=Tensor.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Tensor,
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
            pixel_values=Tensor.zeros(shape=(0, 0, 0), dtype=DType.float32).to(
                self.devices[0]
            ),
            attention_mask=Tensor.zeros(
                shape=(0, 1, 0, 0), dtype=DType.float32
            ).to(self.devices[0]),
            kv_cache_inputs=prev_model_inputs.kv_cache_inputs,
            return_n_logits=prev_model_inputs.return_n_logits,
        )

    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        return PixtralConfig.get_num_layers(huggingface_config)

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return PixtralConfig.get_kv_params(
            huggingface_config=huggingface_config,
            n_devices=n_devices,
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
                default=pipeline_config.max_length,
            )
        except ValueError as e:
            raise ValueError(
                "Unable to infer max_length for Pixtral, the provided "
                f"max_length ({pipeline_config.max_length}) exceeds the "
                f"model's max_position_embeddings "
                f"({huggingface_config.text_config.max_position_embeddings})."
            ) from e

    def load_kv_manager(
        self,
        session: InferenceSession,
        available_cache_memory: int,
    ) -> PagedKVCacheManager:
        return load_kv_manager(
            params=self.get_kv_params(
                huggingface_config=self.huggingface_config,
                n_devices=len(self.devices),
                kv_cache_config=self.kv_cache_config,
                cache_dtype=self.encoding.cache_dtype,
            ),
            max_batch_size=self.pipeline_config.max_batch_size,
            max_seq_len=self.calculate_max_seq_len(
                self.pipeline_config, huggingface_config=self.huggingface_config
            ),
            num_layers=self.get_num_layers(
                huggingface_config=self.huggingface_config
            ),
            devices=self.devices,
            available_cache_memory=available_cache_memory,
            page_size=self.kv_cache_config.kv_cache_page_size,
            session=session,
        )

    @classmethod
    def estimate_kv_cache_size(
        cls,
        pipeline_config: PipelineConfig,
        available_cache_memory: int,
        devices: list[Device],
        huggingface_config: AutoConfig,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> int:
        """Estimates the size of the kv cache in bytes."""
        return estimate_kv_cache_size(
            params=cls.get_kv_params(
                huggingface_config=huggingface_config,
                n_devices=len(devices),
                kv_cache_config=kv_cache_config,
                cache_dtype=cache_dtype,
            ),
            max_batch_size=pipeline_config.max_batch_size,
            max_seq_len=cls.calculate_max_seq_len(
                pipeline_config, huggingface_config=huggingface_config
            ),
            num_layers=cls.get_num_layers(
                huggingface_config=huggingface_config
            ),
            available_cache_memory=available_cache_memory,
            devices=devices,
        )

    def _get_state_dict(
        self,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
    ) -> dict[str, WeightData]:
        pipeline_config = self.pipeline_config
        huggingface_config = self.huggingface_config
        if self.adapter:
            state_dict = self.adapter(
                dict(self.weights.items()),
                huggingface_config=huggingface_config,
                pipeline_config=self.pipeline_config,
            )
        else:
            state_dict = {
                key: value.data() for key, value in self.weights.items()
            }
        return state_dict

    def graph_inputs(self) -> tuple[TensorType]:
        # Generate DeviceRef
        device_ref = DeviceRef.from_device(self.devices[0])

        # Construct general input types
        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )

        kv_inputs = self.kv_manager.input_symbols()

        input_ids_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=DeviceRef.GPU()
        )
        # TODO: should be changed to add "batch_size", "n_images" dims when working with multiple images
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

        return (
            input_ids_type,
            pixel_values_type,
            attention_mask_type,
            input_row_offsets_type,
            return_n_logits_type,
            *kv_inputs[0],
        )

    @traced
    def _build_graph(
        self, weights: Weights, adapter: WeightsAdapter | None = None
    ) -> Graph:
        # Retrieve config
        state_dict = self._get_state_dict(weights, adapter)

        kv_params = PixtralConfig.get_kv_params(
            huggingface_config=self.huggingface_config,
            n_devices=len(self.devices),
            kv_cache_config=self.kv_cache_config,
            cache_dtype=self.encoding.cache_dtype,
        )
        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in self.pipeline_config.model_config.device_specs
        ]
        model_config = PixtralConfig(
            image_token_index=self.huggingface_config.image_token_index,
            vocab_size=self.huggingface_config.text_config.vocab_size,
            hidden_size=self.huggingface_config.text_config.hidden_size,
            feed_forward_length=self.huggingface_config.text_config.intermediate_size,
            num_hidden_layers=self.huggingface_config.text_config.num_hidden_layers,
            num_attention_heads=self.huggingface_config.text_config.num_attention_heads,
            num_key_value_heads=self.huggingface_config.text_config.num_key_value_heads,
            head_dim=self.huggingface_config.text_config.head_dim,
            rms_norm_eps=self.huggingface_config.text_config.rms_norm_eps,
            rope_theta=self.huggingface_config.text_config.rope_theta,
            max_seq_len=self.huggingface_config.text_config.max_position_embeddings,
            patch_size=self.huggingface_config.vision_config.patch_size,
            image_size=self.huggingface_config.vision_config.image_size,
            num_channels=self.huggingface_config.vision_config.num_channels,
            vision_hidden_size=self.huggingface_config.vision_config.hidden_size,
            attention_multiplier=math.sqrt(1 / kv_params.head_dim),
            vision_num_attention_heads=self.huggingface_config.vision_config.num_attention_heads,
            vision_rope_theta=self.huggingface_config.vision_config.rope_theta,
            vision_num_hidden_layers=self.huggingface_config.vision_config.num_hidden_layers,
            vision_intermediate_size=self.huggingface_config.vision_config.intermediate_size,
            vision_head_dim=self.huggingface_config.vision_config.head_dim,
            dtype=self.dtype,
            devices=device_refs,
            return_logits=self.return_logits,
            kv_params=kv_params,
        )

        # Get Graph Inputs
        graph_inputs = self.graph_inputs()

        # Build Graph
        nn_model: Module
        if len(self.devices) > 1:
            raise NotImplementedError(
                "Pixtral does not support distributed inference"
            )

        else:
            nn_model = Pixtral(model_config)
            nn_model.load_state_dict(
                state_dict,
                weight_alignment=1,
                strict=True,
            )
            self.state_dict = nn_model.state_dict()

            with Graph("mistral", input_types=graph_inputs) as graph:
                (
                    input_ids,
                    pixel_values,
                    attention_mask,
                    input_row_offsets,
                    return_n_logits,
                    *kv_cache_inputs,
                ) = graph.inputs
                kv_collection = PagedCacheValues(
                    kv_blocks=kv_cache_inputs[0].buffer,
                    cache_lengths=kv_cache_inputs[1].tensor,
                    lookup_table=kv_cache_inputs[2].tensor,
                    max_lengths=kv_cache_inputs[3].tensor,
                )
                outputs = nn_model(
                    input_ids=input_ids.tensor,
                    pixel_values=pixel_values.tensor,
                    attention_mask=attention_mask.tensor,
                    kv_collection=kv_collection,
                    return_n_logits=return_n_logits.tensor,
                    input_row_offsets=input_row_offsets.tensor,
                )
                graph.output(*outputs)
                return graph

    @traced
    def load_model(
        self,
        session: InferenceSession,
    ) -> Model:
        if self.pipeline_config.enable_echo:
            raise ValueError(
                "Pixtral model does not currently implement enable echo."
            )

        # Pre-allocate a buffer for input_row_offsets in multistep execution.
        # We do this to avoid materializing and copying a buffer with each multistep step
        assert self.pipeline_config.max_batch_size, (
            "Expected max_batch_size to be set"
        )
        self._input_row_offsets_prealloc = Tensor.from_numpy(
            np.arange(self.pipeline_config.max_batch_size + 1, dtype=np.uint32)
        ).to(self.devices[0])

        if not isinstance(self.weights, SafetensorWeights):
            raise ValueError(
                "only safetensors weights are currently supported in Pixtral models."
            )

        logger.info("Building and compiling model...")
        before = time.perf_counter()
        graph = self._build_graph(self.weights, self.adapter)
        after_build = time.perf_counter()

        logger.info(f"Building graph took {after_build - before:.6f} seconds")

        before_compile = time.perf_counter()
        model = session.load(graph, weights_registry=self.state_dict)
        after = time.perf_counter()

        logger.info(
            f"Compiling model took {after - before_compile:.6f} seconds"
        )

        logger.info(
            f"Building and compiling model took {after - before:.6f} seconds"
        )

        return model
