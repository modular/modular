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
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, cast

import numpy as np
from max.driver import Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph.weights import SafetensorWeights, Weights, WeightsAdapter
from max.pipelines import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
    SupportedEncoding,
    TextAndVisionContext,
    upper_bounded_default,
)
from max.pipelines.kv_cache import (
    KVCacheInputs,
    KVCacheManager,
    KVCacheParams,
    estimate_kv_cache_size,
    load_kv_manager,
)
from transformers import AutoConfig

from .model.graph import _build_text_graph, _build_vision_graph
from .model_config import PixtralConfig
from .vision_encoder.attention_utils import causal_attention_mask_2d_from_imgs

logger = logging.getLogger("max.pipelines")


class PixtralInputs(ModelInputs):
    """Holds inputs for the Pixtral model."""

    input_ids: Tensor
    input_row_offsets: Tensor

    # Image inputs
    _pixel_values: Tensor | None = None
    _attention_mask: Tensor | None = None

    def __init__(
        self,
        input_ids: Tensor,
        input_row_offsets: Tensor,
        pixel_values: Tensor | None = None,
        attention_mask: Tensor | None = None,
        kv_cache_inputs: KVCacheInputs | None = None,
    ):
        self.input_ids = input_ids
        self.input_row_offsets = input_row_offsets
        self._pixel_values = pixel_values
        self._attention_mask = attention_mask
        self.kv_cache_inputs = kv_cache_inputs

    @property
    def has_vision_inputs(self) -> bool:
        """Returns true iff this includes vision model inputs."""
        return self._pixel_values is not None

    @property
    def pixel_values(self) -> Tensor:
        assert self._pixel_values is not None
        return self._pixel_values

    @property
    def attention_mask(self) -> Tensor:
        assert self._attention_mask is not None
        return self._attention_mask


class PixtralModel(PipelineModel[TextAndVisionContext]):
    """The overall interface to the Pixtral model."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        huggingface_config: AutoConfig,
        encoding: SupportedEncoding,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: Optional[WeightsAdapter] = None,
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
        )
        self.vision_model, self.language_model = self.load_model(session)
        # Note that in a multimodal model, the language model is the last model in the
        # pipeline. Unfortunately, self.model is still being used (and exposed)
        # in the token generation code, so we still need to set it here.
        self.model = self.language_model

    def execute(
        self,
        model_inputs: ModelInputs,
    ) -> ModelOutputs:
        model_inputs = cast(PixtralInputs, model_inputs)
        if model_inputs.has_vision_inputs:
            image_embeds = self.vision_model.execute(
                model_inputs.pixel_values,
                model_inputs.attention_mask,
                copy_inputs_to_device=False,
            )[0]
        else:
            # batch_size * num_concurrent_media * num_patches are set to 0 here to imitate a dummy tensor (used in text-only mode).
            image_embeds = Tensor.zeros(
                shape=[
                    0,
                    0,
                    self.huggingface_config.text_config.hidden_size,
                ],
                dtype=self.dtype,
            ).to(self.devices[0])
        assert model_inputs.kv_cache_inputs is not None, (
            "Pixtral has KV cache inputs, but none were provided"
        )
        model_outputs = self.language_model.execute(
            model_inputs.input_ids,
            image_embeds,
            model_inputs.input_row_offsets,
            *model_inputs.kv_cache_inputs,
            copy_inputs_to_device=False,
        )
        assert not self.pipeline_config.enable_echo
        assert isinstance(model_outputs[0], Tensor)
        return ModelOutputs(next_token_logits=model_outputs[0])

    def prepare_initial_token_inputs(
        self,
        context_batch: Sequence[TextAndVisionContext],
        kv_cache_inputs: KVCacheInputs | None = None,
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
        if context_batch[
            0
        ].pixel_values:  # check if the request has pixel_values
            # Get first image in first batch and permute the order to (HWC).
            # Pixtral processor returns CHW images.
            image = np.ascontiguousarray(
                np.transpose(context_batch[0].pixel_values[0], (1, 2, 0))
            )
            pixel_values = Tensor.from_numpy(image).to(self.devices[0])
            # TODO(KERN-782): This should be -inf but softmax saturates with NaNs.
            fill_val = -10000.0
            attention_mask = causal_attention_mask_2d_from_imgs(
                [image],
                self.huggingface_config.vision_config.patch_size,
                1,
                fill_val,
            )
            attention_mask = Tensor.from_numpy(attention_mask).to(
                self.devices[0]
            )
            return PixtralInputs(
                input_ids=input_ids,
                input_row_offsets=input_row_offsets,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                kv_cache_inputs=kv_cache_inputs,
            )

        return PixtralInputs(
            input_ids=input_ids,
            input_row_offsets=input_row_offsets,
            kv_cache_inputs=kv_cache_inputs,
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: ModelInputs,
    ) -> PixtralInputs:
        prev_model_inputs = cast(PixtralInputs, prev_model_inputs)
        # input_ids, old_row_offsets, Optional: [pixel_values, attention_mask]
        old_row_offsets = prev_model_inputs.input_row_offsets

        row_offsets_size = old_row_offsets.shape[0]
        next_row_offsets = self._input_row_offsets_prealloc[:row_offsets_size]
        # In multi-step execution, don't re-pass the pixel_values and attention_mask.
        return PixtralInputs(
            input_ids=next_tokens,
            input_row_offsets=next_row_offsets,
            kv_cache_inputs=prev_model_inputs.kv_cache_inputs,
        )

    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        return huggingface_config.text_config.num_hidden_layers

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
            msg = (
                "Unable to infer max_length for Pixtral, the provided "
                f"max_length ({pipeline_config.max_length}) exceeds the "
                f"model's max_position_embeddings "
                f"({huggingface_config.text_config.max_position_embeddings})."
            )
            raise ValueError(msg) from e

    def load_kv_manager(
        self,
        session: InferenceSession,
        available_cache_memory: int,
    ) -> KVCacheManager:
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

    def load_model(
        self,
        session: InferenceSession,
    ) -> tuple[Model, Model]:
        if self.pipeline_config.enable_echo:
            msg = "Pixtral model does not currently implement enable echo."
            raise ValueError(msg)

        # Pre-allocate a buffer for input_row_offsets in multistep execution.
        # We do this to avoid materializing and copying a buffer with each multistep step
        assert self.pipeline_config.max_batch_size, (
            "Expected max_batch_size to be set"
        )
        self._input_row_offsets_prealloc = Tensor.from_numpy(
            np.arange(self.pipeline_config.max_batch_size + 1, dtype=np.uint32)
        ).to(self.devices[0])

        if not isinstance(self.weights, SafetensorWeights):
            msg = (
                "only safetensors weights are currently supported in Pixtral"
                " models."
            )
            raise ValueError(msg)

        def build_and_compile_model(build, label):
            logger.info(f"Building and compiling {label} model...")
            graph = build()
            before = time.perf_counter()
            model = session.load(
                graph,
                weights_registry=self.weights.allocated_weights,
            )
            after = time.perf_counter()
            logger.info(
                f"Building and compiling {label} model took {after - before:.6f} seconds"
            )
            return model

        with ThreadPoolExecutor(max_workers=2) as executor:
            build = lambda: _build_vision_graph(
                pipeline_config=self.pipeline_config,
                weights=self.weights,
                huggingface_config=self.huggingface_config,
                dtype=self.dtype,
            )
            vision_model_future = executor.submit(
                build_and_compile_model, build, "vision"
            )

            assert isinstance(self.weights, SafetensorWeights), (
                "weights provided must be SafetensorWeights"
            )

            build = lambda: _build_text_graph(
                pipeline_config=self.pipeline_config,
                weights=self.weights,
                max_seq_len=self.calculate_max_seq_len(
                    self.pipeline_config,
                    huggingface_config=self.huggingface_config,
                ),
                kv_params=self.get_kv_params(
                    huggingface_config=self.huggingface_config,
                    n_devices=len(self.devices),
                    kv_cache_config=self.kv_cache_config,
                    cache_dtype=self.encoding.cache_dtype,
                ),
                kv_manager=self.kv_manager,
                huggingface_config=self.huggingface_config,
                dtype=self.dtype,
            )
            text_model_future = executor.submit(
                build_and_compile_model, build, "text"
            )

            vision_model = vision_model_future.result()
            text_model = text_model_future.result()

        return vision_model, text_model
