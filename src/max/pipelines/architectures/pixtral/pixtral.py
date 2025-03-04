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
from typing import cast

import numpy as np
from max.driver import Device, Tensor
from max.engine import InferenceSession, Model
from max.graph.weights import SafetensorWeights
from max.pipelines import (
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
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

from .model.graph import _build_text_graph, _build_vision_graph
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
        self, pipeline_config: PipelineConfig, session: InferenceSession
    ) -> None:
        super().__init__(pipeline_config, session)
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
                    self.pipeline_config.huggingface_config.text_config.hidden_size,
                ],
                dtype=self.pipeline_config.dtype,
            ).to(self.pipeline_config.devices[0])
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
        ).to(self.pipeline_config.devices[0])

        # Input Ids: ["total_seq_len"], Int64
        # Create a ragged token vector of length: sum(len(t) for t in tokens).
        tokens = np.ascontiguousarray(
            np.concatenate([ctx.next_tokens for ctx in context_batch])
        )
        input_ids = Tensor.from_numpy(tokens).to(
            self.pipeline_config.devices[0]
        )

        # TODO: change this to work with all contexts in the batch.

        if context_batch[
            0
        ].pixel_values:  # check if the request has pixel_values
            # Get first image in first batch and permute the order to (HWC).
            # Pixtral processor returns CHW images.
            image = np.ascontiguousarray(
                np.transpose(context_batch[0].pixel_values[0], (1, 2, 0))
            )
            pixel_values = Tensor.from_numpy(image).to(
                self.pipeline_config.devices[0]
            )
            # TODO(KERN-782): This should be -inf but softmax saturates with NaNs.
            fill_val = -10000.0
            attention_mask = causal_attention_mask_2d_from_imgs(
                [image],
                self.pipeline_config.huggingface_config.vision_config.patch_size,
                1,
                fill_val,
            )
            attention_mask = Tensor.from_numpy(attention_mask).to(
                self.pipeline_config.devices[0]
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
    def get_num_layers(cls, pipeline_config: PipelineConfig) -> int:
        return pipeline_config.huggingface_config.text_config.num_hidden_layers

    @classmethod
    def get_kv_params(cls, pipeline_config: PipelineConfig) -> KVCacheParams:
        return KVCacheParams(
            page_size=pipeline_config.kv_cache_config.kv_cache_page_size,
            dtype=pipeline_config.cache_dtype,
            n_kv_heads=pipeline_config.huggingface_config.text_config.num_key_value_heads,
            head_dim=pipeline_config.huggingface_config.text_config.head_dim,
            cache_strategy=pipeline_config.kv_cache_config.cache_strategy,
            enable_prefix_caching=pipeline_config.kv_cache_config.enable_prefix_caching,
        )

    @classmethod
    def calculate_max_seq_len(cls, pipeline_config: PipelineConfig) -> int:
        try:
            return upper_bounded_default(
                upper_bound=pipeline_config.huggingface_config.text_config.max_position_embeddings,
                default=pipeline_config.max_length,
            )
        except ValueError as e:
            msg = (
                "Unable to infer max_length for Pixtral, the provided "
                f"max_length ({pipeline_config.max_length}) exceeds the "
                f"model's max_position_embeddings "
                f"({pipeline_config.huggingface_config.text_config.max_position_embeddings})."
            )
            raise ValueError(msg) from e

    def load_kv_manager(
        self,
        session: InferenceSession,
        available_cache_memory: int,
    ) -> KVCacheManager:
        return load_kv_manager(
            params=self.get_kv_params(self.pipeline_config),
            max_batch_size=self.pipeline_config.max_batch_size,
            max_seq_len=self.calculate_max_seq_len(self.pipeline_config),
            num_layers=self.get_num_layers(self.pipeline_config),
            devices=self.pipeline_config.devices,
            available_cache_memory=available_cache_memory,
            page_size=self.pipeline_config.kv_cache_config.kv_cache_page_size,
            session=session,
        )

    @classmethod
    def estimate_kv_cache_size(
        cls,
        pipeline_config: PipelineConfig,
        available_cache_memory: int,
        devices: list[Device],
    ) -> int:
        """Estimates the size of the kv cache in bytes."""
        return estimate_kv_cache_size(
            params=cls.get_kv_params(pipeline_config),
            max_batch_size=pipeline_config.max_batch_size,
            max_seq_len=cls.calculate_max_seq_len(pipeline_config),
            num_layers=cls.get_num_layers(pipeline_config),
            available_cache_memory=available_cache_memory,
            devices=devices,
        )

    def load_model(self, session: InferenceSession) -> tuple[Model, Model]:
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
        ).to(self.pipeline_config.devices[0])

        weights = self.pipeline_config.load_weights()

        if not isinstance(weights, SafetensorWeights):
            msg = (
                "only safetensors weights are currently supported in Pixtral"
                " models."
            )
            raise ValueError(msg)

        self._weights = weights

        if serialized_path := self.pipeline_config.serialized_model_path:
            # Hydrate all weights to be referenced by the serialized path.
            weights_registry = {}
            for name, weight in self._weights.items():
                weights_registry[name] = weight.raw_tensor()

            def serialized_load(serialized_path):
                logger.info("Loading serialized model from %s", serialized_path)
                model = session.load(
                    f"{serialized_path}", weights_registry=weights_registry
                )
                return model

            vision_model = serialized_load(f"{serialized_path}.vision")
            text_model = serialized_load(f"{serialized_path}.text")

        else:

            def build_and_compile_model(build, label, export_path=None):
                logger.info(f"Building and compiling {label} model...")
                graph = build()
                before = time.perf_counter()
                model = session.load(
                    graph,
                    weights_registry=self._weights.allocated_weights,
                )
                after = time.perf_counter()
                logger.info(
                    f"Building and compiling {label} model took {after - before:.6f} seconds"
                )
                if export_path:
                    mef_path = f"{export_path}.{label}"
                    logger.info(
                        f"Exporting serialized {label} model to {mef_path}"
                    )
                    model._export_mef(mef_path)
                return model

            export_path = self.pipeline_config.save_to_serialized_model_path
            with ThreadPoolExecutor(max_workers=2) as executor:
                build = lambda: _build_vision_graph(
                    pipeline_config=self.pipeline_config,
                    weights=self._weights,
                )
                vision_model_future = executor.submit(
                    build_and_compile_model, build, "vision", export_path
                )

                build = lambda: _build_text_graph(
                    pipeline_config=self.pipeline_config,
                    weights=self._weights,
                    max_seq_len=self.calculate_max_seq_len(
                        self.pipeline_config
                    ),
                    kv_params=self.get_kv_params(self.pipeline_config),
                    kv_manager=self.kv_manager,
                )
                text_model_future = executor.submit(
                    build_and_compile_model, build, "text", export_path
                )

                vision_model = vision_model_future.result()
                text_model = text_model_future.result()

        return vision_model, text_model
