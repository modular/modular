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
from functools import cached_property
from typing import Any

import numpy as np
import numpy.typing as npt
from max._core.engine import Model
from max.driver import Device, DLPackArray, Tensor
from max.dtype import DType
from max.engine.api import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.graph.weights import Weights, WeightsAdapter
from max.nn import (
    Module,
    ReturnLogits,
)
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheManager,
    KVCacheParams,
    estimate_kv_cache_size,
    load_kv_manager,
)
from max.pipelines.core import TextAndVisionContext
from max.pipelines.lib import (
    KVCacheConfig,
    KVCacheMixin,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
    SupportedEncoding,
)
from transformers import AutoConfig

from .model_config import Qwen2_5VLConfig
from .nn.data_processing import (
    generate_attention_mask,
    get_rope_index,
    get_window_index,
    mrope_pos_ids_3d,
)
from .qwen2_5vl import Qwen2_5VL

logger = logging.getLogger("max.pipelines")


class Qwen2_5VLInputs(ModelInputs):
    """A class representing inputs for the Qwen2.5VL model.

    This class encapsulates the input tensors required for the Qwen2.5VL model execution:
    - input_ids: A tensor containing the input token IDs
    - input_row_offsets: Tensors containing the offsets for each row in the ragged input sequence
    - pixel_values: Image pixel values for vision processing
    - window_index: Window indices for vision attention mechanism
    - position_ids: 3D RoPE position IDs for vision inputs
    - attention_mask_full: Full attention masks for vision inputs
    - attention_mask_window: Window attention masks for vision inputs
    - max_grid_size: Maximum grid size for vision inputs
    """

    input_ids: Tensor
    """Tensor containing the input token IDs."""

    input_row_offsets: Tensor
    """Per-device tensors containing the offsets for each row in the ragged input sequence."""

    position_ids: Tensor
    """3D RoPE position IDs for the decoder."""

    return_n_logits: Tensor
    """Number of logits to return, used by speculative decoding for example."""

    image_token_indices: Tensor
    """Per-device pre-computed indices of image tokens in the input sequence."""

    kv_cache_inputs: KVCacheInputs
    """KV cache inputs for the model."""

    # Vision inputs.
    pixel_values: Tensor | None = None
    """Pixel values for vision inputs."""

    window_index: Tensor | None = None
    """Window indices for vision attention mechanism."""

    vision_position_ids: Tensor | None = None
    """1D RoPE position IDs for the visual inputs."""

    attention_mask_full: Tensor | None = None
    """Full attention masks for vision inputs."""

    attention_mask_window: Tensor | None = None
    """Window attention masks for vision inputs."""

    max_grid_size: Tensor | None = None
    """Maximum grid size for vision inputs."""

    def __init__(
        self,
        input_ids: Tensor,
        input_row_offsets: Tensor,
        position_ids: Tensor,
        return_n_logits: Tensor,
        kv_cache_inputs: KVCacheInputs,
        image_token_indices: Tensor,
        pixel_values: Tensor | None = None,
        window_index: Tensor | None = None,
        vision_position_ids: Tensor | None = None,
        attention_mask_full: Tensor | None = None,
        attention_mask_window: Tensor | None = None,
        max_grid_size: Tensor | None = None,
    ) -> None:
        self.input_ids = input_ids
        self.input_row_offsets = input_row_offsets
        self.position_ids = position_ids
        self.return_n_logits = return_n_logits
        self.kv_cache_inputs = kv_cache_inputs
        self.image_token_indices = image_token_indices
        self.pixel_values = pixel_values
        self.window_index = window_index
        self.vision_position_ids = vision_position_ids
        self.attention_mask_full = attention_mask_full
        self.attention_mask_window = attention_mask_window
        self.max_grid_size = max_grid_size

    @property
    def has_vision_inputs(self) -> bool:
        """Check if this input contains vision data."""
        return self.pixel_values is not None


class Qwen2_5VLModel(PipelineModel[TextAndVisionContext], KVCacheMixin):
    """A Qwen2.5VL pipeline model for multimodal text generation."""

    vision_model: Model
    """The compiled vision model for processing images."""

    language_model: Model
    """The compiled language model for text generation."""

    model_config: Qwen2_5VLConfig | None
    """The Qwen2.5VL model configuration."""

    _input_row_offsets_prealloc: Tensor
    """Pre-allocated per-device tensors for input row offsets in multi-step execution."""

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

        self.model_config = None

        self.vision_model, self.language_model = self.load_model(session)

    @staticmethod
    def calculate_max_seq_len(
        pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        """Calculates the maximum sequence length for the Qwen2.5VL model."""
        return Qwen2_5VLConfig.calculate_max_seq_len(
            pipeline_config, huggingface_config
        )

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        """Gets the parameters required to configure the KV cache for Qwen2.5VL."""
        return Qwen2_5VLConfig.get_kv_params(
            huggingface_config, n_devices, kv_cache_config, cache_dtype
        )

    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        """Gets the number of hidden layers from the HuggingFace configuration."""
        return Qwen2_5VLConfig.get_num_layers(huggingface_config)

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
        """Estimates the size of the KV cache required for the Qwen2.5VL model in bytes."""
        return estimate_kv_cache_size(
            params=Qwen2_5VLConfig.get_kv_params(
                huggingface_config=huggingface_config,
                n_devices=len(devices),
                kv_cache_config=kv_cache_config,
                cache_dtype=cache_dtype,
            ),
            max_batch_size=pipeline_config.max_batch_size,
            max_seq_len=cls.calculate_max_seq_len(
                pipeline_config, huggingface_config=huggingface_config
            ),
            num_layers=Qwen2_5VLConfig.get_num_layers(
                huggingface_config=huggingface_config
            ),
            available_cache_memory=available_cache_memory,
            devices=devices,
        )

    def _unflatten_kv_inputs(
        self, kv_inputs_flat: Sequence[TensorValue]
    ) -> list[tuple[TensorValue, ...]]:
        """Unflatten KV cache inputs from flat list to per-device structure."""
        fetch_types = self.kv_manager.input_symbols()[0]
        len_of_kv_tuple_per_dev = len(list(fetch_types))
        n_devices = len(self.devices)

        kv_caches_per_dev = [
            tuple(
                kv_inputs_flat[
                    i * len_of_kv_tuple_per_dev : (i + 1)
                    * len_of_kv_tuple_per_dev
                ]
            )
            for i in range(n_devices)
        ]

        return kv_caches_per_dev

    def load_model(self, session: InferenceSession) -> tuple[Model, Model]:
        """Loads the compiled Qwen2.5VL models into the MAX Engine session.

        Returns:
            A tuple of (vision_model, language_model).
        """
        # Pre-allocation for multi-step execution
        assert self.pipeline_config.max_batch_size, (
            "Expected max_batch_size to be set"
        )
        self._input_row_offsets_prealloc = Tensor.from_numpy(
            np.arange(self.pipeline_config.max_batch_size + 1, dtype=np.uint32)
        ).to(self.devices[0])

        # Get LLM weights dictionary. Needed before model config generation
        # because we need to know if word embeddings are tied or not.
        if self.adapter:
            llm_state_dict = self.adapter(
                dict(self.weights.items()),
            )
        else:
            llm_state_dict = {
                key: value.data() for key, value in self.weights.items()
            }

        # Generate Qwen2.5VL config from HuggingFace config
        qwen2_5vl_config = Qwen2_5VLConfig.generate(
            pipeline_config=self.pipeline_config,
            huggingface_config=self.huggingface_config,
            llm_state_dict=llm_state_dict,
            dtype=self.dtype,
            n_devices=len(self.devices),
            logits_postprocessor=None,
            cache_dtype=self.encoding.cache_dtype,
            kv_cache_config=self.kv_cache_config,
            return_logits=self.return_logits,
        )
        self.model_config = qwen2_5vl_config

        self.model: Module = Qwen2_5VL(self.model_config)
        self.model.load_state_dict(llm_state_dict, strict=True)

        logger.info("Building and compiling vision model...")
        before = time.perf_counter()
        vision_graph, vision_model_state_dict = self._build_vision_graph()
        vision_model = session.load(
            vision_graph, weights_registry=vision_model_state_dict
        )
        after = time.perf_counter()
        logger.info(
            f"Building and compiling vision model took {after - before:.6f} seconds"
        )

        logger.info("Building and compiling language model...")
        before = time.perf_counter()
        language_graph, language_model_state_dict = self._build_language_graph()
        language_model = session.load(
            language_graph, weights_registry=language_model_state_dict
        )
        after = time.perf_counter()
        logger.info(
            f"Building and compiling language model took {after - before:.6f} seconds"
        )

        return vision_model, language_model

    def _build_vision_graph(self) -> tuple[Graph, dict[str, DLPackArray]]:
        """Build the vision model graph for processing images."""

        # Create Qwen2.5VL model and vision encoder
        assert isinstance(self.model, Qwen2_5VL)
        vision_encoder = self.model.vision_encoder

        # Generate DeviceRef
        device_ref = DeviceRef.from_device(self.devices[0])

        # Define vision graph input types
        pixel_values_type = TensorType(
            DType.float32,
            shape=["vision_seq_len", vision_encoder.patch_embed.image_dim],
            device=device_ref,
        )

        rot_pos_ids_type = TensorType(
            DType.int64,
            shape=["vision_seq_len", 2],
            device=device_ref,
        )

        window_index_type = TensorType(
            DType.int64,
            shape=["window_seq_len"],
            device=device_ref,
        )

        attention_mask_window_type = TensorType(
            DType.float32,
            shape=[1, "vision_seq_len", "vision_seq_len"],
            device=device_ref,
        )

        attention_mask_full_type = TensorType(
            DType.float32,
            shape=[1, "vision_seq_len", "vision_seq_len"],
            device=device_ref,
        )

        max_grid_size_type = TensorType(
            DType.int32,
            shape=[],
            device=DeviceRef.CPU(),
        )

        # Build the vision graph
        with Graph(
            "qwen2_5vl_vision",
            input_types=(
                pixel_values_type,
                rot_pos_ids_type,
                window_index_type,
                attention_mask_window_type,
                attention_mask_full_type,
                max_grid_size_type,
            ),
        ) as graph:
            (
                pixel_values,
                rot_pos_ids,
                window_index,
                attention_mask_window,
                attention_mask_full,
                max_grid_size,
            ) = graph.inputs

            # Execute vision transformer using the vision encoder module
            vision_outputs = vision_encoder(
                pixel_values.tensor,
                rot_pos_ids.tensor,
                window_index.tensor,
                attention_mask_window.tensor,
                attention_mask_full.tensor,
                max_grid_size.tensor,
            )

            # Ensure we have a valid output
            assert vision_outputs is not None, (
                "Vision encoder must return a valid output"
            )
            graph.output(vision_outputs)

        # Get state dict for the vision encoder
        state_dict = self.model.state_dict()
        vision_state_dict: dict[str, DLPackArray] = {}
        for key, value in state_dict.items():
            if key.startswith("vision_encoder."):
                vision_state_dict[key] = value

        return graph, vision_state_dict

    def _build_language_graph(self) -> tuple[Graph, dict[str, DLPackArray]]:
        """Build the language model graph for text generation with image embeddings."""

        assert isinstance(self.model, Qwen2_5VL)
        language_model = self.model.language_model

        # Generate DeviceRef
        device_ref = DeviceRef.from_device(self.devices[0])

        language_state_dict: dict[str, DLPackArray] = {}
        for key, value in self.model.state_dict().items():
            if key.startswith("language_model."):
                language_state_dict[key] = value

        input_ids_type = TensorType(
            DType.int64,
            shape=["seq_len"],
            device=device_ref,
        )
        return_n_logits_type = TensorType(
            DType.int64,
            shape=["return_n_logits"],
            device=DeviceRef.CPU(),
        )
        input_row_offsets_type = TensorType(
            DType.uint32,
            shape=["batch_size_plus_1"],
            device=device_ref,
        )

        assert self.model_config is not None, "Model config must be initialized"

        image_embeddings_type = TensorType(
            self.dtype,
            shape=["vision_seq_len", self.model_config.llm_config.hidden_size],
            device=device_ref,
        )
        image_token_indices_type = TensorType(
            DType.int32, shape=["total_image_tokens"], device=device_ref
        )
        position_ids_type = TensorType(
            DType.uint32,
            shape=[len(self.model_config.mrope_section), "seq_len"],
            device=device_ref,
        )

        kv_inputs = self.kv_manager.input_symbols()
        flattened_kv_types = [
            kv_type for sublist in kv_inputs for kv_type in sublist
        ]

        with Graph(
            "qwen2_5vl_language",
            input_types=(
                input_ids_type,
                return_n_logits_type,
                input_row_offsets_type,
                image_embeddings_type,
                image_token_indices_type,
                position_ids_type,
                *flattened_kv_types,
            ),
        ) as graph:
            (
                input_ids,
                return_n_logits,
                input_row_offsets,
                image_embeddings,
                image_token_indices,
                position_ids,
                *kv_cache_inputs,
            ) = graph.inputs

            kv_cache_inputs_unflattened = [kv.tensor for kv in kv_cache_inputs]

            # Execute language model: text + image embeddings -> logits
            outputs = language_model(
                tokens=input_ids.tensor,
                kv_cache_inputs=kv_cache_inputs_unflattened,
                return_n_logits=return_n_logits.tensor,
                input_row_offsets=input_row_offsets.tensor,
                image_embeddings=image_embeddings.tensor,
                image_token_indices=image_token_indices.tensor,
                position_ids=position_ids.tensor,
                # TODO: add mrope_section to the model config
                mrope_section=self.model_config.mrope_section,
            )

            graph.output(*outputs)

        return graph, language_state_dict

    def _prepare_vision_inputs(
        self, context_batch: Sequence[TextAndVisionContext]
    ) -> dict[str, Tensor] | None:
        """Prepares vision inputs for vision processing including pixel values, window index, and position IDs."""
        pixel_values_list: list[npt.NDArray[np.floating[Any]]] = []
        image_grid_thw: list[npt.NDArray[np.integer[Any]]] = []

        for context in context_batch:
            if context.pixel_values and context.needs_vision_encoding:
                pixel_values_list.extend(context.pixel_values)
                assert context.extra_model_args["image_grid_thw"] is not None, (
                    "image_grid_thw must be present"
                )
                image_grid_thw.append(
                    context.extra_model_args["image_grid_thw"]
                )

        if not pixel_values_list or not image_grid_thw:
            return None

        # Stack pixel values and image_grid_thw
        stacked_pixel_values = np.concatenate(pixel_values_list, axis=0)
        stacked_image_grid_thw = np.concatenate(image_grid_thw, axis=0)

        pixel_values_tensor = Tensor.from_numpy(stacked_pixel_values)

        # Get vision config parameters
        assert self.model_config is not None, "Model config must be initialized"

        vision_config = self.model_config.vision_config
        spatial_merge_size = vision_config.spatial_merge_size
        window_size = vision_config.window_size
        patch_size = vision_config.patch_size

        # Calculate spatial_merge_unit
        spatial_merge_unit = spatial_merge_size * spatial_merge_size

        # Generate window index and cumulative window sequence lengths
        window_index, cu_window_seqlens = get_window_index(
            grid_thw=stacked_image_grid_thw,
            window_size=window_size,
            spatial_merge_size=spatial_merge_size,
            patch_size=patch_size,
            spatial_merge_unit=spatial_merge_unit,
        )

        # Generate 3D RoPE position IDs
        vision_position_ids = mrope_pos_ids_3d(
            grid_thw=stacked_image_grid_thw,
            spatial_merge_size=spatial_merge_size,
        )

        # Calculate sequence length and max grid size
        vision_seq_length = stacked_pixel_values.shape[0]

        vision_max_grid_size = int(
            np.max(stacked_image_grid_thw[:, 1:])
        )  # Max of height and width dimensions

        # Generate attention masks
        attention_mask_full, attention_mask_window = generate_attention_mask(
            grid_thw=stacked_image_grid_thw,
            seq_length=vision_seq_length,
            cu_win_seqlens=cu_window_seqlens,
        )

        # Convert to tensors
        window_index_tensor = Tensor.from_numpy(window_index.astype(np.int64))
        position_ids_tensor = Tensor.from_numpy(
            vision_position_ids.astype(np.int64)
        )
        attention_mask_full_tensor = Tensor.from_numpy(
            attention_mask_full.astype(np.float32)
        )
        attention_mask_window_tensor = Tensor.from_numpy(
            attention_mask_window.astype(np.float32)
        )
        max_grid_size_tensor = Tensor.from_numpy(
            np.array(vision_max_grid_size, dtype=np.int32)
        )

        # Return all vision inputs as tensors distributed across devices
        vision_inputs = {
            "pixel_values": pixel_values_tensor.to(self.devices[0]),
            "window_index": window_index_tensor.to(self.devices[0]),
            "position_ids": position_ids_tensor.to(self.devices[0]),
            "attention_mask_full": attention_mask_full_tensor.to(
                self.devices[0]
            ),
            "attention_mask_window": attention_mask_window_tensor.to(
                self.devices[0]
            ),
            "max_grid_size": max_grid_size_tensor,
        }

        return vision_inputs

    @cached_property
    def _empty_image_embeddings(self) -> Tensor:
        """Create empty image embeddings for text-only inputs on single GPU."""
        return Tensor.zeros(
            shape=[0, self.huggingface_config.text_config.hidden_size],
            dtype=self.dtype,
        ).to(self.devices[0])

    @cached_property
    def _empty_image_token_indices(self) -> Tensor:
        """Create empty image token indices for text-only inputs on single GPU."""
        return Tensor.zeros(
            shape=[0],
            dtype=DType.int32,
        ).to(self.devices[0])

    def _batch_image_token_indices(
        self, context_batch: Sequence[TextAndVisionContext]
    ) -> Tensor:
        """Batch image token indices from multiple contexts, adjusting for
        position in batch.

        This method efficiently combines image token indices from multiple
        contexts using vectorized operations.

        Args:
            context_batch: Sequence of contexts that may contain image token
                indices

        Returns:
            Tensor containing all batched indices, or None if no indices found
        """
        # Collect indices and offsets.
        indices_and_offsets = []
        batch_offset = 0

        assert self.model_config is not None, "Model config must be initialized"

        for ctx in context_batch:
            input_ids = ctx.next_tokens
            # make sure image_token_id is correct in model config
            special_image_token_mask = (
                input_ids == self.model_config.image_token_id
            )
            indices = np.where(special_image_token_mask)[0].tolist()

            indices_and_offsets.append([idx + batch_offset for idx in indices])
            batch_offset += ctx.active_length

        if not indices_and_offsets:
            np_indices = np.array([], dtype=np.int32)
        else:
            np_indices = np.concatenate(indices_and_offsets).astype(
                np.int32, copy=False
            )

        # Create tensor and distribute to device
        return Tensor.from_numpy(np_indices).to(self.devices[0])

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Executes the Qwen2.5VL model with the prepared inputs."""
        assert isinstance(model_inputs, Qwen2_5VLInputs)
        assert model_inputs.kv_cache_inputs is not None, (
            "Qwen2.5VL requires KV cache inputs"
        )

        # Process vision inputs if present
        image_embeddings: Tensor

        if model_inputs.has_vision_inputs:
            assert model_inputs.pixel_values is not None
            assert model_inputs.vision_position_ids is not None
            assert model_inputs.window_index is not None
            assert model_inputs.attention_mask_window is not None
            assert model_inputs.attention_mask_full is not None
            assert model_inputs.max_grid_size is not None

            # Execute vision model: pixel_values -> image_embeddings
            vision_outputs = self.vision_model.execute(
                model_inputs.pixel_values,
                model_inputs.vision_position_ids,
                model_inputs.window_index,
                model_inputs.attention_mask_window,
                model_inputs.attention_mask_full,
                model_inputs.max_grid_size,
            )

            assert isinstance(vision_outputs[0], Tensor)
            image_embeddings = vision_outputs[0]
        else:
            # Initialize empty tensors for text-only mode
            image_embeddings = self._empty_image_embeddings

        # Execute language model with text and image embeddings
        language_outputs = self.language_model.execute(
            model_inputs.input_ids,
            model_inputs.return_n_logits,
            model_inputs.input_row_offsets,
            image_embeddings,
            # TODO: calculate image_token_indices
            model_inputs.image_token_indices,
            model_inputs.position_ids,
            *model_inputs.kv_cache_inputs,
        )

        # Return model outputs based on what the language model returns
        if len(language_outputs) == 3:
            assert isinstance(language_outputs[0], Tensor)
            assert isinstance(language_outputs[1], Tensor)
            assert isinstance(language_outputs[2], Tensor)
            return ModelOutputs(
                next_token_logits=language_outputs[0],
                logits=language_outputs[1],
                logit_offsets=language_outputs[2],
            )
        else:
            assert isinstance(language_outputs[0], Tensor)
            return ModelOutputs(
                next_token_logits=language_outputs[0],
                logits=language_outputs[0],
            )

    def prepare_initial_token_inputs(
        self,
        context_batch: Sequence[TextAndVisionContext],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> Qwen2_5VLInputs:
        """Prepares the initial inputs for the first execution pass of the Qwen2.5VL model."""
        assert kv_cache_inputs is not None, "KV cache inputs must be provided"

        # Prepare vision inputs
        vision_inputs = self._prepare_vision_inputs(context_batch)

        # Extract individual vision input components
        pixel_values = vision_inputs["pixel_values"] if vision_inputs else None
        window_index = vision_inputs["window_index"] if vision_inputs else None

        # we have to change position_ids to vision_position_ids
        vision_position_ids = (
            vision_inputs["position_ids"] if vision_inputs else None
        )

        attention_mask_full = (
            vision_inputs["attention_mask_full"] if vision_inputs else None
        )
        attention_mask_window = (
            vision_inputs["attention_mask_window"] if vision_inputs else None
        )
        max_grid_size = (
            vision_inputs["max_grid_size"] if vision_inputs else None
        )

        # Input row offsets
        input_row_offsets = Tensor.from_numpy(
            np.cumsum(
                [0] + [ctx.active_length for ctx in context_batch],
                dtype=np.uint32,
            )
        ).to(self.devices[0])

        # we should generate position_ids for the decoder using numpy get_rope_index we have in data_processing.py
        # make sure these values are correct in model config
        assert self.model_config is not None, "Model config must be initialized"

        decoder_position_ids = []
        for ctx in context_batch:
            if ctx.extra_model_args.get("rope_delta", None) is None:
                extra_model_args = ctx.extra_model_args
                image_grid_thw = extra_model_args.get("image_grid_thw", None)
                video_grid_thw = extra_model_args.get("video_grid_thw", None)
                second_per_grid_ts = extra_model_args.get(
                    "second_per_grid_ts", None
                )
                attention_mask = extra_model_args.get("attention_mask", None)

                temp_position_ids, rope_delta = get_rope_index(
                    spatial_merge_size=self.model_config.spatial_merge_size,
                    image_token_id=self.model_config.image_token_id,
                    video_token_id=self.model_config.video_token_id,
                    vision_start_token_id=self.model_config.vision_start_token_id,
                    tokens_per_second=self.model_config.tokens_per_second,
                    # get rope index expects a input_ids in this shape: [batch_size, seq_len] --> [1, seq_len]
                    input_ids=ctx.next_tokens.reshape(1, -1),
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask,
                )
                # Store rope delta in extra_model_args, this is used later to
                # compute the position ids for the next token.
                ctx.extra_model_args["rope_delta"] = rope_delta
                # the temp_position_ids is a 3D tensor, we need to flatten it to 2D

                temp_position_ids = temp_position_ids.squeeze(1)
            else:
                temp_position_ids = np.full(
                    shape=(3, 1),  # hardcode to 3 for temporal, height, width
                    fill_value=ctx.extra_model_args["rope_delta"].item()
                    + ctx.current_length,
                )
            decoder_position_ids.append(temp_position_ids)

        decoder_position_ids = np.concatenate(decoder_position_ids, axis=1)

        # position_ids is a 2D tensor that is passed down to the 2D RoPE kernel in the decoder along with mrope_section
        # Convert to uint32 to match the expected dtype in the language model
        position_ids = Tensor.from_numpy(
            decoder_position_ids.astype(np.uint32)
        ).to(self.devices[0])

        # Input IDs
        tokens = np.concatenate([ctx.next_tokens for ctx in context_batch])
        input_ids = Tensor.from_numpy(tokens).to(self.devices[0])

        # Batch image token indices, offsetting for position in the batch.
        image_token_indices = self._batch_image_token_indices(context_batch)

        # Mark that vision encoding is complete for all contexts in the batch
        for ctx in context_batch:
            ctx.needs_vision_encoding = False

        return Qwen2_5VLInputs(
            input_ids=input_ids,
            input_row_offsets=input_row_offsets,
            position_ids=position_ids,
            return_n_logits=Tensor.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            kv_cache_inputs=kv_cache_inputs,
            image_token_indices=image_token_indices,
            pixel_values=pixel_values,
            window_index=window_index,
            vision_position_ids=vision_position_ids,
            attention_mask_full=attention_mask_full,
            attention_mask_window=attention_mask_window,
            max_grid_size=max_grid_size,
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: ModelInputs,
    ) -> Qwen2_5VLInputs:
        """Prepares the inputs for subsequent execution steps in a multi-step generation."""
        # TODO: This is still buggy. Use max_num_steps=1 until this is fixed.
        assert isinstance(prev_model_inputs, Qwen2_5VLInputs)

        # input_ids, old_row_offsets, Optional: [pixel_values, attention_mask]
        old_row_offsets = prev_model_inputs.input_row_offsets

        row_offsets_size = old_row_offsets.shape[0]
        next_row_offsets = self._input_row_offsets_prealloc[:row_offsets_size]

        old_row_offsets_np = old_row_offsets.to_numpy()
        old_position_ids_np = prev_model_inputs.position_ids.to_numpy()

        # Compute new position ids by adding 1 to the previous final position id
        # for each element in the batch.
        position_ids_np = (
            old_position_ids_np[..., old_row_offsets_np[1:] - 1] + 1
        )
        position_ids = Tensor.from_numpy(position_ids_np).to(self.devices[0])

        return Qwen2_5VLInputs(
            input_ids=next_tokens,
            input_row_offsets=next_row_offsets,
            position_ids=position_ids,
            kv_cache_inputs=prev_model_inputs.kv_cache_inputs,
            return_n_logits=prev_model_inputs.return_n_logits,
            image_token_indices=self._empty_image_token_indices,
            # Leave vision inputs empty since they are only processed on the
            # first step.
        )

    def load_kv_manager(
        self, session: InferenceSession, available_cache_memory: int | None
    ) -> KVCacheManager:
        """Loads and initializes the KVCacheManager for the Qwen2.5VL model."""
        return load_kv_manager(
            params=Qwen2_5VLConfig.get_kv_params(
                huggingface_config=self.huggingface_config,
                n_devices=len(self.devices),
                kv_cache_config=self.kv_cache_config,
                cache_dtype=self.encoding.cache_dtype,
            ),
            max_batch_size=self.pipeline_config.max_batch_size,
            max_seq_len=self.calculate_max_seq_len(
                self.pipeline_config, huggingface_config=self.huggingface_config
            ),
            num_layers=Qwen2_5VLConfig.get_num_layers(
                huggingface_config=self.huggingface_config
            ),
            devices=self.devices,
            available_cache_memory=available_cache_memory,
            page_size=self.kv_cache_config.kv_cache_page_size,
            session=session,
        )

    @classmethod
    def estimate_activation_memory(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        # TODO: Make this more robust
        return 5 * 1024 * 1024 * 1024  # 5 GiB
