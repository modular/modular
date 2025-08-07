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
from typing import Optional, cast

import numpy as np
from max._core.engine import Model
from max.driver import Device, DLPackArray, Tensor
from max.dtype import DType
from max.engine.api import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.graph.weights import Weights, WeightsAdapter
from max.nn import Module, ReturnLogits, Signals
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheManager,
    KVCacheParams,
    KVCacheStrategy,
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

    input_row_offsets: list[Tensor]
    """Per-device tensors containing the offsets for each row in the ragged input sequence."""

    signal_buffers: list[Tensor]
    """Device buffers used for synchronization in communication collectives."""

    # Vision inputs.
    pixel_values: list[Tensor] | None = None
    """Pixel values for vision inputs."""

    window_index: list[Tensor] | None = None
    """Window indices for vision attention mechanism."""

    position_ids: list[Tensor] | None = None
    """3D RoPE position IDs for vision inputs."""

    attention_mask_full: list[Tensor] | None = None
    """Full attention masks for vision inputs."""

    attention_mask_window: list[Tensor] | None = None
    """Window attention masks for vision inputs."""

    max_grid_size: list[Tensor] | None = None
    """Maximum grid size for vision inputs."""

    return_n_logits: Tensor
    """Number of logits to return, used by speculative decoding for example."""

    def __init__(
        self,
        input_ids: Tensor,
        input_row_offsets: list[Tensor],
        signal_buffers: list[Tensor],
        return_n_logits: Tensor,
        pixel_values: list[Tensor] | None = None,
        window_index: list[Tensor] | None = None,
        position_ids: list[Tensor] | None = None,
        attention_mask_full: list[Tensor] | None = None,
        attention_mask_window: list[Tensor] | None = None,
        max_grid_size: list[Tensor] | None = None,
        kv_cache_inputs: KVCacheInputs | None = None,
    ) -> None:
        self.input_ids = input_ids
        self.input_row_offsets = input_row_offsets
        self.signal_buffers = signal_buffers
        self.return_n_logits = return_n_logits
        self.pixel_values = pixel_values
        self.window_index = window_index
        self.position_ids = position_ids
        self.attention_mask_full = attention_mask_full
        self.attention_mask_window = attention_mask_window
        self.max_grid_size = max_grid_size
        self.kv_cache_inputs = kv_cache_inputs

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

    _input_row_offsets_prealloc: list[Tensor]
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
        adapter: Optional[WeightsAdapter] = None,
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

        # Initialize signal buffers for distributed communication.
        self.signal_buffers = [
            Tensor.zeros(
                shape=(Signals.NUM_BYTES,), dtype=DType.uint8, device=dev
            )
            for dev in self.devices
        ]

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

    @property
    def vision_max_seq_len(self) -> int:
        """Returns the maximum number of vision tokens."""
        # TODO: Calculate based on Qwen2.5VL vision configuration
        # For now, use a reasonable default based on typical vision transformers
        return 1024  # Placeholder value

    def load_model(self, session: InferenceSession) -> tuple[Model, Model]:
        """Loads the compiled Qwen2.5VL models into the MAX Engine session.

        Returns:
            A tuple of (vision_model, language_model).
        """
        # Pre-allocation for multi-step execution
        assert self.pipeline_config.max_batch_size, (
            "Expected max_batch_size to be set"
        )
        input_row_offsets_prealloc_host = Tensor.from_numpy(
            np.arange(self.pipeline_config.max_batch_size + 1, dtype=np.uint32)
        )
        self._input_row_offsets_prealloc = [
            input_row_offsets_prealloc_host.to(dev) for dev in self.devices
        ]

        # Get LLM weights dictionary
        if self.adapter:
            llm_state_dict = self.adapter(
                dict(self.weights.items()),
                huggingface_config=self.huggingface_config,
                pipeline_config=self.pipeline_config,
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

        model: Module = Qwen2_5VL(self.model_config)

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
        # Get vision config parameters
        assert self.model_config is not None, "Model config is not set"
        vision_config = self.model_config.vision_config

        # Create Qwen2.5VL model and vision encoder
        model = Qwen2_5VL(self.model_config)
        vision_encoder: Module = model.vision_encoder()

        # Generate DeviceRef
        device_ref = DeviceRef.from_device(self.devices[0])

        # Define vision graph input types
        pixel_values_type = TensorType(
            DType.float32,
            shape=["seq_len", "embed_dim"],
            device=device_ref,
        )

        rot_pos_ids_type = TensorType(
            DType.int64,
            shape=["seq_len", 2],
            device=device_ref,
        )

        window_index_type = TensorType(
            DType.int64,
            shape=["window_seq_len"],
            device=device_ref,
        )

        attention_mask_window_type = TensorType(
            DType.float32,
            shape=[1, "seq_len", "seq_len"],
            device=device_ref,
        )

        attention_mask_full_type = TensorType(
            DType.float32,
            shape=[1, "seq_len", "seq_len"],
            device=device_ref,
        )

        max_grid_size_type = TensorType(
            DType.int64,
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
                pixel_values,
                rot_pos_ids,
                window_index,
                attention_mask_window,
                attention_mask_full,
                max_grid_size,
            )

            # Ensure we have a valid output
            assert vision_outputs is not None, (
                "Vision encoder must return a valid output"
            )
            graph.output(vision_outputs)

        # Get state dict for the vision encoder
        vision_state_dict: dict[str, DLPackArray] = {}
        # TODO: Extract vision-specific weights from the full state dict

        return graph, vision_state_dict

    def _build_language_graph(self) -> tuple[Graph, dict[str, DLPackArray]]:
        """Build the language model graph for text generation with image embeddings."""
        # TODO: Implement language graph building
        # This should use the Qwen2.5VL decoder transformer
        raise NotImplementedError("Language graph building not yet implemented")

    def _prepare_vision_inputs(
        self, context_batch: Sequence[TextAndVisionContext]
    ) -> dict[str, list[Tensor]] | None:
        """Prepares vision inputs for vision processing including pixel values, window index, and position IDs."""
        pixel_values_list: list[np.ndarray] = []
        image_grid_thw = None

        for context in context_batch:
            if context.needs_vision_encoding:
                # Extract pixel values
                if hasattr(context, "pixel_values") and context.pixel_values:
                    pixel_values_list.extend(context.pixel_values)

                # Extract image_grid_thw from extra_model_args (pass as is)
                if "image_grid_thw" in context.extra_model_args:
                    image_grid_thw = context.extra_model_args["image_grid_thw"]

        if not pixel_values_list or image_grid_thw is None:
            return None

        # Stack pixel values
        stacked_pixel_values = np.stack(pixel_values_list)
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
            grid_thw=image_grid_thw,
            window_size=window_size,
            spatial_merge_size=spatial_merge_size,
            patch_size=patch_size,
            spatial_merge_unit=spatial_merge_unit,
        )

        # Generate 3D RoPE position IDs
        position_ids = mrope_pos_ids_3d(
            grid_thw=image_grid_thw,
            spatial_merge_size=spatial_merge_size,
        )

        # Calculate sequence length and max grid size
        seq_length = stacked_pixel_values.shape[0]
        max_grid_size = int(
            np.max(image_grid_thw[:, 1:])
        )  # Max of height and width dimensions

        # Generate attention masks
        attention_mask_full, attention_mask_window = generate_attention_mask(
            grid_thw=image_grid_thw,
            seq_length=seq_length,
            cu_win_seqlens=cu_window_seqlens,
        )

        # Convert to tensors
        window_index_tensor = Tensor.from_numpy(window_index.astype(np.int64))
        position_ids_tensor = Tensor.from_numpy(position_ids.astype(np.int64))
        attention_mask_full_tensor = Tensor.from_numpy(
            attention_mask_full.astype(np.float32)
        )
        attention_mask_window_tensor = Tensor.from_numpy(
            attention_mask_window.astype(np.float32)
        )
        max_grid_size_tensor = Tensor.from_numpy(
            np.array([max_grid_size], dtype=np.int64)
        )

        # Return all vision inputs as tensors distributed across devices
        vision_inputs = {
            "pixel_values": [
                pixel_values_tensor.to(dev) for dev in self.devices
            ],
            "window_index": [
                window_index_tensor.to(dev) for dev in self.devices
            ],
            "position_ids": [
                position_ids_tensor.to(dev) for dev in self.devices
            ],
            "attention_mask_full": [
                attention_mask_full_tensor.to(dev) for dev in self.devices
            ],
            "attention_mask_window": [
                attention_mask_window_tensor.to(dev) for dev in self.devices
            ],
            "max_grid_size": [
                max_grid_size_tensor.to(dev) for dev in self.devices
            ],
        }

        return vision_inputs

    def _create_empty_image_embeddings(self) -> list[Tensor]:
        """Create empty image embeddings for text-only inputs."""
        hidden_size = getattr(self.huggingface_config, "hidden_size", 4096)
        return [
            Tensor.zeros(
                shape=[0, hidden_size],
                dtype=self.dtype,
            ).to(dev)
            for dev in self.devices
        ]

    def execute(
        self,
        model_inputs: ModelInputs,
    ) -> ModelOutputs:
        """Executes the Qwen2.5VL model with the prepared inputs."""
        assert isinstance(model_inputs, Qwen2_5VLInputs)
        assert model_inputs.kv_cache_inputs is not None, (
            "Qwen2.5VL requires KV cache inputs"
        )

        # Process vision inputs if present
        image_embeddings: list[Tensor]

        if model_inputs.has_vision_inputs:
            assert model_inputs.pixel_values is not None

            # Execute vision model: pixel_values -> image_embeddings
            vision_outputs = self.vision_model.execute(
                *model_inputs.pixel_values, *model_inputs.signal_buffers
            )

            image_embeddings = [
                output
                for output in vision_outputs
                if isinstance(output, Tensor)
            ]
        else:
            # Initialize empty tensors for text-only mode
            image_embeddings = self._create_empty_image_embeddings()

        # Prepare KV cache inputs as list of tensors
        kv_cache_inputs_list = list(model_inputs.kv_cache_inputs)

        # Execute language model with text and image embeddings
        language_outputs = self.language_model.execute(
            model_inputs.input_ids,
            model_inputs.return_n_logits,
            *model_inputs.input_row_offsets,
            *image_embeddings,
            *model_inputs.signal_buffers,
            *kv_cache_inputs_list,
        )

        # Return model outputs based on what the language model returns
        if len(language_outputs) == 3:
            return ModelOutputs(
                next_token_logits=cast(Tensor, language_outputs[0]),
                logits=cast(Tensor, language_outputs[1]),
                logit_offsets=cast(Tensor, language_outputs[2]),
            )
        else:
            return ModelOutputs(
                next_token_logits=cast(Tensor, language_outputs[0]),
                logits=cast(Tensor, language_outputs[0]),
            )

    def prepare_initial_token_inputs(
        self,
        context_batch: Sequence[TextAndVisionContext],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> Qwen2_5VLInputs:
        """Prepares the initial inputs for the first execution pass of the Qwen2.5VL model."""
        if self.kv_cache_config.cache_strategy != KVCacheStrategy.CONTINUOUS:
            msg = "Qwen2.5VL only supports continuous batching"
            raise ValueError(msg)

        # Prepare vision inputs
        vision_inputs = self._prepare_vision_inputs(context_batch)

        # Extract individual vision input components
        pixel_values = vision_inputs["pixel_values"] if vision_inputs else None
        window_index = vision_inputs["window_index"] if vision_inputs else None
        position_ids = vision_inputs["position_ids"] if vision_inputs else None
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
        input_row_offsets_host = Tensor.from_numpy(
            np.cumsum(
                [0] + [ctx.active_length for ctx in context_batch],
                dtype=np.uint32,
            )
        )
        input_row_offsets = [
            input_row_offsets_host.to(dev) for dev in self.devices
        ]

        # Input IDs
        tokens = np.concatenate([ctx.next_tokens for ctx in context_batch])
        input_ids = Tensor.from_numpy(tokens).to(self.devices[0])

        # Mark that vision encoding is complete for all contexts in the batch
        for ctx in context_batch:
            ctx.needs_vision_encoding = False

        return Qwen2_5VLInputs(
            input_ids=input_ids,
            input_row_offsets=input_row_offsets,
            signal_buffers=self.signal_buffers,
            return_n_logits=Tensor.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            pixel_values=pixel_values,
            window_index=window_index,
            position_ids=position_ids,
            attention_mask_full=attention_mask_full,
            attention_mask_window=attention_mask_window,
            max_grid_size=max_grid_size,
            kv_cache_inputs=kv_cache_inputs,
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: ModelInputs,
    ) -> Qwen2_5VLInputs:
        """Prepares the inputs for subsequent execution steps in a multi-step generation."""
        assert isinstance(prev_model_inputs, Qwen2_5VLInputs)

        # Use pre-allocated row offsets for next token
        offset = prev_model_inputs.input_row_offsets[0].shape[0]
        next_row_offsets = [
            offsets_prealloc[:offset]
            for offsets_prealloc in self._input_row_offsets_prealloc
        ]

        return Qwen2_5VLInputs(
            input_ids=next_tokens,
            input_row_offsets=next_row_offsets,
            signal_buffers=self.signal_buffers,
            return_n_logits=prev_model_inputs.return_n_logits,
            # Set vision model inputs to None after the first step
            pixel_values=None,
            window_index=None,
            position_ids=None,
            attention_mask_full=None,
            attention_mask_window=None,
            max_grid_size=None,
            kv_cache_inputs=prev_model_inputs.kv_cache_inputs,
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
