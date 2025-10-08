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
from max.driver import Device, Tensor
from max.dtype import DType
from max.engine.api import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, Value
from max.graph.weights import (
    SafetensorWeights,
    WeightData,
    Weights,
    WeightsAdapter,
)
from max.nn import Module, ReturnLogits, Signals
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheParams,
    PagedCacheValues,
    PagedKVCacheManager,
    estimate_kv_cache_size,
    load_kv_manager,
)
from max.nn.parallel import ParallelArrayOps
from max.pipelines.architectures.internvl.model import (
    assert_image_embeddings_invariant,
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
from max.profiler import Tracer, traced
from transformers import AutoConfig

from .model_config import Qwen2_5VLConfig
from .qwen2_5vl import Qwen2_5VL

logger = logging.getLogger("max.pipelines")


class Qwen2_5VLInputs(ModelInputs):
    """A class representing inputs for the Qwen2.5VL model.

    This class encapsulates the input tensors required for the Qwen2.5VL model execution:
    - input_ids: A tensor containing the input token IDs
    - input_row_offsets: Per-device tensors containing the offsets for each row in the ragged input sequence
    - pixel_values: image pixel values for vision processing
    - window_index: window indices for vision attention mechanism
    - position_ids: 3D RoPE position IDs for vision inputs
    - attention_mask_full:  full attention masks for vision inputs
    - attention_mask_window:  window attention masks for vision inputs
    - max_grid_size: Maximum grid size for vision inputs
    """

    input_ids: Tensor
    """Tensor containing the input token IDs."""

    input_row_offsets: list[Tensor]
    """Per-device tensors containing the offsets for each row in the ragged input sequence."""
    signal_buffers: list[Tensor]
    """Device buffers used for synchronization in communication collectives."""

    position_ids: Tensor
    """3D RoPE position IDs for the decoder."""

    return_n_logits: Tensor
    """Number of logits to return, used by speculative decoding for example."""

    kv_cache_inputs: KVCacheInputs
    """KV cache inputs for the model."""

    image_token_indices: list[Tensor] | None = None
    """Per-device pre-computed indices of image tokens in the input sequence."""
    # Vision inputs.
    pixel_values: list[Tensor] | None = None
    """Pixel values for vision inputs."""

    window_index: list[Tensor] | None = None
    """Window indices for vision attention mechanism."""

    vision_position_ids: list[Tensor] | None = None
    """1D RoPE position IDs for the visual inputs."""

    max_grid_size: list[Tensor] | None = None
    """Maximum grid size for vision inputs."""

    cu_seqlens: list[Tensor] | None = None
    """Cumulative sequence lengths for full attention."""

    cu_window_seqlens: list[Tensor] | None = None
    """Cumulative window sequence lengths for window attention."""

    max_seqlen: list[Tensor] | None = None
    """Maximum sequence length for full attention for vision inputs."""

    max_window_seqlen: list[Tensor] | None = None
    """Maximum sequence length for window attention for vision inputs."""

    def __init__(
        self,
        input_ids: Tensor,
        input_row_offsets: list[Tensor],
        signal_buffers: list[Tensor],
        position_ids: Tensor,
        return_n_logits: Tensor,
        kv_cache_inputs: KVCacheInputs,
        image_token_indices: list[Tensor] | None = None,
        pixel_values: list[Tensor] | None = None,
        window_index: list[Tensor] | None = None,
        vision_position_ids: list[Tensor] | None = None,
        max_grid_size: list[Tensor] | None = None,
        cu_seqlens: list[Tensor] | None = None,
        cu_window_seqlens: list[Tensor] | None = None,
        max_seqlen: list[Tensor] | None = None,
        max_window_seqlen: list[Tensor] | None = None,
    ) -> None:
        self.signal_buffers = signal_buffers
        self.input_ids = input_ids
        self.input_row_offsets = input_row_offsets
        self.position_ids = position_ids
        self.return_n_logits = return_n_logits
        self.kv_cache_inputs = kv_cache_inputs
        self.image_token_indices = image_token_indices
        self.pixel_values = pixel_values
        self.window_index = window_index
        self.vision_position_ids = vision_position_ids
        self.max_grid_size = max_grid_size
        self.cu_seqlens = cu_seqlens
        self.cu_window_seqlens = cu_window_seqlens
        self.max_seqlen = max_seqlen
        self.max_window_seqlen = max_window_seqlen

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

        # Initialize signal buffers for distributed execution
        self.signal_buffers = [
            Tensor.zeros(
                shape=(Signals.NUM_BYTES,), dtype=DType.uint8, device=dev
            )
            for dev in self.devices
        ]

        self.vision_model, self.language_model = self.load_model(session)

        self._parallel_ops = ParallelArrayOps(max_workers=24)

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
        self, kv_inputs_flat: Sequence[Value[Any]]
    ) -> list[PagedCacheValues]:
        """Unflatten KV cache inputs from flat list to per-device structure."""
        fetch_types = self.kv_manager.input_symbols()[0]
        len_of_kv_tuple_per_dev = len(list(fetch_types))
        n_devices = len(self.devices)

        kv_caches_per_dev: list[PagedCacheValues] = []
        for i in range(n_devices):
            start_idx = i * len_of_kv_tuple_per_dev
            kv_caches_per_dev.append(
                PagedCacheValues(
                    kv_blocks=kv_inputs_flat[start_idx].buffer,
                    cache_lengths=kv_inputs_flat[start_idx + 1].tensor,
                    lookup_table=kv_inputs_flat[start_idx + 2].tensor,
                    max_lengths=kv_inputs_flat[start_idx + 3].tensor,
                )
            )
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
        self._input_row_offsets_prealloc = [
            Tensor.from_numpy(
                np.arange(
                    self.pipeline_config.max_batch_size + 1, dtype=np.uint32
                )
            ).to(dev)
            for dev in self.devices
        ]

        # Get LLM weights dictionary. Needed before model config generation
        # because we need to know if word embeddings are tied or not.
        if not isinstance(self.weights, SafetensorWeights):
            raise ValueError(
                "Qwen2.5VL currently only supports safetensors weights"
            )
        if self.adapter:
            model_state_dict = self.adapter(
                dict(self.weights.items()),
            )
        else:
            model_state_dict = {
                key: value.data() for key, value in self.weights.items()
            }
        # Get state dict for the vision encoder
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

        # Generate Qwen2.5VL config from HuggingFace config
        qwen2_5vl_config = Qwen2_5VLConfig.generate(
            pipeline_config=self.pipeline_config,
            huggingface_config=self.huggingface_config,
            llm_state_dict=llm_state_dict,
            vision_state_dict=vision_state_dict,
            dtype=self.dtype,
            n_devices=len(self.devices),
            cache_dtype=self.encoding.cache_dtype,
            kv_cache_config=self.kv_cache_config,
            return_logits=self.return_logits,
        )
        self.model_config = qwen2_5vl_config

        assert self.model_config is not None, "Model config must be initialized"
        self.model: Module = Qwen2_5VL(self.model_config)
        self.model.load_state_dict(model_state_dict, strict=True)

        logger.info("Building and compiling vision model...")
        before = time.perf_counter()
        vision_graph = self._build_vision_graph()
        vision_model = session.load(
            vision_graph, weights_registry=vision_state_dict
        )
        after = time.perf_counter()
        logger.info(
            f"Building and compiling vision model took {after - before:.6f} seconds"
        )

        logger.info("Building and compiling language model...")
        before = time.perf_counter()
        language_graph = self._build_language_graph()
        language_model = session.load(
            language_graph, weights_registry=llm_state_dict
        )
        after = time.perf_counter()
        logger.info(
            f"Building and compiling language model took {after - before:.6f} seconds"
        )

        return vision_model, language_model

    def _build_vision_graph(self) -> Graph:
        """Build the vision model graph for processing images.

        Now supports multi-GPU processing for the vision encoder.
        """

        # Create Qwen2.5VL model and vision encoder
        assert isinstance(self.model, Qwen2_5VL)
        vision_encoder = self.model.vision_encoder
        # Define vision graph input types - one per device
        # vision_seq_len is the number of patches in all images and videos in the request
        pixel_values_types = [
            TensorType(
                DType.float32,
                shape=["vision_seq_len", vision_encoder.patch_embed.patch_dim],
                device=DeviceRef.from_device(device),
            )
            for device in self.devices
        ]

        rot_pos_ids_types = [
            TensorType(
                DType.int64,
                shape=["vision_seq_len", 2],
                device=DeviceRef.from_device(device),
            )
            for device in self.devices
        ]

        window_index_types = [
            TensorType(
                DType.int64,
                shape=["window_seq_len"],
                device=DeviceRef.from_device(device),
            )
            for device in self.devices
        ]

        max_grid_size_types = [
            TensorType(
                DType.int32,
                shape=[],
                device=DeviceRef.CPU(),
            )
            for device in self.devices
        ]

        # Create signal types for distributed communication
        signals = Signals(
            devices=(DeviceRef(d.label, d.id) for d in self.devices)
        )

        cu_seqlens_types = [
            TensorType(
                DType.uint32,
                shape=["n_seqlens"],
                device=DeviceRef.from_device(device),
            )
            for device in self.devices
        ]

        cu_window_seqlens_types = [
            TensorType(
                DType.uint32,
                shape=["n_window_seqlens"],
                device=DeviceRef.from_device(device),
            )
            for device in self.devices
        ]

        max_seqlen_types = [
            TensorType(
                DType.uint32,
                shape=[1],
                device=DeviceRef.CPU(),
            )
            for _ in self.devices
        ]

        max_window_seqlen_types = [
            TensorType(
                DType.uint32,
                shape=[1],
                device=DeviceRef.CPU(),
            )
            for _ in self.devices
        ]

        # Build the vision graph
        with Graph(
            "qwen2_5vl_vision",
            input_types=tuple(
                [
                    *pixel_values_types,
                    *rot_pos_ids_types,
                    *window_index_types,
                    *cu_seqlens_types,
                    *cu_window_seqlens_types,
                    *max_seqlen_types,
                    *max_window_seqlen_types,
                    *max_grid_size_types,
                    *signals.input_types(),
                ]
            ),
        ) as graph:
            # Extract inputs
            all_inputs = graph.inputs
            n_devices = len(self.devices)

            pixel_values_list = [inp.tensor for inp in all_inputs[:n_devices]]
            rot_pos_ids_list = [
                inp.tensor for inp in all_inputs[n_devices : 2 * n_devices]
            ]
            window_index_list = [
                inp.tensor for inp in all_inputs[2 * n_devices : 3 * n_devices]
            ]
            cu_seqlens_list = [
                inp.tensor for inp in all_inputs[3 * n_devices : 4 * n_devices]
            ]
            cu_window_seqlens_list = [
                inp.tensor for inp in all_inputs[4 * n_devices : 5 * n_devices]
            ]
            max_seqlen_list = [
                inp.tensor for inp in all_inputs[5 * n_devices : 6 * n_devices]
            ]
            max_window_seqlen_list = [
                inp.tensor for inp in all_inputs[6 * n_devices : 7 * n_devices]
            ]
            max_grid_size_list = [
                inp.tensor for inp in all_inputs[7 * n_devices : 8 * n_devices]
            ]
            signal_buffers = [inp.buffer for inp in all_inputs[8 * n_devices :]]

            # Execute vision transformer using the vision encoder module with multi-GPU support
            # For now, use the first device's inputs (keeping compatibility with single GPU approach)
            # TODO: Implement proper multi-GPU execution when vision encoder is fully parallelized
            vision_outputs = vision_encoder(
                pixel_values=pixel_values_list,
                rot_pos_ids=rot_pos_ids_list,
                window_index=window_index_list,
                cu_seqlens=cu_seqlens_list,
                cu_window_seqlens=cu_window_seqlens_list,
                max_seqlen=max_seqlen_list,
                max_window_seqlen=max_window_seqlen_list,
                max_grid_size=max_grid_size_list,
                signal_buffers=signal_buffers,
            )

            # Ensure we have a valid output
            assert vision_outputs is not None, (
                "Vision encoder must return a valid output"
            )

            graph.output(*vision_outputs)

        return graph

    def _build_language_graph(self) -> Graph:
        """Build the language model graph for text generation with image embeddings."""

        assert isinstance(self.model, Qwen2_5VL)
        language_model = self.model.language_model

        # Generate DeviceRef
        device_ref = DeviceRef.from_device(self.devices[0])

        input_ids_type = TensorType(
            DType.int64,
            shape=["total_seq_len"],
            device=device_ref,
        )
        return_n_logits_type = TensorType(
            DType.int64,
            shape=["return_n_logits"],
            device=DeviceRef.CPU(),
        )
        # Create input_row_offsets_type for each device
        input_row_offsets_types = [
            TensorType(
                DType.uint32,
                shape=["input_row_offsets_len"],
                device=DeviceRef.from_device(dev),
            )
            for dev in self.devices
        ]

        signals = Signals(
            devices=(DeviceRef(d.label, d.id) for d in self.devices)
        )
        assert self.model_config is not None, "Model config must be initialized"

        # Add image embeddings type - one per device, can be empty for text-only inputs
        image_embeddings_types = [
            TensorType(
                self.dtype,
                shape=[
                    "vision_seq_len",
                    self.model_config.llm_config.hidden_size,
                ],
                device=DeviceRef.from_device(device),
            )
            for device in self.devices
        ]

        # Add image token indices type - one per device
        image_token_indices_types = [
            TensorType(
                DType.int32,
                shape=["total_image_tokens"],
                device=DeviceRef.from_device(device),
            )
            for device in self.devices
        ]
        position_ids_type = TensorType(
            DType.uint32,
            shape=[len(self.model_config.mrope_section), "total_seq_len"],
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
                *input_row_offsets_types,
                *image_embeddings_types,
                *image_token_indices_types,
                position_ids_type,
                *signals.input_types(),
                *flattened_kv_types,
            ),
        ) as graph:
            (
                input_ids,
                return_n_logits,
                *variadic_args,
            ) = graph.inputs

            # Extract input_row_offsets (one per device)
            input_row_offsets = [
                v.tensor for v in variadic_args[: len(self.devices)]
            ]
            variadic_args = variadic_args[len(self.devices) :]

            # Extract image embeddings (one per device)
            image_embeddings = [
                v.tensor for v in variadic_args[: len(self.devices)]
            ]
            variadic_args = variadic_args[len(self.devices) :]

            # Extract image token indices (one per device)
            image_token_indices = [
                v.tensor for v in variadic_args[: len(self.devices)]
            ]
            variadic_args = variadic_args[len(self.devices) :]

            # Extract position_ids
            position_ids = variadic_args[0].tensor
            variadic_args = variadic_args[1:]

            # Extract signal buffers (one per device)
            signal_buffers = [
                v.buffer for v in variadic_args[: len(self.devices)]
            ]

            # Unmarshal the remaining arguments, which are for KV cache.
            variadic_args = variadic_args[len(self.devices) :]
            kv_collections = self._unflatten_kv_inputs(variadic_args)

            # Execute language model: text + image embeddings -> logits
            outputs = language_model(
                tokens=input_ids.tensor,
                return_n_logits=return_n_logits.tensor,
                image_embeddings=image_embeddings,
                image_token_indices=image_token_indices,
                position_ids=position_ids,
                signal_buffers=signal_buffers,
                kv_collections=kv_collections,
                input_row_offsets=input_row_offsets,
                mrope_section=self.model_config.mrope_section,
            )

            graph.output(*outputs)

        return graph

    @cached_property
    def _empty_image_embeddings(self) -> list[Tensor]:
        """Create empty image embeddings for text-only inputs on multi-device."""
        return [
            Tensor.zeros(
                shape=[0, self.huggingface_config.text_config.hidden_size],
                dtype=self.dtype,
            ).to(device)
            for device in self.devices
        ]

    @cached_property
    def _empty_image_token_indices(self) -> list[Tensor]:
        """Create empty image token indices for text-only inputs on multi-device."""
        return [
            Tensor.zeros(
                shape=[0],
                dtype=DType.int32,
            ).to(device)
            for device in self.devices
        ]

    def _batch_image_token_indices(
        self, context_batch: Sequence[TextAndVisionContext]
    ) -> list[Tensor]:
        """Batch image token indices from multiple contexts, adjusting for
        position in batch.

        This method efficiently combines image token indices from multiple
        contexts using vectorized operations.

        Args:
            context_batch: Sequence of contexts that may contain image token
                indices

        Returns:
            List of tensors containing all batched indices distributed across devices
        """
        # Collect indices and offsets.
        indices_and_offsets = []
        batch_offset = 0

        assert self.model_config is not None, "Model config must be initialized"

        for ctx in context_batch:
            if "image_token_indices" in ctx.extra_model_args:
                # We have to pop this, as we can only compute it once for CE in advance.
                indices = ctx.extra_model_args.pop("image_token_indices")
                indices_and_offsets.append(indices + batch_offset)
            else:
                input_ids = ctx.next_tokens
                # make sure image_token_id is correct in model config
                special_image_token_mask = (
                    input_ids == self.model_config.image_token_id
                )
                indices = np.where(special_image_token_mask)[0].tolist()
                indices_and_offsets.append(
                    [idx + batch_offset for idx in indices]  # type: ignore
                )

            batch_offset += ctx.active_length

        if not indices_and_offsets:
            np_indices = np.array([], dtype=np.int32)
        else:
            np_indices = np.concatenate(indices_and_offsets).astype(
                np.int32, copy=False
            )

        # Create tensor and distribute to devices
        return [
            Tensor.from_numpy(np_indices).to(device) for device in self.devices
        ]

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Executes the Qwen2.5VL model with the prepared inputs."""
        assert isinstance(model_inputs, Qwen2_5VLInputs)
        assert model_inputs.kv_cache_inputs is not None, (
            "Qwen2.5VL requires KV cache inputs"
        )

        # Process vision inputs if present
        image_embeddings: list[Tensor]

        if model_inputs.has_vision_inputs:
            assert model_inputs.image_token_indices is not None
            assert model_inputs.pixel_values is not None
            assert model_inputs.vision_position_ids is not None
            assert model_inputs.window_index is not None
            assert model_inputs.cu_seqlens is not None
            assert model_inputs.cu_window_seqlens is not None
            assert model_inputs.max_seqlen is not None
            assert model_inputs.max_window_seqlen is not None
            assert model_inputs.max_grid_size is not None
            assert model_inputs.image_token_indices is not None

            # Execute vision model: pixel_values -> image_embeddings (multi-GPU)

            vision_outputs = self.vision_model.execute(
                *model_inputs.pixel_values,
                *model_inputs.vision_position_ids,
                *model_inputs.window_index,
                *model_inputs.cu_seqlens,
                *model_inputs.cu_window_seqlens,
                *model_inputs.max_seqlen,
                *model_inputs.max_window_seqlen,
                *model_inputs.max_grid_size,
                *model_inputs.signal_buffers,
            )

            # Extract image embeddings from vision outputs (one per device)
            assert len(vision_outputs) == len(self.devices)
            image_embeddings = [
                output
                for output in vision_outputs
                if isinstance(output, Tensor)
            ]
            image_token_indices = model_inputs.image_token_indices
            assert_image_embeddings_invariant(
                image_embeddings, image_token_indices
            )
        else:
            # Initialize empty tensors for text-only mode
            image_embeddings = self._empty_image_embeddings
            image_token_indices = self._empty_image_token_indices

        # Execute language model with text and image embeddings
        language_outputs = self.language_model.execute(
            model_inputs.input_ids,
            model_inputs.return_n_logits,
            *model_inputs.input_row_offsets,
            *image_embeddings,
            *image_token_indices,
            model_inputs.position_ids,
            *model_inputs.signal_buffers,
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

    @traced
    def prepare_initial_token_inputs(
        self,
        context_batch: Sequence[TextAndVisionContext],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> Qwen2_5VLInputs:
        """Prepares the initial inputs for the first execution pass of the Qwen2.5VL model."""

        if kv_cache_inputs is None:
            raise ValueError("KV Cache Inputs must be provided")

        # Prepare Inputs Needed Regardless of Images
        with Tracer("prepare_input_ids"):
            input_ids = Tensor.from_numpy(
                self._parallel_ops.concatenate(
                    [ctx.next_tokens for ctx in context_batch]
                )
            ).to(self.devices[0])

        with Tracer("prepare_input_row_offsets"):
            input_row_offsets = np.cumsum(
                [0] + [ctx.active_length for ctx in context_batch],
                dtype=np.uint32,
            )
            input_row_offsets_tensors = [
                Tensor.from_numpy(input_row_offsets).to(device)
                for device in self.devices
            ]

        with Tracer("prepare_decoder_position_ids"):
            position_ids_list = []

            for ctx in context_batch:
                ctx_decoder_position_ids: npt.NDArray[np.int32] | None = None
                if "decoder_position_ids" in ctx.extra_model_args:
                    ctx_decoder_position_ids = ctx.extra_model_args.pop(
                        "decoder_position_ids"
                    )
                    if ctx_decoder_position_ids.shape[1] == ctx.active_length:
                        position_ids_list.append(ctx_decoder_position_ids)
                    else:
                        ctx_decoder_position_ids = None

                if ctx_decoder_position_ids is None:
                    # Recompute this value on the fly.
                    context_seq_length = ctx.next_tokens.shape[0]
                    temp_position_ids = np.arange(context_seq_length)
                    temp_position_ids = temp_position_ids.reshape(1, 1, -1)  # type: ignore
                    temp_position_ids = np.tile(temp_position_ids, (3, 1, 1))
                    delta = (
                        ctx.current_length
                        + ctx.extra_model_args["rope_delta"].item()
                    )
                    temp_position_ids = temp_position_ids + delta
                    temp_position_ids = temp_position_ids.squeeze(1)
                    position_ids_list.append(temp_position_ids)

            decoder_position_ids = Tensor.from_numpy(
                self._parallel_ops.concatenate(
                    position_ids_list, axis=1
                ).astype(np.uint32)
            ).to(self.devices[0])

        with Tracer("prepare_image_token_indices"):
            image_token_indices = self._batch_image_token_indices(context_batch)

        if not any(ctx.needs_vision_encoding for ctx in context_batch):
            return Qwen2_5VLInputs(
                input_ids=input_ids,
                input_row_offsets=input_row_offsets_tensors,
                position_ids=decoder_position_ids,
                signal_buffers=self.signal_buffers,
                return_n_logits=Tensor.from_numpy(
                    np.array([return_n_logits], dtype=np.int64)
                ),
                kv_cache_inputs=kv_cache_inputs,
                image_token_indices=image_token_indices,
                pixel_values=None,
                window_index=None,
                vision_position_ids=None,
                max_grid_size=None,
                cu_seqlens=None,
                cu_window_seqlens=None,
                max_seqlen=None,
                max_window_seqlen=None,
            )

        # From here on, assume that all inputs are available in extra_model_args
        # due to context validators
        with Tracer("preparing_pixel_values"):
            # pixel_values is a tuple of tensors, that is always length 1 with
            # Qwen, so we can just take the first element.
            pixel_values_tensor = Tensor.from_numpy(
                self._parallel_ops.concatenate(
                    [
                        ctx.extra_model_args["concatenated_pixel_values"]
                        for ctx in context_batch
                        if ctx.needs_vision_encoding
                    ]
                )
            )
            pixel_values = [
                pixel_values_tensor.to(device) for device in self.devices
            ]

        with Tracer("preparing_window_index"):
            # Concatenate per-context window_index with cross-context offsets so indices are unique
            window_index_parts: list[npt.NDArray[np.int64]] = []
            index_offset = 0
            for ctx in context_batch:
                if ctx.needs_vision_encoding:
                    per_ctx_index = ctx.extra_model_args["window_index"].astype(
                        np.int64
                    )
                    window_index_parts.append(per_ctx_index + index_offset)
                    index_offset += int(per_ctx_index.shape[0])
            window_index_np = np.concatenate(window_index_parts, axis=0)
            window_index_tensor = Tensor.from_numpy(window_index_np)
            window_index = [
                window_index_tensor.to(device) for device in self.devices
            ]

        with Tracer("preparing_vision_position_ids"):
            vision_position_ids_tensor = Tensor.from_numpy(
                self._parallel_ops.concatenate(
                    [
                        ctx.extra_model_args["vision_position_ids"]
                        for ctx in context_batch
                        if ctx.needs_vision_encoding
                    ]
                ).astype(np.int64)
            )
            vision_position_ids = [
                vision_position_ids_tensor.to(device) for device in self.devices
            ]

        with Tracer("preparing_max_grid_size"):
            max_grid_size_value = max(
                ctx.extra_model_args["max_grid_size"].item()
                for ctx in context_batch
                if ctx.needs_vision_encoding
            )
            max_grid_size_tensor = Tensor.from_numpy(
                np.array(max_grid_size_value, dtype=np.int32)
            )
            max_grid_size = [max_grid_size_tensor for _ in self.devices]

        with Tracer("preparing_cu_seqlens"):
            # Handle cumulative offsets properly when batching
            cu_seqlens_list = []
            offset = 0
            for ctx in context_batch:
                if ctx.needs_vision_encoding:
                    seqlens = ctx.extra_model_args["cu_seqlens"]
                    adjusted = seqlens.copy()
                    adjusted[1:] += offset
                    cu_seqlens_list.append(adjusted[1:])
                    offset = adjusted[-1]

            cu_seqlens_tensor = Tensor.from_numpy(
                np.concatenate(
                    [np.array([0], dtype=np.uint32), *cu_seqlens_list]
                ).astype(np.uint32)
            )
            cu_seqlens = [
                cu_seqlens_tensor.to(device) for device in self.devices
            ]

        with Tracer("preparing_cu_window_seqlens"):
            # cu_window_seqlens_unique is already scaled by spatial_merge_unit per-context.
            # We only need to add cross-context offsets and concatenate.
            cu_window_seqlens_parts: list[npt.NDArray[np.uint32]] = []
            offset = 0
            for ctx in context_batch:
                if ctx.needs_vision_encoding:
                    seqlens_unique = ctx.extra_model_args[
                        "cu_window_seqlens_unique"
                    ].astype(np.uint32)
                    cu_window_seqlens_parts.append(
                        (seqlens_unique[1:] + offset).astype(np.uint32)
                    )
                    offset = offset + seqlens_unique[-1]

            cu_window_seqlens_np = np.concatenate(
                [np.array([0], dtype=np.uint32), *cu_window_seqlens_parts]
            ).astype(np.uint32)
            cu_window_seqlens_unique_tensor = Tensor.from_numpy(
                cu_window_seqlens_np
            )
            cu_window_seqlens = [
                cu_window_seqlens_unique_tensor.to(device)
                for device in self.devices
            ]

        with Tracer("preparing_max_seqlen"):
            max_seqlen_value = max(
                ctx.extra_model_args["max_seqlen"].item()
                for ctx in context_batch
                if ctx.needs_vision_encoding
            )
            max_seqlen_tensor = Tensor.from_numpy(
                np.array([max_seqlen_value], dtype=np.uint32)
            )
            max_seqlen = [max_seqlen_tensor for _ in self.devices]

        with Tracer("preparing_max_window_seqlen"):
            window_max_seqlen_value = max(
                ctx.extra_model_args["window_max_seqlen"].item()
                for ctx in context_batch
                if ctx.needs_vision_encoding
            )
            window_max_seqlen_tensor = Tensor.from_numpy(
                np.array([window_max_seqlen_value], dtype=np.uint32)
            )
            max_window_seqlen = [window_max_seqlen_tensor for _ in self.devices]

        return Qwen2_5VLInputs(
            input_ids=input_ids,
            input_row_offsets=input_row_offsets_tensors,
            signal_buffers=self.signal_buffers,
            position_ids=decoder_position_ids,
            return_n_logits=Tensor.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            kv_cache_inputs=kv_cache_inputs,
            image_token_indices=image_token_indices,
            pixel_values=pixel_values,
            window_index=window_index,
            vision_position_ids=vision_position_ids,
            max_grid_size=max_grid_size,
            cu_seqlens=cu_seqlens,
            cu_window_seqlens=cu_window_seqlens,
            max_seqlen=max_seqlen,
            max_window_seqlen=max_window_seqlen,
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

        row_offsets_size = old_row_offsets[0].shape[0]
        next_row_offsets = [
            offsets_prealloc[:row_offsets_size]
            for offsets_prealloc in self._input_row_offsets_prealloc
        ]

        old_row_offsets_np = old_row_offsets[0].to_numpy()
        old_position_ids_np = prev_model_inputs.position_ids.to_numpy()

        # Compute new position ids by adding 1 to the previous final position id
        # for each element in the batch.
        # TODO: check this is correct for multi-gpu
        position_ids_np = (
            old_position_ids_np[..., old_row_offsets_np[1:] - 1] + 1
        )
        position_ids = Tensor.from_numpy(position_ids_np).to(self.devices[0])

        return Qwen2_5VLInputs(
            signal_buffers=self.signal_buffers,
            input_ids=next_tokens,
            input_row_offsets=next_row_offsets,
            position_ids=position_ids,
            kv_cache_inputs=prev_model_inputs.kv_cache_inputs,
            return_n_logits=prev_model_inputs.return_n_logits,
            image_token_indices=None,
            # Leave vision inputs empty since they are only processed on the
            # first step.
            pixel_values=None,
            window_index=None,
            vision_position_ids=None,
            cu_seqlens=None,
            cu_window_seqlens=None,
            max_seqlen=None,
            max_window_seqlen=None,
            max_grid_size=None,
        )

    def load_kv_manager(
        self, session: InferenceSession, available_cache_memory: int | None
    ) -> PagedKVCacheManager:
        """Loads and initializes the PagedKVCacheManager for the Qwen2.5VL model."""
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
