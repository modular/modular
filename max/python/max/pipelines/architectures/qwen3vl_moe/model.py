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
from dataclasses import dataclass
from typing import Any

import numpy as np
from max.driver import Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, Type, Value
from max.graph.weights import (
    SafetensorWeights,
    WeightData,
    Weights,
    WeightsAdapter,
)
from max.kv_cache import (
    NullKVCacheManager,
    PagedKVCacheManager,
    estimate_kv_cache_size,
    load_kv_manager,
)
from max.nn import Module, ReturnLogits, Signals
from max.nn.kv_cache import KVCacheInputs, KVCacheParams, PagedCacheValues
from max.nn.parallel import ParallelArrayOps
from max.pipelines.lib import (
    AlwaysSignalBuffersMixin,
    KVCacheConfig,
    KVCacheMixin,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
    SupportedEncoding,
)
from max.profiler import Tracer
from transformers import AutoConfig

from .context import Qwen3VLTextAndVisionContext, VisionEncodingData
from .model_config import Qwen3VLConfig
from .qwen3vl import Qwen3VL
from .weight_adapters import convert_qwen3vl_model_state_dict

logger = logging.getLogger("max.pipelines")


@dataclass(eq=False)
class Qwen3VLInputs(ModelInputs):
    """A class representing inputs for the Qwen3VL model.

    This class encapsulates the input tensors required for the Qwen3VL model execution,
    including both text and vision inputs. Vision inputs are optional and can be None
    for text-only processing.
    """

    input_ids: Tensor
    """Tensor containing the input token IDs."""

    input_row_offsets: list[Tensor]
    """Per-device tensors containing the offsets for each row in the ragged input sequence."""

    signal_buffers: list[Tensor]
    """Device buffers used for synchronization in communication collectives."""

    decoder_position_ids: Tensor
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

    vision_position_ids: list[Tensor] | None = None
    """Vision rotary position IDs."""

    weights: list[Tensor] | None = None
    """Bilinear interpolation weights for vision position embeddings."""

    indices: list[Tensor] | None = None
    """Bilinear interpolation indices for vision position embeddings."""

    max_grid_size: list[Tensor] | None = None
    """Maximum grid size for vision inputs."""

    cu_seqlens: list[Tensor] | None = None
    """Cumulative sequence lengths for full attention."""

    max_seqlen: list[Tensor] | None = None
    """Maximum sequence length for full attention for vision inputs."""

    grid_thw: list[Tensor] | None = None
    """Grid dimensions (temporal, height, width) for each image/video, shape (n_images, 3)."""

    @property
    def has_vision_inputs(self) -> bool:
        """Check if this input contains vision data."""
        return self.pixel_values is not None


class Qwen3VLModel(
    AlwaysSignalBuffersMixin,
    PipelineModel[Qwen3VLTextAndVisionContext],
    KVCacheMixin,
):
    """A Qwen3VL pipeline model for multimodal text generation."""

    vision_model: Model
    """The compiled vision model for processing images."""

    language_model: Model
    """The compiled language model for text generation."""

    model_config: Qwen3VLConfig | None
    """The Qwen3VL model configuration."""

    _input_row_offsets_prealloc: list[Tensor]
    """Pre-allocated per-device tensors for input row offsets in multi-step execution."""

    _parallel_ops: ParallelArrayOps
    """Parallel array operations for parallel execution of concatenations."""

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

        # self.vision_model, self.language_model = self.load_model(session)
        self.vision_model, _ = self.load_model(session)
        self._parallel_ops = ParallelArrayOps(max_workers=24)

    # TODO: Seems like a common pattern. Implement in a base class?
    @staticmethod
    def calculate_max_seq_len(
        pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        """Calculates the maximum sequence length for the Qwen3VL model."""
        return Qwen3VLConfig.calculate_max_seq_len(
            pipeline_config, huggingface_config
        )

    # TODO: Seems like a common pattern. Implement in a base class?
    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        """Gets the parameters required to configure the KV cache for Qwen3VL."""
        return Qwen3VLConfig.get_kv_params(
            huggingface_config, n_devices, kv_cache_config, cache_dtype
        )

    # TODO: Seems like a common pattern. Implement in a base class?
    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        """Gets the number of hidden layers from the HuggingFace configuration."""
        return Qwen3VLConfig.get_num_layers(huggingface_config)

    # TODO: Seems like a common pattern. Implement in a base class?
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
        """Estimates the size of the KV cache required for the Qwen3VL model in bytes."""
        return estimate_kv_cache_size(
            params=Qwen3VLConfig.get_kv_params(
                huggingface_config=huggingface_config,
                n_devices=len(devices),
                kv_cache_config=kv_cache_config,
                cache_dtype=cache_dtype,
            ),
            max_batch_size=pipeline_config.max_batch_size,
            max_seq_len=cls.calculate_max_seq_len(
                pipeline_config, huggingface_config=huggingface_config
            ),
            available_cache_memory=available_cache_memory,
        )

    # TODO: Seems like a common pattern. Implement in a base class?
    def _unflatten_kv_inputs(
        self, kv_inputs_flat: Sequence[Value[Any]]
    ) -> list[PagedCacheValues]:
        """Unflatten KV cache inputs from flat list to per-device structure."""
        fetch_types = self.kv_manager.get_symbolic_inputs()[0]
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

    def load_model(
        self, session: InferenceSession
    ) -> tuple[Model, Model | None]:
        """Loads the compiled Qwen3VL models into the MAX Engine session.

        Returns:
            A tuple of (vision_model, language_model).
        """
        # TODO: Pre-allocation Seems like a common pattern. Implement in a base class?
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

        # Split state dict into vision and language model components
        vision_state_dict: dict[str, WeightData] = {}
        llm_state_dict: dict[str, WeightData] = {}
        for key, value in model_state_dict.items():
            if key.startswith("vision_encoder."):
                # TODO: update this to keep the vision_encoder prefix once the language model is implemented
                vision_state_dict[key[len("vision_encoder.") :]] = value
            elif key.startswith("language_model."):
                llm_state_dict[key] = value
            else:
                raise ValueError(
                    f"Key: {key} is not part of the vision or language model"
                )

        # Generate Qwen3VL config from HuggingFace config
        qwen3vl_config = Qwen3VLConfig.generate(
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
        self.model_config = qwen3vl_config

        # TODO: load weights into the model
        self.model: Module = Qwen3VL(self.model_config)
        # self.model.load_state_dict(model_state_dict, strict=True)
        # For now, load weights into the vision model only.
        self.model.vision_encoder.load_state_dict(
            state_dict=vision_state_dict,
            strict=True,
        )

        # Build and compile vision model
        logger.info("Building and compiling vision model...")
        before = time.perf_counter()
        vision_graph = self._build_vision_graph(
            qwen3vl_config, vision_state_dict
        )
        after_build = time.perf_counter()

        logger.info(
            f"Building vision graph took {after_build - before:.6f} seconds"
        )

        before_compile = time.perf_counter()
        vision_model = session.load(
            vision_graph, weights_registry=vision_state_dict
        )
        after = time.perf_counter()

        logger.info(
            f"Compiling vision model took {after - before_compile:.6f} seconds"
        )

        logger.info(
            f"Building and compiling vision model took {after - before:.6f} seconds"
        )

        # TODO: Build and compile language model
        language_model = None
        # logger.info("Building and compiling language model...")
        # before = time.perf_counter()
        # language_graph, language_model_state_dict = self._build_language_graph(
        #     qwen3vl_config, llm_state_dict
        # )
        # after_build = time.perf_counter()

        # logger.info(
        #     f"Building language graph took {after_build - before:.6f} seconds"
        # )

        # before_compile = time.perf_counter()
        # language_model = session.load(
        #     language_graph, weights_registry=language_model_state_dict
        # )
        # after = time.perf_counter()

        # logger.info(
        #     f"Compiling language model took {after - before_compile:.6f} seconds"
        # )

        # logger.info(
        #     f"Building and compiling language model took {after - before:.6f} seconds"
        # )

        return vision_model, language_model

    def _build_vision_graph(
        self, config: Qwen3VLConfig, state_dict: dict[str, WeightData]
    ) -> Graph:
        """Build the vision model graph for processing images."""
        assert isinstance(self.model, Qwen3VL)
        vision_encoder = self.model.vision_encoder

        # Define vision graph input types - one per device
        pixel_values_types = [
            TensorType(
                DType.float32,
                shape=["vision_seq_len", vision_encoder.patch_embed.patch_dim],
                device=DeviceRef.from_device(device),
            )
            for device in self.devices
        ]

        weights_types = [
            TensorType(
                DType.float32,
                shape=[4, "vision_seq_len", 1],
                device=DeviceRef.from_device(device),
            )
            for device in self.devices
        ]

        indices_types = [
            TensorType(
                DType.int64,
                shape=[4, "vision_seq_len"],
                device=DeviceRef.from_device(device),
            )
            for device in self.devices
        ]

        vision_rot_pos_ids_types = [
            TensorType(
                DType.int32,
                shape=["vision_seq_len", 2],
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
            for _ in self.devices
        ]

        grid_thw_types = [
            TensorType(
                DType.int64,
                shape=["n_images", 3],
                device=DeviceRef.from_device(device),
            )
            for device in self.devices
        ]

        cu_seqlens_types = [
            TensorType(
                DType.uint32,
                shape=["n_seqlens"],
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

        # Create signal types for distributed communication
        signals = Signals(
            devices=(DeviceRef(d.label, d.id) for d in self.devices)
        )

        # Build the vision graph
        with Graph(
            "qwen3vl_vision",
            input_types=tuple(
                [
                    *pixel_values_types,
                    *weights_types,
                    *indices_types,
                    *vision_rot_pos_ids_types,
                    *max_grid_size_types,
                    *grid_thw_types,
                    *cu_seqlens_types,
                    *max_seqlen_types,
                    *signals.input_types(),
                ]
            ),
        ) as graph:
            # Extract inputs
            all_inputs = graph.inputs
            n_devices = len(self.devices)

            pixel_values_list = [inp.tensor for inp in all_inputs[:n_devices]]
            all_inputs = all_inputs[n_devices:]

            weights_list = [inp.tensor for inp in all_inputs[:n_devices]]
            all_inputs = all_inputs[n_devices:]

            indices_list = [inp.tensor for inp in all_inputs[:n_devices]]
            all_inputs = all_inputs[n_devices:]

            rot_pos_ids_list = [inp.tensor for inp in all_inputs[:n_devices]]
            all_inputs = all_inputs[n_devices:]

            max_grid_size_list = [inp.tensor for inp in all_inputs[:n_devices]]
            all_inputs = all_inputs[n_devices:]

            grid_thw_list = [inp.tensor for inp in all_inputs[:n_devices]]
            all_inputs = all_inputs[n_devices:]

            cu_seqlens_list = [inp.tensor for inp in all_inputs[:n_devices]]
            all_inputs = all_inputs[n_devices:]

            max_seqlen_list = [inp.tensor for inp in all_inputs[:n_devices]]
            all_inputs = all_inputs[n_devices:]

            signal_buffers = [inp.buffer for inp in all_inputs]

            # Execute vision transformer
            image_embeddings, deepstack_features = vision_encoder(
                pixel_values=pixel_values_list,
                idxs=indices_list,
                weights=weights_list,
                grid_thw=grid_thw_list,
                rot_pos_ids=rot_pos_ids_list,
                max_grid_size=max_grid_size_list,
                cu_seqlens=cu_seqlens_list,
                max_seqlen=max_seqlen_list,
                signal_buffers=signal_buffers,
            )
            # Ensure we have a valid output
            assert (
                image_embeddings is not None and deepstack_features is not None
            ), "Vision encoder must return a valid output"

            graph.output(
                *[
                    *image_embeddings,
                    *[
                        item
                        for sublist in deepstack_features
                        for item in sublist
                    ],
                ]
            )

            return graph

    def _language_graph_input_types(self) -> tuple[Type[Any], ...]:
        """Generate input types for the language model graph."""
        device_ref = DeviceRef.from_device(self.devices[0])

        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )

        kv_inputs = self.kv_manager.get_symbolic_inputs()

        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        input_row_offsets_types = [
            TensorType(
                DType.uint32,
                shape=["input_row_offsets_len"],
                device=DeviceRef.from_device(dev),
            )
            for dev in self.devices
        ]

        # Add image embeddings type - one per device, can be empty for text-only inputs
        assert self.model_config is not None, "Model config must be initialized"
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

        # Add image token indices type
        image_token_indices_types = [
            TensorType(
                DType.int32,
                shape=["total_image_tokens"],
                device=DeviceRef.from_device(device),
            )
            for device in self.devices
        ]

        # Add decoder position IDs type (3D for mrope)
        position_ids_type = TensorType(
            DType.int64,
            shape=[len(self.model_config.mrope_section), "total_seq_len"],
            device=device_ref,
        )

        # Flatten kv types for each device
        flattened_kv_types = [
            kv_type for sublist in kv_inputs for kv_type in sublist
        ]

        signals = Signals(
            devices=(DeviceRef(d.label, d.id) for d in self.devices)
        )

        return (
            tokens_type,
            return_n_logits_type,
            *input_row_offsets_types,
            *image_embeddings_types,
            *image_token_indices_types,
            position_ids_type,
            *signals.input_types(),
            *flattened_kv_types,
        )

    def _build_language_graph(
        self, config: Qwen3VLConfig, state_dict: dict[str, WeightData]
    ) -> tuple[Graph, dict[str, Any]]:
        """Build the language model graph for text generation with image embeddings."""
        # TODO: Implement Qwen3VLLanguageModel that handles image embeddings merging
        # For now, this is a placeholder that will need to be completed
        # The language model should merge image embeddings with text embeddings
        # at image token positions, similar to how InternVL or Qwen2.5VL does it

        raise NotImplementedError(
            "Language model graph building for Qwen3VL is not yet implemented. "
            "A Qwen3VLLanguageModel class that handles image embeddings merging "
            "needs to be created first."
        )

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Executes the Qwen3VL model with the prepared inputs."""
        assert isinstance(model_inputs, Qwen3VLInputs)
        assert model_inputs.kv_cache_inputs is not None, (
            "Qwen3VL requires KV cache inputs"
        )

        # Process vision inputs if present
        image_embeddings: list[Tensor]
        image_token_indices: list[Tensor]
        if model_inputs.has_vision_inputs:
            assert model_inputs.pixel_values is not None
            assert model_inputs.vision_position_ids is not None
            assert model_inputs.weights is not None
            assert model_inputs.indices is not None
            assert model_inputs.max_grid_size is not None
            assert model_inputs.cu_seqlens is not None
            assert model_inputs.max_seqlen is not None
            assert model_inputs.grid_thw is not None

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
            assert len(vision_outputs) == len(self.devices)

            image_embeddings = [
                output
                for output in vision_outputs
                if isinstance(output, Tensor)
            ]
            image_token_indices = model_inputs.image_token_indices or [
                Tensor.zeros(shape=[0], dtype=DType.int32).to(dev)
                for dev in self.devices
            ]
        else:
            # Initialize empty tensors for text-only mode
            assert self.model_config is not None
            image_embeddings = [
                Tensor.zeros(
                    shape=[0, self.model_config.llm_config.hidden_size],
                    dtype=self.dtype,
                ).to(dev)
                for dev in self.devices
            ]
            image_token_indices = [
                Tensor.zeros(shape=[0], dtype=DType.int32).to(dev)
                for dev in self.devices
            ]

        # Prepare KV cache inputs as list of tensors
        assert model_inputs.kv_cache_inputs
        kv_cache_inputs_list = list(model_inputs.kv_cache_inputs)

        # Execute language model with text and image embeddings and deepstack features
        # TODO: Execute language model with text and image embeddings
        language_outputs: list[Tensor] = []

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
        context_batch: Sequence[Qwen3VLTextAndVisionContext],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> Qwen3VLInputs:
        """Prepares the initial inputs for the first execution pass of the Qwen3VL model."""
        if kv_cache_inputs is None:
            raise ValueError("KV Cache Inputs must be provided")

        # Gather all vision data from contexts that need vision encoding
        vision_datas: list[VisionEncodingData] = []
        for ctx in context_batch:
            # Validate all contexts are the correct type
            assert isinstance(ctx, Qwen3VLTextAndVisionContext), (
                f"Expected Qwen3VLTextAndVisionContext, got {type(ctx).__name__}"
            )
            if ctx.needs_vision_encoding:
                assert ctx.vision_data is not None, (
                    "vision_data must be present when needs_vision_encoding is True"
                )
                vision_datas.append(ctx.vision_data)
        any_needs_vision_encoding = len(vision_datas) > 0

        # Prepare Inputs Needed Regardless of Images
        with Tracer("prepare_input_ids"):
            input_ids = Tensor.from_numpy(
                np.concatenate([ctx.next_tokens for ctx in context_batch])
            ).to(self.devices[0])

        with Tracer("prepare_input_row_offsets"):
            input_row_offsets_host = Tensor.from_numpy(
                np.cumsum(
                    [0] + [ctx.active_length for ctx in context_batch],
                    dtype=np.uint32,
                ),
            )
            input_row_offsets = [
                input_row_offsets_host.to(dev) for dev in self.devices
            ]

        with Tracer("prepare_decoder_position_ids"):
            decoder_position_ids_list = []
            for ctx in context_batch:
                ctx_decoder_position_ids = ctx.decoder_position_ids
                if (
                    ctx.needs_vision_encoding
                    and ctx_decoder_position_ids.shape[1] == ctx.current_length
                ):
                    decoder_position_ids_list.append(
                        ctx_decoder_position_ids[
                            :, ctx.start_idx : ctx.active_idx
                        ]
                    )
                else:
                    # Recompute or use simple position IDs
                    # TODO: Implement proper position ID computation for Qwen3VL
                    context_seq_length = ctx.active_length
                    # Qwen3VL uses 3D position IDs (mrope)
                    temp_pos_ids = np.tile(
                        np.arange(context_seq_length).reshape(1, 1, -1),
                        (
                            len(self.model_config.mrope_section)
                            if self.model_config
                            else 3,
                            1,
                            1,
                        ),
                    )
                    delta = ctx.start_idx + ctx.rope_delta
                    temp_position_ids = (temp_pos_ids + delta).squeeze(1)
                    decoder_position_ids_list.append(temp_position_ids)

            decoder_position_ids = Tensor.from_numpy(
                np.concatenate(decoder_position_ids_list, axis=1).astype(
                    np.int64
                )
            ).to(self.devices[0])

        # Batch image token indices
        with Tracer("prepare_image_token_indices"):
            image_token_indices_list = []
            batch_offset = 0
            for ctx in context_batch:
                if ctx.needs_vision_encoding:
                    indices = ctx.image_token_indices
                    image_token_indices_list.append(indices + batch_offset)
                batch_offset += ctx.active_length

            if image_token_indices_list:
                np_image_token_indices = np.concatenate(
                    image_token_indices_list
                ).astype(np.int32, copy=False)
                image_token_indices = [
                    Tensor.from_numpy(np_image_token_indices).to(dev)
                    for dev in self.devices
                ]
            else:
                image_token_indices = None

        if not any_needs_vision_encoding:
            return Qwen3VLInputs(
                input_ids=input_ids,
                input_row_offsets=input_row_offsets,
                signal_buffers=self.signal_buffers,
                decoder_position_ids=decoder_position_ids,
                return_n_logits=Tensor.from_numpy(
                    np.array([return_n_logits], dtype=np.int64)
                ),
                kv_cache_inputs=kv_cache_inputs,
                image_token_indices=image_token_indices,
                pixel_values=None,
                vision_position_ids=None,
                weights=None,
                indices=None,
                max_grid_size=None,
                cu_seqlens=None,
                max_seqlen=None,
                grid_thw=None,
            )

        # From here on, assume that all inputs are available in vision_data
        # Prepare vision inputs
        pixel_values_list = [
            vision_data.concatenated_pixel_values
            for vision_data in vision_datas
        ]
        pixel_values_tensor = Tensor.from_numpy(
            np.concatenate(pixel_values_list)
        )
        pixel_values = [pixel_values_tensor.to(dev) for dev in self.devices]

        # Prepare bilinear interpolation weights and indices
        weights_tensor = Tensor.from_numpy(
            np.concatenate(
                [vision_data.weights for vision_data in vision_datas]
            )
        )
        weights_list = [weights_tensor.to(dev) for dev in self.devices]

        indices_tensor = Tensor.from_numpy(
            np.concatenate(
                [vision_data.indices for vision_data in vision_datas]
            )
        )
        indices_list = [indices_tensor.to(dev) for dev in self.devices]

        # Prepare vision position IDs
        vision_position_ids_list = [
            vision_data.vision_position_ids for vision_data in vision_datas
        ]
        vision_position_ids_tensor = Tensor.from_numpy(
            np.concatenate(vision_position_ids_list).astype(np.int32)
        )
        vision_position_ids = [
            vision_position_ids_tensor.to(dev) for dev in self.devices
        ]

        # Prepare grid_thw
        grid_thw_list = [
            vision_data.image_grid_thw for vision_data in vision_datas
        ]
        grid_thw_tensor = Tensor.from_numpy(
            np.concatenate(grid_thw_list).astype(np.int64)
        )
        grid_thw = [grid_thw_tensor.to(dev) for dev in self.devices]

        # Prepare max_grid_size
        max_grid_size_value = max(
            vision_data.max_grid_size.item() for vision_data in vision_datas
        )
        max_grid_size_tensor = Tensor.from_numpy(
            np.array(max_grid_size_value, dtype=np.int32)
        )
        max_grid_size = [max_grid_size_tensor for _ in self.devices]

        # Prepare cu_seqlens
        cu_seqlens_list = []
        offset = 0
        for vision_data in vision_datas:
            seqlens = vision_data.cu_seqlens
            adjusted = seqlens.copy()
            adjusted[1:] += offset
            cu_seqlens_list.append(adjusted[1:])
            offset = adjusted[-1]

        cu_seqlens_tensor = Tensor.from_numpy(
            np.concatenate(
                [np.array([0], dtype=np.uint32), *cu_seqlens_list]
            ).astype(np.uint32)
        )
        cu_seqlens = [cu_seqlens_tensor.to(dev) for dev in self.devices]

        # Prepare max_seqlen
        max_seqlen_value = max(
            vision_data.max_seqlen.item() for vision_data in vision_datas
        )
        max_seqlen_tensor = Tensor.from_numpy(
            np.array([max_seqlen_value], dtype=np.uint32)
        )
        max_seqlen = [max_seqlen_tensor for _ in self.devices]

        return Qwen3VLInputs(
            input_ids=input_ids,
            input_row_offsets=input_row_offsets,
            signal_buffers=self.signal_buffers,
            decoder_position_ids=decoder_position_ids,
            return_n_logits=Tensor.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            kv_cache_inputs=kv_cache_inputs,
            image_token_indices=image_token_indices,
            pixel_values=pixel_values,
            vision_position_ids=vision_position_ids,
            weights=weights_list,
            indices=indices_list,
            max_grid_size=max_grid_size,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            grid_thw=grid_thw,
        )

    def prepare_next_token_inputs(
        self, next_tokens: Tensor, prev_model_inputs: ModelInputs
    ) -> Qwen3VLInputs:
        """Prepares the inputs for subsequent execution steps in a multi-step generation."""
        assert isinstance(prev_model_inputs, Qwen3VLInputs)
        prev_inputs = prev_model_inputs

        # Use pre-allocated row offsets for next token
        offset = prev_inputs.input_row_offsets[0].shape[0]
        next_row_offsets = [
            offsets_prealloc[:offset]
            for offsets_prealloc in self._input_row_offsets_prealloc
        ]

        # Compute new position ids by adding 1 to the previous final position id
        old_row_offsets_np = prev_inputs.input_row_offsets[0].to_numpy()
        old_position_ids_np = prev_inputs.decoder_position_ids.to_numpy()

        # For 3D position IDs (mrope), update each dimension
        position_ids_np = old_position_ids_np[:, old_row_offsets_np[1:] - 1] + 1
        decoder_position_ids = Tensor.from_numpy(position_ids_np).to(
            self.devices[0]
        )

        return Qwen3VLInputs(
            signal_buffers=self.signal_buffers,
            input_ids=next_tokens,
            input_row_offsets=next_row_offsets,
            decoder_position_ids=decoder_position_ids,
            kv_cache_inputs=prev_inputs.kv_cache_inputs,
            return_n_logits=prev_inputs.return_n_logits,
            # Set vision model inputs to None after the first step
            image_token_indices=None,
            pixel_values=None,
            vision_position_ids=None,
            weights=None,
            indices=None,
            cu_seqlens=None,
            max_seqlen=None,
            max_grid_size=None,
            grid_thw=None,
        )

    def load_kv_manager(
        self, session: InferenceSession, available_cache_memory: int | None
    ) -> PagedKVCacheManager | NullKVCacheManager:
        """Loads and initializes the PagedKVCacheManager for the Qwen3VL model."""
        return load_kv_manager(
            params=Qwen3VLConfig.get_kv_params(
                huggingface_config=self.huggingface_config,
                n_devices=len(self.devices),
                kv_cache_config=self.kv_cache_config,
                cache_dtype=self.encoding.cache_dtype,
            ),
            max_batch_size=self.pipeline_config.max_batch_size,
            max_seq_len=self.calculate_max_seq_len(
                self.pipeline_config, huggingface_config=self.huggingface_config
            ),
            devices=self.devices,
            available_cache_memory=available_cache_memory,
            session=session,
        )
