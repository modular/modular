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
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np
import numpy.typing as npt
from max._core.engine import Model
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine.api import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, Value
from max.graph.buffer_utils import cast_tensors_to
from max.graph.weights import (
    SafetensorWeights,
    WeightData,
    Weights,
    WeightsAdapter,
)
from max.nn.comm import Signals
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheParams,
    PagedCacheValues,
)
from max.nn.layer import Module
from max.nn.parallel import ParallelArrayOps
from max.nn.transformer import ReturnLogits
from max.pipelines.core import TextAndVisionContext
from max.pipelines.lib import (
    AlwaysSignalBuffersMixin,
    CompilationTimer,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModelWithKVCache,
)
from max.profiler import Tracer, traced
from transformers import AutoConfig

from .context import Qwen2_5VLTextAndVisionContext, VisionEncodingData
from .model_config import Qwen2_5VLConfig
from .nn.data_processing import get_rope_index
from .qwen2_5vl import Qwen2_5VL
from .util import compute_multimodal_merge_indices

logger = logging.getLogger("max.pipelines")


@dataclass
class Qwen2_5VLInputs(ModelInputs):
    """A class representing inputs for the Qwen2.5VL model.

    This class encapsulates the input tensors required for the Qwen2.5VL model execution,
    including both text and vision inputs. Vision inputs are optional and can be None
    for text-only processing."""

    input_ids: Buffer
    """Tensor containing the input token IDs."""

    input_row_offsets: list[Buffer]
    """Per-device tensors containing the offsets for each row in the ragged input sequence."""

    signal_buffers: list[Buffer]
    """Device buffers used for synchronization in communication collectives."""

    position_ids: Buffer
    """3D RoPE position IDs for the decoder."""

    return_n_logits: Buffer
    """Number of logits to return, used by speculative decoding for example."""

    kv_cache_inputs: KVCacheInputs = field(kw_only=True)
    """KV cache inputs for the model."""

    image_token_indices: list[Buffer] | None = None
    """Per-device pre-computed multimodal merge indices for the image embeddings.

    These are the locations of the image_token_id in the inputs fed to the model.

    Some indices may be negative, which means that they are ignored by the multimodal merge."""

    # Vision inputs.
    pixel_values: list[Buffer] | None = None
    """Pixel values for vision inputs."""

    window_index: list[Buffer] | None = None
    """Window indices for vision attention mechanism."""

    vision_position_ids: list[Buffer] | None = None
    """1D RoPE position IDs for the visual inputs."""

    max_grid_size: list[Buffer] | None = None
    """Maximum grid size for vision inputs."""

    cu_seqlens: list[Buffer] | None = None
    """Cumulative sequence lengths for full attention."""

    cu_window_seqlens: list[Buffer] | None = None
    """Cumulative window sequence lengths for window attention."""

    max_seqlen: list[Buffer] | None = None
    """Maximum sequence length for full attention for vision inputs."""

    max_window_seqlen: list[Buffer] | None = None
    """Maximum sequence length for window attention for vision inputs."""

    @property
    def has_vision_inputs(self) -> bool:
        """Check if this input contains vision data."""
        return self.pixel_values is not None


class Qwen2_5VLModel(
    AlwaysSignalBuffersMixin, PipelineModelWithKVCache[TextAndVisionContext]
):
    """A Qwen2.5VL pipeline model for multimodal text generation."""

    vision_model: Model
    """The compiled vision model for processing images."""

    language_model: Model
    """The compiled language model for text generation."""

    model_config: Qwen2_5VLConfig | None
    """The Qwen2.5VL model configuration."""

    _input_row_offsets_prealloc: list[Buffer]
    """Pre-allocated per-device tensors for input row offsets in multi-step execution."""

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

        self.model_config = None
        self._session = session  # reuse for on-device casts

        gpu0 = devices[0]
        if gpu0.is_host:
            raise ValueError("Qwen2.5VL currently only supports GPU devices")
        self._parallel_ops = ParallelArrayOps(accelerator=gpu0, max_workers=24)

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
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        """Gets the parameters required to configure the KV cache for Qwen2.5VL."""
        return Qwen2_5VLConfig.construct_kv_params(
            huggingface_config,
            pipeline_config,
            devices,
            kv_cache_config,
            cache_dtype,
        )

    def _unflatten_kv_inputs(
        self, kv_inputs_flat: Sequence[Value[Any]]
    ) -> list[PagedCacheValues]:
        """Unflatten KV cache inputs from flat list to per-device structure."""
        fetch_types = self.kv_params.get_symbolic_inputs()[0]
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
        self._input_row_offsets_prealloc = Buffer.from_numpy(
            np.arange(self.pipeline_config.max_batch_size + 1, dtype=np.uint32)
        ).to(self.devices)

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
        qwen2_5vl_config = Qwen2_5VLConfig.initialize_from_config(
            pipeline_config=self.pipeline_config,
            huggingface_config=self.huggingface_config,
        )
        qwen2_5vl_config.finalize(
            huggingface_config=self.huggingface_config,
            pipeline_config=self.pipeline_config,
            llm_state_dict=llm_state_dict,
            vision_state_dict=vision_state_dict,
            return_logits=self.return_logits,
        )
        self.model_config = qwen2_5vl_config

        assert self.model_config is not None, "Model config must be initialized"
        self.model: Module = Qwen2_5VL(self.model_config)
        self.model.load_state_dict(model_state_dict, strict=True)

        timer = CompilationTimer("vision model")
        vision_graph = self._build_vision_graph()
        timer.mark_build_complete()
        vision_model = session.load(
            vision_graph, weights_registry=vision_state_dict
        )
        timer.done()

        timer = CompilationTimer("language model")
        language_graph = self._build_language_graph()
        timer.mark_build_complete()
        language_model = session.load(
            language_graph, weights_registry=llm_state_dict
        )
        timer.done()

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
                DType.bfloat16,
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

        kv_inputs = self.kv_params.get_symbolic_inputs()
        flattened_kv_types = kv_inputs.flatten()

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
    def _empty_image_embeddings(self) -> list[Buffer]:
        """Create empty image embeddings for text-only inputs on multi-device."""
        return Buffer.zeros(
            shape=[0, self.huggingface_config.text_config.hidden_size],
            dtype=self.dtype,
        ).to(self.devices)

    @cached_property
    def _empty_image_image_token_indices(self) -> list[Buffer]:
        """Create empty image scatter indices for text-only inputs on multi-device."""
        return Buffer.zeros(
            shape=[0],
            dtype=DType.int32,
        ).to(self.devices)

    def _batch_image_token_indices(
        self, context_batch: Sequence[Qwen2_5VLTextAndVisionContext]
    ) -> list[Buffer]:
        """Batch image token indices from multiple contexts, adjusting for
        position in batch.

        This method efficiently combines image token indices from multiple
        contexts using vectorized operations.

        Args:
            context_batch: Sequence of contexts that may contain image token indices

        Returns:
            List of tensors containing all multimodal merge indices distributed
            across devices
        """
        assert self.model_config is not None, "Model config must be initialized"

        np_image_token_indices = compute_multimodal_merge_indices(context_batch)

        # Create tensor and distribute to devices
        return Buffer.from_numpy(np_image_token_indices).to(self.devices)

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Executes the Qwen2.5VL model with the prepared inputs."""
        assert isinstance(model_inputs, Qwen2_5VLInputs)
        assert model_inputs.kv_cache_inputs is not None, (
            "Qwen2.5VL requires KV cache inputs"
        )

        # Process vision inputs if present
        image_embeddings: list[Buffer]

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
                if isinstance(output, Buffer)
            ]
            image_embeddings = cast_tensors_to(
                image_embeddings, self.dtype, self._session
            )

            image_token_indices = model_inputs.image_token_indices

            # The size of scatter indices must match the number of image embeddings.
            assert (
                image_token_indices[0].shape[0] == image_embeddings[0].shape[0]
            )

            # Normalize index dtypes to match the language graph contract.
            image_token_indices = cast_tensors_to(
                image_token_indices, DType.int32, self._session
            )
        else:
            # Initialize empty tensors for text-only mode
            image_embeddings = self._empty_image_embeddings
            image_token_indices = self._empty_image_image_token_indices

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

    @staticmethod
    def prepare_decoder_position_ids(
        context_batch: Sequence[Qwen2_5VLTextAndVisionContext],
        devices: list[Device],
    ) -> Buffer:
        """Prepare decoder position IDs for a batch of contexts.

        This function computes position IDs for decoder tokens, handling three cases:
        1. Vision encoding with pre-computed position IDs (use stored values)
        2. Vision encoding requiring recomputation (after preemption)
        3. Text-only generation (simple arange with offset)

        Optimized implementation: pre-allocates output array and writes directly,
        avoiding concatenation overhead for better performance.

        Args:
            context_batch: Sequence of Qwen2.5VL contexts to process
            devices: List of devices to place the output tensor on

        Returns:
            Buffer containing decoder position IDs with shape [n_rope_sections, total_seq_len]
        """
        # Optimize concatenation: pre-allocate output array and write directly
        # Calculate total output size first
        total_seq_len = sum(ctx.tokens.active_length for ctx in context_batch)
        n_rope_sections = 3  # Fixed for Qwen2.5VL

        # Fast path for single context: avoid concatenation overhead
        if len(context_batch) == 1:
            ctx = context_batch[0]
            ctx_decoder_position_ids = ctx.decoder_position_ids

            # - For each text token, the position id increases by one each time.
            # - Each image token of same image has the same position id as they
            #   occupy the same "position" in the sequence.
            # - The entire image takes up some number of positions so there may
            #   by a jump > 1 at the image end boundary.
            #
            # eg:
            #               token_ids = [10, 11, 12, 13, IMG, IMG, IMG, IMG, IMG, 14, 15]
            # temp_position_ids[0, :] = [0, 1, 2, 3, 4, 4, 4, 4, 4, 7, 8]
            #                                                 jump ^
            if ctx.needs_vision_encoding and ctx_decoder_position_ids.shape[
                1
            ] == len(ctx.tokens):
                result_array = ctx_decoder_position_ids[
                    :, ctx.tokens.processed_length : ctx.tokens.current_position
                ].astype(np.uint32, copy=False)
            elif ctx.needs_vision_encoding:
                # Recompute decoder_position_ids using get_rope_index
                # This handles the case after preemption where we need to recompute the prompt

                # Extract required parameters from context
                spatial_merge_size = ctx.spatial_merge_size
                image_token_id = ctx.image_token_id
                video_token_id = ctx.video_token_id
                vision_start_token_id = ctx.vision_start_token_id
                tokens_per_second = ctx.tokens_per_second
                image_grid_thw = (
                    ctx.vision_data.image_grid_thw
                    if ctx.vision_data is not None
                    else None
                )

                # Always create a fresh attention mask based on current context length
                # The stored attention_mask in extra_model_args may be outdated if tokens
                # were added after context creation (e.g., during generation before reset)
                attention_mask = np.ones((1, len(ctx.tokens)), dtype=np.float32)

                # Recompute position_ids using get_rope_index (same logic as tokenizer)
                temp_position_ids, rope_delta_array = get_rope_index(
                    spatial_merge_size=spatial_merge_size,
                    image_token_id=image_token_id,
                    video_token_id=video_token_id,
                    vision_start_token_id=vision_start_token_id,
                    tokens_per_second=tokens_per_second,
                    input_ids=ctx.tokens[: len(ctx.tokens)].reshape(1, -1),
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=None,
                    second_per_grid_ts=None,
                    attention_mask=attention_mask,
                )
                temp_position_ids = temp_position_ids.squeeze(1)

                # Update rope_delta in context if needed
                ctx.rope_delta = int(rope_delta_array.item())

                # Slice to get only the active portion and convert type
                result_array = temp_position_ids[
                    :, ctx.tokens.processed_length : ctx.tokens.current_position
                ].astype(np.uint32, copy=False)
            else:
                # This case should only happen during Token Generation

                # Recompute this value on the fly.
                # This assumes that there are no image placeholder tokens in
                # next_tokens so it is a simple arange operation.
                context_seq_length = ctx.tokens.active_length
                temp_position_ids = np.arange(context_seq_length)
                temp_position_ids = temp_position_ids.reshape(1, 1, -1)
                temp_position_ids = np.tile(temp_position_ids, (3, 1, 1))
                # Offset by the number of previous tokens (start_idx).
                delta = ctx.tokens.processed_length + ctx.rope_delta
                temp_position_ids = temp_position_ids + delta
                result_array = temp_position_ids.squeeze(1).astype(
                    np.uint32, copy=False
                )

            decoder_position_ids = Buffer.from_numpy(result_array).to(
                devices[0]
            )
            return decoder_position_ids

        # Multi-context path: pre-allocate output and write directly
        # Pre-allocate output array with correct dtype to avoid conversion overhead
        out_array = np.empty(
            (n_rope_sections, total_seq_len), dtype=np.uint32, order="C"
        )

        # Track current write position
        write_offset = 0

        for ctx in context_batch:
            ctx_decoder_position_ids = ctx.decoder_position_ids
            active_len = ctx.tokens.active_length

            # - For each text token, the position id increases by one each time.
            # - Each image token of same image has the same position id as they
            #   occupy the same "position" in the sequence.
            # - The entire image takes up some number of positions so there may
            #   by a jump > 1 at the image end boundary.
            #
            # eg:
            #               token_ids = [10, 11, 12, 13, IMG, IMG, IMG, IMG, IMG, 14, 15]
            # temp_position_ids[0, :] = [0, 1, 2, 3, 4, 4, 4, 4, 4, 7, 8]
            #                                                 jump ^
            if ctx.needs_vision_encoding and ctx_decoder_position_ids.shape[
                1
            ] == len(ctx.tokens):
                # Direct slice and write to output
                src_slice = ctx_decoder_position_ids[
                    :, ctx.tokens.processed_length : ctx.tokens.current_position
                ]
                out_array[:, write_offset : write_offset + active_len] = (
                    src_slice.astype(np.uint32, copy=False)
                )
            elif ctx.needs_vision_encoding:
                # Recompute decoder_position_ids using get_rope_index
                # This handles the case after preemption where we need to recompute the prompt

                # Extract required parameters from context
                spatial_merge_size = ctx.spatial_merge_size
                image_token_id = ctx.image_token_id
                video_token_id = ctx.video_token_id
                vision_start_token_id = ctx.vision_start_token_id
                tokens_per_second = ctx.tokens_per_second
                image_grid_thw = (
                    ctx.vision_data.image_grid_thw
                    if ctx.vision_data is not None
                    else None
                )

                # Always create a fresh attention mask based on current context length
                # The stored attention_mask in extra_model_args may be outdated if tokens
                # were added after context creation (e.g., during generation before reset)
                attention_mask = np.ones((1, len(ctx.tokens)), dtype=np.float32)

                # Recompute position_ids using get_rope_index (same logic as tokenizer)
                temp_position_ids, rope_delta_array = get_rope_index(
                    spatial_merge_size=spatial_merge_size,
                    image_token_id=image_token_id,
                    video_token_id=video_token_id,
                    vision_start_token_id=vision_start_token_id,
                    tokens_per_second=tokens_per_second,
                    input_ids=ctx.tokens[: len(ctx.tokens)].reshape(1, -1),
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=None,
                    second_per_grid_ts=None,
                    attention_mask=attention_mask,
                )
                temp_position_ids = temp_position_ids.squeeze(1)

                # Update rope_delta in context if needed
                ctx.rope_delta = int(rope_delta_array.item())

                # Slice to get only the active portion and write directly
                src_slice = temp_position_ids[
                    :, ctx.tokens.processed_length : ctx.tokens.current_position
                ]
                out_array[:, write_offset : write_offset + active_len] = (
                    src_slice.astype(np.uint32, copy=False)
                )
            else:
                # This case should only happen during Token Generation

                # Recompute this value on the fly.
                # This assumes that there are no image placeholder tokens in
                # next_tokens so it is a simple arange operation.
                context_seq_length = ctx.tokens.active_length
                temp_position_ids = np.arange(context_seq_length)
                temp_position_ids = temp_position_ids.reshape(1, 1, -1)
                temp_position_ids = np.tile(temp_position_ids, (3, 1, 1))
                # Offset by the number of previous tokens (start_idx).
                delta = ctx.tokens.processed_length + ctx.rope_delta
                temp_position_ids = temp_position_ids + delta
                temp_position_ids = temp_position_ids.squeeze(1)
                out_array[:, write_offset : write_offset + active_len] = (
                    temp_position_ids.astype(np.uint32, copy=False)
                )

            write_offset += active_len

        decoder_position_ids = Buffer.from_numpy(out_array).to(devices[0])

        return decoder_position_ids

    @traced
    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[Qwen2_5VLTextAndVisionContext]],  # type: ignore[override]
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> Qwen2_5VLInputs:
        """Prepares the initial inputs for the first execution pass of the Qwen2.5VL model."""

        if len(replica_batches) > 1:
            raise ValueError("Model does not support DP>1")

        context_batch = replica_batches[0]

        if kv_cache_inputs is None:
            raise ValueError("KV Cache Inputs must be provided")

        # Gather all vision data from contexts that need vision encoding
        vision_datas: list[VisionEncodingData] = []
        for ctx in context_batch:
            # Validate all contexts are the correct type
            assert isinstance(ctx, Qwen2_5VLTextAndVisionContext), (
                f"Expected Qwen2_5VLTextAndVisionContext, got {type(ctx).__name__}"
            )
            if ctx.needs_vision_encoding:
                assert ctx.vision_data is not None, (
                    "vision_data must be present when needs_vision_encoding is True"
                )
                vision_datas.append(ctx.vision_data)
        any_needs_vision_encoding = len(vision_datas) > 0

        # Prepare Inputs Needed Regardless of Images
        with Tracer("prepare_input_ids"):
            input_ids = Buffer.from_numpy(
                np.concatenate([ctx.tokens.active for ctx in context_batch])
            ).to(self.devices[0])

        with Tracer("prepare_input_row_offsets"):
            input_row_offsets = np.cumsum(
                [0] + [ctx.tokens.active_length for ctx in context_batch],
                dtype=np.uint32,
            )
            input_row_offsets_tensors = Buffer.from_numpy(input_row_offsets).to(
                self.devices
            )

        with Tracer("prepare_decoder_position_ids"):
            decoder_position_ids = self.prepare_decoder_position_ids(
                context_batch, self.devices
            )

        with Tracer("prepare_image_token_indices"):
            image_token_indices = self._batch_image_token_indices(context_batch)

        if not any_needs_vision_encoding:
            return Qwen2_5VLInputs(
                input_ids=input_ids,
                input_row_offsets=input_row_offsets_tensors,
                position_ids=decoder_position_ids,
                signal_buffers=self.signal_buffers,
                return_n_logits=Buffer.from_numpy(
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

        # From here on, assume that all inputs are available in vision_data
        # due to context validators
        with Tracer("preparing_pixel_values"):
            pixel_values_list = [
                vision_data.concatenated_pixel_values
                for vision_data in vision_datas
            ]
            pixel_values_tensor = self._parallel_ops.concatenate(
                pixel_values_list
            )

            # If uint16, interpret as bfloat16 to work around lack of Numpy
            # bfloat16 support.
            if pixel_values_tensor.dtype == DType.uint16:
                pixel_values_tensor = pixel_values_tensor.view(
                    DType.bfloat16, pixel_values_tensor.shape
                )

            pixel_values = pixel_values_tensor.to(self.devices)

        with Tracer("preparing_window_index"):
            # Concatenate per-context window_index with cross-context offsets so indices are unique
            window_index_parts: list[npt.NDArray[np.int64]] = []
            index_offset = 0
            for ctx in context_batch:
                if ctx.needs_vision_encoding:
                    assert ctx.vision_data is not None, (
                        "vision_data must be present when needs_vision_encoding is True"
                    )
                    per_ctx_index = ctx.vision_data.window_index.astype(
                        np.int64
                    )
                    window_index_parts.append(per_ctx_index + index_offset)
                    index_offset += int(per_ctx_index.shape[0])
            window_index_np = np.concatenate(window_index_parts, axis=0)
            window_index_tensor = Buffer.from_numpy(window_index_np)
            window_index = window_index_tensor.to(self.devices)

        with Tracer("preparing_vision_position_ids"):
            vision_position_ids_list = [
                vision_data.vision_position_ids.astype(np.int64)
                for vision_data in vision_datas
            ]
            vision_position_ids_tensor = self._parallel_ops.concatenate(
                vision_position_ids_list
            )

            vision_position_ids = vision_position_ids_tensor.to(self.devices)

        with Tracer("preparing_max_grid_size"):
            max_grid_size_value = max(
                vision_data.max_grid_size.item() for vision_data in vision_datas
            )
            max_grid_size_tensor = Buffer.from_numpy(
                np.array(max_grid_size_value, dtype=np.int32)
            )
            max_grid_size = [max_grid_size_tensor for _ in self.devices]

        with Tracer("preparing_cu_seqlens"):
            # Handle cumulative offsets properly when batching
            cu_seqlens_list = []
            offset = 0
            for vision_data in vision_datas:
                seqlens = vision_data.cu_seqlens
                adjusted = seqlens.copy()
                adjusted[1:] += offset
                cu_seqlens_list.append(adjusted[1:])
                offset = adjusted[-1]

            cu_seqlens_tensor = Buffer.from_numpy(
                np.concatenate(
                    [np.array([0], dtype=np.uint32), *cu_seqlens_list]
                ).astype(np.uint32)
            )
            cu_seqlens = cu_seqlens_tensor.to(self.devices)

        with Tracer("preparing_cu_window_seqlens"):
            # cu_window_seqlens_unique is already scaled by spatial_merge_unit per-context.
            # We only need to add cross-context offsets and concatenate.
            cu_window_seqlens_parts: list[npt.NDArray[np.uint32]] = []
            offset = 0
            for vision_data in vision_datas:
                seqlens_unique = vision_data.cu_window_seqlens_unique.astype(
                    np.uint32
                )
                cu_window_seqlens_parts.append(
                    (seqlens_unique[1:] + offset).astype(np.uint32)
                )
                offset = offset + seqlens_unique[-1]

            cu_window_seqlens_np = np.concatenate(
                [np.array([0], dtype=np.uint32), *cu_window_seqlens_parts]
            ).astype(np.uint32)
            cu_window_seqlens_unique_tensor = Buffer.from_numpy(
                cu_window_seqlens_np
            )
            cu_window_seqlens = cu_window_seqlens_unique_tensor.to(self.devices)

        with Tracer("preparing_max_seqlen"):
            max_seqlen_value = max(
                vision_data.max_seqlen.item() for vision_data in vision_datas
            )
            max_seqlen_tensor = Buffer.from_numpy(
                np.array([max_seqlen_value], dtype=np.uint32)
            )
            max_seqlen = [max_seqlen_tensor for _ in self.devices]

        with Tracer("preparing_max_window_seqlen"):
            window_max_seqlen_value = max(
                vision_data.window_max_seqlen.item()
                for vision_data in vision_datas
            )
            window_max_seqlen_tensor = Buffer.from_numpy(
                np.array([window_max_seqlen_value], dtype=np.uint32)
            )
            max_window_seqlen = [window_max_seqlen_tensor for _ in self.devices]

        return Qwen2_5VLInputs(
            input_ids=input_ids,
            input_row_offsets=input_row_offsets_tensors,
            signal_buffers=self.signal_buffers,
            position_ids=decoder_position_ids,
            return_n_logits=Buffer.from_numpy(
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
        next_tokens: Buffer,
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
        position_ids = Buffer.from_numpy(position_ids_np).to(self.devices[0])

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

    @classmethod
    def estimate_activation_memory(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        # TODO: Make this more robust
        return 5 * 1024 * 1024 * 1024  # 5 GiB
