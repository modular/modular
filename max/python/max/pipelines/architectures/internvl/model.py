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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import numpy.typing as npt
from max.driver import Device, DLPackArray, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, Type, Value
from max.graph.weights import (
    SafetensorWeights,
    WeightData,
    Weights,
    WeightsAdapter,
)
from max.nn import ReturnLogits, Signals
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
    KVCacheMixin,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
    SupportedEncoding,
)
from transformers.models.auto.configuration_auto import AutoConfig

from .internvl import InternVLLanguageModel, InternVLVisionModel
from .model_config import InternVLConfig
from .tokenizer import (
    IMAGE_NDIMS,
    InternVLImageConfig,
    _get_image_context_token_id,
)
from .weight_adapters import (
    convert_internvl_language_model_state_dict,
    convert_internvl_vision_model_state_dict,
)

logger = logging.getLogger("max.pipelines")


class _VisionStacker:
    """Helper class for efficient parallel stacking of vision patches.

    Uses ThreadPoolExecutor for thread management and bulk numpy operations
    for optimal memory bandwidth utilization.
    """

    def __init__(self, max_workers: int = 24) -> None:
        """Initialize the vision stacker with a thread pool.

        Args:
            max_workers: Maximum number of worker threads (default: 24).
        """
        self._pool = ThreadPoolExecutor(max_workers=max_workers)

    def stack(
        self, images: list[npt.NDArray[np.floating[Any]]]
    ) -> npt.NDArray[np.floating[Any]]:
        """Stack images using parallel bulk copy operations.

        Args:
            images: List of numpy arrays to stack.

        Returns:
            Stacked numpy array.
        """
        n = len(images)
        if n == 0:
            return np.empty((0,), dtype=np.float32)

        # Pre-allocate output.
        out = np.empty((n, *images[0].shape), dtype=images[0].dtype)

        # Divide work evenly among threads.
        # ThreadPoolExecutor will handle cases where n < workers.
        workers = self._pool._max_workers
        step = math.ceil(n / workers)
        slices = [slice(i, min(i + step, n)) for i in range(0, n, step)]

        # Launch parallel bulk copy tasks.
        futures = [
            self._pool.submit(self._copy_block, out, images, sl)
            for sl in slices
        ]

        # Wait for completion and propagate any exceptions.
        for f in as_completed(futures):
            f.result()

        return out

    @staticmethod
    def _copy_block(
        out: npt.NDArray[np.floating[Any]],
        images: list[npt.NDArray[np.floating[Any]]],
        sl: slice,
    ) -> None:
        """Copy a block of images using bulk numpy operations.

        This method performs a C-level bulk copy that releases the GIL,
        allowing true parallel execution.
        """
        # Convert slice of list to temporary array view and bulk copy.
        np.copyto(out[sl], np.asarray(images[sl], dtype=images[0].dtype))


class InternVLInputs(ModelInputs):
    """A class representing inputs for the InternVL model."""

    input_ids: Tensor
    """Tensor containing the input token IDs."""

    input_row_offsets: list[Tensor]
    """Per-device tensors containing the offsets for each row in the ragged
    input sequence.
    """

    signal_buffers: list[Tensor]
    """Device buffers used for synchronization in communication collectives."""

    # Vision inputs.
    pixel_values: list[Tensor] | None = None
    """Pixel values for vision inputs."""

    image_token_indices: list[Tensor] | None = None
    """Per-device pre-computed indices of image tokens in the input sequence."""

    return_n_logits: Tensor
    """Number of logits to return, used by speculative decoding for example."""

    def __init__(
        self,
        input_ids: Tensor,
        input_row_offsets: list[Tensor],
        signal_buffers: list[Tensor],
        return_n_logits: Tensor,
        pixel_values: list[Tensor] | None = None,
        kv_cache_inputs: KVCacheInputs | None = None,
        image_token_indices: list[Tensor] | None = None,
    ) -> None:
        self.input_ids = input_ids
        self.input_row_offsets = input_row_offsets
        self.signal_buffers = signal_buffers
        self.return_n_logits = return_n_logits
        self.pixel_values = pixel_values
        self.kv_cache_inputs = kv_cache_inputs
        self.image_token_indices = image_token_indices

    @property
    def has_vision_inputs(self) -> bool:
        """Check if this input contains vision data."""
        return self.pixel_values is not None


def assert_image_embeddings_invariant(
    image_embeddings: Sequence[Tensor], image_token_indices: Sequence[Tensor]
) -> None:
    # Check for shape mismatch that causes scatter_nd OOB access.
    for i, (embed, indices) in enumerate(
        zip(image_embeddings, image_token_indices, strict=True)
    ):
        embed_count = embed.shape[0]
        indices_count = indices.shape[0]
        if embed_count != indices_count:
            logger.error(
                f"[CRITICAL] Device {i}: Vision embedding count ({embed_count}) "
                f"!= image token indices count ({indices_count})."
            )
        assert embed_count == indices_count, (
            f"Vision embedding shape mismatch on device {i}: {embed_count} embeddings "
            f"but {indices_count} indices."
        )


class InternVLModel(PipelineModel[TextAndVisionContext], KVCacheMixin):
    """An InternVL pipeline model for multimodal text generation."""

    vision_model: Model
    """The compiled vision model for processing images."""

    language_model: Model
    """The compiled language model for text generation."""

    _input_row_offsets_prealloc: list[Tensor]
    """Pre-allocated per-device tensors for input row offsets in multi-step
    execution.
    """

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

        self.vision_model, self.language_model = self.load_model(session)

        # Initialize signal buffers for distributed communication.
        # InternVL is natively distributed, so we always need these.
        self.signal_buffers = [
            Tensor.zeros(
                shape=(Signals.NUM_BYTES,), dtype=DType.uint8, device=dev
            )
            for dev in self.devices
        ]

        # Initialize vision stacker for optimized parallel stacking.
        self._stacker = _VisionStacker()

    @staticmethod
    def calculate_max_seq_len(
        pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        """Calculates the maximum sequence length for the InternVL model."""
        max_seq_len = pipeline_config.max_length
        if max_seq_len:
            return max_seq_len

        # Get `max_position_embeddings` from the `llm_config`.
        llm_config = getattr(
            huggingface_config, "llm_config", huggingface_config
        )
        return getattr(llm_config, "max_position_embeddings", 4096)

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        """Gets the parameters required to configure the KV cache for InternVL."""
        return InternVLConfig.get_kv_params(
            huggingface_config, n_devices, kv_cache_config, cache_dtype
        )

    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        """Gets the number of hidden layers from the HuggingFace configuration."""
        return InternVLConfig.get_num_layers(huggingface_config)

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
        """Estimates the size of the KV cache required for the InternVL model in bytes."""
        return estimate_kv_cache_size(
            params=InternVLConfig.get_kv_params(
                huggingface_config=huggingface_config,
                n_devices=len(devices),
                kv_cache_config=kv_cache_config,
                cache_dtype=cache_dtype,
            ),
            max_batch_size=pipeline_config.max_batch_size,
            max_seq_len=cls.calculate_max_seq_len(
                pipeline_config, huggingface_config=huggingface_config
            ),
            num_layers=InternVLConfig.get_num_layers(
                huggingface_config=huggingface_config
            ),
            available_cache_memory=available_cache_memory,
            devices=devices,
        )

    @classmethod
    def estimate_activation_memory(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        """Estimates the activation memory required for InternVL model execution.

        This accounts for the temporary memory buffers used during model execution,
        particularly for the vision encoder and language model activations.

        Based on empirical analysis of MGP buffer plans (GEX-2365):
        - Vision encoder uses ~128MiB per image.
        - Language model uses ~100KB per token for intermediate activations.

        These values come from printing the high water mark from the
        `mgp.buffer.plan` op, and verifying with GPU free memory at runtime.

        The vision encoder memory scales with the number of images that can be
        processed concurrently, which is limited by prefill_chunk_size / num_image_tokens
        where num_image_tokens=256 for InternVL.

        TODO(GEX-2365): Replace this with a more general solution that analyzes
        the compiled graph's memory requirements directly.

        Args:
            pipeline_config: Pipeline configuration
            huggingface_config: HuggingFace model configuration

        Returns:
            Estimated activation memory in bytes
        """
        # Vision encoder memory estimation.
        vision_memory_per_image = 128 * 1024 * 1024  # 128 MiB per image

        image_config = InternVLImageConfig(
            huggingface_config,
            pipeline_config.model_config.vision_config_overrides,
        )

        # Maximum number of images that can be processed is limited by
        # how many image tokens fit in the target new tokens
        max_images = (
            pipeline_config.prefill_chunk_size // image_config.num_image_token
        )
        # Ensure at least 1 image worth of memory.
        max_images = max(1, max_images)

        # Note: Each image can use up to max_dynamic_patch patches (default 12)
        # plus 1 for thumbnail if applicable.
        if not pipeline_config.enable_chunked_prefill:
            # When there's no chunked prefill, the number of images may overhang
            # by the maximum in a single request.
            # Since we only support a single image per request for now,
            # TODO(MODELS-638, E2EOPT-350): Adjust this after supporting
            # multi-image requests.
            max_images += image_config.max_dynamic_patch + 1

        vision_activation_memory = max_images * vision_memory_per_image

        # Language model memory estimation
        # ~100KB per token for intermediate activations
        llm_memory_per_token = 100 * 1024  # 100 KiB
        llm_activation_memory = (
            pipeline_config.prefill_chunk_size * llm_memory_per_token
        )

        total_activation_memory = (
            vision_activation_memory + llm_activation_memory
        )

        # Multiply by the number of devices since the above analysis is per
        # device, but memory estimation uses total memory across all devices.
        return (
            len(pipeline_config.model_config.device_specs)
            * total_activation_memory
        )

    def load_model(self, session: InferenceSession) -> tuple[Model, Model]:
        """Loads the compiled InternVL models into the MAX Engine session.

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

        # Validate SafetensorWeights requirement
        if not isinstance(self.weights, SafetensorWeights):
            raise ValueError(
                "InternVL currently only supports safetensors weights"
            )

        # Get processed state dict for language and vision models.
        # NOTE: use weights_dict to mean WeightData, and state dict to mean
        # DLPack arrays, since state dict is overloaded.
        weights_dict = dict(self.weights.items())
        llm_weights_dict = convert_internvl_language_model_state_dict(
            weights_dict
        )
        vision_model_weights_dict = convert_internvl_vision_model_state_dict(
            weights_dict
        )

        # Generate InternVL config from HuggingFace config
        internvl_config = InternVLConfig.generate(
            pipeline_config=self.pipeline_config,
            huggingface_config=self.huggingface_config,
            llm_state_dict=llm_weights_dict,
            vision_state_dict=vision_model_weights_dict,
            dtype=self.dtype,
            n_devices=len(self.devices),
            cache_dtype=self.encoding.cache_dtype,
            kv_cache_config=self.kv_cache_config,
            return_logits=self.return_logits,
        )

        # Build and compile vision model
        logger.info("Building and compiling vision model...")
        before = time.perf_counter()
        vision_graph, vision_model_state_dict = self._build_vision_graph(
            internvl_config, vision_model_weights_dict
        )
        after_build = time.perf_counter()

        logger.info(
            f"Building vision graph took {after_build - before:.6f} seconds"
        )

        before_compile = time.perf_counter()
        vision_model = session.load(
            vision_graph, weights_registry=vision_model_state_dict
        )
        after = time.perf_counter()

        logger.info(
            f"Compiling vision model took {after - before_compile:.6f} seconds"
        )

        logger.info(
            f"Building and compiling vision model took {after - before:.6f} seconds"
        )

        # Build and compile language model
        logger.info("Building and compiling language model...")
        before = time.perf_counter()
        language_graph, language_model_state_dict = self._build_language_graph(
            internvl_config, llm_weights_dict
        )
        after_build = time.perf_counter()

        logger.info(
            f"Building language graph took {after_build - before:.6f} seconds"
        )

        before_compile = time.perf_counter()
        language_model = session.load(
            language_graph, weights_registry=language_model_state_dict
        )
        after = time.perf_counter()

        logger.info(
            f"Compiling language model took {after - before_compile:.6f} seconds"
        )

        logger.info(
            f"Building and compiling language model took {after - before:.6f} seconds"
        )

        return vision_model, language_model

    def _build_vision_graph(
        self, config: InternVLConfig, state_dict: dict[str, WeightData]
    ) -> tuple[Graph, dict[str, DLPackArray]]:
        """Build the vision model graph for processing images."""
        # Define input types for the vision model
        # Use static dimensions from the vision config
        image_size = config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        # Calculate number of patches in each dimension
        height_patches = image_size // patch_size
        width_patches = image_size // patch_size

        # Expect pre-extracted patches from the tokenizer.
        # Use bfloat16 to match the tokenizer's output.
        pixel_values_types = [
            TensorType(
                DType.bfloat16,
                shape=[
                    "batch_size",
                    height_patches,
                    width_patches,
                    3,
                    patch_size,
                    patch_size,
                ],
                device=DeviceRef.from_device(dev),
            )
            for dev in self.devices
        ]

        # Create signal types for distributed communication
        signals = Signals(
            devices=(DeviceRef(d.label, d.id) for d in self.devices)
        )

        # Initialize graph with input types
        with Graph(
            "internvl_vision",
            input_types=[*pixel_values_types, *signals.input_types()],
        ) as graph:
            # Build vision model architecture.
            vision_model = InternVLVisionModel(config)
            vision_model.load_state_dict(
                state_dict=state_dict,
                override_quantization_encoding=True,
                weight_alignment=1,
                strict=True,
            )

            # Unpack inputs (one per device).
            pixel_values = [
                inp.tensor for inp in graph.inputs[: len(self.devices)]
            ]

            # Extract signal buffers (one per device).
            signal_buffers = [
                inp.buffer for inp in graph.inputs[len(self.devices) :]
            ]

            # Execute vision model: pixel_values -> image_embeddings.
            image_embeddings = vision_model(pixel_values, signal_buffers)

            # Set graph outputs.
            graph.output(*image_embeddings)

            return graph, vision_model.state_dict()

    def _language_graph_input_types(self) -> tuple[Type[Any], ...]:
        # Generate DeviceRef.
        device_ref = DeviceRef.from_device(self.devices[0])

        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )

        kv_inputs = self.kv_manager.input_symbols()

        # Construct Graph Inputs
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
        image_embeddings_types = [
            TensorType(
                self.dtype,
                shape=[
                    "num_image_tokens",
                    self.huggingface_config.llm_config.hidden_size,
                ],
                device=DeviceRef.from_device(dev),
            )
            for dev in self.devices
        ]

        # Add image token indices type
        image_token_indices_types = [
            TensorType(
                DType.int32,
                shape=["total_image_tokens"],
                device=DeviceRef.from_device(dev),
            )
            for dev in self.devices
        ]

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
            *signals.input_types(),
            *flattened_kv_types,
        )

    def _unflatten_kv_inputs(
        self, kv_inputs_flat: Sequence[Value[Any]]
    ) -> list[PagedCacheValues]:
        kv_params = InternVLConfig.get_kv_params(
            huggingface_config=self.huggingface_config,
            n_devices=len(self.devices),
            kv_cache_config=self.kv_cache_config,
            cache_dtype=self.encoding.cache_dtype,
        )
        n_devices = kv_params.n_devices
        fetch_types = self.kv_manager.input_symbols()[0]
        len_of_kv_tuple_per_dev = len(list(fetch_types))
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

    def _build_language_graph(
        self, config: InternVLConfig, state_dict: dict[str, WeightData]
    ) -> tuple[Graph, dict[str, DLPackArray]]:
        """Build the language model graph for text generation with image embeddings."""
        # Initialize graph with input types.
        with Graph(
            "internvl_language", input_types=self._language_graph_input_types()
        ) as graph:
            image_context_token_id = _get_image_context_token_id(
                self.huggingface_config
            )
            language_model = InternVLLanguageModel(
                config, image_context_token_id
            )
            language_model.load_state_dict(
                state_dict=state_dict,
                override_quantization_encoding=True,
                weight_alignment=1,
            )

            # Unpack inputs.
            (tokens, return_n_logits, *variadic_args) = graph.inputs

            # Extract input_row_offsets (one per device).
            input_row_offsets = [
                v.tensor for v in variadic_args[: len(self.devices)]
            ]
            variadic_args = variadic_args[len(self.devices) :]

            # Extract image embeddings (one per device).
            image_embeddings = [
                v.tensor for v in variadic_args[: len(self.devices)]
            ]
            variadic_args = variadic_args[len(self.devices) :]

            # Extract image token indices.
            image_token_indices = [
                v.tensor for v in variadic_args[: len(self.devices)]
            ]
            variadic_args = variadic_args[len(self.devices) :]

            # Multi-GPU passes a signal buffer per device: unmarshal these.
            signal_buffers = [
                v.buffer for v in variadic_args[: len(self.devices)]
            ]

            # Unmarshal the remaining arguments, which are for KV cache.
            kv_cache = self._unflatten_kv_inputs(
                variadic_args[len(self.devices) :]
            )

            # Execute language model: text + image embeddings -> logits
            outputs = language_model(
                tokens=tokens.tensor,
                signal_buffers=signal_buffers,
                kv_collections=kv_cache,
                return_n_logits=return_n_logits.tensor,
                input_row_offsets=input_row_offsets,
                image_embeddings=image_embeddings,
                image_token_indices=image_token_indices,
            )

            graph.output(*outputs)

            return graph, language_model.state_dict()

    def _prepare_vision_inputs(
        self, context_batch: Sequence[TextAndVisionContext]
    ) -> list[Tensor] | None:
        """Batches up pixel_values for vision processing."""
        images = []
        for context in context_batch:
            if context.needs_vision_encoding:
                # TODO(MODELS-638): Support multiple images per request for InternVL
                next_images = context.next_images
                if len(next_images) != 1:
                    raise ValueError(
                        "InternVL only supports one image per request"
                    )
                image = next_images[0].pixel_values

                if len(image.shape) != IMAGE_NDIMS:
                    raise ValueError(
                        "InternVL vision model expects image shape to be [num_patches, height_patches, width_patches, channels, patch_size, patch_size]"
                    )

                # Each image patch group needs to be processed separately by the vision model
                # So we add each patch group as a separate "batch" item
                for patch_group in image:
                    images.append(patch_group)

        if not images:
            return None

        final_images = self._stacker.stack(images)

        tensor = Tensor.from_numpy(final_images)

        # If uint16, interpret as bfloat16 to work around lack of NumPy
        # bfloat16 support.
        if final_images.dtype == np.uint16:
            tensor = tensor.view(DType.bfloat16, tensor.shape)

        return [tensor.to(dev) for dev in self.devices]

    def _batch_image_token_indices(
        self, context_batch: Sequence[TextAndVisionContext]
    ) -> list[Tensor] | None:
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

        for ctx in context_batch:
            if "image_token_indices" in ctx.extra_model_args:
                indices = ctx.extra_model_args["image_token_indices"]
                indices_and_offsets.append(indices + batch_offset)
            batch_offset += ctx.active_length

        if not indices_and_offsets:
            return None

        np_indices = np.concatenate(indices_and_offsets).astype(
            np.int32, copy=False
        )

        # Create tensor and distribute to devices.
        return [Tensor.from_numpy(np_indices).to(dev) for dev in self.devices]

    def _create_empty_image_embeddings(self) -> list[Tensor]:
        """Create empty image embeddings for text-only inputs."""
        return [
            Tensor.zeros(
                shape=[0, self.huggingface_config.llm_config.hidden_size],
                dtype=self.dtype,
            ).to(dev)
            for dev in self.devices
        ]

    def _create_empty_indices(self) -> list[Tensor]:
        """Create empty image token indices tensor."""
        return [
            Tensor.zeros(shape=[0], dtype=DType.int32).to(dev)
            for dev in self.devices
        ]

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Executes the InternVL model with the prepared inputs."""
        assert model_inputs.kv_cache_inputs is not None, (
            "InternVL requires KV cache inputs"
        )
        assert isinstance(model_inputs, InternVLInputs)

        # Process vision inputs if present.
        image_embeddings: list[Tensor]
        image_token_indices: list[Tensor]
        if model_inputs.has_vision_inputs:
            assert model_inputs.pixel_values is not None
            assert model_inputs.image_token_indices is not None

            # Execute vision model: pixel_values -> image_embeddings.
            vision_outputs = self.vision_model.execute(
                *model_inputs.pixel_values, *model_inputs.signal_buffers
            )
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
            # Initialize empty tensors for text-only mode.
            image_embeddings = self._create_empty_image_embeddings()
            image_token_indices = self._create_empty_indices()

        # Prepare KV cache inputs as list of tensors
        assert model_inputs.kv_cache_inputs
        kv_cache_inputs_list = list(model_inputs.kv_cache_inputs)

        # Execute language model with text and image embeddings
        language_outputs = self.language_model.execute(
            model_inputs.input_ids,
            model_inputs.return_n_logits,
            *model_inputs.input_row_offsets,
            *image_embeddings,
            *image_token_indices,
            *model_inputs.signal_buffers,
            *kv_cache_inputs_list,
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
    ) -> ModelInputs:
        """Prepares the initial inputs for the first execution pass of the InternVL model."""

        # First marshal out the pixel values, since we'll overwrite them.
        pixel_values = self._prepare_vision_inputs(context_batch)

        # Input row offset type: ["input_row_offsets_len"], UInt32
        input_row_offsets_host = Tensor.from_numpy(
            np.cumsum(
                [0] + [ctx.active_length for ctx in context_batch],
                dtype=np.uint32,
            ),
        )
        input_row_offsets = [
            input_row_offsets_host.to(dev) for dev in self.devices
        ]

        # Input Ids: ["total_seq_len"], Int64
        # Create a ragged token vector of length: sum(len(t) for t in tokens).
        tokens = np.concatenate([ctx.next_tokens for ctx in context_batch])
        input_ids = Tensor.from_numpy(tokens).to(self.devices[0])

        # Batch image token indices, offsetting for position in the batch.
        image_token_indices = self._batch_image_token_indices(context_batch)

        return InternVLInputs(
            input_ids=input_ids,
            input_row_offsets=input_row_offsets,
            signal_buffers=self.signal_buffers,
            return_n_logits=Tensor.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            pixel_values=pixel_values,
            kv_cache_inputs=kv_cache_inputs,
            image_token_indices=image_token_indices,
        )

    def prepare_next_token_inputs(
        self, next_tokens: Tensor, prev_model_inputs: ModelInputs
    ) -> ModelInputs:
        """Prepares the inputs for subsequent execution steps in a multi-step generation."""
        assert isinstance(prev_model_inputs, InternVLInputs)
        prev_inputs = prev_model_inputs

        # Use pre-allocated row offsets for next token.
        # Since the pre-allocated array has length max_batch_size, slice out
        # only the current step's batch size.
        offset = prev_inputs.input_row_offsets[0].shape[0]
        next_row_offsets = [
            offsets_prealloc[:offset]
            for offsets_prealloc in self._input_row_offsets_prealloc
        ]

        return InternVLInputs(
            input_ids=next_tokens,
            input_row_offsets=next_row_offsets,
            signal_buffers=self.signal_buffers,
            return_n_logits=prev_model_inputs.return_n_logits,
            # Set vision model inputs to None after the first step
            pixel_values=None,
            kv_cache_inputs=prev_inputs.kv_cache_inputs,
        )

    def load_kv_manager(
        self, session: InferenceSession, available_cache_memory: int | None
    ) -> PagedKVCacheManager:
        """Loads and initializes the KVCacheManager for the InternVL model."""
        return load_kv_manager(
            params=InternVLConfig.get_kv_params(
                huggingface_config=self.huggingface_config,
                n_devices=len(self.devices),
                kv_cache_config=self.kv_cache_config,
                cache_dtype=self.encoding.cache_dtype,
            ),
            max_batch_size=self.pipeline_config.max_batch_size,
            max_seq_len=self.calculate_max_seq_len(
                self.pipeline_config, huggingface_config=self.huggingface_config
            ),
            num_layers=InternVLConfig.get_num_layers(
                huggingface_config=self.huggingface_config
            ),
            devices=self.devices,
            available_cache_memory=available_cache_memory,
            page_size=self.kv_cache_config.kv_cache_page_size,
            session=session,
        )
