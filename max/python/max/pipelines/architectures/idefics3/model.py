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
import math
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from max.driver import Buffer, Device, DLPackArray
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, DeviceRef, Graph, TensorType, Value
from max.graph.buffer_utils import cast_dlpack_to
from max.graph.weights import (
    SafetensorWeights,
    WeightData,
    Weights,
    WeightsAdapter,
)
from max.nn.legacy.kv_cache import (
    KVCacheInputs,
    KVCacheParams,
    PagedCacheValues,
)
from max.nn.legacy.transformer import ReturnLogits
from max.pipelines.core import TextAndVisionContext
from max.pipelines.lib import (
    CompilationTimer,
    KVCacheConfig,
    KVCacheMixin,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
)
from transformers.models.auto.configuration_auto import AutoConfig

from .model_config import Idefics3Config
from .text_model.idefics3_text import Idefics3LanguageModel
from .vision_model.idefics3_vision import Idefics3VisionModel
from .weight_adapters import (
    convert_idefics3_language_model_state_dict,
    convert_idefics3_vision_model_state_dict,
)

logger = logging.getLogger("max.pipelines")


def _assert_image_embeddings_invariant(
    image_embeddings: Buffer, image_token_indices: Buffer
) -> None:
    """Validates that image embeddings count matches image token indices count.

    This prevents out-of-bounds access during scatter operations where image
    embeddings are placed at specific token positions.

    Args:
        image_embeddings: Single tensor of image embeddings
        image_token_indices: Single tensor of image token indices

    Raises:
        AssertionError: If embedding count doesn't match indices count
    """
    embed_count = image_embeddings.shape[0]
    indices_count = image_token_indices.shape[0]

    if embed_count != indices_count:
        logger.error(
            f"[CRITICAL] Vision embedding count ({embed_count}) "
            f"!= image token indices count ({indices_count})."
        )

    assert embed_count == indices_count, (
        f"Vision embedding shape mismatch: {embed_count} embeddings "
        f"but {indices_count} indices."
    )


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


@dataclass
class Idefics3Inputs(ModelInputs):
    """A class representing inputs for the Idefics3 model."""

    input_ids: Buffer
    """Tensor containing the input token IDs."""

    input_row_offsets: Buffer
    """Tensor containing the offsets for each row in the ragged input sequence."""

    return_n_logits: Buffer
    """Number of logits to return, used by speculative decoding for example."""

    # Vision inputs
    pixel_values: Buffer | None = None
    """Pixel values for vision inputs."""

    image_token_indices: Buffer | None = None
    """Pre-computed indices of image tokens in the input sequence."""

    @property
    def has_vision_inputs(self) -> bool:
        """Check if this input contains vision data."""
        return self.pixel_values is not None


class Idefics3Model(PipelineModel[TextAndVisionContext], KVCacheMixin):
    """An Idefics3 pipeline model for multimodal text generation."""

    vision_model: Model
    """The compiled vision model for processing images."""

    language_model: Model
    """The compiled language model for text generation."""

    _input_row_offsets_prealloc: Buffer
    """Pre-allocated tensor for input row offsets in multi-step execution."""

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

        self.vision_model, self.language_model = self.load_model(session)

        self.image_token_id = self.huggingface_config.image_token_id

        self._stacker = _VisionStacker()

    @staticmethod
    def calculate_max_seq_len(
        pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        """Calculates the maximum sequence length for the Idefics3 model."""
        max_seq_len = pipeline_config.model.max_length
        if max_seq_len:
            return max_seq_len

        # Get `max_position_embeddings` from the `text_config`.
        text_config = getattr(
            huggingface_config, "text_config", huggingface_config
        )
        return getattr(text_config, "max_position_embeddings", 4096)

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        """Gets the parameters required to configure the KV cache for Idefics3."""
        return Idefics3Config.construct_kv_params(
            huggingface_config,
            pipeline_config,
            devices,
            kv_cache_config,
            cache_dtype,
        )

    def load_model(self, session: InferenceSession) -> tuple[Model, Model]:
        """Loads the compiled Idefics3 models into the MAX Engine session.

        Returns:
            A tuple of (vision_model, language_model).
        """
        # Pre-allocation for multi-step execution
        assert self.pipeline_config.max_batch_size, (
            "Expected max_batch_size to be set"
        )
        self._input_row_offsets_prealloc = Buffer.from_numpy(
            np.arange(self.pipeline_config.max_batch_size + 1, dtype=np.uint32)
        ).to(self.devices[0])

        # Validate SafetensorWeights requirement
        if not isinstance(self.weights, SafetensorWeights):
            raise ValueError(
                "Idefics3 currently only supports safetensors weights"
            )

        # Get processed state dict for language and vision models.
        # NOTE: use weights_dict to mean WeightData, and state dict to mean
        # DLPack arrays, since state dict is overloaded.
        weights_dict = dict(self.weights.items())
        llm_weights_dict = convert_idefics3_language_model_state_dict(
            weights_dict
        )
        vision_model_weights_dict = convert_idefics3_vision_model_state_dict(
            weights_dict
        )

        # Generate Idefics3 config from HuggingFace config
        idefics3_config = Idefics3Config.initialize(self.pipeline_config)
        idefics3_config.finalize(
            huggingface_config=self.huggingface_config,
            llm_state_dict=llm_weights_dict,
            return_logits=self.return_logits,
        )

        # Build and compile vision model
        timer = CompilationTimer("vision model")
        vision_graph, vision_model_state_dict = self._build_vision_graph(
            idefics3_config, vision_model_weights_dict
        )
        timer.mark_build_complete()
        vision_model = session.load(
            vision_graph, weights_registry=vision_model_state_dict
        )
        timer.done()

        # Build and compile language model
        timer = CompilationTimer("language model")
        language_graph, language_model_state_dict = self._build_language_graph(
            idefics3_config, llm_weights_dict
        )
        timer.mark_build_complete()
        language_model = session.load(
            language_graph, weights_registry=language_model_state_dict
        )
        timer.done()

        return vision_model, language_model

    def _build_vision_graph(
        self, config: Idefics3Config, state_dict: dict[str, WeightData]
    ) -> tuple[Graph, dict[str, DLPackArray]]:
        """Build the vision model graph for processing images."""
        # Define input types for the vision model
        # Use static dimensions from the vision config
        image_size = config.vision_config.image_size
        patch_size = config.vision_config.patch_size

        # Expect pre-extracted patches from the tokenizer.
        pixel_values_type = TensorType(
            DType.bfloat16,
            shape=[
                "batch_size",
                3,
                image_size,
                image_size,
            ],
            # Expect the input on device 0.
            device=DeviceRef.GPU(),
        )

        # Initialize graph with input types
        with Graph("idefics3_vision", input_types=[pixel_values_type]) as graph:
            # Build vision model architecture.
            vision_model = Idefics3VisionModel(
                config.vision_config,
                dtype=self.dtype,
                device=DeviceRef.from_device(self.devices[0]),
            )
            vision_model.load_state_dict(
                state_dict=state_dict,
                weight_alignment=1,
                strict=True,
            )

            # Unpack inputs.
            (pixel_values,) = graph.inputs

            # Execute vision model: pixel_values -> image_embeddings.
            image_embeddings = vision_model(pixel_values.tensor)

            # Set graph outputs.
            graph.output(image_embeddings)

            return graph, vision_model.state_dict()

    def _language_graph_input_types(self) -> Sequence[TensorType | BufferType]:
        # Generate DeviceRef.
        device_ref = DeviceRef.from_device(self.devices[0])

        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )

        # Construct Graph Inputs
        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        )

        # Add image embeddings type - one per device, can be empty for text-only inputs
        image_embeddings_type = TensorType(
            self.dtype,
            shape=[
                "num_image_tokens",
                self.huggingface_config.text_config.hidden_size,
            ],
            device=device_ref,
        )

        # Add image token indices type
        image_token_indices_type = TensorType(
            DType.int32, shape=["total_image_tokens"], device=device_ref
        )

        return (
            tokens_type,
            input_row_offsets_type,
            return_n_logits_type,
            image_embeddings_type,
            image_token_indices_type,
            *self.kv_params.get_symbolic_inputs().flatten(),
        )

    def _unflatten_kv_inputs(
        self, kv_inputs_flat: Sequence[Value[Any]]
    ) -> list[PagedCacheValues]:
        kv_params = Idefics3Config.construct_kv_params(
            huggingface_config=self.huggingface_config,
            pipeline_config=self.pipeline_config,
            devices=[DeviceRef.from_device(d) for d in self.devices],
            kv_cache_config=self.kv_cache_config,
            cache_dtype=self.pipeline_config.model.kv_cache.cache_dtype,
        )
        n_devices = kv_params.n_devices
        fetch_types = self.kv_params.get_symbolic_inputs()[0]
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
        self, config: Idefics3Config, state_dict: dict[str, WeightData]
    ) -> tuple[Graph, dict[str, DLPackArray]]:
        """Build the language model graph for text generation with image embeddings."""
        # Initialize graph with input types.
        with Graph(
            "idefics3_language", input_types=self._language_graph_input_types()
        ) as graph:
            # Build language model architecture.
            language_model = Idefics3LanguageModel(
                config.text_config,
                config.image_token_id,
                self.dtype,
                DeviceRef.from_device(self.devices[0]),
            )
            language_model.load_state_dict(
                state_dict=state_dict,
                override_quantization_encoding=True,
                weight_alignment=1,
                strict=True,
            )

            # Unpack inputs
            (
                tokens,
                input_row_offsets,
                return_n_logits,
                image_embeddings,
                image_token_indices,
                *variadic_args,
            ) = graph.inputs

            # Unmarshal the remaining arguments, which are for KV cache
            kv_cache = self._unflatten_kv_inputs(variadic_args)

            # Execute language model: text + image embeddings -> logits
            outputs = language_model(
                tokens=tokens.tensor,
                kv_collection=kv_cache[0],
                return_n_logits=return_n_logits.tensor,
                input_row_offsets=input_row_offsets.tensor,
                image_embeddings=image_embeddings.tensor,
                image_token_indices=image_token_indices.tensor,
            )

            graph.output(*outputs)

            return graph, language_model.state_dict()

    def _prepare_vision_inputs(
        self, context_batch: Sequence[TextAndVisionContext]
    ) -> Buffer | None:
        """Batches up pixel_values for vision processing."""
        images = []
        for context in context_batch:
            # For Idefics, a single image may be split into multiple "patch_groups"
            # which appear as multiple images in the context object.
            for img in context.next_images:
                patch_group_pixels = img.pixel_values
                images.append(patch_group_pixels)

        if not images:
            return None

        final_images = self._stacker.stack(images)

        return cast_dlpack_to(
            final_images, DType.float32, DType.bfloat16, self.devices[0]
        )

    def _batch_image_token_indices(
        self, context_batch: Sequence[TextAndVisionContext]
    ) -> Buffer | None:
        """Batch image token indices from multiple contexts, adjusting for
        position in batch.

        This method efficiently combines image token indices from multiple
        contexts using vectorized operations.

        Args:
            context_batch: Sequence of contexts that may contain image token
                indices

        Returns:
            Buffer containing all batched indices, or None if no indices found
        """
        # Collect indices and offsets.
        indices_and_offsets = []
        batch_offset = 0

        for ctx in context_batch:
            input_ids = ctx.tokens.active
            special_image_token_mask = input_ids == self.image_token_id
            indices = np.where(special_image_token_mask)[0].tolist()

            indices_and_offsets.append([idx + batch_offset for idx in indices])
            batch_offset += ctx.tokens.active_length

        if not indices_and_offsets:
            return None

        np_indices = np.concatenate(indices_and_offsets).astype(
            np.int32, copy=False
        )

        # Create tensor and distribute to device
        return Buffer.from_numpy(np_indices).to(self.devices[0])

    def _create_empty_image_embeddings(self) -> Buffer:
        """Create empty image embeddings for text-only inputs on single GPU."""
        return Buffer.zeros(
            shape=[0, self.huggingface_config.text_config.hidden_size],
            dtype=self.dtype,
        ).to(self.devices[0])

    def _create_empty_indices(self) -> Buffer:
        """Create empty image token indices tensor for single GPU."""
        return Buffer.zeros(shape=[0], dtype=DType.int32).to(self.devices[0])

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Executes the Idefics3 model with the prepared inputs."""
        assert model_inputs.kv_cache_inputs is not None, (
            "Idefics3 requires KV cache inputs"
        )
        assert isinstance(model_inputs, Idefics3Inputs)

        # Process vision inputs if present.
        image_embeddings: Buffer
        image_token_indices: Buffer
        if model_inputs.has_vision_inputs:
            assert model_inputs.pixel_values is not None
            assert model_inputs.image_token_indices is not None

            # Execute vision model: pixel_values -> image_embeddings.
            vision_outputs = self.vision_model.execute(
                model_inputs.pixel_values,
            )

            assert isinstance(vision_outputs[0], Buffer)

            image_embeddings = vision_outputs[0]
            image_token_indices = model_inputs.image_token_indices

            _assert_image_embeddings_invariant(
                image_embeddings, image_token_indices
            )
        else:
            # Initialize empty tensors for text-only mode.
            image_embeddings = self._create_empty_image_embeddings()
            image_token_indices = self._create_empty_indices()

        # Prepare KV cache inputs as list of tensors
        assert model_inputs.kv_cache_inputs is not None, (
            "Idefics3 has KV cache inputs, but none were provided"
        )

        # Execute language model with text and image embeddings
        language_outputs = self.language_model.execute(
            model_inputs.input_ids,
            model_inputs.input_row_offsets,
            model_inputs.return_n_logits,
            image_embeddings,
            image_token_indices,
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

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextAndVisionContext]],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> ModelInputs:
        """Prepares the initial inputs for the first execution pass of the Idefics3 model."""

        if len(replica_batches) > 1:
            raise ValueError("Model does not support DP>1")

        context_batch = replica_batches[0]

        # First marshal out the pixel values, since we'll overwrite them.
        pixel_values = self._prepare_vision_inputs(context_batch)

        # Input row offset type: ["input_row_offsets_len"], UInt32
        input_row_offsets = Buffer.from_numpy(
            np.cumsum(
                [0] + [ctx.tokens.active_length for ctx in context_batch],
                dtype=np.uint32,
            )
        ).to(self.devices[0])

        # Input Ids: ["total_seq_len"], Int64
        # Create a ragged token vector of length: sum(len(t) for t in tokens).
        tokens = np.concatenate([ctx.tokens.active for ctx in context_batch])
        input_ids = Buffer.from_numpy(tokens).to(self.devices[0])

        # Batch image token indices, offsetting for position in the batch.
        image_token_indices = self._batch_image_token_indices(context_batch)

        return Idefics3Inputs(
            input_ids=input_ids,
            input_row_offsets=input_row_offsets,
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            pixel_values=pixel_values,
            kv_cache_inputs=kv_cache_inputs,
            image_token_indices=image_token_indices,
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Buffer,
        prev_model_inputs: ModelInputs,
    ) -> Idefics3Inputs:
        prev_model_inputs = cast(Idefics3Inputs, prev_model_inputs)
        # input_ids, old_row_offsets, Optional: [pixel_values, attention_mask]
        old_row_offsets = prev_model_inputs.input_row_offsets

        row_offsets_size = old_row_offsets.shape[0]
        next_row_offsets = self._input_row_offsets_prealloc[:row_offsets_size]
        # In multi-step execution, don't re-pass the pixel_values and attention_mask.
        return Idefics3Inputs(
            input_ids=next_tokens,
            input_row_offsets=next_row_offsets,
            kv_cache_inputs=prev_model_inputs.kv_cache_inputs,
            return_n_logits=prev_model_inputs.return_n_logits,
        )
