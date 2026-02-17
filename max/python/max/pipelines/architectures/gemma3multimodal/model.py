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
from max.graph import BufferType, DeviceRef, Graph, TensorType, Type, Value
from max.graph.buffer_utils import cast_dlpack_to
from max.graph.weights import WeightData, Weights, WeightsAdapter
from max.kv_cache import PagedKVCacheManager, load_kv_managers
from max.nn.legacy.comm import Signals
from max.nn.legacy.kv_cache import (
    KVCacheInputs,
    KVCacheInputsSequence,
    KVCacheParamInterface,
    KVCacheParams,
    PagedCacheValues,
)
from max.nn.legacy.transformer import ReturnLogits
from max.pipelines.core import TextAndVisionContext
from max.pipelines.lib import (
    AlwaysSignalBuffersMixin,
    CompilationTimer,
    KVCacheConfig,
    KVCacheMixin,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
    SupportedEncoding,
)
from transformers import AutoConfig

from .model_config import Gemma3ForConditionalGenerationConfig
from .vision_model.gemma3multimodal import (
    Gemma3LanguageModel,
    Gemma3VisionModel,
)
from .weight_adapters import (
    convert_safetensor_language_state_dict,
    convert_safetensor_vision_state_dict,
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


@dataclass
class Gemma3MultiModalModelInputs(ModelInputs):
    """A class representing inputs for the Gemma3 multi modal model.

    This class encapsulates the input tensors required for the Gemma3 multi
    modal model, for text and vision processing.

    Args:
        tokens: Input token IDs.
        input_row_offsets: Input row offsets (ragged tensors).
        return_n_logits: Number of logits to return.
        signal_buffers: Device buffers for distributed communication.
        kv_cache_inputs: Inputs for the KV cache.
        pixel_values: Raw pixel values for vision inputs. Defaults to ``None``.
        image_token_indices: Pre-computed indices of image tokens. Defaults to
            ``None``.
    """

    tokens: npt.NDArray[np.integer[Any]] | Buffer
    """Tensor containing the input token IDs."""

    input_row_offsets: npt.NDArray[np.integer[Any]] | list[Buffer]
    """Tensor containing the offsets for each row in the ragged input sequence,
    or the attention mask for the padded input sequence. For distributed execution,
    this can be a list of tensors, one per device."""

    signal_buffers: list[Buffer]
    """Device buffers used for synchronization in communication collectives."""

    return_n_logits: Buffer
    """Number of logits to return, used by speculative decoding for example."""

    pixel_values: list[Buffer] | None = None
    """Raw pixel values for vision inputs: [batch, channels, height, width]."""

    image_token_indices: list[Buffer] | None = None
    """Pre-computed indices of image tokens in the input sequence."""

    @property
    def has_vision_inputs(self) -> bool:
        """Check if this input contains vision data."""
        return self.pixel_values is not None


class Gemma3_MultiModalModel(
    AlwaysSignalBuffersMixin, PipelineModel[TextAndVisionContext], KVCacheMixin
):
    """Gemma 3 multimodal pipeline model for text generation.

    This class integrates the Gemma 3 multimodal architecture with the MAX
    pipeline infrastructure, handling model loading, KV cache management, and
    input preparation for inference.

    Args:
        pipeline_config: The configuration settings for the entire pipeline.
        session: The MAX inference session managing the runtime.
        huggingface_config: The configuration loaded from HuggingFace
            (:obj:`transformers.AutoConfig`).
        encoding: The quantization and data type encoding used for the model
            (:obj:`max.pipelines.config_enums.SupportedEncoding`).
        devices: A list of MAX devices (:obj:`max.driver.Device`) to
            run the model on.
        kv_cache_config: Configuration settings for the Key-Value cache
            (:obj:`max.pipelines.max_config.KVCacheConfig`).
        weights: The model weights (:obj:`max.graph.weights.Weights`).
        adapter: An optional adapter to modify weights before loading
            (:obj:`max.graph.weights.WeightsAdapter`).
        return_logits: The number of top logits to return from the model
            execution.
    """

    language_model: Model
    """The compiled and initialized MAX Engine model ready for inference."""

    vision_model: Model
    """The compiled and initialized MAX Engine vision model ready for inference."""

    # The vision and text towers are in the same weights file, but are in
    # separate models, so load_state_dict will naturally be loading subsets in
    # each case.
    _strict_state_dict_loading = True

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

        # signal_buffers are provided by AlwaysSignalBuffersMixin as a cached_property
        # to avoid GPU memory allocation during compile-only mode (cross-compilation).
        # Force initialization here to ensure buffers are ready before model execution,
        # preventing potential race conditions in multi-GPU scenarios.
        _ = self.signal_buffers

        self._stacker = _VisionStacker()
        self.vision_model, self.language_model = self.load_model(session)

    @classmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        """Calculates the maximum sequence length for the InternVL model."""
        return Gemma3ForConditionalGenerationConfig.calculate_max_seq_len(
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
        """Gets the parameters required to configure the KV cache for InternVL."""
        return Gemma3ForConditionalGenerationConfig.construct_kv_params(
            huggingface_config,
            pipeline_config,
            devices,
            kv_cache_config,
            cache_dtype,
        )

    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        """Gets the number of hidden layers from the HuggingFace configuration."""
        return Gemma3ForConditionalGenerationConfig.get_num_layers(
            huggingface_config
        )

    def load_model(self, session: InferenceSession) -> tuple[Model, Model]:
        """Loads the compiled Gemma3 MultiModal models into the MAX Engine session.

        Returns:
            A tuple of (vision_model, language_model).
        """
        assert self.pipeline_config.max_batch_size, (
            "Expected max_batch_size to be set"
        )

        # Get processed state dict for language and vision models
        weights_dict = dict(self.weights.items())
        language_weights_dict = convert_safetensor_language_state_dict(
            weights_dict
        )
        vision_weights_dict = convert_safetensor_vision_state_dict(weights_dict)

        raw_state_dict = {k: v.data() for k, v in weights_dict.items()}
        model_config = Gemma3ForConditionalGenerationConfig.initialize(
            self.pipeline_config
        )
        model_config.finalize(
            huggingface_config=self.huggingface_config,
            state_dict=raw_state_dict,
            return_logits=self.return_logits,
        )
        self.config = model_config

        input_row_offsets_prealloc_host = Buffer.from_numpy(
            np.arange(self.pipeline_config.max_batch_size + 1, dtype=np.uint32)
        )
        self._input_row_offsets_prealloc = [
            input_row_offsets_prealloc_host.to(dev) for dev in self.devices
        ]

        # Build and compile language model
        timer = CompilationTimer("language model")
        language_graph, language_weight_dict = self._build_language_graph(
            model_config, language_weights_dict
        )
        timer.mark_build_complete()
        language_model = session.load(
            language_graph, weights_registry=language_weight_dict
        )
        timer.done()

        # Build and compile vision model
        timer = CompilationTimer("vision model")
        vision_graph, vision_model_state_dict = self._build_vision_graph(
            model_config, vision_weights_dict
        )
        timer.mark_build_complete()
        vision_model = session.load(
            vision_graph, weights_registry=vision_model_state_dict
        )
        timer.done()

        return vision_model, language_model

    def _language_model_input_types(
        self, config: Gemma3ForConditionalGenerationConfig
    ) -> Sequence[TensorType | BufferType]:
        """Prepare the Tensor input types that our language graph will work with"""
        device_ref = DeviceRef.from_device(self.devices[0])
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

        image_embeddings_types = [
            TensorType(
                DType.bfloat16,
                shape=[
                    "num_image_tokens",
                    config.text_config.hidden_size,
                ],
                device=DeviceRef.from_device(dev),
            )
            for dev in self.devices
        ]

        image_token_indices_types = [
            TensorType(
                DType.int32,
                shape=["total_image_tokens"],
                device=DeviceRef.from_device(dev),
            )
            for dev in self.devices
        ]

        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )

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
            *self.kv_params.get_symbolic_inputs().flatten(),
        )

    def _build_language_graph(
        self,
        config: Gemma3ForConditionalGenerationConfig,
        state_dict: dict[str, WeightData],
    ) -> tuple[Graph, dict[str, DLPackArray]]:
        """Build the language model with our input types and graph"""
        with Graph(
            getattr(self.huggingface_config, "model_type", "Gemma3"),
            input_types=self._language_model_input_types(config),
        ) as graph:
            language_model = Gemma3LanguageModel(config)
            language_model.load_state_dict(
                state_dict,
                weight_alignment=1,
                strict=self._strict_state_dict_loading,
            )

            # Unpack inputs following InternVL pattern
            (tokens, return_n_logits, *variadic_args) = graph.inputs

            # Extract input_row_offsets (one per device)
            input_row_offsets = [
                v.tensor for v in variadic_args[: len(self.devices)]
            ]
            variadic_args = variadic_args[len(self.devices) :]

            # Extract image embeddings (one per device).
            image_embeddings = [
                v.tensor for v in variadic_args[: len(self.devices)]
            ]
            variadic_args = variadic_args[len(self.devices) :]

            image_token_indices = [
                v.tensor for v in variadic_args[: len(self.devices)]
            ]
            variadic_args = variadic_args[len(self.devices) :]

            # Extract signal buffers (one per device)
            signal_buffers = [
                v.buffer for v in variadic_args[: len(self.devices)]
            ]
            variadic_args = variadic_args[len(self.devices) :]

            # Extract KV cache inputs
            kv_cache = self._unflatten_kv_inputs(variadic_args)

            outputs = language_model(
                tokens=tokens.tensor,
                signal_buffers=signal_buffers,
                return_n_logits=return_n_logits.tensor,
                input_row_offsets=input_row_offsets,
                kv_collections=kv_cache,
                image_embeddings=image_embeddings,
                image_token_indices=image_token_indices,
            )
            graph.output(*outputs)
        return graph, language_model.state_dict()

    def _vision_model_input_types(
        self, config: Gemma3ForConditionalGenerationConfig
    ) -> list[Type[Any]]:
        """Build the vision model graph for processing images."""
        pixel_values_types = [
            TensorType(
                DType.bfloat16,
                shape=[
                    "batch_size",
                    3,
                    config.vision_config.image_size,
                    config.vision_config.image_size,
                ],
                device=DeviceRef.from_device(dev),
            )
            for dev in self.devices
        ]

        # Create signal types for distributed communication
        signals = Signals(
            devices=(DeviceRef(d.label, d.id) for d in self.devices)
        )
        return [*pixel_values_types, *signals.input_types()]

    def _build_vision_graph(
        self,
        config: Gemma3ForConditionalGenerationConfig,
        state_dict: dict[str, WeightData],
    ) -> tuple[Graph, dict[str, DLPackArray]]:
        """Build the vision model with our input types and graph"""
        with Graph(
            getattr(self.huggingface_config, "model_type", "Gemma3"),
            input_types=self._vision_model_input_types(config),
        ) as graph:
            vision_model = Gemma3VisionModel(
                config,
                device=DeviceRef.from_device(self.devices[0]),
            )

            vision_model.load_state_dict(
                state_dict=state_dict,
                override_quantization_encoding=True,
                weight_alignment=1,
                strict=self._strict_state_dict_loading,
            )

            pixel_values = [
                inp.tensor for inp in graph.inputs[: len(self.devices)]
            ]

            signal_buffers = [
                inp.buffer for inp in graph.inputs[len(self.devices) :]
            ]

            image_embeddings = vision_model(pixel_values, signal_buffers)

            graph.output(*image_embeddings)

            return graph, vision_model.state_dict()

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """If required, execute the vision model, then continue to execute the
        language model.  Either pass through image embeddings or create an empty
        placeholder."""
        model_inputs = cast(Gemma3MultiModalModelInputs, model_inputs)

        input_row_offsets = model_inputs.input_row_offsets

        image_embeddings: list[Buffer]
        image_token_indices: list[Buffer]
        if model_inputs.has_vision_inputs:
            assert model_inputs.pixel_values is not None

            # Execute vision model: patched pixel_values -> image_embeddings.
            vision_outputs = self.vision_model.execute(
                *model_inputs.pixel_values, *model_inputs.signal_buffers
            )
            assert len(vision_outputs) == len(self.devices)

            image_embeddings = [
                output
                for output in vision_outputs
                if isinstance(output, Buffer)
            ]
            assert model_inputs.image_token_indices is not None
            image_token_indices = model_inputs.image_token_indices
        else:
            # Initialize empty tensors for text-only mode.
            image_embeddings = self._create_empty_image_embeddings()
            image_token_indices = self._create_empty_indices()

        assert model_inputs.kv_cache_inputs
        curr_kv_cache_inputs = list(model_inputs.kv_cache_inputs)

        model_outputs = self.language_model.execute(
            model_inputs.tokens,
            model_inputs.return_n_logits,
            *input_row_offsets,
            *image_embeddings,
            *image_token_indices,
            *model_inputs.signal_buffers,
            *curr_kv_cache_inputs,
        )

        if len(model_outputs) == 3:
            assert isinstance(model_outputs[0], Buffer)
            assert isinstance(model_outputs[1], Buffer)
            assert isinstance(model_outputs[2], Buffer)
            return ModelOutputs(
                logits=model_outputs[1],
                next_token_logits=model_outputs[0],
                logit_offsets=model_outputs[2],
            )
        else:
            assert isinstance(model_outputs[0], Buffer)
            return ModelOutputs(
                logits=model_outputs[0],
                next_token_logits=model_outputs[0],
            )

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextAndVisionContext]],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> ModelInputs:
        """Prepare our inputs for the first execution pass of the multimodal model."""

        if len(replica_batches) > 1:
            raise ValueError("Model does not support DP>1")

        context_batch = replica_batches[0]

        dev = self.devices[0]
        assert kv_cache_inputs is not None
        kv_cache_inputs = cast(KVCacheInputsSequence, kv_cache_inputs)
        input_row_offsets = Buffer.from_numpy(
            np.cumsum(
                [0] + [ctx.tokens.active_length for ctx in context_batch],
                dtype=np.uint32,
            )
        )
        input_row_offsets_tensors = [
            input_row_offsets.to(device) for device in self.devices
        ]

        tokens = np.concatenate([ctx.tokens.active for ctx in context_batch])

        # stack our images in a list of tensors
        pixel_values = self._prepare_vision_inputs(context_batch)

        # Batch image token indices, offsetting for position in the batch.
        image_token_indices = self._batch_image_token_indices(context_batch)

        return Gemma3MultiModalModelInputs(
            tokens=Buffer.from_numpy(tokens).to(dev),
            input_row_offsets=input_row_offsets_tensors,
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            signal_buffers=self.signal_buffers,
            kv_cache_inputs=kv_cache_inputs,
            pixel_values=pixel_values,
            image_token_indices=image_token_indices,
        )

    def prepare_next_token_inputs(
        self, next_tokens: Buffer, prev_model_inputs: ModelInputs
    ) -> ModelInputs:
        prev_model_inputs = cast(Gemma3MultiModalModelInputs, prev_model_inputs)
        row_offsets_size = prev_model_inputs.input_row_offsets[0].shape[0]

        # Slice each tensor in the list, not the list itself
        next_row_offsets = [
            offsets_prealloc[:row_offsets_size]
            for offsets_prealloc in self._input_row_offsets_prealloc
        ]

        return Gemma3MultiModalModelInputs(
            tokens=next_tokens,
            input_row_offsets=next_row_offsets,
            return_n_logits=prev_model_inputs.return_n_logits,
            signal_buffers=self.signal_buffers,
            kv_cache_inputs=prev_model_inputs.kv_cache_inputs,
            pixel_values=None,
        )

    def _prepare_vision_inputs(
        self, context_batch: Sequence[TextAndVisionContext]
    ) -> list[Buffer] | None:
        """Use the VisionStacker to prepare batched vision inputs from multiple contexts.
        The Tokenizer should have already processed images into pixel_values (pan and scan etc)"""
        images = []
        for context in context_batch:
            for img in context.next_images:
                images.append(img.pixel_values)

        if not images:
            return None

        final_images = self._stacker.stack(images)

        tensor = cast_dlpack_to(
            final_images, DType.float32, DType.bfloat16, self.devices[0]
        )

        return [tensor.to(dev) for dev in self.devices]

    def _batch_image_token_indices(
        self, context_batch: Sequence[TextAndVisionContext]
    ) -> list[Buffer] | None:
        """Batch image token indices from multiple contexts, adjusting for
        position in batch.
        """
        indices_and_offsets = []
        batch_offset = 0

        for ctx in context_batch:
            input_ids = ctx.tokens.active

            # Find where image tokens appear
            special_image_token_mask = (
                input_ids == self.config.image_token_index
            )
            indices = np.where(special_image_token_mask)[0]

            if len(indices) > 0:
                indices_and_offsets.append(indices + batch_offset)

            batch_offset += ctx.tokens.active_length

        if not indices_and_offsets:
            return [
                Buffer.zeros(shape=[0], dtype=DType.int32).to(self.devices[0])
            ]

        np_indices = np.concatenate(indices_and_offsets).astype(
            np.int32, copy=False
        )

        # Create tensor and distribute to device
        return [Buffer.from_numpy(np_indices).to(dev) for dev in self.devices]

    def load_kv_managers(
        self,
        kv_params: KVCacheParamInterface,
        max_batch_size: int,
        max_seq_len: int,
        session: InferenceSession,
        available_cache_memory: int,
    ) -> list[PagedKVCacheManager]:
        return load_kv_managers(
            params=kv_params,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            # FIXME: Decrease KVCache memory usage by 10% to leave headroom for
            # vision processing.
            available_cache_memory=int(available_cache_memory * 0.9),
            session=session,
        )

    def _unflatten_kv_inputs(
        self, kv_inputs_flat: Sequence[Value[Any]]
    ) -> list[PagedCacheValues]:
        """Receives KVCache inputs from the language graph, unflattens them, and
        returns in a list"""
        kv_params = Gemma3ForConditionalGenerationConfig.construct_kv_params(
            huggingface_config=self.huggingface_config,
            pipeline_config=self.pipeline_config,
            devices=[DeviceRef.from_device(d) for d in self.devices],
            kv_cache_config=self.kv_cache_config,
            cache_dtype=self.pipeline_config.model.kv_cache.cache_dtype,
        )
        n_devices = kv_params.n_devices
        fetch_types = kv_params.get_symbolic_inputs()[0]
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

    def _create_empty_image_embeddings(self) -> list[Buffer]:
        """Create empty image embeddings for text-only inputs."""
        return [
            Buffer.zeros(
                shape=[0, self.huggingface_config.text_config.hidden_size],
                dtype=DType.bfloat16,
            ).to(dev)
            for dev in self.devices
        ]

    def _create_empty_indices(self) -> list[Buffer]:
        """Create empty image token indices tensor."""
        return [
            Buffer.zeros(shape=[0], dtype=DType.int32).to(dev)
            for dev in self.devices
        ]
