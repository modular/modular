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
from dataclasses import dataclass
from typing import Any, ClassVar, cast

import numpy as np
import numpy.typing as npt
from max.driver import Buffer, Device, DevicePinnedBuffer, DLPackArray
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, DeviceRef, Graph, Module, TensorType
from max.graph.weights import WeightData, Weights, WeightsAdapter
from max.nn.comm import Signals
from max.nn.kv_cache import MultiKVCacheParams
from max.nn.transformer import ReturnLogits
from max.pipelines.context import ImageMetadata
from max.pipelines.lib import (
    AlwaysSignalBuffersMixin,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    MultiGraphPipelineModelWithKVCache,
    PipelineConfig,
)
from max.pipelines.lib.vision_encoder_cache import (
    VisionEncoderCache,
    VisionEncodeResult,
)
from max.profiler import traced

from .batch_processor import Gemma4BatchProcessor
from .batch_vision_inputs import (
    VideoInputs,
    VisionRawInputs,
    create_empty_embeddings,
    create_empty_indices,
    merge_per_device_buffers,
    pack_uncached_images,
)
from .context import Gemma4Context
from .gemma4 import Gemma4TextModel
from .model_config import Gemma4ForConditionalGenerationConfig
from .vision_model.vision_model import Gemma4VisionModel
from .weight_adapters import (
    convert_safetensor_language_state_dict,
    convert_safetensor_vision_state_dict,
    fuse_gemma4_projection_weights,
    gemma4_uses_fused_projections,
)

logger = logging.getLogger("max.pipelines")


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
        kv_cache_inputs: Combined KV cache inputs (sliding-window + global).
        video: Inputs to the video encoder.

    Image embeddings come from the pipeline-driven encoder cache on the base
    ``vision_embeddings`` / ``vision_scatter_indices`` fields.
    """

    tokens: npt.NDArray[np.integer[Any]] | Buffer
    input_row_offsets: npt.NDArray[np.integer[Any]] | list[Buffer]
    signal_buffers: list[Buffer]
    return_n_logits: Buffer

    video: VideoInputs | None = None

    combined_embeds: list[Buffer] | None = None
    combined_indices: list[Buffer] | None = None

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        """Returns positional Buffer inputs for the language model ABI."""
        assert self.combined_embeds is not None
        assert self.combined_indices is not None
        assert self.kv_cache_inputs is not None
        return (
            self.tokens,
            self.return_n_logits,
            *self.input_row_offsets,
            *self.combined_embeds,
            *self.combined_indices,
            *self.signal_buffers,
            *self.kv_cache_inputs.flatten(),
        )


class Gemma3_MultiModalModel(
    AlwaysSignalBuffersMixin,
    MultiGraphPipelineModelWithKVCache[Gemma4Context],
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

    model_config_cls: ClassVar[type[Any]] = Gemma4ForConditionalGenerationConfig
    batch_processor_cls: ClassVar[type[Gemma4BatchProcessor]] = (
        Gemma4BatchProcessor
    )

    language_model: Model
    """The compiled and initialized MAX Engine model ready for inference."""

    vision_model: Model | None
    """The compiled vision model, or None for text-only ("gemma4_unified")
    checkpoints whose vision embedder is not implemented yet."""
    # The vision and text towers are in the same weights file, but are in
    # separate models, so load_state_dict will naturally be loading subsets in
    # each case.
    _strict_state_dict_loading = True

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
        max_batch_size: int = 1,
    ) -> None:
        self._max_batch_size = max_batch_size
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits,
        )

        self._scatter_buffers: dict[int, tuple[Buffer, list[Buffer]]] = {}

        # signal_buffers are provided by AlwaysSignalBuffersMixin as a cached_property
        # to avoid GPU memory allocation during compile-only mode (cross-compilation).
        # Force initialization here to ensure buffers are ready before model execution,
        # preventing potential race conditions in multi-GPU scenarios.
        _ = self.signal_buffers

        self.vision_model, self.language_model = self.load_model(session)

        # Images are cached by the pipeline-owned VisionEncoderCache. The
        # pipeline hands its cache here via set_video_cache() for the video
        # branch's store (transitional — see execute()).
        self._video_cache: VisionEncoderCache[Gemma4Context] | None = None

        assert isinstance(self.kv_params, MultiKVCacheParams)

        if self._batch_processor is not None:
            assert isinstance(self._batch_processor, Gemma4BatchProcessor)
            self._batch_processor.bind_model_state(config=self.config)

    @property
    def model(self) -> Model:
        """Expose language model for graph capture/replay.

        Only the language model is captured since vision runs
        during prefill only.
        """
        return self.language_model

    def set_video_cache(self, cache: VisionEncoderCache[Gemma4Context]) -> None:
        """Transitional: receive the pipeline-owned cache for the video store.

        Video is a separate scatter path not yet unified into the images
        machinery; until it is, the pipeline hands the single owner's cache
        here so freshly-encoded videos are stored (and released) through it.
        """
        self._video_cache = cache

    def pack_vision_inputs(
        self,
        selection: Sequence[tuple[Gemma4Context, Sequence[ImageMetadata]]],
        devices: list[Device],
    ) -> VisionRawInputs | None:
        """Pack the pipeline-selected uncached image pixels to device.

        Runs in the pipeline's prep-ahead window (pinned host-to-device copy)
        so it overlaps the prior batch, delegating to the shared
        :func:`pack_uncached_images`.
        """
        assert self.config.vision_config is not None
        return pack_uncached_images(
            selection,
            devices,
            self.config.vision_config.pooling_kernel_size,
            self.config.unquantized_dtype,
        )

    def vision_execute(
        self,
        selection: Sequence[tuple[Gemma4Context, Sequence[ImageMetadata]]],
        devices: list[Device],
        packed: VisionRawInputs | None,
    ) -> VisionEncodeResult:
        """Run the vision encoder on the pixels packed by ``pack_vision_inputs``.

        Returns embeddings only; the pipeline derives per-image counts from its
        selection. When ``pack_vision_inputs`` had no packable patches
        (``packed is None``), returns an empty result so the pipeline assembles
        from the cache.
        """
        if packed is None:
            return VisionEncodeResult(
                embeddings=self.empty_vision_embeddings(self.devices)
            )
        return VisionEncodeResult(embeddings=self._run_vision_encoder(packed))

    def empty_vision_embeddings(self, devices: list[Device]) -> list[Buffer]:
        """Per-device zero-row image embeddings for cached / text-only batches.

        Cached: hit on every text-only / decode step, so it must not allocate
        per call.
        """
        if not hasattr(self, "_cached_empty_embeddings"):
            self._cached_empty_embeddings = create_empty_embeddings(
                devices,
                self.huggingface_config.text_config.hidden_size,
                self.config.unquantized_dtype,
            )
        return self._cached_empty_embeddings

    def _load_state_dict(self) -> dict[str, Any]:
        assert self._max_batch_size, "Expected max_batch_size to be set"

        # Get processed state dict for language and vision models
        weights_dict = dict(self.weights.items())
        self._language_weights_dict = convert_safetensor_language_state_dict(
            weights_dict
        )
        self._vision_weights_dict = convert_safetensor_vision_state_dict(
            weights_dict
        )
        return {k: v.data() for k, v in weights_dict.items()}

    def _create_model_config(
        self, state_dict: dict[str, Any]
    ) -> Gemma4ForConditionalGenerationConfig:
        model_config = Gemma4ForConditionalGenerationConfig.initialize(
            self.pipeline_config
        )
        model_config.finalize(
            huggingface_config=self.huggingface_config,
            state_dict=state_dict,
            return_logits=self.return_logits,
        )
        self.config = model_config

        # DISTINF-194: pre-fuse gate/up and qkv/qk projections when configured,
        # matching the FusedMLP / stacked qkv layers the graph builds.
        if gemma4_uses_fused_projections(model_config):
            self._language_weights_dict = fuse_gemma4_projection_weights(
                self._language_weights_dict
            )

        return model_config

    def _include_vision_graph(
        self, model_config: Gemma4ForConditionalGenerationConfig
    ) -> bool:
        return model_config.vision_config is not None

    def _language_model_input_types(
        self, config: Gemma4ForConditionalGenerationConfig
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
                # Match the vision tower's output dtype.
                config.unquantized_dtype,
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
            *self.kv_params.flattened_kv_inputs(),
        )

    def _build_language_graph(
        self,
        config: Gemma4ForConditionalGenerationConfig,
        state_dict: dict[str, WeightData],
        module: Module | None = None,
    ) -> tuple[Graph, dict[str, DLPackArray]]:
        """Build the language model with our input types and graph"""
        with Graph(
            "gemma4_language",
            input_types=self._language_model_input_types(config),
            module=module,
        ) as graph:
            language_model = Gemma4TextModel(config)
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

            # Extract KV cache inputs from the unified {sliding, global} tree.
            kv_cache_local, kv_cache_global = (
                self.kv_params.unflatten_basic_kv_tree(iter(variadic_args))
            )

            outputs = language_model(
                tokens=tokens.tensor,
                signal_buffers=signal_buffers,
                sliding_kv_collections=kv_cache_local,
                global_kv_collections=kv_cache_global,
                return_n_logits=return_n_logits.tensor,
                input_row_offsets=input_row_offsets,
                image_embeddings=image_embeddings,
                image_token_indices=image_token_indices,
            )
            graph.output(*outputs)
        return graph, language_model.state_dict()

    def _build_vision_graph(
        self,
        config: Gemma4ForConditionalGenerationConfig,
        state_dict: dict[str, WeightData],
        module: Module | None = None,
    ) -> tuple[Graph, dict[str, DLPackArray]]:
        """Build the vision model with our input types and graph"""
        vision_model = Gemma4VisionModel(
            config,
            device=DeviceRef.from_device(self.devices[0]),
        )
        vision_model.load_state_dict(
            state_dict=state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
            strict=self._strict_state_dict_loading,
        )
        with Graph(
            "gemma4_vision",
            input_types=vision_model.input_types(),
            module=module,
        ) as vision_graph:
            # Extract inputs
            all_inputs = vision_graph.inputs
            n_devices = len(self.devices)

            patches_flat_list = [inp.tensor for inp in all_inputs[:n_devices]]
            all_inputs = all_inputs[n_devices:]

            pixel_position_ids_list = [
                inp.tensor for inp in all_inputs[:n_devices]
            ]
            all_inputs = all_inputs[n_devices:]

            cu_seqlens_list = [inp.tensor for inp in all_inputs[:n_devices]]
            all_inputs = all_inputs[n_devices:]

            pool_gather_index_list = [
                inp.tensor for inp in all_inputs[:n_devices]
            ]
            all_inputs = all_inputs[n_devices:]

            max_seq_len = all_inputs[0].tensor

            outputs = vision_model(
                patches_flat_list,
                pixel_position_ids_list,
                cu_seqlens_list,
                pool_gather_index_list,
                max_seq_len,
            )
            vision_graph.output(*outputs)

        return vision_graph, vision_model.state_dict()

    def _run_vision_encoder(self, raw: VisionRawInputs) -> list[Buffer]:
        if self.vision_model is None:
            raise ValueError(
                "This checkpoint is served text-only (no vision encoder"
                " is loaded); image and video inputs are not supported."
            )
        return self.vision_model(
            *raw.patches_flat,
            *raw.pixel_position_ids,
            *raw.cu_seqlens,
            *raw.pool_gather_index,
            raw.max_seq_len,
        )

    @traced
    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Execute the vision model (if needed), then the language model."""
        model_inputs = cast(Gemma3MultiModalModelInputs, model_inputs)

        image_embeddings: list[Buffer]
        image_scatter: list[Buffer]
        if model_inputs.vision_embeddings is not None:
            image_embeddings = model_inputs.vision_embeddings
            scatter_np = model_inputs.vision_scatter_indices
            if scatter_np is not None and len(scatter_np) > 0:
                image_scatter = self._scatter_to_devices(scatter_np)
            else:
                image_scatter = self._empty_indices()
        else:
            image_embeddings = self.empty_vision_embeddings(self.devices)
            image_scatter = self._empty_indices()

        # --- video embeddings ---
        video_embeddings: list[Buffer]
        video_scatter: list[Buffer]
        vid = model_inputs.video
        if vid is not None and vid.cached_embeddings is not None:
            # Cache hit: embeddings pre-assembled in build_video_inputs.
            video_embeddings = vid.cached_embeddings
        elif vid is not None and vid.raw is not None:
            # Cache miss: encode, then store so future requests hit the cache.
            video_embeddings = self._run_vision_encoder(vid.raw)
            if vid.cache_hashes:
                assert vid.cache_per_video_token_counts is not None
                assert vid.cache_req_ids is not None
                assert self._video_cache is not None, (
                    "video cache not bound; pipeline must call set_video_cache"
                )
                self._video_cache._cache_and_split(
                    vision_outputs=video_embeddings,
                    per_image_token_counts=vid.cache_per_video_token_counts,
                    image_hashes=vid.cache_hashes,
                    request_ids=vid.cache_req_ids,
                )
        else:
            video_embeddings = self.empty_vision_embeddings(self.devices)

        if vid is not None:
            if vid.token_indices is not None:
                video_scatter = vid.token_indices
            else:
                assert vid.token_indices_np is not None
                video_scatter = self._scatter_to_devices(vid.token_indices_np)
        else:
            video_scatter = self._empty_indices()

        # --- merge image + video ---
        combined_embeds = merge_per_device_buffers(
            image_embeddings, video_embeddings
        )
        combined_indices = merge_per_device_buffers(
            image_scatter, video_scatter
        )

        assert model_inputs.kv_cache_inputs

        model_outputs = self.language_model.execute(
            model_inputs.tokens,
            model_inputs.return_n_logits,
            *model_inputs.input_row_offsets,
            *combined_embeds,
            *combined_indices,
            *model_inputs.signal_buffers,
            *model_inputs.kv_cache_inputs.flatten(),
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

    def _empty_indices(self) -> list[Buffer]:
        if not hasattr(self, "_cached_empty_indices"):
            self._cached_empty_indices = create_empty_indices(self.devices)
        return self._cached_empty_indices

    @traced
    def _scatter_to_devices(
        self, scatter_np: npt.NDArray[np.int32]
    ) -> list[Buffer]:
        """Copy scatter indices to each device using cached pinned buffers."""
        dev = self.devices[0]
        n = len(scatter_np)
        bufs = self._scatter_buffers.get(n)
        host: Buffer
        if bufs is None:
            if not dev.is_host:
                host = DevicePinnedBuffer(
                    dtype=DType.int32, shape=(n,), device=dev
                )
            else:
                host = Buffer(shape=(n,), dtype=DType.int32, device=dev)
            device = [host.to(d) for d in self.devices]
            bufs = (host, device)
            self._scatter_buffers[n] = bufs
        host, device = bufs
        host.to_numpy()[:] = scatter_np.astype(np.int32)
        for d in device:
            d.inplace_copy_from(host)
        return device
