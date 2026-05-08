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
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from max.driver import Buffer, Device, DevicePinnedBuffer, DLPackArray
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, DeviceRef, Graph, Module, TensorType
from max.graph.weights import WeightData, Weights, WeightsAdapter
from max.interfaces import RequestID
from max.kv_cache.paged_kv_cache.increment_cache_lengths import (
    IncrementCacheLengthsProcessor,
)
from max.nn.comm import Signals
from max.nn.kv_cache import KVCacheInputs, MultiKVCacheParams
from max.nn.transformer import ReturnLogits
from max.pipelines.lib import (
    AlwaysSignalBuffersMixin,
    CompilationTimer,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModelWithKVCache,
)
from max.pipelines.lib.vision_encoder_cache import VisionEncoderCache
from max.profiler import traced
from transformers import AutoConfig

from .batch_vision_inputs import (
    ImageInputs,
    VideoInputs,
    VisionRawInputs,
    build_image_inputs,
    build_video_inputs,
    create_empty_embeddings,
    create_empty_indices,
    merge_per_device_buffers,
)
from .context import Gemma4Context
from .gemma4 import Gemma4TextModel
from .model_config import Gemma4ForConditionalGenerationConfig
from .vision_model.vision_model import Gemma4VisionModel
from .weight_adapters import (
    convert_safetensor_language_state_dict,
    convert_safetensor_vision_state_dict,
)

logger = logging.getLogger("max.pipelines")

_GRAPH_CAPTURE_HEADROOM_BYTES = 2 * 1024**3  # 2 GiB


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
        images: Inputs to the image encoder.
        video: Inputs to the video encoder.
    """

    tokens: npt.NDArray[np.integer[Any]] | Buffer
    input_row_offsets: npt.NDArray[np.integer[Any]] | list[Buffer]
    signal_buffers: list[Buffer]
    return_n_logits: Buffer

    images: ImageInputs | None = None
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
    PipelineModelWithKVCache[Gemma4Context],
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

        # signal_buffers are provided by AlwaysSignalBuffersMixin as a cached_property
        # to avoid GPU memory allocation during compile-only mode (cross-compilation).
        # Force initialization here to ensure buffers are ready before model execution,
        # preventing potential race conditions in multi-GPU scenarios.
        _ = self.signal_buffers

        self.vision_model, self.language_model = self.load_model(session)

        self._ve_cache: VisionEncoderCache[Gemma4Context] = VisionEncoderCache(
            max_entries=pipeline_config.runtime.max_vision_cache_entries
        )

        assert isinstance(self.kv_params, MultiKVCacheParams)
        self._increment_global_cache_lengths_processor = (
            IncrementCacheLengthsProcessor(
                session=session, params=self.kv_params.params[1]
            )
        )

    @property
    def model(self) -> Model:
        """Expose language model for graph capture/replay.

        Only the language model is captured since vision runs
        during prefill only.
        """
        return self.language_model

    def release(self, request_id: RequestID) -> None:
        """Release vision encoder cache for a completed request."""
        self._ve_cache.release_request(request_id)

    @classmethod
    def estimate_activation_memory(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        del huggingface_config  # Unused.

        # FIXME: We arbitrarily set some memory for activation memory to leave headroom
        # for vision processing. We should determine this in a more principled way.
        # Update: Bumped to 15 GiB after #80736 removed MemoryManager fallthrough.
        base = 15 * 1024 * 1024 * 1024  # 15 GiB
        if pipeline_config.runtime.device_graph_capture:
            base += _GRAPH_CAPTURE_HEADROOM_BYTES
        return base

    @classmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        """Calculates the maximum sequence length for the InternVL model."""
        return Gemma4ForConditionalGenerationConfig.calculate_max_seq_len(
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
    ) -> MultiKVCacheParams:
        """Gets the parameters required to configure the KV cache for InternVL."""
        return Gemma4ForConditionalGenerationConfig.construct_kv_params(
            huggingface_config,
            pipeline_config,
            devices,
            kv_cache_config,
            cache_dtype,
        )

    def load_model(self, session: InferenceSession) -> tuple[Model, Model]:
        """Loads the compiled Gemma3 MultiModal models into the MAX Engine session.

        Returns:
            A tuple of (vision_model, language_model).
        """
        assert self.pipeline_config.runtime.max_batch_size, (
            "Expected max_batch_size to be set"
        )

        # Get processed state dict for language and vision models
        weights_dict = dict(self.weights.items())
        language_weights_dict = convert_safetensor_language_state_dict(
            weights_dict
        )

        vision_weights_dict = convert_safetensor_vision_state_dict(weights_dict)

        raw_state_dict = {k: v.data() for k, v in weights_dict.items()}
        model_config = Gemma4ForConditionalGenerationConfig.initialize(
            self.pipeline_config
        )
        model_config.finalize(
            huggingface_config=self.huggingface_config,
            state_dict=raw_state_dict,
            return_logits=self.return_logits,
        )
        self.config = model_config

        input_row_offsets_prealloc_host = Buffer.from_numpy(
            np.arange(
                self.pipeline_config.runtime.max_batch_size + 1,
                dtype=np.uint32,
            )
        )
        self._input_row_offsets_prealloc = [
            input_row_offsets_prealloc_host.to(dev) for dev in self.devices
        ]

        # Cache for pinned host + device buffer pairs, keyed by
        # (batch_size, total_seq_len), to avoid per-call h2d allocations
        # in prepare_initial_token_inputs.
        self._execution_input_buffers: dict[
            tuple[int, int],
            tuple[Buffer, Buffer, list[Buffer], Buffer, Buffer],
        ] = {}

        # Cache for scatter-index buffers (pinned host + device), keyed by
        # length, to avoid per-call h2d allocations for image/video scatter.
        self._scatter_buffers: dict[int, tuple[Buffer, list[Buffer]]] = {}

        # Build and compile vision + language model together.
        with CompilationTimer("vision + language model") as timer:
            module = Module()

            vision_graph, vision_model_state_dict = self._build_vision_graph(
                model_config, vision_weights_dict, module=module
            )

            language_graph, language_model_state_dict = (
                self._build_language_graph(
                    model_config, language_weights_dict, module=module
                )
            )
            timer.mark_build_complete()

            combined_weights = {
                **vision_model_state_dict,
                **language_model_state_dict,
            }
            models = session.load_all(module, weights_registry=combined_weights)
            vision_model = models[vision_graph.name]
            language_model = models[language_graph.name]

        return vision_model, language_model

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

            # Extract KV cache inputs
            kv_cache = self._unflatten_kv_inputs(variadic_args)
            kv_cache_local, kv_cache_global = (
                kv_cache[: len(kv_cache) // 2],
                kv_cache[len(kv_cache) // 2 :],
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
        vision_graph = Graph(
            "gemma4_vision",
            vision_model,
            vision_model.input_types(),
            module=module,
        )
        return vision_graph, vision_model.state_dict()

    def _run_vision_encoder(self, raw: VisionRawInputs) -> list[Buffer]:
        return self.vision_model(
            *raw.patches_flat,
            *raw.pixel_position_ids,
            *raw.cu_seqlens,
            *raw.pool_weights,
            raw.max_seq_len,
        )

    @traced
    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Execute the vision model (if needed), then the language model."""
        model_inputs = cast(Gemma3MultiModalModelInputs, model_inputs)

        # --- image embeddings ---
        image_embeddings: list[Buffer]
        image_scatter: list[Buffer]
        img = model_inputs.images
        if img is not None and img.raw is not None:
            raw_embeds = self._run_vision_encoder(img.raw)

            assert img.cache_context_batch is not None
            assert img.cache_uncached_contexts is not None
            assert img.cache_per_image_token_counts is not None
            image_embeddings, scatter_np = (
                self._ve_cache.prepare_vision_outputs(
                    context_batch=img.cache_context_batch,
                    uncached_contexts=img.cache_uncached_contexts,
                    vision_embeds=raw_embeds,
                    per_image_token_counts=img.cache_per_image_token_counts,
                    n_devices=len(self.devices),
                    empty_embeddings=self._empty_embeddings(),
                )
            )
            if len(scatter_np) > 0:
                image_scatter = self._scatter_to_devices(scatter_np)
            else:
                image_scatter = self._empty_indices()
        elif img is not None and img.cached_embeddings is not None:
            image_embeddings = img.cached_embeddings
            if img.cached_token_indices is not None:
                image_scatter = img.cached_token_indices
            else:
                assert img.cached_token_indices_np is not None
                image_scatter = self._scatter_to_devices(
                    img.cached_token_indices_np
                )
        else:
            image_embeddings = self._empty_embeddings()
            image_scatter = self._empty_indices()

        # --- video embeddings ---
        video_embeddings: list[Buffer]
        video_scatter: list[Buffer]
        vid = model_inputs.video
        if vid is not None:
            video_embeddings = self._run_vision_encoder(vid.raw)
            if vid.token_indices is not None:
                video_scatter = vid.token_indices
            else:
                assert vid.token_indices_np is not None
                video_scatter = self._scatter_to_devices(vid.token_indices_np)
        else:
            video_embeddings = self._empty_embeddings()
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

    @traced
    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[Gemma4Context]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> ModelInputs:
        """Prepare inputs for the first execution pass."""
        if len(replica_batches) > 1:
            raise ValueError("Model does not support DP>1")
        context_batch = replica_batches[0]

        dev = self.devices[0]
        pinned = not dev.is_host
        assert kv_cache_inputs is not None

        batch_size = len(context_batch)
        total_seq_len = sum(ctx.tokens.active_length for ctx in context_batch)
        buffer_key = (batch_size, total_seq_len)
        buffers = self._execution_input_buffers.get(buffer_key)
        host_tokens: Buffer
        host_row_offsets: Buffer
        if buffers is None:
            if pinned:
                host_tokens = DevicePinnedBuffer(
                    dtype=DType.int64, shape=(total_seq_len,), device=dev
                )
                host_row_offsets = DevicePinnedBuffer(
                    dtype=DType.uint32,
                    shape=(batch_size + 1,),
                    device=dev,
                )
            else:
                host_tokens = Buffer(
                    shape=(total_seq_len,), dtype=DType.int64, device=dev
                )
                host_row_offsets = Buffer(
                    shape=(batch_size + 1,), dtype=DType.uint32, device=dev
                )
            device_tokens = host_tokens.to(dev)
            device_row_offsets = [
                host_row_offsets.to(device) for device in self.devices
            ]
            return_n_logits_buf = Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            )
            buffers = (
                host_tokens,
                host_row_offsets,
                device_row_offsets,
                device_tokens,
                return_n_logits_buf,
            )
            self._execution_input_buffers[buffer_key] = buffers

        (
            host_tokens,
            host_row_offsets,
            device_row_offsets,
            device_tokens,
            return_n_logits_buf,
        ) = buffers

        # Fill host buffers in-place, then copy to device.
        row_offsets_np = host_row_offsets.to_numpy()
        np.cumsum(
            [0] + [ctx.tokens.active_length for ctx in context_batch],
            dtype=np.uint32,
            out=row_offsets_np,
        )

        tokens_np = host_tokens.to_numpy()
        if context_batch:
            np.concatenate(
                [ctx.tokens.active for ctx in context_batch],
                out=tokens_np,
            )

        device_tokens.inplace_copy_from(host_tokens)
        for d_offsets in device_row_offsets:
            d_offsets.inplace_copy_from(host_row_offsets)

        k = self.config.vision_config.pooling_kernel_size

        needs_images = (
            any(
                getattr(ctx, "needs_vision_encoding", False)
                for ctx in context_batch
            )
            if context_batch
            else False
        )
        if needs_images:
            uncached = self._ve_cache.get_uncached_contexts(context_batch)
            image_inputs = build_image_inputs(
                context_batch=context_batch,
                uncached=uncached,
                devices=self.devices,
                pooling_kernel_size=k,
                ve_cache=self._ve_cache,
                empty_embeddings=self._empty_embeddings(),
            )
        else:
            image_inputs = None

        needs_video = (
            any(
                getattr(ctx, "needs_video_encoding", False)
                for ctx in context_batch
            )
            if context_batch
            else False
        )
        if needs_video:
            video_inputs = build_video_inputs(
                context_batch=context_batch,
                devices=self.devices,
                pooling_kernel_size=k,
            )
        else:
            video_inputs = None

        return Gemma3MultiModalModelInputs(
            tokens=device_tokens,
            input_row_offsets=device_row_offsets,
            return_n_logits=return_n_logits_buf,
            signal_buffers=self.signal_buffers,
            kv_cache_inputs=kv_cache_inputs,
            images=image_inputs,
            video=video_inputs,
            combined_embeds=self._empty_embeddings(),
            combined_indices=self._empty_indices(),
        )

    @traced
    def prepare_next_token_inputs(
        self, next_tokens: Buffer, prev_model_inputs: ModelInputs
    ) -> ModelInputs:
        prev_model_inputs = cast(Gemma3MultiModalModelInputs, prev_model_inputs)

        # Extract the global cache portion from combined kv_cache_inputs.
        # Combined layout per replica: [primary_tp0..tpN, global_tp0..tpN].
        n_devices = len(self.devices)
        assert prev_model_inputs.kv_cache_inputs is not None
        global_kv_inputs = KVCacheInputs(
            inputs=prev_model_inputs.kv_cache_inputs.inputs[
                n_devices : 2 * n_devices
            ]
        )
        self._increment_global_cache_lengths_processor.execute(
            kv_cache_inputs=global_kv_inputs,
            prev_model_inputs=prev_model_inputs,
        )

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
            combined_embeds=self._empty_embeddings(),
            combined_indices=self._empty_indices(),
        )

    def _empty_embeddings(self) -> list[Buffer]:
        if not hasattr(self, "_cached_empty_embeddings"):
            self._cached_empty_embeddings = create_empty_embeddings(
                self.devices,
                self.huggingface_config.text_config.hidden_size,
            )
        return self._cached_empty_embeddings

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
