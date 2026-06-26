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
"""Input batching for Gemma4 pipeline models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from max.driver import Buffer, DevicePinnedBuffer
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.nn.kv_cache import KVCacheInputsInterface
from max.nn.kv_cache.cache_params import KVCacheParamInterface
from max.pipelines.lib.interfaces.batch_processor import (
    BatchProcessor,
    BatchProcessorRuntime,
    process_ragged_kv_outputs,
    ragged_kv_symbolic_inputs,
)
from max.pipelines.lib.interfaces.pipeline_model import ModelOutputs
from max.pipelines.lib.vision_encoder_cache import VisionEncoderCache
from max.profiler import traced

from .batch_vision_inputs import (
    build_image_inputs,
    build_video_inputs,
    create_empty_embeddings,
    create_empty_indices,
)
from .context import Gemma4Context
from .model_config import Gemma4ForConditionalGenerationConfig

if TYPE_CHECKING:
    from .model import Gemma3MultiModalModelInputs


class Gemma4BatchProcessor(
    BatchProcessor[Gemma4Context, "Gemma3MultiModalModelInputs"]
):
    """Ragged batching with optional vision inputs for Gemma4 models."""

    _config: Gemma4ForConditionalGenerationConfig | None = None
    _ve_cache: VisionEncoderCache[Gemma4Context] | None = None

    # Cached pinned host + device buffer pairs keyed by (batch_size, total_seq_len).
    _execution_input_buffers: dict[
        tuple[int, int],
        tuple[Buffer, Buffer, list[Buffer], Buffer, Buffer],
    ]

    def __init__(
        self,
        config: Gemma4ForConditionalGenerationConfig,
        runtime: BatchProcessorRuntime,
    ) -> None:
        """Initialise with empty per-instance caches."""
        super().__init__(config, runtime)
        self._execution_input_buffers = {}

    def bind_model_state(
        self,
        *,
        config: Gemma4ForConditionalGenerationConfig,
        ve_cache: VisionEncoderCache[Gemma4Context],
    ) -> None:
        """Wire model config and vision encoder cache from ``load_model``.

        Args:
            config: Fully-initialised Gemma4 model configuration.
            ve_cache: Shared vision encoder result cache for this pipeline.
        """
        self._config = config
        self._ve_cache = ve_cache

    def get_symbolic_inputs(
        self,
        *,
        kv_params: KVCacheParamInterface,
        device_refs: list[DeviceRef],
    ) -> list[TensorType | BufferType]:
        return ragged_kv_symbolic_inputs(
            kv_params=kv_params,
            device_refs=device_refs,
            include_signal_buffers=True,
        )

    @traced
    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[Gemma4Context]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> Gemma3MultiModalModelInputs:
        """Prepare inputs for the first execution pass."""
        from .model import Gemma3MultiModalModelInputs

        if len(replica_batches) > 1:
            raise ValueError("Model does not support DP>1")
        context_batch = replica_batches[0]

        assert self._config is not None, (
            "config must be bound before prepare_initial_token_inputs(); "
            "call bind_model_state() in load_model()"
        )
        assert self._ve_cache is not None, (
            "ve_cache must be bound before prepare_initial_token_inputs(); "
            "call bind_model_state() in load_model()"
        )
        assert kv_cache_inputs is not None

        devices = self.runtime.devices
        dev = devices[0]
        pinned = not dev.is_host

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
                host_row_offsets.to(device) for device in devices
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

        needs_images = (
            any(
                getattr(ctx, "needs_vision_encoding", False)
                for ctx in context_batch
            )
            if context_batch
            else False
        )
        k = (
            self._config.vision_config.pooling_kernel_size
            if self._config.vision_config is not None
            else 1
        )
        if needs_images:
            uncached = self._ve_cache.get_uncached_contexts(context_batch)
            image_inputs = build_image_inputs(
                context_batch=context_batch,
                uncached=uncached,
                devices=devices,
                pooling_kernel_size=k,
                ve_cache=self._ve_cache,
                empty_embeddings=self._empty_embeddings(),
                dtype=self._config.unquantized_dtype,
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
                devices=devices,
                pooling_kernel_size=k,
                dtype=self._config.unquantized_dtype,
            )
        else:
            video_inputs = None

        return Gemma3MultiModalModelInputs(
            tokens=device_tokens,
            input_row_offsets=device_row_offsets,
            return_n_logits=return_n_logits_buf,
            signal_buffers=list(self.runtime.signal_buffers),
            kv_cache_inputs=kv_cache_inputs,
            images=image_inputs,
            video=video_inputs,
            combined_embeds=self._empty_embeddings(),
            combined_indices=self._empty_indices(),
        )

    def _empty_embeddings(self) -> list[Buffer]:
        assert self._config is not None
        if not hasattr(self, "_cached_empty_embeddings"):
            self._cached_empty_embeddings = create_empty_embeddings(
                self.runtime.devices,
                self._config.text_config.hidden_size,
                self._config.unquantized_dtype,
            )
        return self._cached_empty_embeddings

    def _empty_indices(self) -> list[Buffer]:
        if not hasattr(self, "_cached_empty_indices"):
            self._cached_empty_indices = create_empty_indices(
                self.runtime.devices
            )
        return self._cached_empty_indices

    def process_outputs(
        self, outputs: Sequence[Buffer | object]
    ) -> ModelOutputs:
        return process_ragged_kv_outputs(
            outputs,
            return_logits=self.runtime.return_logits,
            return_hidden_states=self.runtime.return_hidden_states,
        )
