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
"""Input batching for the unified MTP Gemma4 pipeline model."""

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
    ragged_kv_symbolic_inputs,
)
from max.pipelines.lib.interfaces.pipeline_model import ModelOutputs
from max.pipelines.lib.vision_encoder_cache import VisionEncoderCache

from ..gemma4.batch_vision_inputs import (
    build_image_inputs,
    build_video_inputs,
    create_empty_embeddings,
    create_empty_indices,
)
from ..gemma4.context import Gemma4Context

if TYPE_CHECKING:
    from ..gemma4.model_config import Gemma4ForConditionalGenerationConfig
    from .model import UnifiedMTPGemma4Inputs


class UnifiedMTPGemma4BatchProcessor(
    BatchProcessor[Gemma4Context, "UnifiedMTPGemma4Inputs"]
):
    """Ragged batching with signal buffers + optional vision for unified MTP.

    Prepares :class:`UnifiedMTPGemma4Inputs` for each forward pass.
    ``draft_tokens`` and sampling buffers are left as ``None`` and filled
    in by the overlap pipeline after this method returns.
    """

    _config: Gemma4ForConditionalGenerationConfig | None = None
    _ve_cache: VisionEncoderCache[Gemma4Context] | None = None

    def __init__(
        self,
        config: Gemma4ForConditionalGenerationConfig,
        runtime: BatchProcessorRuntime,
    ) -> None:
        super().__init__(config, runtime)

    def bind_model_state(
        self,
        *,
        config: Gemma4ForConditionalGenerationConfig,
        ve_cache: VisionEncoderCache[Gemma4Context],
    ) -> None:
        """Wire model config and vision encoder cache from ``load_model``."""
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

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[Gemma4Context]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> UnifiedMTPGemma4Inputs:
        """Prepare batch inputs for a UnifiedMTPGemma4 forward pass.

        Args:
            replica_batches: One inner list per DP replica containing the
                :class:`Gemma4Context` objects for that shard.
            kv_cache_inputs: Optional KV cache inputs (may be ``None`` during
                compilation warm-up).
            return_n_logits: Number of per-token logit rows to return.

        Returns:
            :class:`UnifiedMTPGemma4Inputs` with tokens, row offsets, signal
            buffers, optional image/video inputs, and ``draft_tokens=None``
            (set later by the overlap pipeline).
        """
        from .model import UnifiedMTPGemma4Inputs

        context_batch = [ctx for batch in replica_batches for ctx in batch]
        devices = self.runtime.devices
        device0 = devices[0]
        pinned = not device0.is_host

        batch_size = len(context_batch)
        total_seq_len = sum(ctx.tokens.active_length for ctx in context_batch)

        buffer_type = DevicePinnedBuffer if pinned else Buffer
        host_tokens = buffer_type(
            dtype=DType.int64, shape=(total_seq_len,), device=device0
        )
        host_row_offsets = buffer_type(
            dtype=DType.uint32,
            shape=(batch_size + 1,),
            device=device0,
        )

        np.concatenate(
            [ctx.tokens.active for ctx in context_batch],
            out=host_tokens.to_numpy(),
        )
        device_tokens = host_tokens.to(device0)

        np.cumsum(
            [0] + [ctx.tokens.active_length for ctx in context_batch],
            dtype=np.uint32,
            out=host_row_offsets.to_numpy(),
        )
        device_row_offsets = host_row_offsets.to(device0)

        host_input_row_offsets = Buffer.from_numpy(
            np.cumsum(
                [0] + [ctx.tokens.active_length for ctx in context_batch],
                dtype=np.uint32,
            )
        )

        return_n_logits_buf = Buffer.from_numpy(
            np.array([return_n_logits], dtype=np.int64)
        )

        data_parallel_splits = Buffer.from_numpy(
            np.array([0, batch_size], dtype=np.int64)
        )

        batch_context_lengths = [
            Buffer.zeros(shape=[1], dtype=DType.int32)
            for _ in range(len(self.runtime.devices))
        ]

        # --- Vision inputs (mirrors Gemma4BatchProcessor) ---
        # The vision encoder runs during prefill only; execute() consumes
        # ``images``/``video`` to produce combined_embeds/indices, which default
        # to empty (a no-op scatter) for text-only and decode steps.
        assert self._config is not None, (
            "config must be bound before prepare_initial_token_inputs(); "
            "call bind_model_state() in load_model()"
        )
        assert self._ve_cache is not None, (
            "ve_cache must be bound before prepare_initial_token_inputs(); "
            "call bind_model_state() in load_model()"
        )
        k = (
            self._config.vision_config.pooling_kernel_size
            if self._config.vision_config is not None
            else 1
        )
        needs_images = any(
            getattr(ctx, "needs_vision_encoding", False)
            for ctx in context_batch
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

        needs_video = any(
            getattr(ctx, "needs_video_encoding", False) for ctx in context_batch
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

        return UnifiedMTPGemma4Inputs(
            tokens=device_tokens,
            input_row_offsets=device_row_offsets,
            host_input_row_offsets=host_input_row_offsets,
            return_n_logits=return_n_logits_buf,
            data_parallel_splits=data_parallel_splits,
            signal_buffers=list(self.runtime.signal_buffers),
            kv_cache_inputs=kv_cache_inputs,
            batch_context_lengths=batch_context_lengths,
            draft_tokens=None,
            structured_output=self.runtime.pipeline_config.needs_bitmask_constraints,
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
        assert isinstance(outputs[0], Buffer)
        return ModelOutputs(logits=outputs[0])
