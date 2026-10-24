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
from max import tree
from max.driver import Buffer, Device
from max.engine import InferenceSession, Model
from max.graph import BufferType, DeviceRef, TensorType
from max.nn.kv_cache import KVCacheInputs
from max.nn.kv_cache.cache_params import KVCacheParamInterface
from max.pipelines.context import ImageMetadata
from max.pipelines.lib.interfaces.arch_config import ArchConfig
from max.pipelines.lib.interfaces.batch_processor import (
    RAGGED_INPUT_ROW_OFFSETS,
    RAGGED_INPUT_TOKENS,
    BatchProcessor,
    BatchProcessorRuntime,
    process_ragged_kv_outputs,
    ragged_token_descriptors,
)
from max.pipelines.lib.interfaces.pipeline_model import ModelOutputs
from max.pipelines.lib.vision_encoder_cache import VisionEncodeResult
from max.profiler import traced

from .batch_vision_inputs import (
    VisionRawInputs,
    create_empty_embeddings,
    pack_uncached_images,
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
    _vision_model: Model | None = None
    _empty_embeddings: list[Buffer] | None = None

    def __init__(
        self,
        config: ArchConfig,
        runtime: BatchProcessorRuntime,
    ) -> None:
        super().__init__(config, runtime, ragged_token_descriptors(runtime))
        # The vision encoder reads only the config's vision half, which the
        # HuggingFace config fixes before the language model finalizes.
        if isinstance(config, Gemma4ForConditionalGenerationConfig):
            self._config = config

    def bind_model_state(
        self,
        *,
        config: Gemma4ForConditionalGenerationConfig,
    ) -> None:
        """Wire the model config from ``load_model``.

        Images go through the pipeline-owned ``VisionEncoderCache``; this
        processor only builds tokens/offsets and video inputs.

        Args:
            config: Fully-initialised Gemma4 model configuration.
        """
        self._config = config

    def bind_vision_encoder(
        self, *, vision_model: Model, session: InferenceSession
    ) -> None:
        """Wire the compiled vision encoder."""
        del session
        self._vision_model = vision_model

    def pack_vision_inputs(
        self,
        selection: Sequence[tuple[Gemma4Context, Sequence[ImageMetadata]]],
        devices: list[Device],
    ) -> VisionRawInputs | None:
        """Pack the pipeline-selected uncached image pixels to device.

        Runs in the pipeline's prep-ahead window (pinned host-to-device copy)
        so it overlaps the prior batch.
        """
        assert self._config is not None
        assert self._config.vision_config is not None
        return pack_uncached_images(
            selection,
            devices,
            self._config.vision_config.pooling_kernel_size,
            self._config.unquantized_dtype,
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
                embeddings=self.empty_vision_embeddings(devices)
            )
        if self._vision_model is None:
            raise ValueError(
                "This checkpoint is served text-only (no vision encoder"
                " is loaded); image and video inputs are not supported."
            )
        return VisionEncodeResult(
            embeddings=self._vision_model(
                *packed.patches_flat,
                *packed.pixel_position_ids,
                *packed.cu_seqlens,
                *packed.pool_gather_index,
                packed.max_seq_len,
            )
        )

    def empty_vision_embeddings(self, devices: list[Device]) -> list[Buffer]:
        """Per-device zero-row image embeddings for cached / text-only batches.

        Cached: hit on every text-only / decode step, so it must not allocate
        per call.
        """
        if self._empty_embeddings is None:
            assert self._config is not None
            self._empty_embeddings = create_empty_embeddings(
                devices,
                self._config.text_config.hidden_size,
                self._config.unquantized_dtype,
            )
        return self._empty_embeddings

    def get_symbolic_inputs(
        self,
        *,
        kv_params: KVCacheParamInterface,
        device_refs: list[DeviceRef],
    ) -> list[TensorType | BufferType]:
        assert self.input_tree is not None, "the model binds its inputs first"
        return tree.leaves(self.input_tree)

    @traced
    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[Gemma4Context]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> Gemma3MultiModalModelInputs:
        """Prepare inputs for the first execution pass."""
        from .model import Gemma3MultiModalModelInputs

        if len(replica_batches) > 1:
            raise ValueError("Model does not support DP>1")
        context_batch = replica_batches[0]

        assert kv_cache_inputs is not None

        batch_size = len(context_batch)
        total_seq_len = sum(ctx.tokens.active_length for ctx in context_batch)

        with self._stager.stage() as staging:
            host_tokens, (device_tokens,) = staging.get(
                RAGGED_INPUT_TOKENS, (total_seq_len,)
            )
            host_row_offsets, (device_row_offsets,) = staging.get(
                RAGGED_INPUT_ROW_OFFSETS, (batch_size + 1,)
            )
            np.cumsum(
                [0] + [ctx.tokens.active_length for ctx in context_batch],
                dtype=np.uint32,
                out=host_row_offsets.to_numpy(),
            )
            if context_batch:
                np.concatenate(
                    [ctx.tokens.active for ctx in context_batch],
                    out=host_tokens.to_numpy(),
                )

        return_n_logits_buf = Buffer.from_numpy(
            np.array([return_n_logits], dtype=np.int64)
        )

        return Gemma3MultiModalModelInputs(
            tokens=device_tokens,
            input_row_offsets=device_row_offsets,
            return_n_logits=return_n_logits_buf,
            signal_buffers=list(self.runtime.signal_buffers),
            kv_cache_inputs=kv_cache_inputs,
        )

    def process_outputs(
        self, outputs: Sequence[Buffer | object]
    ) -> ModelOutputs:
        return process_ragged_kv_outputs(
            outputs,
            return_logits=self.runtime.return_logits,
            return_hidden_states=self.runtime.return_hidden_states,
        )
