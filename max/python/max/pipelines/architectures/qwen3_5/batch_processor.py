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
"""Input batching for Qwen3.5 pipeline models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from max import tree
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, DeviceRef, TensorType
from max.graph.buffer_utils import cast_tensors_to
from max.nn.kv_cache import KVCacheInputs
from max.nn.kv_cache.cache_params import KVCacheParamInterface
from max.pipelines.architectures.llama3.batch_processor import (
    Llama3BatchProcessor,
)
from max.pipelines.architectures.qwen3vl_moe.context import (
    Qwen3VLTextAndVisionContext,
)
from max.pipelines.context import ImageMetadata, TextContext
from max.pipelines.lib.interfaces.batch_processor import BatchProcessorRuntime
from max.pipelines.lib.vision_encoder_cache import VisionEncodeResult

from .model_config import Qwen3_5Config
from .vision_packing import Qwen3_5VisionInputs, pack_uncached_images

if TYPE_CHECKING:
    from .model import Qwen3_5Inputs


_MROPE_AXES = 3
"""Temporal, height and width, the three axes M-RoPE positions carry."""


def context_position_rows(
    ctx: TextContext, length: int | None = None
) -> npt.NDArray[np.int64]:
    """Returns one request's ``[3, length]`` M-RoPE positions.

    They start at the processed length and cover the active tokens, or
    ``length`` tokens where a verify window adds the drafts. A prompt whose
    images are still to be encoded takes its slice of the positions the
    tokenizer precomputed. Otherwise positions count on from the processed
    length, offset by the request's rope delta. The two agree past the last
    image.
    """
    start = ctx.tokens.processed_length
    if length is None:
        length = ctx.tokens.active_length
    rope_delta = 0
    if isinstance(ctx, Qwen3VLTextAndVisionContext):
        precomputed = ctx.decoder_position_ids
        if ctx.needs_vision_encoding and precomputed.shape[1] == len(
            ctx.tokens
        ):
            return precomputed[:, start : start + length]
        rope_delta = ctx.rope_delta
    flat = np.arange(start, start + length, dtype=np.int64) + rope_delta
    return np.tile(flat, (_MROPE_AXES, 1))


class Qwen3_5BatchProcessor(Llama3BatchProcessor):
    """Ragged batching with linear-attention state pools and optional vision inputs.

    It encodes each batch's new images with the vision graph its model, or a
    speculative pairing of it, binds.
    """

    mrope_enabled: bool = False
    """Whether the compiled graph takes M-RoPE positions."""

    vision_enabled: bool = True
    """Whether the compiled graph takes vision-merge inputs."""

    compute_dtype: DType
    """The dtype of the image rows, which the decoder graph merges.

    Only ``finalize`` resolves it for a quantized checkpoint, so a pipeline
    model that builds this batch processor first sets it from the finalized
    config."""

    _vision_model: Model | None = None
    _session: InferenceSession | None = None
    _empty_embeddings: list[Buffer] | None = None

    def __init__(
        self, config: Qwen3_5Config, runtime: BatchProcessorRuntime
    ) -> None:
        super().__init__(config, runtime)
        self._config = config
        self.mrope_enabled = config.mrope_enabled
        self.vision_enabled = config.vision_config is not None
        self.compute_dtype = config.compute_dtype

    def bind_vision_encoder(
        self, *, vision_model: Model, session: InferenceSession
    ) -> None:
        """Binds the compiled vision graph that encodes each batch's images."""
        self._vision_model = vision_model
        self._session = session

    def pack_vision_inputs(
        self,
        selection: Sequence[
            tuple[Qwen3VLTextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
    ) -> Qwen3_5VisionInputs | None:
        """Packs the pipeline-selected cache-miss images to device.

        Runs in the pipeline's prep-ahead window so the host-to-device copy
        overlaps the prior batch.
        """
        return pack_uncached_images(selection, devices)

    def vision_execute(
        self,
        selection: Sequence[
            tuple[Qwen3VLTextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
        packed: Qwen3_5VisionInputs | None,
    ) -> VisionEncodeResult:
        """Runs the vision encoder on the media ``pack_vision_inputs`` packed.

        Returns explicit per-item token counts rather than letting the driver
        derive them from placeholder spans. A clip's span covers ``grid_t``
        placeholder runs AND the timestamp text between them, so its width
        exceeds the rows the encoder emits; an image's span happens to match.
        """
        if packed is None:
            return VisionEncodeResult(
                embeddings=self.empty_vision_embeddings(devices)
            )
        assert self._vision_model is not None
        assert self._session is not None
        vision_outputs = self._vision_model.execute(
            *packed.graph_inputs(devices), *self.runtime.signal_buffers
        )
        assert isinstance(vision_outputs[0], Buffer)
        embeddings = cast_tensors_to(
            [vision_outputs[0]], self.compute_dtype, self._session
        )[0]
        # The hidden state is replicated across devices, so every replica
        # merges the same embeddings.
        return VisionEncodeResult(
            embeddings=[embeddings.to(device) for device in devices],
            per_image_token_counts=[
                img.embedding_rows
                for _ctx, miss_images in selection
                for img in miss_images
            ],
        )

    def empty_vision_embeddings(self, devices: list[Device]) -> list[Buffer]:
        """Per-device zero-row image embeddings for cached / text-only batches.

        Cached: hit on every text-only / decode step, so it must not allocate
        per call, and graph-capture replay only skips an input refresh for an
        identical buffer object.
        """
        if self._empty_embeddings is None:
            self._empty_embeddings = [
                Buffer.zeros(
                    shape=[0, self._config.hidden_size],
                    dtype=self.compute_dtype,
                ).to(device)
                for device in devices
            ]
        return self._empty_embeddings

    def get_symbolic_inputs(
        self,
        *,
        kv_params: KVCacheParamInterface,
        device_refs: list[DeviceRef],
    ) -> list[TensorType | BufferType]:
        """The graph's symbolic inputs, in the order the tree declares them."""
        assert self.input_tree is not None, "the model binds its inputs first"
        return tree.leaves(self.input_tree)

    def _decoder_position_ids(self, contexts: Sequence[TextContext]) -> Buffer:
        """Returns this step's ``[3, total_seq_len]`` M-RoPE positions."""
        rows = [context_position_rows(ctx) for ctx in contexts]
        return Buffer.from_numpy(
            np.concatenate(rows, axis=1).astype(np.int64)
        ).to(self.runtime.devices[0])

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> Qwen3_5Inputs:
        """Builds the first step's inputs from the replica batches."""
        from .model import Qwen3_5Inputs

        all_contexts = [ctx for batch in replica_batches for ctx in batch]

        # Checked before anything is staged: this batch cannot be served at
        # all, so there is no point building its inputs first.
        #
        # TODO(kevinbi): nothing between here and the model worker's main
        # loop catches this, so it ends the worker process rather than the
        # one request that asked for the impossible. Failing just the
        # request needs the scheduler's per-request path
        # (`SchedulerResult.failed`) to cover batch preparation, not only
        # batch construction.
        if not self.mrope_enabled and any(
            isinstance(ctx, Qwen3VLTextAndVisionContext)
            and ctx.needs_vision_encoding
            for ctx in all_contexts
        ):
            raise ValueError(
                "Qwen3.5 cannot serve image prompts for this checkpoint: "
                "it declares no vision config or no mrope_section, so "
                "M-RoPE positions are not wired into the compiled graph "
                "and every token after an image would get a flat position."
            )

        base_inputs = super().prepare_initial_token_inputs(
            replica_batches,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=return_n_logits,
        )
        request_ids = [ctx.request_id for ctx in all_contexts]

        return Qwen3_5Inputs(
            tokens=base_inputs.tokens,
            input_row_offsets=base_inputs.input_row_offsets,
            signal_buffers=base_inputs.signal_buffers,
            kv_cache_inputs=base_inputs.kv_cache_inputs,
            return_n_logits=base_inputs.return_n_logits,
            request_ids=request_ids,
            vision_inputs_in_graph=self.vision_enabled,
            decoder_position_ids=(
                self._decoder_position_ids(all_contexts)
                if self.mrope_enabled
                else None
            ),
        )
