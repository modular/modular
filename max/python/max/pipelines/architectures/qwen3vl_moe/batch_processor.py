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
"""Batching for the registered Qwen3-VL decoder and its vision tower.

The framework batches the text and the cache. This adds each token's 3-axis
M-RoPE position and encodes the batch's new images and video frames on the
first device, copying the rows to the others.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from max.driver import Buffer, Device
from max.engine import InferenceSession
from max.graph import TensorType
from max.pipelines.context import ImageMetadata, TextContext
from max.pipelines.lib.module_registration import DecoderBatchProcessor
from max.pipelines.lib.vision_encoder_cache import VisionEncodeResult
from max.tensor.compilation import CompiledCallable

from .context import Qwen3VLTextAndVisionContext
from .model_config import Qwen3VLConfig
from .vision_packing import Qwen3VLVisionInputs, pack_uncached_images

__all__ = ["Qwen3VLBatchProcessor", "context_position_rows"]

_MROPE_AXES = 3
"""Temporal, height and width, the three axes M-RoPE positions carry."""


def context_position_rows(ctx: TextContext) -> npt.NDArray[np.int64]:
    """Returns one request's ``[3, active_length]`` M-RoPE positions.

    They start at the processed length. A prompt whose images are still to
    be encoded takes its slice of the positions the tokenizer precomputed.
    Otherwise positions count on from the processed length, offset by the
    request's rope delta. The two agree past the last image.
    """
    start = ctx.tokens.processed_length
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


class Qwen3VLBatchProcessor(DecoderBatchProcessor):
    """Batches Qwen3-VL's tokens and encodes their images and clips."""

    _vision_model: CompiledCallable[Any, Any] | None = None
    _empty_vision_embeddings: list[Buffer] | None = None

    @property
    def _qwen_config(self) -> Qwen3VLConfig:
        assert isinstance(self.config, Qwen3VLConfig)
        return self.config

    def token_positions(
        self, context_batch: Sequence[TextContext], position_type: TensorType
    ) -> npt.NDArray[np.integer[Any]]:
        """Returns this step's ``[3, total_seq_len]`` M-RoPE positions."""
        if not context_batch:
            return np.empty((_MROPE_AXES, 0), dtype=np.int64)
        return np.concatenate(
            [context_position_rows(ctx) for ctx in context_batch], axis=1
        ).astype(np.int64)

    def bind_vision_encoder(
        self,
        *,
        vision_model: CompiledCallable[Any, Any],
        session: InferenceSession,
    ) -> None:
        """Binds the compiled vision tower."""
        del session
        self._vision_model = vision_model

    def pack_vision_inputs(
        self,
        selection: Sequence[
            tuple[Qwen3VLTextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
    ) -> Qwen3VLVisionInputs | None:
        """Packs the cache-miss images to the first device.

        Runs in the pipeline's prep-ahead window, so the host-to-device copy
        overlaps the prior batch.
        """
        return pack_uncached_images(selection, devices)

    def vision_execute(
        self,
        selection: Sequence[
            tuple[Qwen3VLTextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
        packed: Qwen3VLVisionInputs | None,
    ) -> VisionEncodeResult:
        """Encodes the packed media and copies its rows to every device.

        Returns each item's row count: a clip's placeholder span also holds
        the timestamp text between its frames, so it is wider than the rows
        the tower emits.
        """
        if packed is None:
            return VisionEncodeResult(
                embeddings=self.empty_vision_embeddings(devices)
            )
        assert self._vision_model is not None
        (rows,) = self._vision_model.execute_raw(*packed.graph_inputs())
        assert isinstance(rows, Buffer)
        return VisionEncodeResult(
            embeddings=[rows.to(device) for device in devices],
            per_image_token_counts=[
                img.embedding_rows
                for _ctx, miss_images in selection
                for img in miss_images
            ],
        )

    def empty_vision_embeddings(self, devices: list[Device]) -> list[Buffer]:
        """Per-device ``[0, hidden]`` rows for a step that encodes nothing.

        Cached: every decode step asks for them, and a captured device
        graph replays only an identical buffer object without a refresh.
        """
        if self._empty_vision_embeddings is None:
            config = self._qwen_config
            self._empty_vision_embeddings = [
                Buffer.zeros(
                    shape=[0, config.vision_config.embedding_width],
                    dtype=config.declared_dtype,
                ).to(device)
                for device in devices
            ]
        return self._empty_vision_embeddings
