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
"""Shared text batching and ragged image packing for registered Pixtral."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from max.driver import Buffer, Device
from max.engine import InferenceSession
from max.pipelines.context import ImageMetadata, TextAndVisionContext
from max.pipelines.lib.module_registration import DecoderBatchProcessor
from max.pipelines.lib.vision_batching import (
    create_empty_image_embeddings_single,
)
from max.pipelines.lib.vision_encoder_cache import VisionEncodeResult
from max.tensor.compilation import CompiledCallable

from .model_config import PixtralConfig


class PixtralModuleV3BatchProcessor(DecoderBatchProcessor):
    """Encodes selected uncached images; the framework batches decoder fields."""

    _vision_model: CompiledCallable[Any, Any] | None = None
    _empty_embeddings: list[Buffer] | None = None

    def bind_vision_encoder(
        self,
        *,
        vision_model: CompiledCallable[Any, Any],
        session: InferenceSession,
    ) -> None:
        """Binds the framework-compiled vision tower."""
        self._vision_model = vision_model

    def pack_vision_inputs(
        self,
        selection: Sequence[
            tuple[TextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
    ) -> tuple[Buffer, Buffer, Buffer]:
        """Extracts image patches and preserves each image's attention block."""
        assert isinstance(self.config, PixtralConfig)
        patch_size = self.config.patch_size
        patches_per_side = self.config.image_size // patch_size
        patches: list[np.ndarray] = []
        positions: list[np.ndarray] = []
        counts: list[int] = []
        for _, images in selection:
            for metadata in images:
                image = np.ascontiguousarray(metadata.pixel_values)
                channels, height, width = image.shape
                rows, columns = height // patch_size, width // patch_size
                count = rows * columns
                image_patches = image.reshape(
                    channels, rows, patch_size, columns, patch_size
                ).transpose(1, 3, 0, 2, 4)
                patches.append(
                    image_patches.reshape(count, -1).astype(np.float32)
                )
                positions.append(
                    (
                        np.repeat(np.arange(rows), columns) * patches_per_side
                        + np.tile(np.arange(columns), rows)
                    ).astype(np.int64)
                )
                counts.append(count)
        total = sum(counts)
        # Matches the previous per-batch graph's softmax saturation limit.
        mask = np.full((1, 1, total, total), -10000.0, dtype=np.float32)
        offset = 0
        for count in counts:
            mask[0, 0, offset : offset + count, offset : offset + count] = 0
            offset += count
        if not patches:
            raise ValueError("Pixtral vision packing requires selected images")
        return (
            Buffer.from_numpy(np.concatenate(patches)).to(devices[0]),
            Buffer.from_numpy(mask).to(devices[0]),
            Buffer.from_numpy(np.concatenate(positions)).to(devices[0]),
        )

    def vision_execute(
        self,
        selection: Sequence[
            tuple[TextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
        packed: tuple[Buffer, Buffer, Buffer],
    ) -> VisionEncodeResult:
        """Encodes the selected images and validates their patch-token spans."""
        assert self._vision_model is not None
        output = self._vision_model(*packed)[0].driver_tensor
        assert isinstance(output, Buffer)
        counts = [
            image.embedding_rows for _, images in selection for image in images
        ]
        if output.shape[0] != sum(counts):
            raise ValueError(
                "Pixtral vision embeddings do not match image token spans"
            )
        return VisionEncodeResult(
            embeddings=[output.to(device) for device in devices],
            per_image_token_counts=counts,
        )

    def empty_vision_embeddings(self, devices: list[Device]) -> list[Buffer]:
        """Returns cached zero-row embeddings for text-only steps."""
        assert isinstance(self.config, PixtralConfig)
        if self._empty_embeddings is None:
            self._empty_embeddings = [
                create_empty_image_embeddings_single(
                    device, self.config.hidden_size
                )
                for device in devices
            ]
        return self._empty_embeddings
