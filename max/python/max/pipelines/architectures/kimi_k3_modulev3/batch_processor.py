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

"""Batches Kimi K3 images while the framework manages decoder state."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph.buffer_utils import cast_tensor_to
from max.pipelines.context import ImageMetadata
from max.pipelines.lib.module_registration import DecoderBatchProcessor
from max.pipelines.lib.vision_encoder_cache import (
    VisionEncodeResult,
    concat_device_buffers,
)
from max.tensor.compilation import CompiledCallable

from .context import KimiK3TextAndVisionContext
from .model_config import KimiK3Config


@dataclass(frozen=True)
class KimiK3ImageInputs:
    """Keeps a selected image's CPU inputs together until encoding."""

    pixels: npt.NDArray[np.float32]
    grid: npt.NDArray[np.int64]
    positions: npt.NDArray[np.int64]
    token_count: int


class KimiK3BatchProcessor(DecoderBatchProcessor):
    """Encodes cache-miss images individually and replicates their rows."""

    _vision_model: CompiledCallable[Any, Any] | None = None
    _session: InferenceSession | None = None
    _empty_vision_embeddings: list[Buffer] | None = None

    def bind_vision_encoder(
        self,
        *,
        vision_model: CompiledCallable[Any, Any],
        session: InferenceSession,
    ) -> None:
        """Binds the registered vision graph and its conversion session."""
        self._vision_model = vision_model
        self._session = session

    def pack_vision_inputs(
        self,
        selection: Sequence[
            tuple[KimiK3TextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
    ) -> list[KimiK3ImageInputs]:
        """Collects each selected image in the cache's requested order."""
        packed = []
        for context, images in selection:
            indices = {
                image.start_idx: index
                for index, image in enumerate(context.images)
            }
            offsets = np.concatenate(
                [[0], np.cumsum(np.prod(context.grid_thws, axis=1))]
            )
            for image in images:
                index = indices[image.start_idx]
                lo, hi = int(offsets[index]), int(offsets[index + 1])
                packed.append(
                    KimiK3ImageInputs(
                        pixels=image.pixel_values,
                        grid=context.grid_thws[index : index + 1],
                        positions=context.position_ids[lo:hi],
                        token_count=image.embedding_rows,
                    )
                )
        return packed

    def vision_execute(
        self,
        selection: Sequence[
            tuple[KimiK3TextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
        packed: list[KimiK3ImageInputs],
    ) -> VisionEncodeResult:
        """Encodes one image at a time, preserving per-image row ownership."""
        if not packed:
            return VisionEncodeResult(
                embeddings=self.empty_vision_embeddings(devices)
            )
        assert self._vision_model is not None
        assert self._session is not None
        assert isinstance(self.config, KimiK3Config)
        assert self.config.vision_config is not None
        pixel_dtype = self.config.vision_config.dtype
        outputs: list[Buffer] = []
        counts: list[int] = []
        for image in packed:
            patches = len(image.pixels)
            pixels = Buffer.from_numpy(image.pixels).to(devices[0])
            if pixels.dtype != pixel_dtype:
                pixels = cast_tensor_to(
                    pixels, pixel_dtype, session=self._session
                )
            # Raw argument order follows the registered named groups, not the
            # encoder's keyword argument declaration.
            (rows,) = self._vision_model.execute_raw(
                pixels,
                Buffer.from_numpy(image.positions).to(devices[0]),
                Buffer.from_numpy(image.grid).to(devices[0]),
                Buffer.from_numpy(np.array([0, patches], dtype=np.uint32)).to(
                    devices[0]
                ),
                Buffer.from_numpy(np.array([patches], dtype=np.uint32)),
            )
            assert isinstance(rows, Buffer)
            if rows.shape[0] != image.token_count:
                raise ValueError(
                    "Kimi K3 image span does not match its encoded rows"
                )
            outputs.append(rows)
            counts.append(image.token_count)
        embeddings = (
            outputs[0] if len(outputs) == 1 else concat_device_buffers(outputs)
        )
        return VisionEncodeResult(
            embeddings=[embeddings.to(device) for device in devices],
            per_image_token_counts=counts,
        )

    def empty_vision_embeddings(self, devices: list[Device]) -> list[Buffer]:
        """Returns stable empty rows for text-only and decode steps."""
        if self._empty_vision_embeddings is None:
            assert isinstance(self.config, KimiK3Config)
            self._empty_vision_embeddings = [
                Buffer.zeros(
                    shape=[0, self.config.model.hidden_size],
                    dtype=DType.bfloat16,
                ).to(device)
                for device in devices
            ]
        return self._empty_vision_embeddings
