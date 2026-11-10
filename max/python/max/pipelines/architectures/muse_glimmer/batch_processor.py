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
"""Shared decoder batching and image-cache hooks for Muse Glimmer roles."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from max.driver import Buffer, Device
from max.engine import InferenceSession
from max.pipelines.context import ImageMetadata, TextAndVisionContext
from max.pipelines.lib.module_registration import DecoderBatchProcessor
from max.pipelines.lib.vision_batching import create_empty_image_embeddings
from max.pipelines.lib.vision_encoder_cache import VisionEncodeResult
from max.tensor.compilation import CompiledCallable

from .batch_vision_inputs import pack_uncached_images
from .model_config import MuseGlimmerConfig


class MuseGlimmerBatchProcessor(DecoderBatchProcessor):
    """Encodes selected images while shared batching prepares decoder fields."""

    _vision_model: CompiledCallable[Any, Any] | None = None

    @property
    def _vision_config(self) -> MuseGlimmerConfig:
        """Returns the architecture config used for image packing."""
        assert isinstance(self.config, MuseGlimmerConfig)
        return self.config

    def bind_vision_encoder(
        self,
        *,
        vision_model: CompiledCallable[Any, Any],
        session: InferenceSession,
    ) -> None:
        """Binds the framework-compiled vision graph."""
        self._vision_model = vision_model

    def pack_vision_inputs(
        self,
        selection: Sequence[
            tuple[TextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
    ) -> list[Buffer] | None:
        """Packs the pipeline-selected uncached images to device."""
        if self._vision_config.vision_config is None:
            return None
        return pack_uncached_images(
            selection,
            devices[0],
            self._vision_config.vision_config,
            self._vision_config.dtype,
        )

    def vision_execute(
        self,
        selection: Sequence[
            tuple[TextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
        packed: list[Buffer] | None,
    ) -> VisionEncodeResult:
        """Runs the vision tower on the images :meth:`pack_vision_inputs` packed."""
        if packed is None:
            return VisionEncodeResult(
                embeddings=self.empty_vision_embeddings(devices)
            )
        assert self._vision_model is not None, (
            "This checkpoint has no vision tower; image inputs are not"
            " supported."
        )
        embeddings = self._vision_model(*packed).local_shards[0].driver_tensor
        return VisionEncodeResult(embeddings=embeddings.to(devices))

    def empty_vision_embeddings(self, devices: list[Device]) -> list[Buffer]:
        """Per-device zero-row image embeddings for text-only batches.

        Cached: this is hit on every step, so it must not allocate per call.
        """
        if not hasattr(self, "_cached_empty_embeddings"):
            self._cached_empty_embeddings = create_empty_image_embeddings(
                devices,
                self._vision_config.text_config.hidden_size,
                self._vision_config.dtype,
            )
        return self._cached_empty_embeddings
