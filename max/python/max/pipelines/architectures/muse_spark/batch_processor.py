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
"""Muse Spark batching and image-cache hooks."""

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
from .model_config import MuseSparkConfig
from .muse_spark import compute_dtype


class MuseSparkBatchProcessor(DecoderBatchProcessor):
    """Encodes selected images while shared batching prepares decoder fields."""

    _vision_model: CompiledCallable[Any, Any] | None = None
    _cached_empty_embeddings: list[Buffer] | None = None

    @property
    def _spark_config(self) -> MuseSparkConfig:
        assert isinstance(self.config, MuseSparkConfig)
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
        config = self._spark_config
        return pack_uncached_images(
            selection, devices[0], config.vision_config, compute_dtype(config)
        )

    def vision_execute(
        self,
        selection: Sequence[
            tuple[TextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
        packed: list[Buffer] | None,
    ) -> VisionEncodeResult:
        """Runs the vision tower on :meth:`pack_vision_inputs`' images."""
        if packed is None:
            return VisionEncodeResult(
                embeddings=self.empty_vision_embeddings(devices)
            )
        assert self._vision_model is not None, (
            "the vision graph was not bound before encoding images"
        )
        embeddings = self._vision_model(*packed).local_shards[0].driver_tensor
        return VisionEncodeResult(embeddings=embeddings.to(devices))

    def empty_vision_embeddings(self, devices: list[Device]) -> list[Buffer]:
        """Per-device zero-row image embeddings for text-only batches.

        Cached: this is hit on every step, so it must not allocate per call.
        """
        if self._cached_empty_embeddings is None:
            config = self._spark_config
            self._cached_empty_embeddings = create_empty_image_embeddings(
                devices,
                config.text_config.hidden_size,
                compute_dtype(config),
            )
        return self._cached_empty_embeddings
