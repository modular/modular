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
"""Text batching and vision encoding for registered Idefics3 modules."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph.buffer_utils import cast_dlpack_to
from max.pipelines.context import ImageMetadata, TextAndVisionContext
from max.pipelines.lib.module_registration import DecoderBatchProcessor
from max.pipelines.lib.vision_batching import (
    VisionStacker,
    create_empty_image_embeddings_single,
)
from max.pipelines.lib.vision_encoder_cache import VisionEncodeResult
from max.tensor.compilation import CompiledCallable

from .model_config import Idefics3Config


class Idefics3ModuleV3BatchProcessor(DecoderBatchProcessor):
    """Encodes cache-selected images; shared batching prepares decoder inputs."""

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
    ) -> Buffer:
        """Stacks the selected images into the vision graph's pixel batch."""
        images = [
            image.pixel_values
            for _, selected in selection
            for image in selected
        ]
        return cast_dlpack_to(
            VisionStacker().stack(images),
            DType.float32,
            DType.bfloat16,
            devices[0],
        )

    def vision_execute(
        self,
        selection: Sequence[
            tuple[TextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
        packed: Buffer,
    ) -> VisionEncodeResult:
        """Encodes images and checks that their token spans match the output."""
        assert self._vision_model is not None
        output = self._vision_model(packed).driver_tensor
        assert isinstance(output, Buffer)
        counts = [
            image.embedding_rows
            for _, selected in selection
            for image in selected
        ]
        if output.shape[0] != sum(counts):
            raise ValueError(
                "Idefics3 vision embedding count does not match image tokens"
            )
        return VisionEncodeResult(
            embeddings=[output.to(device) for device in devices],
            per_image_token_counts=counts,
        )

    def empty_vision_embeddings(self, devices: list[Device]) -> list[Buffer]:
        """Returns the zero-row embeddings bound for text-only steps."""
        assert isinstance(self.config, Idefics3Config)
        if self._empty_embeddings is None:
            self._empty_embeddings = [
                create_empty_image_embeddings_single(
                    device, self.config.text_config.hidden_size
                )
                for device in devices
            ]
        return self._empty_embeddings
