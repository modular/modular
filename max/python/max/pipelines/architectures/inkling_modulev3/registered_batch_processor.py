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
"""Shared decoder batching and cache-selected vision encoding for Inkling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import fields, replace
from typing import Any

from max import tree
from max.driver import Buffer, Device
from max.engine import InferenceSession
from max.nn.kv_cache import KVCacheInputs
from max.pipelines.context import (
    ImageMetadata,
    TextAndVisionContext,
    TextContext,
)
from max.pipelines.lib.module_registration import (
    DecoderBatchProcessor,
    DecoderModelInputs,
)
from max.pipelines.lib.role_contracts import DecoderInputs, VisionInputs
from max.pipelines.lib.vision_batching import (
    create_empty_image_embeddings_single,
)
from max.pipelines.lib.vision_encoder_cache import VisionEncodeResult
from max.tensor.compilation import CompiledCallable

from ..inkling.batch_processor import _pinned_copy
from .model_config import InklingConfig


class _InklingInputs(DecoderModelInputs):
    """Keeps late cache attachments while packing device-zero vision inputs."""

    def input_fields(self) -> DecoderInputs[tree.Tree[Buffer], Buffer]:
        """Selects the image tensors broadcast internally by the decoder."""
        return replace(
            super().input_fields(),
            vision=VisionInputs(
                embeddings=tuple(self.vision_embeddings[:1]),
                scatter_indices=tuple(self.vision_scatter_indices[:1]),
            ),
        )


class InklingModuleV3BatchProcessor(DecoderBatchProcessor):
    """Uses shared text/position batching and encodes only cache-selected images."""

    _vision_callable: CompiledCallable[Any, Any] | None = None
    _empty_embeddings: list[Buffer] | None = None

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> DecoderModelInputs:
        """Uses the shared batcher and retains Inkling's image input placement."""
        inputs = super().prepare_initial_token_inputs(
            replica_batches, kv_cache_inputs, return_n_logits
        )
        return _InklingInputs(
            **{
                field.name: getattr(inputs, field.name)
                for field in fields(inputs)
            }
        )

    def bind_vision_encoder(
        self,
        *,
        vision_model: CompiledCallable[Any, Any],
        session: InferenceSession,
    ) -> None:
        """Binds the framework-compiled image encoder."""
        self._vision_callable = vision_model

    def pack_vision_inputs(
        self,
        selection: Sequence[
            tuple[TextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
    ) -> Buffer:
        """Packs the selected images' raw patch blocks on device zero."""
        return _pinned_copy(
            [image.pixel_values for _, images in selection for image in images],
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
        """Encodes image rows for the framework's vision cache."""
        assert self._vision_callable is not None
        output = self._vision_callable(packed).local_shards[0].driver_tensor
        counts = [
            image.embedding_rows for _, images in selection for image in images
        ]
        if output.shape[0] != sum(counts):
            raise ValueError(
                "Inkling vision rows do not match selected image tokens"
            )
        return VisionEncodeResult(
            embeddings=[output.to(device) for device in devices],
            per_image_token_counts=counts,
        )

    def empty_vision_embeddings(self, devices: list[Device]) -> list[Buffer]:
        """Caches zero-row embeddings for decode and text-only batches."""
        assert isinstance(self.config, InklingConfig)
        if self._empty_embeddings is None:
            self._empty_embeddings = [
                create_empty_image_embeddings_single(
                    device,
                    self.config.text_config.hidden_size,
                    self.config.dtype,
                )
                for device in devices
            ]
        return self._empty_embeddings
