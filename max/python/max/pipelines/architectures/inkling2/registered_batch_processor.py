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
"""The registered batcher with Inkling2's image placement."""

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

from .batch_processor import (
    concat_rows,
    prepare_tower_inputs,
    tower_rows,
)
from .model_config import Inkling2Config

_Selection = Sequence[tuple[TextAndVisionContext, Sequence[ImageMetadata]]]
"""The ``(context, cache-miss images)`` pairs the vision cache selects."""


class _Inkling2Inputs(DecoderModelInputs):
    def input_fields(self) -> DecoderInputs[tree.Tree[Buffer], Buffer]:
        """Selects the image tensors the decoder replicates itself."""
        return replace(
            super().input_fields(),
            vision=VisionInputs(
                embeddings=tuple(self.vision_embeddings[:1]),
                scatter_indices=tuple(self.vision_scatter_indices[:1]),
            ),
        )


class Inkling2BatchProcessor(DecoderBatchProcessor):
    """Uses shared text/position batching and encodes only cache-selected images."""

    _vision_callable: CompiledCallable[Any, Any] | None = None
    _empty_embeddings: list[Buffer] | None = None

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> DecoderModelInputs:
        """Uses the shared batcher and keeps Inkling2's image input placement."""
        inputs = super().prepare_initial_token_inputs(
            replica_batches, kv_cache_inputs, return_n_logits
        )
        return _Inkling2Inputs(
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
        del session
        self._vision_callable = vision_model

    def pack_vision_inputs(
        self, selection: _Selection, devices: list[Device]
    ) -> list[tuple[Buffer, Buffer, Buffer, Buffer]]:
        """Prepares each selected image's four tower operands on device zero."""
        assert isinstance(self.config, Inkling2Config)
        device = devices[0]
        return [
            prepare_tower_inputs(image, self.config.vision_config, device)
            for _, images in selection
            for image in images
        ]

    def vision_execute(
        self,
        selection: _Selection,
        devices: list[Device],
        packed: list[tuple[Buffer, Buffer, Buffer, Buffer]] | None,
    ) -> VisionEncodeResult:
        """Encodes the selected images, one tower call each."""
        assert self._vision_callable is not None
        if packed is None:
            packed = self.pack_vision_inputs(selection, devices)
        rows_per_image = [
            getattr(image, "embedding_rows", None)
            for _, images in selection
            for image in images
        ]
        counts = (
            None
            if None in rows_per_image
            else [int(count) for count in rows_per_image if count is not None]
        )
        rows = [
            tower_rows(self._vision_callable(*operands)) for operands in packed
        ]
        output = concat_rows(rows, devices[0])
        if counts is not None and int(output.shape[0]) != sum(counts):
            raise ValueError(
                f"the vision tower emitted {int(output.shape[0])} rows but "
                f"the batch has {sum(counts)} image-token positions to fill; "
                "each image's rows must equal its placeholder run "
                "(ImageMetadata.embedding_rows)"
            )
        return VisionEncodeResult(
            embeddings=[output.to(device) for device in devices],
            per_image_token_counts=counts,
        )

    def empty_vision_embeddings(self, devices: list[Device]) -> list[Buffer]:
        """Caches zero-row embeddings for decode and text-only batches."""
        assert isinstance(self.config, Inkling2Config)
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
