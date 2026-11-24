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
"""Batching for the registered Inkling 2 decoder and its vision tower."""

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
from max.pipelines.lib.vision_batching import create_empty_image_embeddings
from max.pipelines.lib.vision_encoder_cache import VisionEncodeResult
from max.tensor.compilation import CompiledCallable

from .batch_vision_inputs import pack_uncached_images
from .model_config import Inkling2Config


class _Inkling2Inputs(DecoderModelInputs):
    """Hands the decoder the image inputs of device zero only."""

    def input_fields(self) -> DecoderInputs[tree.Tree[Buffer], Buffer]:
        """Selects the image tensors the decoder broadcasts itself."""
        return replace(
            super().input_fields(),
            vision=VisionInputs(
                embeddings=tuple(self.vision_embeddings[:1]),
                scatter_indices=tuple(self.vision_scatter_indices[:1]),
            ),
        )


class Inkling2BatchProcessor(DecoderBatchProcessor):
    """Batches ragged text and encodes the images the vision cache misses."""

    _vision_model: CompiledCallable[Any, Any] | None = None
    _empty_embeddings: list[Buffer] | None = None

    @property
    def _config(self) -> Inkling2Config:
        assert isinstance(self.config, Inkling2Config)
        return self.config

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> DecoderModelInputs:
        """Uses the shared batcher and keeps the image inputs on device zero."""
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
        """Binds the framework-compiled vision graph."""
        self._vision_model = vision_model

    def pack_vision_inputs(
        self,
        selection: Sequence[
            tuple[TextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
    ) -> list[Buffer] | None:
        """Packs the pipeline-selected uncached images to device zero."""
        return pack_uncached_images(
            selection, devices[0], self._config.vision_config
        )

    def vision_execute(
        self,
        selection: Sequence[
            tuple[TextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
        packed: list[Buffer] | None,
    ) -> VisionEncodeResult:
        """Runs the tower on the images :meth:`pack_vision_inputs` packed."""
        if packed is None:
            return VisionEncodeResult(
                embeddings=self.empty_vision_embeddings(devices)
            )
        assert self._vision_model is not None
        (output,) = self._vision_model.execute_raw(*packed)
        return VisionEncodeResult(
            embeddings=[output.to(device) for device in devices]
        )

    def empty_vision_embeddings(self, devices: list[Device]) -> list[Buffer]:
        """Per-device zero-row image embeddings, cached for text-only steps."""
        if self._empty_embeddings is None:
            self._empty_embeddings = create_empty_image_embeddings(
                devices,
                self._config.vision_config.text_hidden_size,
                self._config.dtype,
            )
        return self._empty_embeddings
