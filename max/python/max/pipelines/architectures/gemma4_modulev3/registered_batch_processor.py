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
"""Input batching for Gemma4 ModuleV3 pipeline models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import fields, replace
from typing import Any

from max import tree
from max.driver import Buffer, Device
from max.engine import InferenceSession
from max.nn.kv_cache import KVCacheInputs
from max.pipelines.architectures.gemma4.batch_vision_inputs import (
    VisionRawInputs,
    create_empty_embeddings,
    pack_uncached_images,
)
from max.pipelines.architectures.gemma4.context import Gemma4Context
from max.pipelines.architectures.gemma4.model_config import (
    Gemma4ForConditionalGenerationConfig,
)
from max.pipelines.context import ImageMetadata, TextContext
from max.pipelines.lib.module_registration import (
    DecoderBatchProcessor,
    DecoderModelInputs,
)
from max.pipelines.lib.role_contracts import DecoderInputs, VisionInputs
from max.pipelines.lib.vision_encoder_cache import VisionEncodeResult
from max.tensor.compilation import CompiledCallable


class _Gemma4Inputs(DecoderModelInputs):
    """Packs only the vision operands the language tower broadcasts."""

    def input_fields(self) -> DecoderInputs[tree.Tree[Buffer], Buffer]:
        """Keeps late vision-cache attachments on their original device."""
        return replace(
            super().input_fields(),
            vision=VisionInputs(
                embeddings=tuple(self.vision_embeddings[:1]),
                scatter_indices=tuple(self.vision_scatter_indices[:1]),
            ),
        )


class Gemma4ModuleV3BatchProcessor(DecoderBatchProcessor):
    """Ragged batching for Gemma4 ModuleV3 models (single GPU, no signals).

    Images go through the pipeline-owned ``VisionEncoderCache``, which this
    processor encodes for through the compiled vision tower it is handed; the
    embeddings and scatter indices land on the base :class:`ModelInputs`
    vision fields.
    """

    _vision_model: CompiledCallable[Any, Any] | None = None
    _empty_embeddings: list[Buffer] | None = None

    @property
    def _config(self) -> Gemma4ForConditionalGenerationConfig:
        """Reads the framework-bound finalized config rather than a cached copy."""
        assert isinstance(self.config, Gemma4ForConditionalGenerationConfig)
        return self.config

    def bind_vision_encoder(
        self,
        *,
        vision_model: CompiledCallable[Any, Any],
        session: InferenceSession,
    ) -> None:
        """Wires the compiled vision encoder."""
        del session
        self._vision_model = vision_model

    def pack_vision_inputs(
        self,
        selection: Sequence[tuple[Gemma4Context, Sequence[ImageMetadata]]],
        devices: list[Device],
    ) -> VisionRawInputs | None:
        """Packs the pipeline-selected uncached image pixels to device."""
        assert self._config.vision_config is not None
        return pack_uncached_images(
            selection,
            devices,
            self._config.vision_config.pooling_kernel_size,
            self._config.unquantized_dtype,
        )

    def vision_execute(
        self,
        selection: Sequence[tuple[Gemma4Context, Sequence[ImageMetadata]]],
        devices: list[Device],
        packed: VisionRawInputs | None,
    ) -> VisionEncodeResult:
        """Runs the vision encoder on the packed pixels."""
        if packed is None:
            return VisionEncodeResult(
                embeddings=self.empty_vision_embeddings(devices)
            )
        if self._vision_model is None:
            raise ValueError(
                "This checkpoint is served text-only (no vision encoder"
                " is loaded); image and video inputs are not supported."
            )
        # The tower runs on device 0, and its embeddings are copied to the
        # other devices.
        outputs = self._vision_model(
            packed.patches_flat[0],
            packed.pixel_position_ids[0],
            packed.cu_seqlens[0],
            packed.pool_gather_index[0],
            packed.max_seq_len,
        )
        device0_embeddings = outputs.local_shards[0].driver_tensor
        return VisionEncodeResult(
            embeddings=[device0_embeddings.to(device) for device in devices]
        )

    def empty_vision_embeddings(self, devices: list[Device]) -> list[Buffer]:
        """Per-device zero-row image embeddings for cached / text-only batches.

        Cached: this is hit on every text-only / decode step, so it must not
        allocate per call.
        """
        if self._empty_embeddings is None:
            self._empty_embeddings = create_empty_embeddings(
                devices,
                self._config.text_config.hidden_size,
                self._config.unquantized_dtype,
            )
        return self._empty_embeddings

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> DecoderModelInputs:
        """Uses shared text batching and preserves device-zero image operands."""
        inputs = super().prepare_initial_token_inputs(
            replica_batches, kv_cache_inputs, return_n_logits
        )
        return _Gemma4Inputs(
            **{
                field.name: getattr(inputs, field.name)
                for field in fields(inputs)
            }
        )
