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

from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, DeviceRef, TensorType
from max.nn.legacy.kv_cache import KVCacheInputs
from max.nn.legacy.kv_cache.cache_params import KVCacheParamInterface
from max.pipelines.architectures.gemma4.batch_vision_inputs import (
    VisionRawInputs,
    create_empty_embeddings,
    pack_uncached_images,
)
from max.pipelines.architectures.gemma4.context import Gemma4Context
from max.pipelines.architectures.gemma4.model_config import (
    Gemma4ForConditionalGenerationConfig,
)
from max.pipelines.context import ImageMetadata
from max.pipelines.lib.interfaces.arch_config import ArchConfig
from max.pipelines.lib.interfaces.batch_processor import (
    BatchProcessorRuntime,
    ModuleV3SingleReplicaBatchProcessor,
    modulev3_gemma_multimodal_language_symbolic_inputs,
)
from max.pipelines.lib.vision_encoder_cache import VisionEncodeResult

from .inputs import Gemma4Inputs


class Gemma4ModuleV3BatchProcessor(
    ModuleV3SingleReplicaBatchProcessor[Gemma4Context, Gemma4Inputs]
):
    """Ragged batching for Gemma4 ModuleV3 models (single GPU, no signals).

    Images go through the pipeline-owned ``VisionEncoderCache``, which this
    processor encodes for through the compiled vision tower it is handed; the
    embeddings and scatter indices land on the base :class:`ModelInputs`
    vision fields.
    """

    _config: Gemma4ForConditionalGenerationConfig | None = None
    _vision_model: Model | None = None
    _empty_embeddings: list[Buffer] | None = None

    def __init__(
        self,
        config: ArchConfig,
        runtime: BatchProcessorRuntime,
    ) -> None:
        super().__init__(config, runtime)
        if isinstance(config, Gemma4ForConditionalGenerationConfig):
            self._config = config

    def bind_model_state(
        self,
        *,
        config: Gemma4ForConditionalGenerationConfig,
    ) -> None:
        """Wires the model config, which finalizing fixed the dtypes of.

        Args:
            config: Fully-initialised Gemma4 model configuration.
        """
        self._config = config

    def bind_vision_encoder(
        self, *, vision_model: Model, session: InferenceSession
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
        assert self._config is not None
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
        device0_embeddings = outputs[0]
        assert isinstance(device0_embeddings, Buffer)
        return VisionEncodeResult(
            embeddings=[device0_embeddings.to(device) for device in devices]
        )

    def empty_vision_embeddings(self, devices: list[Device]) -> list[Buffer]:
        """Per-device zero-row image embeddings for cached / text-only batches.

        Cached: this is hit on every text-only / decode step, so it must not
        allocate per call.
        """
        if self._empty_embeddings is None:
            assert self._config is not None
            self._empty_embeddings = create_empty_embeddings(
                devices,
                self._config.text_config.hidden_size,
                self._config.unquantized_dtype,
            )
        return self._empty_embeddings

    def get_language_symbolic_inputs(
        self,
        *,
        kv_params: KVCacheParamInterface,
        device_ref: DeviceRef,
        hidden_size: int,
        embedding_dtype: DType,
    ) -> list[TensorType | BufferType]:
        """Symbolic inputs for the ModuleV3 language-model ``compile()`` call.

        ``embedding_dtype`` must match the vision tower's output dtype:
        the runtime image-embedding buffers come
        either from the tower or from ``empty_vision_embeddings``, both of
        which use ``config.unquantized_dtype``.
        """
        return modulev3_gemma_multimodal_language_symbolic_inputs(
            kv_params=kv_params,
            device_ref=device_ref,
            hidden_size=hidden_size,
            embedding_dtype=embedding_dtype,
        )

    def _make_inputs(
        self,
        *,
        tokens: Buffer,
        input_row_offsets: Buffer,
        return_n_logits: Buffer,
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer],
    ) -> Gemma4Inputs:
        return Gemma4Inputs(
            tokens=tokens,
            input_row_offsets=input_row_offsets,
            return_n_logits=return_n_logits,
            kv_cache_inputs=kv_cache_inputs,
        )
