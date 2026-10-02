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
"""Muse Glimmer ModuleV3 pipeline model."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, ClassVar

from max.driver import Buffer, Device
from max.engine import InferenceSession
from max.experimental import functional as F
from max.experimental.compilation import CompiledCallable
from max.experimental.sharding import DeviceMesh
from max.experimental.tensor import Tensor, default_dtype
from max.graph import DeviceRef
from max.graph.weights import Weights, WeightsAdapter
from max.nn.transformer import ReturnLogits
from max.pipelines.context import ImageMetadata, TextAndVisionContext
from max.pipelines.lib import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    ModuleV3MultiGraphPipelineModelWithKVCache,
    PipelineConfig,
)
from max.pipelines.lib.interfaces.batch_processor import (
    modulev3_gemma_multimodal_language_symbolic_inputs,
)
from max.pipelines.lib.log_probabilities import LogProbabilitiesMixin
from max.pipelines.lib.memory_estimation import MemoryPlan
from max.pipelines.lib.vision_batching import create_empty_image_embeddings
from max.pipelines.lib.vision_encoder_cache import VisionEncodeResult
from transformers import AutoConfig

from .batch_processor import MuseGlimmerBatchProcessor
from .batch_vision_inputs import pack_uncached_images
from .inputs import MuseGlimmerInputs
from .model_config import MuseGlimmerConfig
from .muse_glimmer import MuseGlimmer
from .vision import MuseGlimmerVisionModel, vision_input_types
from .weight_adapters import (
    convert_safetensor_language_state_dict,
    convert_safetensor_vision_state_dict,
)


def _to_buffer(t: Tensor) -> Buffer:
    """Extracts a Buffer from a potentially distributed Tensor."""
    return t.local_shards[0].driver_tensor


class MuseGlimmerModel(
    LogProbabilitiesMixin,
    ModuleV3MultiGraphPipelineModelWithKVCache[TextAndVisionContext],
):
    """The Muse Glimmer pipeline model (ModuleV3).

    The vision and language towers are compiled as two graphs. The
    pipeline's ``VisionEncoderCache`` drives the
    :class:`SupportsVisionEncoding` hooks and scatters the image embeddings
    into the ``<|patch|>`` positions.
    """

    model_config_cls: ClassVar[type[Any]] = MuseGlimmerConfig
    batch_processor_cls: ClassVar[type[MuseGlimmerBatchProcessor]] = (
        MuseGlimmerBatchProcessor
    )

    config: MuseGlimmerConfig

    language_model: CompiledCallable[Any, Any]
    """The compiled language tower."""

    vision_model: Callable[..., Any] | None
    """The compiled vision tower; ``None`` for a checkpoint without one."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        *,
        memory_plan: MemoryPlan,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
        max_batch_size: int = 1,
    ) -> None:
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter=adapter,
            return_logits=return_logits,
            max_batch_size=max_batch_size,
            memory_plan=memory_plan,
        )
        self.vision_model, language_model = self.load_model()
        assert isinstance(language_model, CompiledCallable)
        self.language_model = language_model

    @property
    def model(self) -> CompiledCallable[Any, Any]:
        """The language tower, for device graph capture and replay.

        Vision runs at prefill through :meth:`vision_execute`, so only the
        language tower is captured.
        """
        return self.language_model

    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        return huggingface_config.text_config.num_hidden_layers

    def _load_state_dict(self) -> dict[str, Any]:
        weights = dict(self.weights.items())
        self._language_weights_dict = convert_safetensor_language_state_dict(
            weights
        )
        self._vision_weights_dict = convert_safetensor_vision_state_dict(
            weights
        )
        return self._language_weights_dict

    def _create_model_config(
        self, state_dict: dict[str, Any]
    ) -> MuseGlimmerConfig:
        model_config = MuseGlimmerConfig.initialize_from_config(
            self.pipeline_config,
            self.huggingface_config,
            max_seq_len=self.max_seq_len,
        )
        model_config.finalize(
            huggingface_config=self.huggingface_config,
            state_dict=state_dict,
            return_logits=self.return_logits,
        )
        self.config = model_config
        return model_config

    def _compile_vision_model(  # type: ignore[override]
        self, model_config: MuseGlimmerConfig, state_dict: dict[str, Any]
    ) -> Callable[..., Any] | None:
        """Compiles the vision tower, or returns ``None`` when the checkpoint
        has none.

        The override widens the base hook's return type, hence the ignore.
        """
        vision_config = model_config.vision_config
        if vision_config is None:
            return None
        text_config = model_config.text_config
        with F.lazy(), default_dtype(model_config.dtype):
            vision_nn = MuseGlimmerVisionModel(
                vision_config,
                text_config.hidden_size,
                text_config.rms_norm_eps,
            )
            vision_nn.to(self.devices[0])
        input_types = vision_input_types(
            vision_config,
            DeviceRef.from_device(self.devices[0]),
            model_config.dtype,
        )
        return vision_nn.compile(*input_types, weights=state_dict)

    def _compile_language_model(
        self, model_config: MuseGlimmerConfig, state_dict: dict[str, Any]
    ) -> Callable[..., Any]:
        mesh = DeviceMesh(tuple(self.devices), (len(self.devices),), ("tp",))
        with F.lazy(), default_dtype(model_config.dtype):
            language_nn = MuseGlimmer(model_config, mesh)
            language_nn.to(mesh)

        input_types = modulev3_gemma_multimodal_language_symbolic_inputs(
            kv_params=self.kv_params,
            device_ref=DeviceRef.from_device(self.devices[0]),
            hidden_size=model_config.text_config.hidden_size,
            # Must match empty_vision_embeddings().
            embedding_dtype=model_config.dtype,
        )
        return language_nn.compile(*input_types, weights=state_dict)

    # --- SupportsVisionEncoding ---

    def pack_vision_inputs(
        self,
        selection: Sequence[
            tuple[TextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
    ) -> list[Buffer] | None:
        """Packs the pipeline-selected uncached images to device."""
        if self.config.vision_config is None:
            return None
        return pack_uncached_images(
            selection, devices[0], self.config.vision_config, self.config.dtype
        )

    def vision_execute(
        self,
        selection: Sequence[
            tuple[TextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
        packed: list[Buffer] | None,
    ) -> VisionEncodeResult:
        """Runs the vision tower on the images :meth:`pack_vision_inputs`
        packed."""
        if packed is None:
            return VisionEncodeResult(
                embeddings=self.empty_vision_embeddings(devices)
            )
        assert self.vision_model is not None, (
            "This checkpoint has no vision tower; image inputs are not"
            " supported."
        )
        embeddings = _to_buffer(self.vision_model(*packed))
        return VisionEncodeResult(embeddings=embeddings.to(devices))

    def empty_vision_embeddings(self, devices: list[Device]) -> list[Buffer]:
        """Per-device zero-row image embeddings for text-only batches.

        Cached: this is hit on every step, so it must not allocate per call.
        """
        if not hasattr(self, "_cached_empty_embeddings"):
            self._cached_empty_embeddings = create_empty_image_embeddings(
                devices,
                self.huggingface_config.text_config.hidden_size,
                self.config.dtype,
            )
        return self._cached_empty_embeddings

    # --- execute ---

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Executes the language tower with the prepared inputs."""
        assert isinstance(model_inputs, MuseGlimmerInputs)
        assert model_inputs.kv_cache_inputs is not None
        assert len(model_inputs.vision_embeddings) == len(self.devices)
        assert len(model_inputs.vision_scatter_indices) == len(self.devices)

        # Replay feeds the same buffers, so eager and captured share one ABI.
        model_outputs = self.language_model.execute_raw(*model_inputs.buffers)
        if len(model_outputs) == 3:
            return ModelOutputs(
                logits=model_outputs[1],
                next_token_logits=model_outputs[0],
                logit_offsets=model_outputs[2],
            )
        return ModelOutputs(
            logits=model_outputs[0],
            next_token_logits=model_outputs[0],
        )
