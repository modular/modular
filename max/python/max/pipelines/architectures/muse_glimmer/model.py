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
"""Muse Glimmer ModuleV3 pipeline model (text only; vision comes later)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, ClassVar

from max import tree
from max.driver import Buffer, Device
from max.engine import InferenceSession
from max.experimental import functional as F
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
from .inputs import MuseGlimmerInputs
from .model_config import MuseGlimmerConfig
from .muse_glimmer import MuseGlimmer
from .weight_adapters import convert_safetensor_language_state_dict


def _to_buffer(t: Tensor) -> Buffer:
    """Extracts a Buffer from a potentially distributed Tensor."""
    return t.local_shards[0].driver_tensor


class MuseGlimmerModel(
    LogProbabilitiesMixin,
    ModuleV3MultiGraphPipelineModelWithKVCache[TextAndVisionContext],
):
    """The Muse Glimmer pipeline model (ModuleV3).

    Only the language tower is compiled. The vision hooks of
    :class:`SupportsVisionEncoding` are implemented so the pipeline's
    ``VisionEncoderCache`` supplies the zero-row image embeddings the
    language graph takes on every step.
    """

    model_config_cls: ClassVar[type[Any]] = MuseGlimmerConfig
    batch_processor_cls: ClassVar[type[MuseGlimmerBatchProcessor]] = (
        MuseGlimmerBatchProcessor
    )

    config: MuseGlimmerConfig

    language_model: Callable[..., Any]
    """The compiled language tower."""

    vision_model: Callable[..., Any] | None
    """``None`` until the vision tower lands."""

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
        self.vision_model, self.language_model = self.load_model()

    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        return huggingface_config.text_config.num_hidden_layers

    def _load_state_dict(self) -> dict[str, Any]:
        self._language_weights_dict = convert_safetensor_language_state_dict(
            dict(self.weights.items())
        )
        self._vision_weights_dict = {}
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
    ) -> None:
        """Returns ``None``: the vision tower is not implemented yet.

        The override narrows the base hook's return type, hence the ignore.
        """
        return

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
    ) -> None:
        """Returns ``None``: there are no pixels to pack without a vision tower."""
        return

    def vision_execute(
        self,
        selection: Sequence[
            tuple[TextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
        packed: None,
    ) -> VisionEncodeResult:
        """Returns zero-row embeddings; image inputs are not supported yet."""
        if any(images for _, images in selection):
            raise NotImplementedError(
                "Muse Glimmer is served text-only; image inputs are not"
                " supported yet."
            )
        return VisionEncodeResult(
            embeddings=self.empty_vision_embeddings(self.devices)
        )

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
        kv_cache_inputs = model_inputs.kv_cache_inputs
        assert kv_cache_inputs is not None
        assert len(model_inputs.vision_embeddings) == len(self.devices)
        assert len(model_inputs.vision_scatter_indices) == len(self.devices)

        model_outputs = self.language_model(
            model_inputs.tokens,
            model_inputs.return_n_logits,
            model_inputs.input_row_offsets,
            model_inputs.vision_embeddings[0],
            model_inputs.vision_scatter_indices[0],
            *tree.leaves(kv_cache_inputs),
        )
        if len(model_outputs) == 3:
            return ModelOutputs(
                logits=_to_buffer(model_outputs[1]),
                next_token_logits=_to_buffer(model_outputs[0]),
                logit_offsets=_to_buffer(model_outputs[2]),
            )
        return ModelOutputs(
            logits=_to_buffer(model_outputs[0]),
            next_token_logits=_to_buffer(model_outputs[0]),
        )
