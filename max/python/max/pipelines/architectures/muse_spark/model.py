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
"""Muse Spark ModuleV3 pipeline model."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, ClassVar

from max.driver import Buffer, Device
from max.engine import InferenceSession
from max.experimental import functional as F
from max.experimental.compilation import CompiledCallable
from max.experimental.sharding import DeviceMesh
from max.experimental.tensor import default_dtype
from max.graph import DeviceRef
from max.graph.weights import Weights, WeightsAdapter
from max.nn.legacy.transformer import ReturnLogits
from max.pipelines.architectures.muse_glimmer.batch_processor import (
    MuseGlimmerBatchProcessor,
)
from max.pipelines.architectures.muse_glimmer.inputs import MuseGlimmerInputs
from max.pipelines.context import ImageMetadata, TextAndVisionContext
from max.pipelines.lib import (
    GraphSpec,
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

from .model_config import MuseSparkConfig, reconcile_expert_dtypes
from .muse_spark import MuseSpark, compute_dtype
from .weight_adapters import convert_safetensor_language_state_dict

_EXPERT_0_GATE = re.compile(
    r"language_model\.layers\.(\d+)\.mlp\.experts\.0\.gate_proj\.weight(\.data)?"
)


class MuseSparkModel(
    LogProbabilitiesMixin,
    ModuleV3MultiGraphPipelineModelWithKVCache[TextAndVisionContext],
):
    """The Muse Spark pipeline model (ModuleV3).

    Only the language tower is compiled; the vision hooks feed it empty image
    embeddings and reject image inputs until the vision tower exists.
    """

    model_config_cls: ClassVar[type[Any]] = MuseSparkConfig
    batch_processor_cls: ClassVar[type[MuseGlimmerBatchProcessor]] = (
        MuseGlimmerBatchProcessor
    )

    config: MuseSparkConfig

    language_model: CompiledCallable[Any, Any]
    """The compiled language tower."""

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
        _, language_model = self._load_towers(session)
        assert isinstance(language_model, CompiledCallable)
        self.language_model = language_model

    @property
    def model(self) -> CompiledCallable[Any, Any]:
        """The language tower, for device graph capture and replay."""
        return self.language_model

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
    ) -> MuseSparkConfig:
        model_config = MuseSparkConfig.initialize_from_config(
            self.pipeline_config,
            self.huggingface_config,
            max_seq_len=self.max_seq_len,
        )
        model_config.finalize(
            huggingface_config=self.huggingface_config,
            state_dict=state_dict,
            return_logits=self.return_logits,
        )
        text_config = model_config.text_config
        if text_config.nvfp4 is not None:
            expert_dtypes = {
                int(match[1]): data.dtype
                for name, data in state_dict.items()
                if (match := _EXPERT_0_GATE.fullmatch(name))
            }
            text_config.nvfp4 = replace(
                text_config.nvfp4,
                quant_config=reconcile_expert_dtypes(
                    text_config.nvfp4.quant_config, expert_dtypes
                ),
            )
        self.config = model_config
        return model_config

    def _vision_spec(
        self, model_config: MuseSparkConfig, state_dict: dict[str, Any]
    ) -> GraphSpec | None:
        """Returns ``None``: there is no vision tower yet."""
        return None

    def _language_spec(
        self, model_config: MuseSparkConfig, state_dict: dict[str, Any]
    ) -> GraphSpec:
        mesh = DeviceMesh(tuple(self.devices), (len(self.devices),), ("tp",))
        dtype = compute_dtype(model_config)
        with F.lazy(), default_dtype(dtype):
            language_nn = MuseSpark(model_config, mesh)
            language_nn.to(mesh)

        input_types = modulev3_gemma_multimodal_language_symbolic_inputs(
            kv_params=self.kv_params,
            device_ref=DeviceRef.from_device(self.devices[0]),
            hidden_size=model_config.text_config.hidden_size,
            # Must match empty_vision_embeddings().
            embedding_dtype=dtype,
        )
        return GraphSpec(
            language_nn,
            type(language_nn).__qualname__,
            input_types,
            weights=state_dict,
        )

    # --- SupportsVisionEncoding ---

    def pack_vision_inputs(
        self,
        selection: Sequence[
            tuple[TextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
    ) -> list[Buffer] | None:
        """Rejects image inputs: the vision tower is not built yet."""
        if any(images for _, images in selection):
            raise ValueError("Muse Spark does not support image inputs yet.")
        return None

    def vision_execute(
        self,
        selection: Sequence[
            tuple[TextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
        packed: list[Buffer] | None,
    ) -> VisionEncodeResult:
        """Returns empty image embeddings; nothing is ever packed."""
        return VisionEncodeResult(
            embeddings=self.empty_vision_embeddings(devices)
        )

    def empty_vision_embeddings(self, devices: list[Device]) -> list[Buffer]:
        """Per-device zero-row image embeddings for text-only batches.

        Cached: this is hit on every step, so it must not allocate per call.
        """
        if not hasattr(self, "_cached_empty_embeddings"):
            self._cached_empty_embeddings = create_empty_image_embeddings(
                devices,
                self.huggingface_config.text_config.hidden_size,
                compute_dtype(self.config),
            )
        return self._cached_empty_embeddings

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Executes the language tower with the prepared inputs."""
        assert isinstance(model_inputs, MuseGlimmerInputs)
        assert model_inputs.kv_cache_inputs is not None

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
