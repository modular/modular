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
"""Inkling pipeline model on ModuleV3: weight split, compile, execution."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, ClassVar

from max.driver import Device
from max.engine import InferenceSession
from max.experimental import functional as F
from max.experimental.nn.common_layers.mesh_axis import TP
from max.experimental.sharding import DeviceMesh
from max.experimental.tensor import default_device, default_dtype
from max.graph.weights import Weights, WeightsAdapter
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.context import TextAndVisionContext
from max.pipelines.lib import (
    GraphSpec,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    ModuleV3MultiGraphPipelineModelWithKVCache,
    ModuleV3Outputs,
    PipelineConfig,
    SupportsSSMStateWarmup,
)
from max.pipelines.lib.log_probabilities import LogProbabilitiesMixin
from max.pipelines.lib.memory_estimation import MemoryPlan
from typing_extensions import override

from ..inkling.batch_processor import InklingInputs
from ..inkling.model_config import InklingConfig
from ..inkling.processor import load_processor_config
from ..inkling.weight_adapters import VISION_PREFIX
from .batch_processor import InklingModuleV3BatchProcessor
from .inkling import Inkling
from .layers.vision import InklingVisionModel

# The checkpoint hangs each stacked expert tensor's scales off the tensor's
# own name; a module attribute cannot be both a tensor and a parent.
_EXPERT_SCALE = re.compile(
    r"(\.experts\.w(?:13|2))_weight\.(weight_scale(?:_2)?|input_scale)$"
)


class InklingModel(
    LogProbabilitiesMixin,
    ModuleV3MultiGraphPipelineModelWithKVCache[
        TextAndVisionContext, InklingConfig
    ],
    SupportsSSMStateWarmup,
):
    """Pipeline model for Inkling's decoder and its vision tower."""

    batch_processor_cls: ClassVar[type[InklingModuleV3BatchProcessor]] = (
        InklingModuleV3BatchProcessor
    )
    model_config_cls: ClassVar[type[Any]] = InklingConfig

    vision_model: Callable[..., Any]
    model: Callable[..., ModuleV3Outputs]

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
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
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
            return_hidden_states=return_hidden_states,
            max_batch_size=max_batch_size,
            memory_plan=memory_plan,
        )
        vision_model, self.model = self._load_towers(session)
        assert vision_model is not None
        self.vision_model = vision_model
        assert isinstance(self._batch_processor, InklingModuleV3BatchProcessor)
        self._batch_processor.bind_vision_callable(vision_model)

    @override
    def _load_state_dict(self) -> dict[str, Any]:
        """Splits the checkpoint between the two modules the towers compile
        from."""
        state_dict = super()._load_state_dict()
        self._vision_weights_dict = {
            name.removeprefix(VISION_PREFIX): data
            for name, data in state_dict.items()
            if name.startswith(VISION_PREFIX)
        }
        self._language_weights_dict = {
            _EXPERT_SCALE.sub(r"\1_\2", name): data
            for name, data in state_dict.items()
            if not name.startswith(VISION_PREFIX)
        }
        return state_dict

    @override
    def _create_model_config(self, state_dict: dict[str, Any]) -> InklingConfig:
        # Quantization is read off the loaded weights; the checkpoint config
        # may not declare it.
        model_config = InklingConfig.initialize(
            self.pipeline_config, max_seq_len=self.max_seq_len
        )
        model_config.finalize(self.huggingface_config, state_dict)
        return model_config

    @override
    def _vision_spec(
        self, model_config: InklingConfig, state_dict: dict[str, Any]
    ) -> GraphSpec:
        image_processor = load_processor_config(
            self.pipeline_config.model.model_path,
            self.pipeline_config.model.huggingface_model_revision,
        )["image_processor"]
        # The tower is small and runs once per image, so it stays on one device.
        with (
            F.lazy(),
            default_dtype(model_config.dtype),
            default_device(self.devices[0]),
        ):
            vision_model = InklingVisionModel(
                model_config.vision_config,
                model_config.dtype,
                image_mean=image_processor["image_mean"],
                image_std=image_processor["image_std"],
            )
        return GraphSpec(
            vision_model,
            "inkling_vision",
            vision_model.input_types(model_config.devices[0]),
            weights=state_dict,
        )

    @override
    def _language_spec(
        self, model_config: InklingConfig, state_dict: dict[str, Any]
    ) -> GraphSpec:
        mesh = DeviceMesh(tuple(self.devices), (len(self.devices),), (TP,))
        with (
            F.lazy(),
            default_dtype(model_config.dtype),
            default_device(mesh),
        ):
            nn_model = Inkling(
                model_config,
                return_logits=self.return_logits,
                return_hidden_states=self.return_hidden_states,
            )
        return GraphSpec(
            nn_model, "inkling", nn_model.input_types(), weights=state_dict
        )

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        assert isinstance(model_inputs, InklingInputs)
        outputs = self.model(*model_inputs.buffers)
        logits = outputs.logits
        if logits is None:
            logits = outputs.next_token_logits
        return ModelOutputs(
            logits=logits.driver_tensor,
            next_token_logits=outputs.next_token_logits.driver_tensor,
            logit_offsets=(
                None
                if outputs.logit_offsets is None
                else outputs.logit_offsets.driver_tensor
            ),
        )
