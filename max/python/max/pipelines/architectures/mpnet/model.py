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
"""Defines the MPNet pipeline model.

Implementation is based on MPNetModel from the transformers library.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from max.driver import Buffer, Device
from max.engine import InferenceSession, Model
from max.graph.weights import Weights, WeightsAdapter
from max.nn.transformer import ReturnLogits
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
)
from max.pipelines.lib.graph_specs import GraphRole, GraphSpec
from max.pipelines.lib.memory_estimation import MemoryPlan

from .batch_processor import MPNetBatchProcessor
from .graph import MPNetModel
from .model_config import MPNetConfig

logger = logging.getLogger("max.pipelines")

PAD_VALUE = 1


@dataclass
class MPNetInputs(ModelInputs):
    """A class representing inputs for the MPNet model.

    This class encapsulates the input tensors required for the MPNet model execution:
    - next_tokens_batch: A tensor containing the input token IDs
    - attention_mask: A tensor containing the extended attention mask
    """

    next_tokens_batch: Buffer
    attention_mask: Buffer


class MPNetPipelineModel(PipelineModel[TextContext, MPNetConfig, Model]):
    model_config_cls: ClassVar[type[MPNetConfig]] = MPNetConfig
    batch_processor_cls: ClassVar[type[MPNetBatchProcessor]] = (
        MPNetBatchProcessor
    )

    state_dict: dict[str, Any]
    """The weights, kept for :meth:`graph_specs`."""

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
        return_logits: ReturnLogits = ReturnLogits.ALL,
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
        self._models = self.load_graphs(session)

    @property
    def model(self) -> Model:
        """The compiled and initialized encoder."""
        return self._models[GraphRole.ENCODER]

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        assert isinstance(model_inputs, MPNetInputs)
        model_outputs = self.model.execute(
            model_inputs.next_tokens_batch, model_inputs.attention_mask
        )
        assert isinstance(model_outputs[0], Buffer)

        return ModelOutputs(logits=model_outputs[0])

    def _load_state_dict(self) -> dict[str, Any]:
        self.state_dict = super()._load_state_dict()
        return self.state_dict

    def _create_model_config(self, state_dict: dict[str, Any]) -> MPNetConfig:
        del state_dict
        return self.arch_config_as(MPNetConfig)

    def graph_specs(self, model_config: MPNetConfig) -> Mapping[str, GraphSpec]:
        """Returns the encoder graph."""
        model = MPNetModel(model_config)
        model.load_state_dict(self.state_dict)
        return {
            GraphRole.ENCODER: GraphSpec(model, "mpnet", model.input_types())
        }
