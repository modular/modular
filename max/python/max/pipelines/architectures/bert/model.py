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
"""Defines the Bert pipeline model.

Implementation is based on BertModel from the transformers library.
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

from .batch_processor import BertBatchProcessor
from .graph import BertModel
from .model_config import BertModelConfig

logger = logging.getLogger("max.pipelines")


@dataclass
class BertInputs(ModelInputs):
    next_tokens_batch: Buffer
    attention_mask: Buffer


class BertPipelineModel(PipelineModel[TextContext, BertModelConfig, Model]):
    batch_processor_cls: ClassVar[type[BertBatchProcessor]] = BertBatchProcessor
    model_config_cls: ClassVar[type[BertModelConfig]] = BertModelConfig

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
        assert isinstance(model_inputs, BertInputs)
        model_outputs = self.model.execute(
            model_inputs.next_tokens_batch, model_inputs.attention_mask
        )
        assert self.batch_processor is not None
        return self.batch_processor.process_outputs(model_outputs)

    def _load_state_dict(self) -> dict[str, Any]:
        self.state_dict = super()._load_state_dict()
        return self.state_dict

    def _create_model_config(
        self, state_dict: dict[str, Any]
    ) -> BertModelConfig:
        del state_dict
        return self.arch_config_as(BertModelConfig)

    def graph_specs(
        self, model_config: BertModelConfig
    ) -> Mapping[str, GraphSpec]:
        """Returns the encoder graph."""
        model = BertModel(model_config)
        model.load_state_dict(self.state_dict)
        return {
            GraphRole.ENCODER: GraphSpec(model, "bert", model.input_types())
        }
