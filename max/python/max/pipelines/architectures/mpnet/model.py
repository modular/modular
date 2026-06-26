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
from dataclasses import dataclass
from typing import ClassVar

from max.driver import Buffer, Device
from max.engine import InferenceSession, Model
from max.graph.weights import Weights, WeightsAdapter
from max.nn.transformer import ReturnLogits
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    CompilationTimer,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
)

from .batch_processor import MPNetBatchProcessor
from .graph import build_graph
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


class MPNetPipelineModel(PipelineModel[TextContext]):
    model_config_cls: ClassVar[type[MPNetConfig]] = MPNetConfig
    batch_processor_cls: ClassVar[type[MPNetBatchProcessor]] = (
        MPNetBatchProcessor
    )

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.ALL,
    ) -> None:
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits,
        )
        self.model = self.load_model(session)

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        assert isinstance(model_inputs, MPNetInputs)
        model_outputs = self.model.execute(
            model_inputs.next_tokens_batch, model_inputs.attention_mask
        )
        assert isinstance(model_outputs[0], Buffer)

        return ModelOutputs(logits=model_outputs[0])

    def load_model(self, session: InferenceSession) -> Model:
        with CompilationTimer("model") as timer:
            if self.adapter:
                state_dict = self.adapter(dict(self.weights.items()))
            else:
                state_dict = {
                    key: value.data() for key, value in self.weights.items()
                }
            config = MPNetConfig.initialize(self.pipeline_config)
            graph = build_graph(config, state_dict)
            timer.mark_build_complete()
            model = session.load(graph, weights_registry=state_dict)

        return model
