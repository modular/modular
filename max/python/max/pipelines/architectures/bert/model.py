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
import time
from dataclasses import dataclass
from typing import ClassVar

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

from .batch_processor import BertBatchProcessor
from .graph import build_graph
from .model_config import BertModelConfig

logger = logging.getLogger("max.pipelines")


@dataclass
class BertInputs(ModelInputs):
    next_tokens_batch: Buffer
    attention_mask: Buffer


class BertPipelineModel(PipelineModel[TextContext]):
    batch_processor_cls: ClassVar[type[BertBatchProcessor]] = BertBatchProcessor
    model_config_cls: ClassVar[type[BertModelConfig]] = BertModelConfig

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
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
            adapter,
            return_logits,
            max_batch_size=max_batch_size,
        )
        self.model = self.load_model(session)

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        assert isinstance(model_inputs, BertInputs)
        model_outputs = self.model.execute(
            model_inputs.next_tokens_batch, model_inputs.attention_mask
        )
        assert self.batch_processor is not None
        return self.batch_processor.process_outputs(model_outputs)

    def load_model(self, session: InferenceSession) -> Model:
        logger.info("Building and compiling model...")
        before = time.perf_counter()
        if self.adapter:
            state_dict = self.adapter(dict(self.weights.items()))
        else:
            state_dict = {
                key: value.data() for key, value in self.weights.items()
            }
        config = BertModelConfig.initialize(self.pipeline_config)
        graph = build_graph(config, state_dict)
        after_build = time.perf_counter()

        logger.info(f"Building graph took {after_build - before:.6f} seconds")

        before_compile = time.perf_counter()
        model = session.load(graph, weights_registry=state_dict)
        after = time.perf_counter()

        logger.info(
            f"Compiling model took {after - before_compile:.6f} seconds"
        )

        logger.info(
            f"Building and compiling model took {after - before:.6f} seconds"
        )
        return model
