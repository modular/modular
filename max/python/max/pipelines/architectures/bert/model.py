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
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from max.driver import Buffer, Device
from max.engine import InferenceSession, Model
from max.graph.weights import Weights, WeightsAdapter
from max.nn.legacy.kv_cache import KVCacheInputs
from max.nn.legacy.transformer import ReturnLogits
from max.pipelines.core import TextContext
from max.pipelines.dataprocessing import collate_batch
from max.pipelines.lib import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
    upper_bounded_default,
)
from transformers import AutoConfig

from .graph import build_graph
from .model_config import BertModelConfig

logger = logging.getLogger("max.pipelines")


@dataclass
class BertInputs(ModelInputs):
    next_tokens_batch: Buffer
    attention_mask: Buffer


class BertPipelineModel(PipelineModel[TextContext]):
    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        huggingface_config: AutoConfig,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.ALL,
    ) -> None:
        super().__init__(
            pipeline_config,
            session,
            huggingface_config,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits,
        )
        self.model = self.load_model(session)

    @classmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        try:
            return upper_bounded_default(
                upper_bound=huggingface_config.max_position_embeddings,
                default=pipeline_config.model.max_length,
            )
        except ValueError as e:
            raise ValueError(
                "Unable to infer max_length for Bert, the provided "
                f"max_length ({pipeline_config.model.max_length}) exceeds the "
                f"model's max_position_embeddings "
                f"({huggingface_config.max_position_embeddings})."
            ) from e

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        assert isinstance(model_inputs, BertInputs)
        model_outputs = self.model.execute(
            model_inputs.next_tokens_batch, model_inputs.attention_mask
        )
        assert isinstance(model_outputs[0], Buffer)

        return ModelOutputs(logits=model_outputs[0])

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> BertInputs:
        if len(replica_batches) > 1:
            raise ValueError("Model does not support DP>1")

        context_batch = replica_batches[0]

        tokens = [ctx.tokens.active for ctx in context_batch]

        pad_value = getattr(self.huggingface_config, "pad_token_id", 0)
        next_tokens_batch, _ = collate_batch(
            tokens,
            pad_value=pad_value,
            batch_size=len(tokens),
        )

        attention_mask = (next_tokens_batch != pad_value).astype(np.float32)

        return BertInputs(
            next_tokens_batch=Buffer.from_numpy(next_tokens_batch).to(
                self.devices[0]
            ),
            attention_mask=Buffer.from_numpy(attention_mask).to(
                self.devices[0]
            ),
        )

    def prepare_next_token_inputs(
        self, next_tokens: Buffer, prev_model_inputs: ModelInputs
    ) -> BertInputs:
        raise NotImplementedError(
            "Bert does not support preparing next tokens inputs."
        )

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
