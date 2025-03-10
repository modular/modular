# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
import time
from collections.abc import Sequence
from typing import cast

import numpy as np
from max.driver import Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph.weights import Weights
from max.pipelines import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
    SupportedEncoding,
    TextContext,
    upper_bounded_default,
)
from max.pipelines.dataprocessing import collate_batch
from max.pipelines.kv_cache import KVCacheInputs, KVCacheParams
from transformers import AutoConfig

from .graph import build_graph

logger = logging.getLogger("max.pipelines")

PAD_VALUE = 1


class MPNetInputs(ModelInputs):
    """A class representing inputs for the MPNet model.

    This class encapsulates the input tensors required for the MPNet model execution:
    - next_tokens_batch: A tensor containing the input token IDs
    - attention_mask: A tensor containing the extended attention mask
    """

    next_tokens_batch: Tensor
    attention_mask: Tensor

    def __init__(
        self,
        next_tokens_batch: Tensor,
        attention_mask: Tensor,
    ) -> None:
        self.next_tokens_batch = next_tokens_batch
        self.attention_mask = attention_mask
        # MPNet does not have KV cache inputs.
        self.kv_cache_inputs = None


class MPNetPipelineModel(PipelineModel[TextContext]):
    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        huggingface_config: AutoConfig,
        encoding: SupportedEncoding,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
    ) -> None:
        super().__init__(
            pipeline_config,
            session,
            huggingface_config,
            encoding,
            devices,
            kv_cache_config,
            weights,
        )
        self.model = self.load_model(session)

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return KVCacheParams(
            dtype=cache_dtype,
            n_kv_heads=huggingface_config.num_attention_heads,
            head_dim=(
                huggingface_config.hidden_size
                // huggingface_config.num_attention_heads
            ),
            cache_strategy=kv_cache_config.cache_strategy,
            n_devices=n_devices,
            enable_prefix_caching=kv_cache_config.enable_prefix_caching,
        )

    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        return huggingface_config.num_hidden_layers

    @classmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        try:
            return upper_bounded_default(
                upper_bound=huggingface_config.max_position_embeddings,
                default=pipeline_config.max_length,
            )
        except ValueError as e:
            msg = (
                "Unable to infer max_length for MPNet, the provided "
                f"max_length ({pipeline_config.max_length}) exceeds the "
                f"model's max_position_embeddings "
                f"({huggingface_config.max_position_embeddings})."
            )
            raise ValueError(msg) from e

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        model_inputs = cast(MPNetInputs, model_inputs)
        model_outputs = self.model.execute(
            model_inputs.next_tokens_batch,
            model_inputs.attention_mask,
            copy_inputs_to_device=False,
        )
        assert isinstance(model_outputs[0], Tensor)
        return ModelOutputs(logits=model_outputs[0])

    def prepare_initial_token_inputs(
        self,
        context_batch: Sequence[TextContext],
        kv_cache_inputs: KVCacheInputs | None = None,
    ) -> MPNetInputs:
        # Get tokens and seq_ids.
        tokens = [ctx.next_tokens for ctx in context_batch]

        # Pad tokens for the batch.
        pad_value = getattr(self.huggingface_config, "pad_token_id", 1)
        next_tokens_batch, _ = collate_batch(
            tokens,
            pad_value=pad_value,
            batch_size=len(tokens),
            pad_to_multiple_of=self.pipeline_config.pad_to_multiple_of,
        )

        # Compute attention mask.
        attention_mask = (next_tokens_batch != pad_value).astype(np.float32)

        return MPNetInputs(
            next_tokens_batch=Tensor.from_numpy(next_tokens_batch).to(
                self.devices[0]
            ),
            attention_mask=Tensor.from_numpy(attention_mask).to(
                self.devices[0]
            ),
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: ModelInputs,
    ) -> MPNetInputs:
        raise NotImplementedError(
            "MPNet does not support preparing next tokens inputs."
        )

    def load_model(
        self,
        session: InferenceSession,
    ) -> Model:
        logger.info("Building and compiling model...")
        before = time.perf_counter()
        graph = build_graph(
            self.pipeline_config,
            self.weights,
            self.huggingface_config,
            self.dtype,
        )
        model = session.load(
            graph, weights_registry=self.weights.allocated_weights
        )
        after = time.perf_counter()
        logger.info(
            f"Building and compiling model took {after - before:.6f} seconds"
        )
        return model
