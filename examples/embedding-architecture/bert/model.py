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
"""Defines the BERT pipeline model.

Implementation is based on BertModel from the transformers library.
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
from max.graph import DeviceRef
from max.graph.weights import Weights, WeightsAdapter
from max.nn import ReturnLogits
from max.nn.kv_cache import KVCacheInputs, KVCacheParams
from max.pipelines.core import TextContext
from max.pipelines.dataprocessing import collate_batch
from max.pipelines.lib import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
    SupportedEncoding,
    upper_bounded_default,
)
from transformers import AutoConfig

from .graph import build_graph
from .model_config import BertConfig

logger = logging.getLogger("max.pipelines")


class BertInputs(ModelInputs):
    """A class representing inputs for the BERT model.

    This class encapsulates the input tensors required for the BERT model execution:
    - input_ids: A tensor containing the input token IDs
    - attention_mask: A tensor containing the attention mask
    - token_type_ids: A tensor containing the token type IDs (optional).
      Also called segment IDs, these distinguish between different segments
      in the input (e.g., sentence A vs sentence B in sentence pair tasks).
      Use 0 for the first segment and 1 for the second segment
    """

    input_ids: Tensor
    attention_mask: Tensor
    token_type_ids: Tensor | None

    def __init__(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor | None = None,
    ) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        # BERT does not have KV cache inputs.
        self.kv_cache_inputs = None


class BertPipelineModel(PipelineModel[TextContext]):
    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        huggingface_config: AutoConfig,
        encoding: SupportedEncoding,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.ALL,
    ) -> None:
        # Auto-select BERT adapter if not provided
        if adapter is None:
            from .weight_adapters import BertWeightsAdapter

            adapter = BertWeightsAdapter()
            logger.info("Using BertWeightsAdapter")

        super().__init__(
            pipeline_config,
            session,
            huggingface_config,
            encoding,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits,
        )
        self.model = self.load_model(session)
        # BERT doesn't use KV cache, but serving infrastructure expects this attribute
        self.kv_manager = None

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return BertConfig.get_kv_params(
            huggingface_config=huggingface_config,
            n_devices=n_devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        return BertConfig.get_num_layers(huggingface_config)

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
            raise ValueError(
                "Unable to infer max_length for BERT, the provided "
                f"max_length ({pipeline_config.max_length}) exceeds the "
                f"model's max_position_embeddings "
                f"({huggingface_config.max_position_embeddings})."
            ) from e

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        assert isinstance(model_inputs, BertInputs)

        if model_inputs.token_type_ids is not None:
            model_outputs = self.model.execute(
                model_inputs.input_ids,
                model_inputs.attention_mask,
                model_inputs.token_type_ids,
            )
        else:
            model_outputs = self.model.execute(
                model_inputs.input_ids, model_inputs.attention_mask
            )

        return ModelOutputs(logits=cast(Tensor, model_outputs[0]))

    def prepare_initial_token_inputs(
        self,
        context_batch: Sequence[TextContext],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> BertInputs:
        # Get tokens and seq_ids.
        tokens = [ctx.next_tokens for ctx in context_batch]

        # Pad tokens for the batch.
        if not hasattr(self.huggingface_config, "pad_token_id"):
            raise ValueError(
                f"Model config for {self.huggingface_config.model_type} "
                "does not have pad_token_id attribute"
            )
        pad_value = self.huggingface_config.pad_token_id
        input_ids_batch, _ = collate_batch(
            tokens,
            pad_value=pad_value,
            batch_size=len(tokens),
        )

        # Compute attention mask.
        attention_mask = (input_ids_batch != pad_value).astype(np.float32)

        # Create token type IDs (segment embeddings).
        # Token type IDs distinguish between different segments in the input.
        # For BERT, this is used to differentiate between sentence A and sentence B
        # in tasks like next sentence prediction or question answering.
        # - 0 indicates tokens from the first segment (sentence A)
        # - 1 indicates tokens from the second segment (sentence B)
        # For single sequence tasks (like text classification with one sentence),
        # all token type IDs are 0. For sentence pair tasks, the first sentence
        # tokens get 0 and the second sentence tokens get 1.
        # Example: [CLS] sentence A [SEP] sentence B [SEP]
        #          [ 0,   0, 0, 0,  0,   1, 1, 1, 1,  1]
        token_type_ids = np.zeros_like(
            input_ids_batch
        )  # All zeros for single sequence

        return BertInputs(
            input_ids=Tensor.from_numpy(input_ids_batch).to(self.devices[0]),
            attention_mask=Tensor.from_numpy(attention_mask).to(
                self.devices[0]
            ),
            token_type_ids=Tensor.from_numpy(token_type_ids).to(
                self.devices[0]
            ),
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: ModelInputs,
    ) -> BertInputs:
        raise NotImplementedError(
            "This BERT model is configured for embeddings generation only. "
            "The prepare_next_token_inputs method is for autoregressive text generation "
            "models (e.g., GPT, Llama) that predict tokens sequentially. "
            "This embeddings model processes the entire input at once to produce "
            "dense vector representations, not generate new tokens."
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
            DeviceRef.from_device(self.devices[0]),
        )
        model = session.load(
            graph, weights_registry=self.weights.allocated_weights
        )
        after = time.perf_counter()
        logger.info(
            f"Building and compiling model took {after - before:.6f} seconds"
        )
        return model
