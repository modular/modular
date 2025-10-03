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
"""BERT graph implementation using shared transformer components."""

from __future__ import annotations

import os

# Import shared components
import sys

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.graph.weights import Weights
from max.nn import Module
from max.pipelines.lib import PipelineConfig
from transformers import AutoConfig

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.embeddings import EmbeddingLayer, PoolingLayer
from common.transformer import TransformerEncoder


class BertModel(Module):
    """The BERT model for embeddings generation using shared components."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        weights: Weights,
        huggingface_config: AutoConfig,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        config = huggingface_config

        # Handle both BERT and RoBERTa weight prefixes
        # Some models (like sentence-transformers) don't have the model prefix
        model_prefix = (
            "bert" if config.model_type == "bert" else config.model_type
        )

        # Try different weight access patterns
        model_weights = weights
        if hasattr(weights, model_prefix):
            # Standard HuggingFace models with prefix (e.g., roberta.embeddings)
            model_weights = getattr(weights, model_prefix)
        elif hasattr(weights, "embeddings"):
            # Sentence-transformers style without prefix (e.g., embeddings)
            model_weights = weights
        else:
            # Fallback to original weights
            model_weights = weights

        self.embeddings = EmbeddingLayer(
            pipeline_config,
            model_weights.embeddings,
            huggingface_config=huggingface_config,
            dtype=dtype,
            device=device,
        )
        self.encoder = TransformerEncoder(
            pipeline_config,
            model_weights.encoder,
            huggingface_config=huggingface_config,
            dtype=dtype,
        )
        # Pooler is optional - some models don't have it
        # Check if pooler weights exist in the model
        try:
            # Try to access pooler weights to see if they exist
            _ = model_weights.pooler.dense.weight
            self.has_pooler = True
            self.pooler = PoolingLayer(
                pipeline_config,
                model_weights.pooler,
                huggingface_config=huggingface_config,
                dtype=dtype,
            )
        except (AttributeError, KeyError):
            self.has_pooler = False
        self.pool_outputs = pipeline_config.pool_embeddings

    def __call__(
        self,
        input_ids: TensorValue,
        attention_mask: TensorValue,
        token_type_ids: TensorValue | None = None,
    ) -> TensorValue:
        embedding_output = self.embeddings(
            input_ids=input_ids, token_type_ids=token_type_ids
        )

        # Extend attention mask to 4D for multi-head attention
        extended_attention_mask = ops.reshape(
            attention_mask, ("batch_size", 1, 1, "seq_len")
        )
        extended_attention_mask = (1 - extended_attention_mask) * ops.constant(
            np.finfo(np.float32).min,
            DType.float32,
            device=attention_mask.device,
        )

        encoder_outputs = self.encoder(
            embedding_output, attention_mask=extended_attention_mask
        )

        if self.pool_outputs:
            if self.has_pooler:
                # Use the pooler layer if available
                return self.pooler(encoder_outputs)
            else:
                # Mean pooling as fallback
                encoder_outputs = encoder_outputs.transpose(1, 2)
                input_mask_expanded = ops.broadcast_to(
                    ops.unsqueeze(attention_mask, 1),
                    ("batch_size", encoder_outputs.shape[1], "seq_len"),
                )
                input_lengths = ops.max(
                    ops.sum(input_mask_expanded),
                    ops.constant(
                        1e-9, DType.float32, device=input_mask_expanded.device
                    ),
                )
                pooled_output = (
                    ops.sum(encoder_outputs * input_mask_expanded)
                    / input_lengths
                )
                return ops.squeeze(pooled_output, 2)
        else:
            return encoder_outputs


def build_graph(
    pipeline_config: PipelineConfig,
    weights: Weights,
    huggingface_config: AutoConfig,
    dtype: DType,
    input_device: DeviceRef,
) -> Graph:
    # Graph input types.
    input_ids_type = TensorType(
        DType.int64, shape=["batch_size", "seq_len"], device=input_device
    )
    attention_mask_type = TensorType(
        DType.float32, shape=["batch_size", "seq_len"], device=input_device
    )
    token_type_ids_type = TensorType(
        DType.int64, shape=["batch_size", "seq_len"], device=input_device
    )

    # Check if we need token type IDs
    use_token_type_ids = hasattr(weights.embeddings, "token_type_embeddings")

    if use_token_type_ids:
        input_types = [input_ids_type, attention_mask_type, token_type_ids_type]
    else:
        input_types = [input_ids_type, attention_mask_type]

    with Graph("bert", input_types=input_types) as graph:
        bert = BertModel(
            pipeline_config,
            weights,
            huggingface_config,
            dtype,
            device=input_device,
        )
        input_ids = graph.inputs[0].tensor
        attention_mask = graph.inputs[1].tensor

        if use_token_type_ids:
            token_type_ids = graph.inputs[2].tensor
            graph.output(bert(input_ids, attention_mask, token_type_ids))
        else:
            graph.output(bert(input_ids, attention_mask))

    return graph
