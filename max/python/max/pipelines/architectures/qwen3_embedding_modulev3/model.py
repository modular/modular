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
"""Qwen3 Embedding pipeline model without KV caching (V3 eager API)."""

from __future__ import annotations

import functools
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental import functional as F
from max.experimental.nn.common_layers.mlp import MLP
from max.experimental.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.experimental.nn.embedding import Embedding
from max.experimental.nn.norm import RMSNorm
from max.graph import DeviceRef, TensorType
from max.graph.weights import Weights, WeightsAdapter
from max.nn.kv_cache import KVCacheInputs
from max.nn.transformer import ReturnLogits
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
)
from transformers import AutoConfig

from .layers import (
    Qwen3AttentionNoCache,
    Qwen3Embedding,
    Qwen3EmbeddingTransformer,
    Qwen3EmbeddingTransformerBlock,
)
from .model_config import Qwen3EmbeddingConfig

logger = logging.getLogger("max.pipelines")


@dataclass
class Qwen3EmbeddingInputs(ModelInputs):
    """Input structure for Qwen3 embedding models."""

    tokens: Buffer
    """Input token IDs [total_seq_len]"""

    input_row_offsets: Buffer
    """Row offsets for ragged tensors [batch_size + 1]"""

    return_n_logits: Buffer
    """Number of logits to return (kept for interface compatibility)"""


class Qwen3EmbeddingModel(PipelineModel[TextContext]):
    """Qwen3 embedding pipeline model without KV caching (V3 eager API).

    Optimized for embedding generation with:
    - No KV cache overhead
    - Single-pass forward computation
    - Flash attention without cache operations
    - Last token pooling with L2 normalization
    """

    model: Callable[..., Any]
    """Compiled model callable."""

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
        self.pipeline_config = pipeline_config
        self.devices = devices
        self.weights = weights
        self.adapter = adapter

        self.model = self.load_model()

    def load_model(self) -> Callable[..., Any]:
        """Build and compile the embedding model using V3 eager API."""
        huggingface_config = self.huggingface_config

        # Get state dict
        if self.adapter:
            state_dict = self.adapter(
                dict(self.weights.items()),
                huggingface_config=huggingface_config,
                pipeline_config=self.pipeline_config,
            )
        else:
            state_dict = {
                key: value.data() for key, value in self.weights.items()
            }

        # Remove lm_head weights — embedding model doesn't use them
        state_dict = {k: v for k, v in state_dict.items() if "lm_head" not in k}

        # Qwen3-Embedding checkpoints store weights without a "model." prefix
        # (e.g. "layers.0.self_attn.q_proj.weight" instead of
        # "model.layers.0.self_attn.q_proj.weight"). The llama3_modulev3
        # adapter maps "model." → "language_model.", but when there is no
        # "model." prefix to replace, keys pass through unchanged.
        # Ensure every key carries the "language_model." prefix that the
        # compiled module tree expects.
        state_dict = {
            k if k.startswith("language_model.") else f"language_model.{k}": v
            for k, v in state_dict.items()
        }

        # Configuration
        head_dim = huggingface_config.head_dim
        max_seq_len = self.pipeline_config.model.max_length or 32768
        norm_eps = getattr(huggingface_config, "rms_norm_eps", 1e-6)
        attention_multiplier = getattr(
            huggingface_config,
            "attention_multiplier",
            1.0 / math.sqrt(float(head_dim)),
        )

        # Create RoPE (needs device at construction time for freqs_cis)
        rope = RotaryEmbedding(
            dim=huggingface_config.hidden_size,
            n_heads=huggingface_config.num_attention_heads,
            theta=huggingface_config.rope_theta,
            max_seq_len=max_seq_len,
            device=self.devices[0],
            head_dim=head_dim,
            interleaved=False,
        )

        create_norm = functools.partial(
            RMSNorm,
            huggingface_config.hidden_size,
            eps=norm_eps,
        )

        with F.lazy():
            # Create transformer layers
            layers = []
            for _layer_idx in range(huggingface_config.num_hidden_layers):
                attention = Qwen3AttentionNoCache(
                    rope=rope,
                    num_attention_heads=huggingface_config.num_attention_heads,
                    num_key_value_heads=huggingface_config.num_key_value_heads,
                    hidden_size=huggingface_config.hidden_size,
                    head_dim=head_dim,
                    scale=attention_multiplier,
                    qk_norm_eps=norm_eps,
                )

                mlp = MLP(
                    hidden_dim=huggingface_config.hidden_size,
                    feed_forward_length=huggingface_config.intermediate_size,
                    bias=False,
                )

                block = Qwen3EmbeddingTransformerBlock(
                    attention=attention,
                    mlp=mlp,
                    attention_norm=create_norm(),
                    mlp_norm=create_norm(),
                    residual_multiplier=1.0,
                )
                layers.append(block)

            embedding = Embedding(
                huggingface_config.vocab_size,
                dim=huggingface_config.hidden_size,
            )

            transformer = Qwen3EmbeddingTransformer(
                layers=layers,
                norm=create_norm(),
                embedding=embedding,
                pool_embeddings=self.pipeline_config.model.pool_embeddings,
                embedding_multiplier=1.0,
            )

            nn_model = Qwen3Embedding(transformer)
            nn_model.to(self.devices[0])

        # Define input types
        device0 = self.devices[0]
        device_ref = DeviceRef(device0.label, device0.id)
        tokens_type = TensorType(
            DType.uint32, shape=["total_seq_len"], device=device_ref
        )
        input_row_offsets_type = TensorType(
            DType.uint32,
            shape=["batch_size_plus_1"],
            device=DeviceRef.CPU(),
        )
        return_n_logits_type = TensorType(
            DType.uint32, shape=(1,), device=DeviceRef.CPU()
        )

        compiled_model = nn_model.compile(
            tokens_type,
            input_row_offsets_type,
            return_n_logits_type,
            weights=state_dict,
        )

        return compiled_model

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Execute the model."""
        assert isinstance(model_inputs, Qwen3EmbeddingInputs)

        model_outputs = self.model(
            model_inputs.tokens,
            model_inputs.input_row_offsets,
            model_inputs.return_n_logits,
        )

        return ModelOutputs(logits=cast(Buffer, model_outputs[0].driver_tensor))

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> Qwen3EmbeddingInputs:
        if len(replica_batches) > 1:
            raise ValueError("Model does not support DP>1")

        context_batch = replica_batches[0]
        device = self.devices[0]

        all_tokens: list[int] = []
        row_offsets = [0]

        for ctx in context_batch:
            tokens = ctx.tokens.active
            all_tokens.extend(tokens)
            row_offsets.append(len(all_tokens))

        tokens_array = np.array(all_tokens, dtype=np.uint32)
        row_offsets_array = np.array(row_offsets, dtype=np.uint32)

        tokens_buffer = Buffer.from_numpy(tokens_array)
        row_offsets_buffer = Buffer.from_numpy(row_offsets_array)
        return_n_logits_buffer = Buffer.from_numpy(
            np.array([return_n_logits], dtype=np.uint32)
        )

        return Qwen3EmbeddingInputs(
            tokens=tokens_buffer.to(device),
            input_row_offsets=row_offsets_buffer,
            return_n_logits=return_n_logits_buffer,
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Buffer,
        prev_model_inputs: ModelInputs,
    ) -> Qwen3EmbeddingInputs:
        raise NotImplementedError(
            "Qwen3 embedding model does not support autoregressive generation"
        )

    @staticmethod
    def calculate_max_seq_len(
        pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        return Qwen3EmbeddingConfig.calculate_max_seq_len(
            pipeline_config, huggingface_config
        )
