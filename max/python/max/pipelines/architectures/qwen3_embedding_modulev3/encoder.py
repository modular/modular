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
"""The Qwen3 embedding encoder an architecture registers for ``ENCODER``."""

from __future__ import annotations

import functools
import math

from max.nn import Module
from max.nn.common_layers.mlp import MLP
from max.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.nn.embedding import Embedding
from max.nn.norm import RMSNorm
from max.pipelines.lib.pipeline_variants.utils import get_rope_theta
from max.pipelines.lib.registration_encoder import EncoderInputs
from max.tensor import Tensor

from .layers import (
    Qwen3AttentionNoCache,
    Qwen3EmbeddingTransformer,
    Qwen3EmbeddingTransformerBlock,
)
from .model_config import Qwen3EmbeddingConfig


class Qwen3Embedding(Module[[EncoderInputs], Tensor]):
    """Qwen3 embedding, registered for the ``ENCODER`` role.

    It takes a ragged batch and returns last-token pooled, normalized
    embeddings, or the float32 hidden states when the pipeline doesn't pool.
    The transformer sits under ``language_model``, where the weight adapter
    puts the checkpoint's weights.
    """

    graph_name = "Qwen3Embedding"
    """The name the graph compiled under before registration, which keys
    precompiled MEFs."""

    ragged_batch = True

    def __init__(self, config: Qwen3EmbeddingConfig) -> None:
        """Builds the transformer from the checkpoint's config."""
        super().__init__()
        hf = config.huggingface_config
        head_dim = hf.head_dim
        norm_eps = getattr(hf, "rms_norm_eps", 1e-6)
        rope = RotaryEmbedding(
            dim=hf.hidden_size,
            n_heads=hf.num_attention_heads,
            theta=get_rope_theta(hf),
            max_seq_len=config.max_seq_len,
            device=config.devices[0].to_device(),
            head_dim=head_dim,
            interleaved=False,
        )
        create_norm = functools.partial(RMSNorm, hf.hidden_size, eps=norm_eps)
        layers = [
            Qwen3EmbeddingTransformerBlock(
                attention=Qwen3AttentionNoCache(
                    rope=rope,
                    num_attention_heads=hf.num_attention_heads,
                    num_key_value_heads=hf.num_key_value_heads,
                    hidden_size=hf.hidden_size,
                    head_dim=head_dim,
                    scale=getattr(
                        hf,
                        "attention_multiplier",
                        1.0 / math.sqrt(float(head_dim)),
                    ),
                    qk_norm_eps=norm_eps,
                ),
                mlp=MLP(
                    hidden_dim=hf.hidden_size,
                    feed_forward_length=hf.intermediate_size,
                    bias=False,
                ),
                attention_norm=create_norm(),
                mlp_norm=create_norm(),
                residual_multiplier=1.0,
            )
            for _ in range(hf.num_hidden_layers)
        ]
        embedding = Embedding(hf.vocab_size, dim=hf.hidden_size)
        self.language_model = Qwen3EmbeddingTransformer(
            layers=layers,
            norm=create_norm(),
            embedding=embedding,
            pool_embeddings=config.pool_embeddings,
            embedding_multiplier=1.0,
        )

    def forward(self, inputs: EncoderInputs) -> Tensor:
        """Runs the transformer over the ragged batch."""
        assert inputs.input_row_offsets is not None
        assert inputs.return_n_logits is not None
        (embeddings,) = self.language_model(
            inputs.tokens, inputs.input_row_offsets, inputs.return_n_logits
        )
        return embeddings
