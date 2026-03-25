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
"""Simplified Transformer for Qwen3 embedding models without KV caching."""

from __future__ import annotations

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.mlp import MLP
from max.experimental.nn.embedding import Embedding
from max.experimental.nn.norm import RMSNorm
from max.experimental.nn.sequential import ModuleList
from max.experimental.tensor import Tensor

from .attention import Qwen3AttentionNoCache
from .pooling import last_token_pool, normalize_embeddings


class Qwen3EmbeddingTransformerBlock(
    Module[[Tensor, Tensor], Tensor],
):
    """Transformer block for embedding models without KV caching."""

    def __init__(
        self,
        attention: Qwen3AttentionNoCache,
        mlp: MLP,
        attention_norm: RMSNorm,
        mlp_norm: RMSNorm,
        residual_multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        self.self_attn = attention
        self.mlp = mlp
        self.input_layernorm = attention_norm
        self.post_attention_layernorm = mlp_norm
        self.residual_multiplier = residual_multiplier

    def forward(
        self,
        x: Tensor,
        input_row_offsets: Tensor,
    ) -> Tensor:
        # Attention with pre-normalization
        attn_out = self.self_attn(
            self.input_layernorm(x),
            input_row_offsets,
        )

        # Only allocate the scalar constant when actually needed, since
        # dtype/device are not available at __init__ time.
        if self.residual_multiplier != 1.0:
            rm = F.constant(self.residual_multiplier, x.dtype, device=x.device)
            attn_out = attn_out * rm

        h = x + attn_out

        # MLP with pre-normalization
        mlp_out = self.mlp(self.post_attention_layernorm(h))

        if self.residual_multiplier != 1.0:
            rm = F.constant(self.residual_multiplier, x.dtype, device=x.device)
            mlp_out = mlp_out * rm

        return h + mlp_out


class Qwen3EmbeddingTransformer(
    Module[[Tensor, Tensor, Tensor], tuple[Tensor, ...]],
):
    """Transformer model for embedding generation without KV caching.

    Returns hidden states (optionally pooled and normalized) for embedding tasks.
    """

    def __init__(
        self,
        layers: list[Qwen3EmbeddingTransformerBlock],
        norm: RMSNorm,
        embedding: Embedding,
        pool_embeddings: bool = True,
        embedding_multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        self.layers = ModuleList(layers)
        self.norm = norm
        self.embed_tokens = embedding
        self.pool_embeddings = pool_embeddings
        self.embedding_multiplier = embedding_multiplier

    def forward(
        self,
        tokens: Tensor,
        input_row_offsets: Tensor,
        return_n_logits: Tensor,
    ) -> tuple[Tensor, ...]:
        h = self.embed_tokens(tokens)

        if self.embedding_multiplier != 1.0:
            h = h * F.constant(
                self.embedding_multiplier, h.dtype, device=h.device
            )

        # Process through transformer layers
        input_row_offsets_device = input_row_offsets.to(h.device)
        for layer in self.layers:
            h = layer(h, input_row_offsets_device)
        h = self.norm(h)

        if self.pool_embeddings:
            embeddings = last_token_pool(h, input_row_offsets_device)
            embeddings_normalized = normalize_embeddings(embeddings)
            return (embeddings_normalized,)
        else:
            return (F.cast(h, DType.float32),)


class Qwen3Embedding(
    Module[[Tensor, Tensor, Tensor], tuple[Tensor, ...]],
):
    """Wrapper module providing ``language_model.`` weight prefix.

    V3's ``compile(weights=state_dict)`` resolves names from the root module.
    The weight adapter maps ``model.`` to ``language_model.``, so the root
    compiled module must store the transformer as ``self.language_model``.
    """

    def __init__(self, transformer: Qwen3EmbeddingTransformer) -> None:
        super().__init__()
        self.language_model = transformer

    def forward(
        self,
        tokens: Tensor,
        input_row_offsets: Tensor,
        return_n_logits: Tensor,
    ) -> tuple[Tensor, ...]:
        return self.language_model(tokens, input_row_offsets, return_n_logits)
