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

from __future__ import annotations

from typing import Callable

from max.dtype import DType
from max.graph import TensorValue, TensorValueLike, ops

from ..attention import NaiveAttentionWithRope
from ..embedding import Embedding, EmbeddingV2
from ..layer import Layer, LayerList, Module
from ..linear import Linear, LinearV2
from .transformer import ReturnLogits


class NaiveTransformerBlock(Module):
    """Max-Graph Only Stack of Attention, FeedForward, and RMSNorm layers."""

    def __init__(
        self,
        attention: NaiveAttentionWithRope,
        mlp: Layer,
        attention_norm: Layer,
        mlp_norm: Layer,
        residual_multiplier: float = 1.0,
    ):
        super().__init__()
        self.self_attn = attention
        self.mlp = mlp
        self.input_layernorm = attention_norm
        self.post_attention_layernorm = mlp_norm
        self.residual_multiplier = residual_multiplier

    def __call__(
        self,
        x: TensorValue,
        attention_mask: TensorValueLike,
        k_cache: TensorValueLike,
        v_cache: TensorValueLike,
        start_pos: TensorValue,
        layer_index: int,
    ) -> tuple[TensorValue, TensorValue, TensorValue]:
        residual_multiplier = ops.constant(self.residual_multiplier, x.dtype)
        attn_out = self.self_attn(
            self.input_layernorm(x),
            attention_mask,
            k_cache,  # type: ignore
            v_cache,  # type: ignore
            start_pos,
            layer_index,
        )

        if self.residual_multiplier != 1.0:
            attn_out = attn_out * residual_multiplier

        h = x + attn_out
        mlp = self.mlp(self.post_attention_layernorm(h))
        if self.residual_multiplier != 1.0:
            mlp = mlp * residual_multiplier

        return h + mlp


class NaiveTransformer(Module):
    """Max-Graph only model consisting of NaiveTransformerBlock layers."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        layers: list[NaiveTransformerBlock],
        norm: Layer,
        output: Linear | LinearV2,
        theta: float,
        embedding: Embedding | EmbeddingV2,
        output_type: DType | None = None,
        embedding_multiplier: float = 1.0,
        logits_postprocessor: Callable[[TensorValue], TensorValue]
        | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.layers = LayerList(layers)
        self.norm = norm
        self.lm_head = output
        self.theta = theta
        self.embed_tokens = embedding
        self.output_type = output_type
        self.embedding_multiplier = embedding_multiplier
        self.logits_postprocessor = logits_postprocessor
        self.return_logits = return_logits

        if self.return_logits != ReturnLogits.LAST_TOKEN:
            msg = (
                "return_logits must be 'last_token', variable token lengths "
                "variable token lengths not supported with NaiveTransformer."
            )
            raise ValueError(msg)

    def _apply_logits_postprocessor(
        self, output: tuple[TensorValue]
    ) -> tuple[TensorValue]:
        if self.logits_postprocessor is None:
            return output
        return tuple(self.logits_postprocessor(elem) for elem in output)  # type:ignore

    def __call__(
        self,
        tokens: TensorValueLike,
        attention_mask: TensorValueLike,
        k_cache: TensorValueLike,
        v_cache: TensorValueLike,
        start_pos: TensorValueLike,
        return_n_logits: TensorValueLike,
    ) -> tuple[TensorValue]:
        h = self.embed_tokens(tokens)

        if self.embedding_multiplier != 1.0:
            h = h * ops.constant(self.embedding_multiplier, h.dtype)

        for i in range(len(self.layers)):
            h = self.layers[i](
                h,
                attention_mask,
                k_cache,
                v_cache,
                start_pos,
                i,
            )

        logits = self.lm_head(self.norm(h[:, -1]))

        if self.output_type is not None:
            casted_logits = ops.cast(logits, self.output_type)
            return self._apply_logits_postprocessor((casted_logits,))

        return self._apply_logits_postprocessor((logits,))
