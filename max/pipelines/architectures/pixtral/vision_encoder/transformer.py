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

from dataclasses import dataclass
from typing import TYPE_CHECKING

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, TensorValueLike, ops
from max.nn.layer import Layer

if TYPE_CHECKING:
    from max.nn.linear import LinearV1
    from max.nn.norm import RMSNormV1

    from .attention import Attention


@dataclass
class MLP(Layer):
    """
    Simple multi-layer perceptron composed of three linear layers.
    Uses Gelu activation function.
    """

    gate_proj: LinearV1
    down_proj: LinearV1
    up_proj: LinearV1

    def __call__(self, x: TensorValueLike) -> TensorValue:
        return self.down_proj(ops.silu(self.gate_proj(x)) * self.up_proj(x))  # type: ignore


@dataclass
class TransformerBlock(Layer):
    """Stack of Attention, FeedForward, and RMSNormV1 layers."""

    attention: Attention
    mlp: MLP
    attention_norm: RMSNormV1
    mlp_norm: RMSNormV1
    residual_multiplier: float = 1.0

    def __call__(
        self,
        x: TensorValue,
        attention_mask: TensorValueLike,
        position_embeddings: tuple[TensorValue, TensorValue],
    ) -> TensorValue:
        residual_multiplier = ops.constant(
            self.residual_multiplier, x.dtype, device=DeviceRef.CPU()
        )
        attn_out = self.attention(
            self.attention_norm(x), attention_mask, position_embeddings
        )

        if self.residual_multiplier != 1.0:
            attn_out = attn_out * residual_multiplier

        h = x + attn_out
        mlp = self.mlp(self.mlp_norm(h))
        if self.residual_multiplier != 1.0:
            mlp = mlp * residual_multiplier

        return h + mlp


@dataclass
class Transformer(Layer):
    """Transformer model consisting of TransformerBlock layers.
    The input is embeddings created using convolution followed by normalization.

    The differences between this transformer and other decoder model transformers:
    1. Input to the transformer is patch embeddings created by convolutions not tokens.
    2. No linear(norm(output)) at the transformer output.
    3. It uses the 2d rotary embeddings defined for images which is different
    from the rotary embeddings defined in other classes as rope: RotaryEmbedding
    """

    n_heads: int
    layers: list[TransformerBlock]
    dtype: DType

    def __call__(
        self,
        patch_embeds: TensorValue,
        attention_mask: TensorValueLike,
        position_embeddings: tuple[TensorValue, TensorValue],
        **kwargs,
    ):
        h = patch_embeds

        for _, layer in enumerate(self.layers):
            h = layer(
                x=h,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        return ops.cast(h, self.dtype)
