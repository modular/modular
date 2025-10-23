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

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, TensorValueLike, ops
from max.nn import LayerList, Linear, Module, RMSNorm

from .attention import Attention


class MLP(Module):
    """
    Simple multi-layer perceptron composed of three linear layers.
    Uses Gelu activation function.
    """

    gate_proj: Linear
    down_proj: Linear
    up_proj: Linear

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()

        self.gate_proj = Linear(
            hidden_size, intermediate_size, dtype=dtype, device=device
        )
        self.down_proj = Linear(
            intermediate_size, hidden_size, dtype=dtype, device=device
        )
        self.up_proj = Linear(
            hidden_size, intermediate_size, dtype=dtype, device=device
        )

    def __call__(self, x: TensorValueLike) -> TensorValue:
        return self.down_proj(ops.silu(self.gate_proj(x)) * self.up_proj(x))  # type: ignore


class TransformerBlock(Module):
    """Stack of Attention, FeedForward, and RMSNormV1 layers."""

    attention: Attention
    feed_forward: MLP
    attention_norm: RMSNorm
    ffn_norm: RMSNorm
    residual_multiplier: float = 1.0

    def __init__(
        self,
        attention: Attention,
        feed_forward: MLP,
        attention_norm: RMSNorm,
        ffn_norm: RMSNorm,
        residual_multiplier: float = 1.0,
    ) -> None:
        super().__init__()

        self.attention = attention
        self.feed_forward = feed_forward
        self.attention_norm = attention_norm
        self.ffn_norm = ffn_norm
        self.residual_multiplier = residual_multiplier

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
            x=self.attention_norm(x),
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )

        if self.residual_multiplier != 1.0:
            attn_out = attn_out * residual_multiplier

        h = x + attn_out
        mlp = self.feed_forward(self.ffn_norm(h))
        if self.residual_multiplier != 1.0:
            mlp = mlp * residual_multiplier

        return h + mlp


class Transformer(Module):
    """Transformer model consisting of TransformerBlock layers.
    The input is embeddings created using convolution followed by normalization.

    The differences between this transformer and other decoder model transformers:
    1. Input to the transformer is patch embeddings created by convolutions not tokens.
    2. No linear(norm(output)) at the transformer output.
    3. It uses the 2d rotary embeddings defined for images which is different
    from the rotary embeddings defined in other classes as rope: RotaryEmbedding
    """

    n_heads: int
    layers: LayerList
    dtype: DType

    def __init__(
        self, n_heads: int, layers: list[TransformerBlock], dtype: DType
    ) -> None:
        super().__init__()

        self.n_heads = n_heads
        self.layers = LayerList(layers)
        self.dtype = dtype

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
