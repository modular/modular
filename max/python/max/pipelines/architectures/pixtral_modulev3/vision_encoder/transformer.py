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


from __future__ import annotations

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.linear import Linear
from max.experimental.nn.norm import RMSNorm
from max.experimental.nn.sequential import ModuleList
from max.experimental.tensor import Tensor

from .attention import Attention


class MLP(Module[[Tensor], Tensor]):
    """
    Simple multi-layer perceptron composed of three linear layers.
    Uses SiLU activation function.
    """

    gate_proj: Linear
    down_proj: Linear
    up_proj: Linear

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        super().__init__()

        self.gate_proj = Linear(
            in_dim=hidden_size, out_dim=intermediate_size, bias=False
        )
        self.down_proj = Linear(
            in_dim=intermediate_size, out_dim=hidden_size, bias=False
        )
        self.up_proj = Linear(
            in_dim=hidden_size, out_dim=intermediate_size, bias=False
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(Module[..., Tensor]):
    """Stack of Attention, FeedForward, and RMSNorm layers."""

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

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
    ) -> Tensor:
        attn_out = self.attention(
            x=self.attention_norm(x),
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )

        if self.residual_multiplier != 1.0:
            multiplier = F.constant(
                self.residual_multiplier, x.dtype, device=x.device
            )
            attn_out = attn_out * multiplier

        h = x + attn_out
        mlp = self.feed_forward(self.ffn_norm(h))
        if self.residual_multiplier != 1.0:
            multiplier = F.constant(
                self.residual_multiplier, x.dtype, device=x.device
            )
            mlp = mlp * multiplier

        return h + mlp


class Transformer(Module[..., Tensor]):
    """Transformer model consisting of TransformerBlock layers.
    The input is embeddings created using convolution followed by normalization.

    The differences between this transformer and other decoder model transformers:
    1. Input to the transformer is patch embeddings created by convolutions not tokens.
    2. No linear(norm(output)) at the transformer output.
    3. It uses the 2d rotary embeddings defined for images which is different
    from the rotary embeddings defined in other classes as rope: RotaryEmbedding
    """

    n_heads: int
    layers: ModuleList
    _dtype: DType

    def __init__(
        self, n_heads: int, layers: list[TransformerBlock], dtype: DType
    ) -> None:
        super().__init__()

        self.n_heads = n_heads
        self.layers = ModuleList(layers)
        self._dtype = dtype

    def forward(
        self,
        patch_embeds: Tensor,
        attention_mask: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
    ) -> Tensor:
        h = patch_embeds

        for _, layer in enumerate(self.layers):
            h = layer(
                h,
                attention_mask,
                position_embeddings,
            )

        return F.cast(h, self._dtype)
