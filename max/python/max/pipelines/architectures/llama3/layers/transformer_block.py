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

"""Llama3 transformer block."""

from __future__ import annotations

from max import functional as F
from max.nn import Module
from max.nn.legacy.kv_cache import PagedCacheValues
from max.tensor import Tensor

from ...common_layers.attention import AttentionWithRope


class LlamaTransformerBlock(Module[..., Tensor]):
    """Stack of Attention, MLP, and normalization layers.

    Supports an optional ``residual_multiplier`` that scales the attention
    and MLP outputs before adding them to the residual stream.
    """

    def __init__(
        self,
        attention: AttentionWithRope,
        mlp: Module[[Tensor], Tensor],
        input_layernorm: Module[[Tensor], Tensor],
        post_attention_layernorm: Module[[Tensor], Tensor],
        residual_multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        self.self_attn = attention
        self.mlp = mlp
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm
        self.residual_multiplier = residual_multiplier

    def forward(
        self,
        layer_idx: Tensor,
        x: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
        **kwargs,
    ) -> Tensor:
        attn_out = self.self_attn(
            self.input_layernorm(x),
            kv_collection,
            input_row_offsets=input_row_offsets,
            **kwargs,
        )

        if self.residual_multiplier != 1.0:
            multiplier = F.constant(
                self.residual_multiplier, x.dtype, device=x.device
            )
            attn_out = attn_out * multiplier

        h = x + attn_out

        mlp_out = self.mlp(self.post_attention_layernorm(h))

        if self.residual_multiplier != 1.0:
            multiplier = F.constant(
                self.residual_multiplier, x.dtype, device=x.device
            )
            mlp_out = mlp_out * multiplier

        return h + mlp_out
