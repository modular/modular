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

"""Gemma3 transformer block for the ModuleV3 API."""

from __future__ import annotations

from max.experimental.nn import Module
from max.experimental.tensor import Tensor
from max.nn.kv_cache import PagedCacheValues


class Gemma3TransformerBlock(Module[..., Tensor]):
    """Gemma3 transformer block with four normalization layers.

    Unlike standard pre-LN blocks, Gemma3 applies:
    - ``input_layernorm`` before attention
    - ``post_attention_layernorm`` to the attention output (before adding residual)
    - ``pre_feedforward_layernorm`` before the MLP
    - ``post_feedforward_layernorm`` to the MLP output (before adding residual)
    """

    def __init__(
        self,
        attention: Module[..., Tensor],
        mlp: Module[[Tensor], Tensor],
        input_layernorm: Module[[Tensor], Tensor],
        post_attention_layernorm: Module[[Tensor], Tensor],
        pre_feedforward_layernorm: Module[[Tensor], Tensor],
        post_feedforward_layernorm: Module[[Tensor], Tensor],
    ) -> None:
        super().__init__()
        self.self_attn = attention
        self.mlp = mlp
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm
        self.pre_feedforward_layernorm = pre_feedforward_layernorm
        self.post_feedforward_layernorm = post_feedforward_layernorm

    def forward(
        self,
        layer_idx: Tensor,
        x: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
        **kwargs,
    ) -> Tensor:
        residual = x
        attn_out = self.self_attn(
            self.input_layernorm(x),
            kv_collection,
            input_row_offsets=input_row_offsets,
            **kwargs,
        )
        hidden_states = residual + self.post_attention_layernorm(attn_out)

        residual = hidden_states
        mlp_out = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        return residual + self.post_feedforward_layernorm(mlp_out)
