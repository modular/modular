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

"""Implements the Olmo3 transformer block."""

from __future__ import annotations

from max.nn import Module
from max.nn.legacy.kv_cache import PagedCacheValues
from max.nn.norm import RMSNorm
from max.tensor import Tensor

from .attention import Olmo3Attention
from .mlp import MLP


class Olmo3TransformerBlock(
    Module[[Tensor, Tensor, PagedCacheValues, Tensor], Tensor]
):
    """Stack of Attention, MLP, and RMSNorm layers for Olmo3.

    Block's attention type (full or window) is specified in the model config.
    """

    def __init__(
        self,
        attention: Olmo3Attention,
        mlp: MLP,
        post_attention_layernorm: RMSNorm,
        post_feedforward_layernorm: RMSNorm,
    ) -> None:
        super().__init__()
        self.self_attn = attention
        self.mlp = mlp

        self.post_attention_layernorm = post_attention_layernorm
        self.post_feedforward_layernorm = post_feedforward_layernorm

    def forward(
        self,
        layer_idx: Tensor,
        x: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
    ) -> Tensor:
        # Post-norm architecture: norm is applied after the sub-layer output,
        # before adding the residual.
        residual = x
        attn_out = self.self_attn(
            x,
            kv_collection,
            input_row_offsets,
        )
        h = residual + self.post_attention_layernorm(attn_out)

        residual = h
        mlp_out = self.mlp(h)
        return residual + self.post_feedforward_layernorm(mlp_out)
