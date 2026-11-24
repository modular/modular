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
"""Qwen2 transformer block for the ModuleV3 API."""

from __future__ import annotations

from max import functional as F
from max.nn import Module
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.common_layers.mlp import MLP
from max.nn.norm import RMSNorm
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping

from .attention import Qwen2Attention


def _replicated(x: Tensor) -> Tensor:
    # The row-parallel output projections leave a partial sum under TP;
    # reduce it before it joins the replicated residual stream.
    return F.transfer_to(x, DeviceMapping.replicated(x.mesh))


class Qwen2TransformerBlock(Module[..., Tensor]):
    """A pre-norm decoder block: attention, then a gated MLP."""

    def __init__(
        self,
        attention: Qwen2Attention,
        mlp: MLP,
        input_layernorm: RMSNorm,
        post_attention_layernorm: RMSNorm,
    ) -> None:
        super().__init__()
        self.self_attn = attention
        self.mlp = mlp
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm

    def forward(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
    ) -> Tensor:
        """Runs the block's attention, then its MLP."""
        attn_out = self.self_attn(
            self.input_layernorm(x), kv_collection, input_row_offsets
        )
        h = x + _replicated(attn_out)
        mlp_out = self.mlp(self.post_attention_layernorm(h))
        return h + _replicated(mlp_out)
