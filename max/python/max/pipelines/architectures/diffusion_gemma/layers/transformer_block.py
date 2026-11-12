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
"""DiffusionGemma transformer block."""

from __future__ import annotations

from max import functional as F
from max.nn import Module
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping, Replicated

from .rms_norm import Gemma4RMSNorm


def _replicated(x: Tensor) -> Tensor:
    # The row-parallel output projections leave a Partial sum under TP, and
    # RMSNorm is not linear, so reduce before the post-norms.
    return F.transfer_to(
        x, DeviceMapping(x.mesh, (Replicated(),) * x.mesh.ndim)
    )


class DiffusionGemmaTransformerBlock(Module[..., Tensor]):
    """Gemma4 decoder layer with the always-on parallel MLP + MoE branch.

    Mirrors the graph-API ``Gemma4TextDecoderLayer`` with
    ``enable_moe_block=True`` (DiffusionGemma removes the dense-only
    variant): 4 norms, the parallel dense-MLP and routed-expert branches
    with their own post-norms, and the learned ``layer_scalar``.
    """

    def __init__(
        self,
        attention: Module[..., Tensor],
        mlp: Module[[Tensor], Tensor],
        moe_block: Module[[Tensor], Tensor],
        hidden_size: int,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()
        self.self_attn = attention
        self.mlp = mlp
        self.moe_block = moe_block
        self.input_layernorm = Gemma4RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(
            hidden_size, eps=rms_norm_eps
        )
        self.pre_feedforward_layernorm = Gemma4RMSNorm(
            hidden_size, eps=rms_norm_eps
        )
        self.post_feedforward_layernorm = Gemma4RMSNorm(
            hidden_size, eps=rms_norm_eps
        )
        self.post_feedforward_layernorm_1 = Gemma4RMSNorm(
            hidden_size, eps=rms_norm_eps
        )
        self.post_feedforward_layernorm_2 = Gemma4RMSNorm(
            hidden_size, eps=rms_norm_eps
        )
        self.layer_scalar = Tensor.ones([1])

    def forward(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
    ) -> Tensor:
        """Runs the block's attention, then its parallel MLP + MoE branches."""
        residual = x
        attn_out = self.self_attn(
            self.input_layernorm(x),
            kv_collection,
            input_row_offsets=input_row_offsets,
        )
        h = residual + self.post_attention_layernorm(_replicated(attn_out))

        residual = h
        mlp_out = self.mlp(self.pre_feedforward_layernorm(h))
        mlp_out = self.post_feedforward_layernorm_1(_replicated(mlp_out))

        # The routed branch's pre-expert norm runs in the MoE, after routing.
        moe_out = self.moe_block(h)
        moe_out = self.post_feedforward_layernorm_2(_replicated(moe_out))

        h = residual + self.post_feedforward_layernorm(
            _replicated(mlp_out + moe_out)
        )
        return h * self.layer_scalar
