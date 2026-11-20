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

"""Gemma4 transformer block for the ModuleV3 API."""

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


class Gemma4TransformerBlock(Module[..., Tensor]):
    """Gemma4 decoder layer: 4 norms, dense MLP, learned layer_scalar.

    With a ``moe_block``, the routed experts run beside the dense MLP, each
    output post-normed before their sum.
    """

    def __init__(
        self,
        attention: Module[..., Tensor],
        mlp: Module[[Tensor], Tensor],
        hidden_size: int,
        rms_norm_eps: float,
        moe_block: Module[[Tensor], Tensor] | None = None,
        post_feedforward_norm_eps: float | None = None,
    ) -> None:
        super().__init__()
        # post_feedforward_norm_eps applies to the norm right after the
        # dense MLP: post_feedforward_layernorm_1 when experts run beside it.
        dense_norm_eps = (
            rms_norm_eps
            if post_feedforward_norm_eps is None
            else post_feedforward_norm_eps
        )
        self.self_attn = attention
        self.mlp = mlp
        self.moe_block = moe_block
        if moe_block is not None:
            self.post_feedforward_layernorm_1 = Gemma4RMSNorm(
                hidden_size, eps=dense_norm_eps
            )
            self.post_feedforward_layernorm_2 = Gemma4RMSNorm(
                hidden_size, eps=rms_norm_eps
            )
        self.input_layernorm = Gemma4RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(
            hidden_size, eps=rms_norm_eps
        )
        self.pre_feedforward_layernorm = Gemma4RMSNorm(
            hidden_size, eps=rms_norm_eps
        )
        self.post_feedforward_layernorm = Gemma4RMSNorm(
            hidden_size,
            eps=rms_norm_eps if moe_block is not None else dense_norm_eps,
        )
        self.layer_scalar = Tensor.ones([1])

    def forward(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
    ) -> Tensor:
        """Runs the block's attention, then its MLP and any routed experts."""
        residual = x
        attn_out = self.self_attn(
            self.input_layernorm(x),
            kv_collection,
            input_row_offsets=input_row_offsets,
        )
        h = residual + self.post_attention_layernorm(_replicated(attn_out))

        residual = h
        ff_out = _replicated(self.mlp(self.pre_feedforward_layernorm(h)))
        if self.moe_block is not None:
            dense = self.post_feedforward_layernorm_1(ff_out)
            routed = _replicated(self.moe_block(h))
            ff_out = dense + self.post_feedforward_layernorm_2(routed)
        h = residual + self.post_feedforward_layernorm(ff_out)
        return h * self.layer_scalar
