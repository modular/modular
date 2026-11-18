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
"""Hybrid transformer block for the LFM2-VL decoder.

HF modeling_lfm2.py::Lfm2DecoderLayer (and the checkpoint's own
remotecode hybrid.py `DecoderLayer` with `sandwich=False`): pre-norm on
the mixer input (`operator_norm`), mixer = gated short conv (conv
layers) or GQA attention with q/k layernorm (full_attention layers),
residual; pre-norm on the FFN (`ffn_norm`), dense SwiGLU MLP, residual.

`block__name_mlp=parallel_mlp_merged` does NOT make the FFN parallel to
the mixer: the checkpoint ships plain separate
`feed_forward.{w1,w2,w3}.weight` matrices (verified in the safetensors
header) and the reference residual order is sequential.
"""

from __future__ import annotations

from max import functional as F
from max.nn import Module
from max.nn.kv_cache import PagedCacheValues
from max.nn.norm import RMSNorm
from max.tensor import Tensor


class Lfm2VlTransformerBlock(Module[..., Tensor]):
    """One hybrid decoder layer: conv or attention, then the dense MLP."""

    def __init__(
        self,
        layer_type: str,
        operator: Module[..., Tensor],
        feed_forward: Module[[Tensor], Tensor],
        operator_norm: RMSNorm,
        ffn_norm: RMSNorm,
    ) -> None:
        super().__init__()
        self.layer_type = layer_type
        # Exactly one mixer is live per layer; the other slot stays None
        # (the landed lfm2 block's declaration idiom).
        self.self_attn: Module[..., Tensor] | None
        self.conv: Module[..., Tensor] | None
        if layer_type == "full_attention":
            self.self_attn = operator
            self.conv = None
        else:
            self.conv = operator
            self.self_attn = None
        self.feed_forward = feed_forward
        self.operator_norm = operator_norm
        self.ffn_norm = ffn_norm

    def forward(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        conv_pool: Tensor,
        conv_rows: Tensor,
        conv_layer: Tensor,
        query_start_loc: Tensor,
        has_initial_state: Tensor,
        input_row_offsets: Tensor,
    ) -> Tensor:
        """One hybrid block: the operator's pre-norm branch, then the FFN's."""
        residual = x
        normed = self.operator_norm(x)
        if self.layer_type == "full_attention":
            assert self.self_attn is not None
            op_out = self.self_attn(
                normed,
                kv_collection,
                input_row_offsets=input_row_offsets,
            )
        else:
            assert self.conv is not None
            op_out = self.conv(
                normed,
                conv_pool,
                conv_rows,
                conv_layer,
                query_start_loc,
                has_initial_state,
            )
        h = residual + op_out
        out = self.feed_forward(self.ffn_norm(h))
        return F.rebind(h + out, x.shape)
