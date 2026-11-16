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
"""Beam MoE router (ModuleV3)."""

from __future__ import annotations

from max import functional as F
from max.dtype import DType
from max.nn._graph_ops import (
    moe_sigmoid_gemv_group_limited_router_packed_supports,
)
from max.nn.kernels import moe_sigmoid_gemv_group_limited_router_packed
from max.tensor import Tensor, default_dtype

from ...deepseekV3_2_modulev3.layers.moe_gate import DeepseekV3_2TopKRouter


class BeamTopKRouter(DeepseekV3_2TopKRouter):
    """DeepSeek-V3.2's group-limited sigmoid router on float32 weights.

    The checkpoint stores the gate weight and the correction bias in float32.
    ``routed_scaling_factor`` is folded into the routing weights; the
    reference scales the summed expert output instead, which is the same
    product.

    The gate GEMM is IEEE float32 only up to 64 rows: above that the SM100
    float32 matmul runs as TF32, which changes the top-8 set at 0.1-1% of
    prefill positions.

    Args:
        hidden_dim: Dimension of the hidden state.
        num_experts: Number of routed experts.
        num_experts_per_token: Top-k experts selected per token.
        n_group: Number of expert groups.
        topk_group: Groups kept per token.
        norm_topk_prob: Whether the selected weights are renormalized.
        routed_scaling_factor: Scale folded into the routing weights.
    """

    # TODO(MXTOOLS-957): route the gate GEMM through an IEEE float32 kernel
    # at every M once `mo.matmul` can opt out of TF32; until then long
    # prefills route a few tokens differently from the reference.
    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        *,
        n_group: int,
        topk_group: int,
        norm_topk_prob: bool,
        routed_scaling_factor: float,
    ) -> None:
        with default_dtype(DType.float32):
            super().__init__(
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                num_experts_per_token=num_experts_per_token,
                routed_scaling_factor=routed_scaling_factor,
                scoring_func="sigmoid",
                topk_method="noaux_tc",
                n_group=n_group,
                topk_group=topk_group,
                norm_topk_prob=norm_topk_prob,
                correction_bias_dtype=DType.float32,
            )

    def packed(self, hidden_states: Tensor) -> Tensor:
        """Routes ``hidden_states`` and packs each row's routing.

        Returns each row's expert ids, as float32, then their weights,
        ``[rows, 2 * num_experts_per_token]``, the routing
        :meth:`~max.nn.common_layers.quant_moe.ExpertParallelMoE.local_expert_partial_sums`
        reads. A routing config the fused gate GEMV and router takes runs as
        one op, whose GEMV is IEEE float32 at every row count; any other
        runs :meth:`forward`.
        """
        if not moe_sigmoid_gemv_group_limited_router_packed_supports(
            self.num_experts,
            self.num_experts_per_token,
            self.n_group,
            self.topk_group,
            self.hidden_dim,
            DType.float32,
        ):
            topk_idx, topk_weight = self(hidden_states)
            return F.concat(
                [topk_idx.cast(DType.float32), topk_weight.cast(DType.float32)],
                axis=-1,
            )
        return moe_sigmoid_gemv_group_limited_router_packed(
            hidden_states,
            self.gate_score.weight.cast(DType.float32),
            self.e_score_correction_bias.cast(DType.float32),
            self.num_experts_per_token,
            self.n_group,
            self.topk_group,
            self.norm_topk_prob,
            self.routed_scaling_factor,
        )
