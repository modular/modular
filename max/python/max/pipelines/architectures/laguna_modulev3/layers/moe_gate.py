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
"""Laguna MoE gate (ModuleV3)."""

from __future__ import annotations

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn.common_layers.moe import MoEGate
from max.experimental.tensor import Tensor, default_dtype


class LagunaTopKRouter(MoEGate):
    """Sigmoid top-k router with a per-expert score-correction bias.

    Experts are selected by ``sigmoid(logits) + e_score_correction_bias`` and
    weighted by the unbiased sigmoid score, renormalized when
    ``norm_topk_prob`` and then scaled by ``routed_scaling_factor``. Laguna has
    no expert groups, so this is plain top-k: the group-limited router kernel
    the DeepSeek gate uses caps top-k at 32 per group.

    Args:
        hidden_dim: Dimension of the hidden state.
        num_experts: Number of routed experts.
        num_experts_per_token: Top-k experts selected per token.
        norm_topk_prob: Whether the selected weights are renormalized to sum
            to one.
        gate_dtype: Storage dtype of the gate projection.
        correction_bias_dtype: Storage dtype of ``e_score_correction_bias``,
            and the dtype the routing weights are returned in.
        routed_scaling_factor: Scalar folded into the routing weights, which
            is the reference's scaling of the summed routed output.
        router_logit_softcapping: When positive, the logits pass through
            ``softcap * tanh(logits / softcap)`` before the sigmoid.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        *,
        norm_topk_prob: bool,
        gate_dtype: DType,
        correction_bias_dtype: DType,
        routed_scaling_factor: float = 1.0,
        router_logit_softcapping: float = 0.0,
    ) -> None:
        with default_dtype(gate_dtype):
            super().__init__(
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                num_experts_per_token=num_experts_per_token,
            )
        self.norm_topk_prob = norm_topk_prob
        self.correction_bias_dtype = correction_bias_dtype
        self.routed_scaling_factor = routed_scaling_factor
        self.router_logit_softcapping = router_logit_softcapping
        self.e_score_correction_bias = Tensor.zeros(
            [num_experts], dtype=correction_bias_dtype
        )

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        """Computes expert routing indices and weights.

        Args:
            hidden_states: Input of shape ``(seq_len, hidden_dim)``.

        Returns:
            A pair ``(topk_idx, topk_weight)``, each of shape
            ``(seq_len, num_experts_per_token)``.
        """
        # Routing runs in float32 whatever the bias dtype: across 256 experts
        # bfloat16 sigmoid scores collapse into tie groups, and top-k then
        # picks among them arbitrarily.
        logits = self.gate_score(hidden_states).cast(DType.float32)
        if self.router_logit_softcapping > 0.0:
            softcap = self.router_logit_softcapping
            logits = F.tanh(logits / softcap) * softcap
        scores = F.sigmoid(logits)
        correction_bias = self.e_score_correction_bias.cast(DType.float32)

        topk_sel, topk_idx = F.top_k(
            scores + correction_bias, k=self.num_experts_per_token, axis=-1
        )
        topk_weight = topk_sel - F.gather(correction_bias, topk_idx, axis=0)
        if self.norm_topk_prob:
            topk_weight = topk_weight / F.sum(topk_weight, axis=-1)
        topk_weight = topk_weight * self.routed_scaling_factor
        return topk_idx, topk_weight.cast(self.correction_bias_dtype)
