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

"""Sparse MoE block for Flint."""

from __future__ import annotations

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn.common_layers.moe import MoE, MoEGate
from max.experimental.tensor import Tensor


class FlintRouter(MoEGate):
    """Routes tokens to the top-k experts by sigmoid score."""

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        norm_topk_prob: bool,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
        )
        self.norm_topk_prob = norm_topk_prob

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        """Returns the top-k expert indices and weights, each ``[seq, top_k]``.

        HF computes router logits in fp32 even under autocast, so the
        hidden state and the weight are cast to fp32 for the matmul.
        """
        h = hidden_states.cast(DType.float32)
        w = self.gate_score.weight.cast(DType.float32)
        scores = F.sigmoid(h @ w.T)
        # `moe_router_group_limited` renormalizes whenever there is a single
        # group regardless of its `norm_weights` argument, so the selection is
        # done here where `norm_topk_prob` can actually take effect.
        topk_weight, topk_idx = F.top_k(
            scores, k=self.num_experts_per_token, axis=-1
        )
        if self.norm_topk_prob:
            topk_weight = topk_weight / F.sum(topk_weight, axis=-1)
        return (
            topk_idx.rebind_mapping(scores.mapping),
            topk_weight.rebind_mapping(scores.mapping),
        )


class FlintMoE(MoE):
    """The built-in MoE with Flint's router and shared expert."""
