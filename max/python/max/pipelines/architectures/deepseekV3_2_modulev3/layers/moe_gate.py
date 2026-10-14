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
"""Mixture of Experts gate for DeepSeek V3.2, in the ModuleV3 API."""

from __future__ import annotations

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn.common_layers.functional_kernels import (
    moe_router_group_limited,
)
from max.experimental.tensor import Tensor

from ...deepseekV3_modulev3.layers.moe_gate import DeepseekV3TopKRouter


class DeepseekV3_2TopKRouter(DeepseekV3TopKRouter):
    """V3 router whose gate score and sigmoid are computed in float32."""

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        """Compute expert routing weights and indices for input hidden states.

        Args:
            hidden_states: Input tensor of shape ``(seq_len, hidden_dim)``.

        Returns:
            A pair ``(topk_idx, topk_weight)`` of selected expert indices and
            their routing weights, each of shape
            ``(seq_len, num_experts_per_token)``.
        """
        # V3.2 runs the gate matmul itself in float32, not just the sigmoid.
        weight = self.gate_score.weight.cast(DType.float32)
        logits = hidden_states.cast(DType.float32) @ weight.T
        scores = F.sigmoid(logits)
        topk_idx, topk_weight = moe_router_group_limited(
            scores,
            self.e_score_correction_bias.cast(DType.float32),
            self.num_experts,
            self.num_experts_per_token,
            self.n_group,
            self.topk_group,
            self.norm_topk_prob,
            self.routed_scaling_factor,
        )
        return topk_idx, topk_weight
