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

"""Mixture of Experts Gate Layer for DeepSeek V3.2."""

from __future__ import annotations

from max.dtype import DType
from max.graph import TensorValue, ops
from max.nn.kernels import moe_router_group_limited
from max.nn.linear import Linear
from max.pipelines.architectures.deepseekV3.layers.moe_gate import (
    DeepseekV3TopKRouter,
)


class FloatCastingLinear(Linear):
    """Linear layer that casts to Dtype for intermediate operations."""

    def __call__(self, x: TensorValue) -> TensorValue:
        x = x.cast(DType.float32)
        weight = self.weight.to(x.device).cast(DType.float32)
        return x @ weight.T


class DeepseekV3_2TopKRouter(DeepseekV3TopKRouter):
    def __init__(self, *args, **kwargs):
        kwargs.pop("linear_cls")
        kwargs["linear_cls"] = FloatCastingLinear
        super().__init__(*args, **kwargs)

    def __call__(
        self, hidden_states: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        """Compute expert routing weights and indices for input hidden states.
        Differs from V3_2 router in that it performs intermediate operations in float32.

        Args:
            hidden_states: Input tensor of shape (seq_len, hidden_dim)

        Returns:
            tuple containing:
                - topk_idx: Indices of top-k selected experts of shape (seq_len, num_experts_per_token)
                - topk_weight: Routing weights for selected experts of shape (seq_len, num_experts_per_token)
        """
        logits = self.gate_score(hidden_states)

        scores = ops.sigmoid(logits)

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
