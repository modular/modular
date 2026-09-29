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
"""The Nemotron-H relu2 MLP and mixture of experts."""

from __future__ import annotations

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Linear, Module
from max.experimental.nn.common_layers.functional_kernels import (
    grouped_matmul_ragged,
    moe_create_indices,
)
from max.experimental.nn.sequential import ModuleList
from max.experimental.tensor import Tensor

from ..model_config import NemotronHConfig


def _relu2(x: Tensor) -> Tensor:
    r = F.relu(x)
    return r * r


class NemotronHMLP(Module[[Tensor], Tensor]):
    """Non-gated MLP: ``down(relu(up(x)) ** 2)``."""

    def __init__(self, hidden_dim: int, feed_forward_length: int) -> None:
        self.up_proj = Linear(hidden_dim, feed_forward_length, bias=False)
        self.down_proj = Linear(feed_forward_length, hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(_relu2(self.up_proj(x)))


class NemotronHRouter(Module[[Tensor], tuple[Tensor, Tensor]]):
    """Sigmoid top-k router with a selection-only score bias.

    With one expert group, DeepSeek-V3's group-limited routing reduces to a
    plain top-k.
    """

    def __init__(self, config: NemotronHConfig) -> None:
        self.num_experts_per_tok = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.weight = Tensor.zeros(
            [config.num_experts, config.hidden_size], dtype=DType.float32
        )
        self.e_score_correction_bias = Tensor.zeros(
            [config.num_experts], dtype=DType.float32
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        scores = F.sigmoid(F.cast(x, DType.float32) @ self.weight.T)
        bias = self.e_score_correction_bias
        biased, experts = F.top_k(
            scores + bias, k=self.num_experts_per_tok, axis=-1
        )
        # The weights are the unbiased scores of the selected experts. Gathers
        # from the [num_experts] bias rather than the [seq, num_experts]
        # scores.
        weights = biased - F.gather(bias, experts, axis=0)
        if self.norm_topk_prob:
            weights = weights / F.sum(weights, axis=-1)
        return experts, weights * self.routed_scaling_factor


class NemotronHMoE(Module[[Tensor], Tensor]):
    """Routed relu2 experts plus one always-on shared expert."""

    def __init__(self, config: NemotronHConfig) -> None:
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.gate = NemotronHRouter(config)
        self.experts = ModuleList(
            [
                NemotronHMLP(config.hidden_size, config.moe_intermediate_size)
                for _ in range(config.num_experts)
            ]
        )
        self.shared_experts = NemotronHMLP(
            config.hidden_size, config.moe_shared_expert_intermediate_size
        )

    def forward(self, x: Tensor) -> Tensor:
        seq_len, hidden_dim = x.shape
        experts, weights = self.gate(x)
        (
            token_expert_order,
            expert_start_indices,
            restore_token_order,
            expert_ids,
            expert_usage_stats,
        ) = moe_create_indices(
            F.cast(F.reshape(experts, [-1]), DType.int32), self.num_experts
        )
        permuted = F.gather(
            x,
            F.cast(
                F.floor_div(token_expert_order, self.num_experts_per_tok),
                DType.int32,
            ),
            axis=0,
        )
        up = grouped_matmul_ragged(
            permuted,
            F.stack([e.up_proj.weight for e in self.experts], axis=0),
            expert_start_indices,
            expert_ids,
            expert_usage_stats,
        )
        down = grouped_matmul_ragged(
            _relu2(up),
            F.stack([e.down_proj.weight for e in self.experts], axis=0),
            expert_start_indices,
            expert_ids,
            expert_usage_stats,
        )
        down = F.gather(down, restore_token_order, axis=0).reshape(
            [seq_len, self.num_experts_per_tok, hidden_dim]
        )
        routed = F.unsqueeze(F.cast(weights, x.dtype), axis=1) @ down
        return F.squeeze(routed, axis=1) + self.shared_experts(x)
