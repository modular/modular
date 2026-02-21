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
"""A generalized Mixture of Experts (MoE) module.

These classes need to be cleaned up before moving to `max.nn`.
"""

from __future__ import annotations

from collections.abc import Callable

from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.nn import Linear
from max.nn.module import Module
from max.nn.sequential import ModuleList
from max.tensor import Tensor

from ...common_layers.functional_kernels import (
    grouped_matmul_ragged,
    moe_create_indices,
)
from ...common_layers.mlp import MLP


class MoEGate(Module[[Tensor], tuple[Tensor, Tensor]]):
    """Gate module for MoE."""

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
    ) -> None:
        """
        Args:
            hidden_dim: The dimension of the hidden state.
            num_experts: The number of experts.
            num_experts_per_token: The number of experts per token.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token

        self.gate_score = Linear(
            in_dim=hidden_dim, out_dim=num_experts, bias=False
        )

    def forward(self, hidden_state: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            hidden_state: The hidden state of the model.

        Returns:
            A tuple of the topk indices and scores.
        """
        scores = self.gate_score(hidden_state)
        topk_scores, topk_indices = F.top_k(
            scores, k=self.num_experts_per_token, axis=-1
        )

        return topk_indices, topk_scores


class MoE(Module[[Tensor], Tensor]):
    """Implementation of Mixture of Experts (MoE)."""

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        moe_dim: int,
        gate_cls: Callable[..., MoEGate] = MoEGate,
        has_shared_experts: bool = False,
        shared_experts_dim: int = 0,
        ep_size: int = 1,
        apply_router_weight_first: bool = False,
    ):
        """
        Args:
            hidden_dim: The dimension of the hidden state.
            num_experts: The number of experts.
            num_experts_per_token: The number of experts per token.
            moe_dim: The intermediate dimension of each expert.
            gate_cls: The model specific gate implementation.
            has_shared_experts: Whether to use shared experts.
            shared_experts_dim: The dimension of the shared experts.
            ep_size: The expert parallelism size.
            apply_router_weight_first: Whether to apply the router weight first.
            ep_batch_manager: The expert parallel batch manager.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.moe_dim = moe_dim
        self.gate_cls = gate_cls
        self.has_shared_experts = has_shared_experts
        self.shared_experts_dim = shared_experts_dim
        self.ep_size = ep_size
        self.apply_router_weight_first = apply_router_weight_first
        self.gate = gate_cls(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
        )
        self.num_local_experts = num_experts // ep_size

        if has_shared_experts:
            assert shared_experts_dim > 0, (
                "shared_experts_dim must be greater than 0"
            )
            self.shared_experts = MLP(
                hidden_dim=self.hidden_dim,
                feed_forward_length=self.shared_experts_dim,
                bias=False,
            )

        self._init_experts()

    def _init_experts(self) -> None:
        self.experts: ModuleList[MLP] = ModuleList(
            [
                MLP(
                    hidden_dim=self.hidden_dim,
                    feed_forward_length=self.moe_dim,
                    bias=False,
                )
                for _ in range(self.num_experts)
            ]
        )

    @property
    def gate_up_proj(self) -> Tensor:
        gate_list = [expert.gate_proj.weight for expert in self.experts]
        up_list = [expert.up_proj.weight for expert in self.experts]

        gate_up_list: list[Tensor] = []
        for tensors in zip(gate_list, up_list, strict=True):
            gate_up_list.extend(tensors)

        return F.stack(gate_up_list, axis=0).reshape(
            [self.num_local_experts, -1, self.hidden_dim]
        )

    @property
    def down_proj(self) -> Tensor:
        down_proj = F.stack(
            [expert.down_proj.weight for expert in self.experts], axis=0
        )
        return down_proj

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (seq_len, hidden_dim)

        Returns:
            (seq_len, hidden_dim)
        """
        seq_len = x.shape[0]

        # Get the topk experts per token and their weights
        router_idx, router_weight = self.gate(x)
        router_idx = F.reshape(
            router_idx, [-1]
        )  # (seq_len * n_expert_per_token,)

        (
            token_expert_order,
            expert_start_indices,
            restore_token_order,
            expert_ids,
            expert_usage_stats,
        ) = moe_create_indices(
            F.cast(router_idx, DType.int32), self.num_experts
        )

        permutated_states = F.gather(
            x,
            F.cast(
                token_expert_order // self.num_experts_per_token, DType.int32
            ),
            axis=0,
        )

        if self.apply_router_weight_first:
            permutated_states = permutated_states * F.gather(
                router_weight.reshape([-1, 1]), token_expert_order, axis=0
            ).cast(x.dtype)

        gate_up_projs = grouped_matmul_ragged(
            permutated_states,
            self.gate_up_proj,
            expert_start_indices,
            expert_ids,
            expert_usage_stats.to(CPU()),
        )

        gate_up_projs = (
            F.silu(gate_up_projs[:, : self.moe_dim])
            * gate_up_projs[:, self.moe_dim :]
        )

        down_projs = grouped_matmul_ragged(
            gate_up_projs,
            self.down_proj,
            expert_start_indices,
            expert_ids,
            expert_usage_stats.to(CPU()),
        )

        down_projs = F.gather(down_projs, restore_token_order, axis=0).reshape(
            [seq_len, self.num_experts_per_token, -1]
        )

        if not self.apply_router_weight_first:
            # (seq_len, 1, n_expert) @ (seq_len, n_expert, hidden_dim) -> (seq_len, 1, hidden_dim)
            routed_expert_out = F.unsqueeze(router_weight, axis=1) @ down_projs
            routed_expert_out = F.squeeze(routed_expert_out, axis=1).cast(
                x.dtype
            )
        else:
            routed_expert_out = down_projs.transpose(1, 2)
            routed_expert_out = F.squeeze(
                F.sum(routed_expert_out, axis=2), axis=2
            ).cast(x.dtype)

        if self.has_shared_experts:
            routed_expert_out += self.shared_experts(x)

        return routed_expert_out
