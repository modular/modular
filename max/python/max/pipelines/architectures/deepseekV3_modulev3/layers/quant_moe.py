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

"""Quantize-aware MoE Layer."""

from __future__ import annotations

from collections.abc import Callable

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.functional_kernels import (
    fused_silu,
    moe_create_indices,
)
from max.experimental.nn.common_layers.moe import MoEGate
from max.experimental.nn.sequential import ModuleList
from max.experimental.tensor import Tensor
from max.nn.quant_config import QuantConfig

from . import quant_ops
from .quant_linear import QuantizedMLP
from .quant_ops import QuantAwareTensor
from .quant_tensor import FP8BlockTensor


class QuantizedMoE(Module[[Tensor], Tensor]):
    """Mixture of Experts with quantize-aware expert weights."""

    gate: MoEGate
    experts: ModuleList
    shared_experts: QuantizedMLP | None

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        moe_dim: int,
        gate_cls: Callable[..., MoEGate] = MoEGate,
        has_shared_experts: bool = False,
        shared_experts_dim: int = 0,
        apply_router_weight_first: bool = False,
        quant_config: QuantConfig | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.apply_router_weight_first = apply_router_weight_first
        self.moe_dim = moe_dim
        self.quant_config = quant_config

        self.gate = gate_cls(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
        )

        self.shared_experts: QuantizedMLP | None = None
        if has_shared_experts:
            assert shared_experts_dim > 0
            self.shared_experts = QuantizedMLP(
                hidden_dim=hidden_dim,
                feed_forward_length=shared_experts_dim,
                quant_config=quant_config,
            )

        self.experts = ModuleList(
            [
                QuantizedMLP(
                    hidden_dim=hidden_dim,
                    feed_forward_length=moe_dim,
                    quant_config=quant_config,
                )
                for _ in range(num_experts)
            ]
        )

    def _stack_proj(
        self, per_expert: list[QuantAwareTensor]
    ) -> QuantAwareTensor:
        """Stack per-expert weights into a single leading-dim weight."""
        first = per_expert[0]
        if isinstance(first, FP8BlockTensor):
            return FP8BlockTensor(
                data=F.stack([w.data for w in per_expert], axis=0),  # type: ignore[union-attr]
                scale_inv=F.stack(
                    [w.scale_inv for w in per_expert],  # type: ignore[union-attr]
                    axis=0,
                ),
                block_size=first.block_size,
            )
        return F.stack(per_expert, axis=0)

    @property
    def gate_up_proj(self) -> QuantAwareTensor:
        """Stacked ``[gate, up]`` expert weights."""
        per_expert: list[QuantAwareTensor] = []
        for expert in self.experts:
            assert isinstance(expert, QuantizedMLP)
            per_expert.append(
                quant_ops.concat_weights(
                    expert.gate_proj.weight, expert.up_proj.weight, axis=0
                )
            )
        return self._stack_proj(per_expert)

    @property
    def down_proj(self) -> QuantAwareTensor:
        """Stacked down-projection expert weights (bf16 or FP8)."""
        per_expert: list[QuantAwareTensor] = []
        for expert in self.experts:
            assert isinstance(expert, QuantizedMLP)
            per_expert.append(expert.down_proj.weight)
        return self._stack_proj(per_expert)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.shape[0]

        router_idx, router_weight = self.gate(x)
        router_idx = F.reshape(router_idx, [-1])

        (
            token_expert_order,
            expert_start_indices,
            restore_token_order,
            expert_ids,
            expert_usage_stats,
        ) = moe_create_indices(
            F.cast(router_idx, DType.int32), self.num_experts
        )

        permuted_states = F.gather(
            x,
            F.cast(
                token_expert_order // self.num_experts_per_token, DType.int32
            ),
            axis=0,
        )

        if self.apply_router_weight_first:
            permuted_states = permuted_states * F.gather(
                router_weight.reshape([-1, 1]), token_expert_order, axis=0
            ).cast(x.dtype)

        gate_up_projs = quant_ops.grouped_matmul(
            permuted_states,
            self.gate_up_proj,
            expert_start_indices,
            expert_ids,
            expert_usage_stats.to(CPU()),
        )
        gate_up_projs = fused_silu(gate_up_projs, expert_start_indices)

        down_projs = quant_ops.grouped_matmul(
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
            routed_expert_out = F.unsqueeze(router_weight, axis=1) @ down_projs
            routed_expert_out = F.squeeze(routed_expert_out, axis=1).cast(
                x.dtype
            )
        else:
            routed_expert_out = down_projs.transpose(1, 2)
            routed_expert_out = F.squeeze(
                F.sum(routed_expert_out, axis=2), axis=2
            ).cast(x.dtype)

        if self.shared_experts is not None:
            routed_expert_out = routed_expert_out + self.shared_experts(x)

        return routed_expert_out
