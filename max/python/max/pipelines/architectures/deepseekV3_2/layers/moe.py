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
"""Mixture of Experts (MoE) module for DeepseekV3.2."""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn.comm.ep.ep_kernels import fused_silu
from max.nn.kernels import grouped_matmul_ragged, moe_create_indices
from max.nn.moe import MoE


class DeepseekV3_2MoE(MoE):
    """DeepSeek V3 uses the nn module MoE layer.
    This class implements changes needed for V3.2.

    DeepSeek modified MoE datatypes for V3 to V3.2 upgrade.
    Accumulations are performed in float32.
    Compare .astype calls to find the changes.

    DeepSeek V3 and V3.2 reference implementations:
    - V3: self.w2(F.silu(self.w1(x)) * self.w3(x))
    - V3.2: self.w2((F.silu(self.w1(x).float()) * self.w3(x).float()).type_as(x))
    """

    def __init__(self, *args, **kwargs):
        # minimize code paths, KISS principle
        assert not kwargs.get("apply_router_weight_first", True), (
            "apply_router_weight_first argument not supported"
        )
        assert kwargs.get("has_shared_experts", False), (
            "has_shared_experts argument not supported"
        )

        super().__init__(*args, **kwargs)

    def _ep_call(
        self,
        x: TensorValue,
        router_idx: TensorValue,
        router_weight: TensorValue,
    ) -> TensorValue:
        device_id = self.devices[0].id

        expert_inputs = self.ep_batch_manager.ep_dispatch(
            x, router_idx, device_id
        )

        gate_up_projs = grouped_matmul_ragged(
            expert_inputs[0],
            self.gate_up_proj,
            *expert_inputs[1:],
        ).cast(DType.float32)

        silu_out = fused_silu(gate_up_projs, expert_inputs[1]).cast(x.dtype)

        down_projs = grouped_matmul_ragged(
            silu_out,
            self.down_proj,
            *expert_inputs[1:],
        )

        routed_expert_out = self.ep_batch_manager.ep_combine(
            down_projs.cast(DType.float32), router_weight, device_id
        )

        routed_expert_out += self.shared_experts(x).cast(DType.float32)

        return routed_expert_out.cast(x.dtype)

    def __call__(self, x: TensorValue) -> TensorValue:
        """
        Args:
            x: (seq_len, hidden_dim)

        Returns:
            (seq_len, hidden_dim)
        """
        seq_len = x.shape[0]

        # Get the topk experts per token and their weights
        router_idx, router_weight = self.gate(x)

        if self._ep_batch_manager:
            return self._ep_call(
                x, ops.cast(router_idx, DType.int32), router_weight
            )

        router_idx = ops.reshape(
            router_idx, [-1]
        )  # (seq_len * n_expert_per_token,)

        (
            token_expert_order,
            expert_start_indices,
            restore_token_order,
            expert_ids,
            expert_usage_stats,
        ) = moe_create_indices(
            ops.cast(router_idx, DType.int32), self.num_experts
        )

        permutated_states = ops.gather(
            x,
            ops.cast(
                token_expert_order // self.num_experts_per_token, DType.int32
            ),
            axis=0,
        )

        gate_up_projs = grouped_matmul_ragged(
            permutated_states,
            self.gate_up_proj,
            expert_start_indices,
            expert_ids,
            expert_usage_stats.to(DeviceRef.CPU()),
        ).cast(DType.float32)

        gate_up_projs = (
            ops.silu(gate_up_projs[:, : self.moe_dim])
            * gate_up_projs[:, self.moe_dim :]
        ).cast(x.dtype)

        down_projs = grouped_matmul_ragged(
            gate_up_projs,
            self.down_proj,
            expert_start_indices,
            expert_ids,
            expert_usage_stats.to(DeviceRef.CPU()),
        )

        down_projs = (
            ops.gather(down_projs, restore_token_order, axis=0)
            .reshape([seq_len, self.num_experts_per_token, self.hidden_dim])
            .cast(DType.float32)
        )

        routed_expert_out = ops.unsqueeze(router_weight, axis=1) @ down_projs
        routed_expert_out = ops.squeeze(routed_expert_out, axis=1)

        routed_expert_out += self.shared_experts(x).cast(DType.float32)

        return routed_expert_out.cast(x.dtype)
