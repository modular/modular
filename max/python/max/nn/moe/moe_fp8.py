# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""A generalized Mixture of Experts (MoE) module."""

from __future__ import annotations

from max.dtype import DType
from max.graph import (
    DeviceRef,
    TensorValue,
    ops,
)

from ..kernels import (
    grouped_dynamic_scaled_fp8_matmul,
    moe_create_indices,
    quantize_dynamic_scaled_float8,
)
from .moe import MoE


class MoEFp8(MoE):
    """Implementation of Mixture of Experts (MoE) with FP8 quantization."""

    @property
    def gate_up_proj_scales(self) -> TensorValue:
        assert self.float8_config is not None
        assert self.float8_config.weight_scale.block_size is not None
        assert self.float8_config.weight_scale.block_size == (128, 128), (
            "Only support block_size=[128, 128] for weights."
        )
        gate_proj_scales_list = [
            expert.gate_proj.weight_scale for expert in self.experts
        ]
        up_proj_scales_list = [
            expert.up_proj.weight_scale for expert in self.experts
        ]

        gate_up_proj_scales_list: list[TensorValue] = []
        for tensors in zip(
            gate_proj_scales_list, up_proj_scales_list, strict=True
        ):
            gate_up_proj_scales_list.extend(tensors)

        return ops.stack(gate_up_proj_scales_list, axis=0).reshape(
            [
                self.num_local_experts,
                -1,
                self.hidden_dim
                // self.float8_config.weight_scale.block_size[1],
            ]
        )

    @property
    def down_proj_scales(self) -> TensorValue:
        down_proj_scales = ops.stack(
            [expert.down_proj.weight_scale for expert in self.experts], axis=0
        )
        return down_proj_scales

    def _ep_call(
        self,
        x: TensorValue,
        router_idx: TensorValue,
        router_weight: TensorValue,
    ) -> TensorValue:
        assert self.float8_config is not None
        assert self.float8_config.input_scale.block_size is not None
        token_group_size = self.float8_config.input_scale.block_size[1]
        device_id = self.devices[0].id
        self.ep_batch_manager.ep_dispatch(x, router_idx, device_id)
        expert_inputs = self.ep_batch_manager.ep_dispatch_cb(device_id)

        gate_up_projs = grouped_dynamic_scaled_fp8_matmul(
            expert_inputs[0],
            self.gate_up_proj,
            expert_inputs[1],
            self.gate_up_proj_scales,
            expert_inputs[2],
            expert_inputs[3],
            expert_inputs[4],
            self.float8_config.input_scale,
            self.float8_config.weight_scale,
        )

        gate_up_projs = (
            ops.silu(gate_up_projs[:, : self.moe_dim])
            * gate_up_projs[:, self.moe_dim :]
        )
        gate_up_projs_fp8, gate_up_projs_scales = (
            quantize_dynamic_scaled_float8(
                gate_up_projs,
                self.float8_config.input_scale,
                self.float8_config.weight_scale,
                group_size_or_per_token=token_group_size,
                out_type=self.dtype,
                scales_type=self.float8_config.weight_scale.dtype,
            )
        )

        down_projs = grouped_dynamic_scaled_fp8_matmul(
            gate_up_projs_fp8,
            self.down_proj,
            gate_up_projs_scales,
            self.down_proj_scales,
            expert_inputs[2],
            expert_inputs[3],
            expert_inputs[4],
            self.float8_config.input_scale,
            self.float8_config.weight_scale,
        )

        self.ep_batch_manager.ep_combine(down_projs, device_id)
        combined_down_projs = self.ep_batch_manager.ep_combine_cb(device_id)

        routed_expert_out = (
            ops.unsqueeze(router_weight, axis=1) @ combined_down_projs
        )
        routed_expert_out = ops.squeeze(routed_expert_out, axis=1).cast(x.dtype)

        if self.has_shared_experts:
            routed_expert_out += self.shared_experts(x)

        return routed_expert_out

    def __call__(self, x: TensorValue) -> TensorValue:
        """
        Args:
            x: (seq_len, hidden_dim)

        Returns:
            (seq_len, hidden_dim)
        """
        assert self.float8_config is not None
        assert self.float8_config.input_scale.block_size is not None
        token_group_size = self.float8_config.input_scale.block_size[1]
        assert not self.apply_router_weight_first, (
            "apply_router_weight_first must be False for MoE with FP8 quantization"
        )
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

        permutated_states_fp8, permutated_states_scales = (
            quantize_dynamic_scaled_float8(
                permutated_states,
                self.float8_config.input_scale,
                self.float8_config.weight_scale,
                group_size_or_per_token=token_group_size,
                out_type=self.dtype,
                scales_type=self.float8_config.weight_scale.dtype,
            )
        )

        gate_up_projs = grouped_dynamic_scaled_fp8_matmul(
            permutated_states_fp8,
            self.gate_up_proj,
            permutated_states_scales,
            self.gate_up_proj_scales,
            expert_start_indices,
            expert_ids,
            expert_usage_stats.to(DeviceRef.CPU()),
            self.float8_config.input_scale,
            self.float8_config.weight_scale,
        )

        gate_up_projs = (
            ops.silu(gate_up_projs[:, : self.moe_dim])
            * gate_up_projs[:, self.moe_dim :]
        )
        gate_up_projs_fp8, gate_up_projs_scales = (
            quantize_dynamic_scaled_float8(
                gate_up_projs,
                self.float8_config.input_scale,
                self.float8_config.weight_scale,
                group_size_or_per_token=token_group_size,
                out_type=self.dtype,
                scales_type=self.float8_config.weight_scale.dtype,
            )
        )

        down_projs = grouped_dynamic_scaled_fp8_matmul(
            gate_up_projs_fp8,
            self.down_proj,
            gate_up_projs_scales,
            self.down_proj_scales,
            expert_start_indices,
            expert_ids,
            expert_usage_stats.to(DeviceRef.CPU()),
            self.float8_config.input_scale,
            self.float8_config.weight_scale,
        )

        down_projs = ops.gather(
            down_projs, restore_token_order, axis=0
        ).reshape([seq_len, self.num_experts_per_token, down_projs.shape[-1]])

        # (seq_len, 1, n_expert) @ (seq_len, n_expert, hidden_dim) -> (seq_len, 1, hidden_dim)
        routed_expert_out = ops.unsqueeze(router_weight, axis=1) @ down_projs
        routed_expert_out = ops.squeeze(routed_expert_out, axis=1).cast(x.dtype)

        if self.has_shared_experts:
            routed_expert_out += self.shared_experts(x)

        return routed_expert_out
