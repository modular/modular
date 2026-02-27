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
from max.nn.kernels import moe_create_indices
from max.nn.moe import MoEQuantized
from max.nn.moe.quant_strategy import (
    silu_gate,
)


class DeepseekV3_2MoE(MoEQuantized):
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
        """Executes the expert-parallel quantized MoE path."""
        strategy = self._strategy()
        nvfp4 = self._nvfp4_scales() if self._is_nvfp4 else None

        device_id = self.devices[0].id
        expert_inputs = self.ep_batch_manager.ep_dispatch(
            x, router_idx, device_id, nvfp4.gate_up_input if nvfp4 else None
        )

        gate_up_scales, down_scales = strategy.prepare_weight_scales(
            self.gate_up_proj_scales, self.down_proj_scales, x.device
        )

        gate_up_projs = strategy.grouped_matmul(
            self.gate_up_proj,
            gate_up_scales,
            expert_scales=nvfp4.gate_up_expert if nvfp4 else None,
            tokens_padded_per_expert=True,
            expert_inputs=expert_inputs,
        )
        # V3.2: Cast to float32 at the silu
        # https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/87e509a2e5a100d221c97df52c6e8be7835f0057/inference/model.py#L744
        gate_up_projs = gate_up_projs.cast(DType.float32)

        down_in, silu_scales = strategy.fused_silu_quantize(
            gate_up_projs,
            input_scales=nvfp4.down_input if nvfp4 else None,
            expert_inputs=expert_inputs,
        )

        down_inputs = (down_in, silu_scales) + expert_inputs[2:]
        down_projs = strategy.grouped_matmul(
            self.down_proj,
            down_scales,
            expert_scales=nvfp4.down_expert if nvfp4 else None,
            tokens_padded_per_expert=True,
            expert_inputs=down_inputs,
        )

        routed_expert_out = self.ep_batch_manager.ep_combine(
            down_projs, router_weight, device_id
        )
        if self.has_shared_experts and not self._fused_shared_expert:
            routed_expert_out += self.shared_experts(x)

        return routed_expert_out.cast(x.dtype)

    def __call__(self, x: TensorValue) -> TensorValue:
        """Runs quantized MoE routing and expert computation."""
        strategy = self._strategy()
        nvfp4 = self._nvfp4_scales() if self._is_nvfp4 else None

        assert not self.apply_router_weight_first, (
            "apply_router_weight_first must be False for quantized MoE"
        )

        router_idx, router_weight = self.gate(x)
        if self._ep_batch_manager:
            return self._ep_call(
                x, ops.cast(router_idx, DType.int32), router_weight
            )

        router_idx = ops.reshape(router_idx, [-1])
        seq_len = x.shape[0]

        (
            token_order,
            expert_start,
            restore_order,
            expert_ids,
            usage_stats,
        ) = moe_create_indices(
            ops.cast(router_idx, DType.int32), self.num_experts
        )

        permutated_states = ops.gather(
            x,
            ops.cast(token_order // self.num_experts_per_token, DType.int32),
            axis=0,
        )

        permuted_quant, permuted_scales = strategy.quantize(
            permutated_states,
            self._token_group_size,
            nvfp4.gate_up_input if nvfp4 else None,
        )

        gate_up_scales, down_scales = strategy.prepare_weight_scales(
            self.gate_up_proj_scales,
            self.down_proj_scales,
            permutated_states.device,
        )

        expert_inputs: tuple[TensorValue, ...] = (
            permuted_quant,
            permuted_scales,
            expert_start,
            expert_ids,
            usage_stats.to(DeviceRef.CPU()),
        )

        if nvfp4:
            a_scale_offsets = ops.constant(
                0, dtype=DType.uint32, device=x.device
            ).broadcast_to([expert_ids.shape[0]])
            expert_inputs = (
                *expert_inputs[:3],
                a_scale_offsets,
                *expert_inputs[3:],
            )

        gate_up_projs = strategy.grouped_matmul(
            self.gate_up_proj,
            gate_up_scales,
            expert_scales=nvfp4.gate_up_expert if nvfp4 else None,
            expert_inputs=expert_inputs,
        )
        # V3.2: Cast to float32 at the silu
        # https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/87e509a2e5a100d221c97df52c6e8be7835f0057/inference/model.py#L744
        gate_up_projs = gate_up_projs.cast(DType.float32)
        gate_up_projs = silu_gate(gate_up_projs, self.moe_dim)
        gate_up_quant, gate_up_scales = strategy.quantize(
            gate_up_projs,
            self._token_group_size,
            nvfp4.down_input if nvfp4 else None,
        )

        down_inputs = (gate_up_quant, gate_up_scales) + expert_inputs[2:]

        down_projs = strategy.grouped_matmul(
            self.down_proj,
            down_scales,
            expert_scales=nvfp4.down_expert if nvfp4 else None,
            expert_inputs=down_inputs,
        )

        down_projs = ops.gather(down_projs, restore_order, axis=0).reshape(
            [seq_len, self.num_experts_per_token, down_projs.shape[-1]]
        )

        routed_expert_out = ops.unsqueeze(router_weight, axis=1) @ down_projs
        routed_expert_out = ops.squeeze(routed_expert_out, axis=1)

        if self.has_shared_experts:
            routed_expert_out += self.shared_experts(x).cast(DType.float32)

        return routed_expert_out.cast(x.dtype)
