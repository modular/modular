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
"""Mixture of Experts with FP8/NVFP4 quantization."""

from __future__ import annotations

from typing import TypeVar

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops

from ..kernels import moe_create_indices
from .moe import MoE
from .quant_strategy import (
    Fp8Strategy,
    Mxfp4Strategy,
    Nvfp4Scales,
    Nvfp4Strategy,
    QuantStrategy,
    gated_activation,
)

_T = TypeVar("_T")


def _scalar_max(t: TensorValue) -> TensorValue:
    """Reduces a tensor to a rank-0 scalar max value."""
    return ops.max(t).reshape([])


class MoEQuantized(MoE):
    """Mixture of Experts with FP8 or NVFP4 quantization."""

    @property
    def _fused_shared_expert(self) -> bool:
        """Whether shared expert is fused into expert list."""
        return bool(
            self._ep_batch_manager
            and self.ep_batch_manager.config.fused_shared_expert
        )

    def _strategy(self) -> QuantStrategy:
        """Selects the quantization strategy for this MoE."""
        assert self.quant_config is not None
        if self.quant_config.is_nvfp4:
            return Nvfp4Strategy(self.quant_config, self.dtype)
        elif self.quant_config.is_mxfp4:
            return Mxfp4Strategy(self.quant_config, self.dtype)
        return Fp8Strategy(self.quant_config, self.dtype)

    @property
    def _token_group_size(self) -> int:
        """Returns the activation token-group size for quantization."""
        assert self.quant_config is not None
        assert self.quant_config.input_scale.block_size is not None
        return self.quant_config.input_scale.block_size[1]

    def _with_shared_expert(
        self, values: list[_T], shared: _T | None
    ) -> list[_T]:
        """Prepends shared expert value if fused shared expert is enabled."""
        if self._fused_shared_expert and shared is not None:
            assert self.has_shared_experts
            return [shared] + values
        return values

    def _nvfp4_scales(self) -> Nvfp4Scales:
        """Collects NVFP4 input and expert scales for matmuls."""
        gate_up_input = self._collect_input_scale("gate_proj", collect_all=True)
        down_input = self._collect_input_scale("down_proj")

        # For gate/up projs, current EP communication kernels only support one
        # global input scale for all experts, hence we use the max input scale
        # across all experts.
        gate_up_max_scale = ops.max(gate_up_input, axis=0)
        gate_up_input = ops.broadcast_to(gate_up_max_scale, gate_up_input.shape)
        local_gate_up_input = ops.broadcast_to(
            gate_up_max_scale, down_input.shape
        )

        return Nvfp4Scales(
            gate_up_input=gate_up_input,
            down_input=down_input,
            gate_up_expert=self._collect_scale_2("gate_proj")
            * local_gate_up_input,
            down_expert=self._collect_scale_2("down_proj") * down_input,
        )

    def _collect_scale_2(self, proj_name: str) -> TensorValue:
        """Stacks per-expert secondary scales for NVFP4 kernels."""
        scales = [getattr(e, proj_name).weight_scale_2 for e in self.experts]
        shared_scale = (
            getattr(self.shared_experts, proj_name).weight_scale_2
            if self.has_shared_experts
            else None
        )
        scales = self._with_shared_expert(scales, shared_scale)
        return ops.stack(scales, axis=0)

    def _collect_input_scale(
        self, proj_name: str, collect_all: bool = False
    ) -> TensorValue:
        """Stacks per-expert input scales for NVFP4 kernels."""
        expert_collect = self._all_experts if collect_all else self.experts
        scales = [getattr(e, proj_name).input_scale for e in expert_collect]
        shared_scale = (
            getattr(self.shared_experts, proj_name).input_scale
            if self.has_shared_experts
            else None
        )
        scales = self._with_shared_expert(scales, shared_scale)
        return ops.stack(scales, axis=0)

    @property
    def gate_up_proj_scales(self) -> TensorValue:
        """Returns stacked gate/up weight scales for grouped matmul."""
        assert self.quant_config is not None
        assert self.quant_config.weight_scale.block_size is not None
        if not self.quant_config.is_fp4:
            assert self.quant_config.weight_scale.block_size == (128, 128), (
                "Only support block_size=[128, 128] for weights."
            )

        gate_scales = [e.gate_proj.weight_scale for e in self.experts]
        up_scales = [e.up_proj.weight_scale for e in self.experts]
        gate_shared = (
            self.shared_experts.gate_proj.weight_scale
            if self.has_shared_experts
            else None
        )
        up_shared = (
            self.shared_experts.up_proj.weight_scale
            if self.has_shared_experts
            else None
        )
        gate_scales = self._with_shared_expert(gate_scales, gate_shared)
        up_scales = self._with_shared_expert(up_scales, up_shared)

        # Interleave gate and up scales: [g0, u0, g1, u1, ...]
        interleaved = [
            s for pair in zip(gate_scales, up_scales, strict=True) for s in pair
        ]

        scale_k_dim = gate_scales[0].shape[-1]
        if self.shard_devices:
            shard = ops.shard_and_stack(
                interleaved, devices=self.shard_devices
            )[self.shard_index]
        else:
            shard = ops.stack(interleaved, axis=0)

        return shard.reshape([len(gate_scales), -1, scale_k_dim]).to(
            self.devices[0]
        )

    @property
    def down_proj_scales(self) -> TensorValue:
        """Returns stacked down-projection weight scales."""
        scales = [e.down_proj.weight_scale for e in self.experts]
        down_shared = (
            self.shared_experts.down_proj.weight_scale
            if self.has_shared_experts
            else None
        )
        scales = self._with_shared_expert(scales, down_shared)

        if self.shard_devices:
            devices = [DeviceRef.CPU()] * len(self.shard_devices)
            return ops.shard_and_stack(scales, devices=devices, axis=-1)[
                self.shard_index
            ].to(self.devices[0])
        return ops.stack(scales, axis=0).to(self.devices[0])

    @property
    def _is_nvfp4(self) -> bool:
        """Whether the current quant config uses NVFP4."""
        return self.quant_config is not None and self.quant_config.is_nvfp4

    def _ep_dispatch_input_scales(self) -> TensorValue | None:
        """Returns NVFP4 input scales for EP dispatch, or ``None``."""
        if self._is_nvfp4:
            return self._nvfp4_scales().gate_up_input
        return None

    def _local_ep_compute(
        self,
        expert_inputs: tuple[TensorValue, ...],
        x: TensorValue,
        estimated_total_m: TensorValue,
    ) -> TensorValue:
        """Runs quantized local expert matmuls on dispatched tokens."""
        # TODO: swiglu_limit is not supported here because
        # fused_silu_quantize fuses silu + multiply + quantize into one
        # kernel, leaving no place to insert the clamp.  For NVFP4 the
        # kernel also produces expert-aware padded scales that cannot be
        # replicated with unfused ops.
        if self.swiglu_limit > 0:
            raise ValueError(
                "swiglu_limit is not supported in the expert-parallel"
                " quantized MoE path because fused_silu_quantize does not"
                " support clamping."
            )
        strategy = self._strategy()
        nvfp4 = self._nvfp4_scales() if self._is_nvfp4 else None

        gate_up_scales, down_scales = strategy.prepare_weight_scales(
            self.gate_up_proj_scales, self.down_proj_scales, x.device
        )

        gate_up = strategy.grouped_matmul(
            self.gate_up_proj,
            gate_up_scales,
            expert_scales=nvfp4.gate_up_expert if nvfp4 else None,
            tokens_padded_per_expert=True,
            expert_inputs=expert_inputs,
            estimated_total_m=estimated_total_m,
        )

        down_in, silu_scales = strategy.fused_silu_quantize(
            gate_up,
            input_scales=nvfp4.down_input if nvfp4 else None,
            expert_inputs=expert_inputs,
        )

        down_inputs = (down_in, silu_scales) + expert_inputs[2:]
        return strategy.grouped_matmul(
            self.down_proj,
            down_scales,
            expert_scales=nvfp4.down_expert if nvfp4 else None,
            tokens_padded_per_expert=True,
            expert_inputs=down_inputs,
            estimated_total_m=estimated_total_m,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        """Runs quantized MoE routing and expert computation."""
        if self._ep_batch_manager:
            raise ValueError(
                "Use forward_moe_sharded_layers for expert-parallel inference "
                "instead of calling MoEQuantized directly."
            )

        strategy = self._strategy()
        nvfp4 = self._nvfp4_scales() if self._is_nvfp4 else None

        assert not self.apply_router_weight_first, (
            "apply_router_weight_first must be False for quantized MoE"
        )

        router_idx, router_weight = self.gate(x)

        router_idx = ops.reshape(router_idx, [-1])
        seq_len = x.shape[0]

        create_indices_result = moe_create_indices(
            ops.cast(router_idx, DType.int32),
            self.num_experts,
            needs_scales_offset=self._is_nvfp4,
        )
        token_order, expert_start, restore_order, expert_ids, usage_stats = (
            create_indices_result[:5]
        )
        scales_offset = create_indices_result[5] if nvfp4 else None

        if self.pre_expert_norm is not None:
            x = self.pre_expert_norm(x)

        permuted = ops.gather(
            x,
            ops.cast(token_order // self.num_experts_per_token, DType.int32),
            axis=0,
        )

        total_m = ops.shape_to_tensor(permuted.shape)[0].cast(DType.uint32)

        if nvfp4:
            assert scales_offset is not None
            permuted_quant, permuted_scales = strategy.grouped_quantize(
                permuted,
                self._token_group_size,
                nvfp4.gate_up_input,
                expert_start,
                scales_offset,
                expert_ids,
            )
        else:
            permuted_quant, permuted_scales = strategy.quantize(
                permuted,
                self._token_group_size,
            )

        gate_up_scales, down_scales = strategy.prepare_weight_scales(
            self.gate_up_proj_scales, self.down_proj_scales, permuted.device
        )

        expert_inputs: tuple[TensorValue, ...] = (
            permuted_quant,
            permuted_scales,
            expert_start,
            expert_ids,
            usage_stats,
        )

        if nvfp4:
            assert scales_offset is not None
            expert_inputs = (
                *expert_inputs[:3],
                scales_offset,
                *expert_inputs[3:],
            )

        gate_up = strategy.grouped_matmul(
            self.gate_up_proj,
            gate_up_scales,
            expert_scales=nvfp4.gate_up_expert if nvfp4 else None,
            expert_inputs=expert_inputs,
            estimated_total_m=total_m,
        )

        if self.gate_activation == "silu" and self.swiglu_limit > 0:
            gate = ops.silu(gate_up[:, : self.moe_dim])
            up = gate_up[:, self.moe_dim :]
            lim = ops.constant(
                self.swiglu_limit, gate.dtype, device=gate.device
            )
            neg_lim = ops.constant(
                -self.swiglu_limit, up.dtype, device=up.device
            )
            gate = ops.min(gate, lim)
            up = ops.min(ops.max(up, neg_lim), lim)
            gate_up = gate * up
        else:
            gate_up = gated_activation(
                gate_up, self.moe_dim, self.gate_activation
            )

        if nvfp4:
            assert scales_offset is not None
            gate_up_quant, gate_up_scales = strategy.grouped_quantize(
                gate_up,
                self._token_group_size,
                nvfp4.down_input,
                expert_start,
                scales_offset,
                expert_ids,
            )
        else:
            gate_up_quant, gate_up_scales = strategy.quantize(
                gate_up,
                self._token_group_size,
            )

        down_inputs = (gate_up_quant, gate_up_scales) + expert_inputs[2:]

        down = strategy.grouped_matmul(
            self.down_proj,
            down_scales,
            expert_scales=nvfp4.down_expert if nvfp4 else None,
            expert_inputs=down_inputs,
            estimated_total_m=total_m,
        )

        down = ops.gather(down, restore_order, axis=0).reshape(
            [seq_len, self.num_experts_per_token, down.shape[-1]]
        )

        out = ops.unsqueeze(router_weight, axis=1) @ down
        out = ops.squeeze(out, axis=1).cast(x.dtype)

        if self.has_shared_experts:
            out += self.shared_experts(x)

        return out
