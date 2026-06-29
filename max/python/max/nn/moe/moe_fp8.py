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

from ..comm.ep.ep_kernels import ep_mxfp4_max_padded_m, fused_silu
from ..kernels import _is_pre_sm100_nvidia_gpu, moe_create_indices
from .moe import MoE
from .quant_strategy import (
    Fp8Strategy,
    Mxfp4Strategy,
    Nvfp4DequantStrategy,
    Nvfp4Scales,
    NvMxf4f8Strategy,
    QuantStrategy,
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
        if self._uses_nvidia_block_scaled_ep_layout:
            return NvMxf4f8Strategy(self.quant_config, self.dtype)
        if self.quant_config.is_nvfp4:
            if _is_pre_sm100_nvidia_gpu():
                if self._ep_batch_manager:
                    raise NotImplementedError(
                        "Expert-parallel NVFP4 is not supported on"
                        " pre-SM100 GPUs (EP dispatch quantizes"
                        " activations to FP4)."
                    )
                # Pre-Blackwell fallback: no FP4 tensor cores -- dequantize
                # expert weights to BF16 and use the BF16 grouped matmul.
                return Nvfp4DequantStrategy(self.quant_config, self.dtype)
            # Native Blackwell NVFP4 path (main renamed Nvfp4Strategy ->
            # NvMxf4f8Strategy while this branch was open).
            return NvMxf4f8Strategy(self.quant_config, self.dtype)
        elif self.quant_config.is_mxfp4:
            return Mxfp4Strategy(
                self.quant_config,
                self.dtype,
                preshuffled_b=self.quant_config.mxfp4_preshuffled_b,
            )
        return Fp8Strategy(self.quant_config, self.dtype)

    def configure_ep_scale_fusion(self, dispatch_supports_fold: bool) -> None:
        """Enable the MXFP4 EP A-scale preshuffle fold on the shared EP config
        so the dispatch ops emit slot-sized scales. Must run BEFORE the dispatch
        op (the dispatch output shape depends on this flag); the EP forward
        driver calls it once per layer before dispatch.

        The fold writes the up-proj (KS224, ``ep_wait``) and down-proj (KS64,
        ``fused_silu``) A-scale directly into the grouped-matmul slot layout,
        dropping the standalone preshuffle kernels from the decode critical
        path. It is enabled whenever this is an MXFP4 preshuffled-B EP layer and
        the selected dispatch path wires the fold. It implements standard SiLU
        only, so OAI-clamped SwiGLU (e.g. gpt-oss) is excluded and routed
        through the generic quantize path.

        Args:
            dispatch_supports_fold: Whether the dispatch path selected for this
                forward threads the fold params. The multi-device single-op
                ``call_distributed_ep_dispatch`` does not, so the fold stays off
                there and the standalone preshuffle runs.
        """
        if self._ep_batch_manager is None:
            return
        self.ep_batch_manager.config.mxfp4_a_scales_preshuffled = bool(
            dispatch_supports_fold
            and self.quant_config is not None
            and self.quant_config.is_mxfp4
            and self.quant_config.mxfp4_preshuffled_b
            and not self.use_swigluoai
        )

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

        if _is_pre_sm100_nvidia_gpu():
            # Dequant fallback: activations are never quantized and the
            # BF16 grouped matmul has no scale epilogue; the per-tensor
            # weight scale is folded into the dequant block scales instead
            # (no activation input-scale factor anywhere).
            return Nvfp4Scales(
                gate_up_input=gate_up_input,
                down_input=down_input,
                gate_up_expert=self._collect_scale_2("gate_proj"),
                down_expert=self._collect_scale_2("down_proj"),
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
            if self.has_shared_experts and self._shared_experts_use_quant
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
            if self.has_shared_experts and self._shared_experts_use_quant
            else None
        )
        scales = self._with_shared_expert(scales, shared_scale)
        return ops.stack(scales, axis=0)

    @property
    def gate_up_proj_scales(self) -> TensorValue:
        """Returns stacked gate/up weight scales for grouped matmul."""
        assert self.quant_config is not None
        assert self.quant_config.weight_scale.block_size is not None
        if not (self.quant_config.is_fp4 or self.quant_config.is_mxfp8):
            assert self.quant_config.weight_scale.block_size == (128, 128), (
                "Only support block_size=[128, 128] for weights."
            )

        gate_scales = [e.gate_proj.weight_scale for e in self.experts]
        up_scales = [e.up_proj.weight_scale for e in self.experts]
        gate_shared = (
            self.shared_experts.gate_proj.weight_scale
            if self.has_shared_experts and self._shared_experts_use_quant
            else None
        )
        up_shared = (
            self.shared_experts.up_proj.weight_scale
            if self.has_shared_experts and self._shared_experts_use_quant
            else None
        )
        gate_scales = self._with_shared_expert(gate_scales, gate_shared)
        up_scales = self._with_shared_expert(up_scales, up_shared)

        scale_k_dim = gate_scales[0].shape[-1]

        # Interleave gate and up scales: [g0, u0, g1, u1, ...]
        interleaved = [
            s for pair in zip(gate_scales, up_scales, strict=True) for s in pair
        ]

        if self.shard_devices:
            shard = ops.shard_and_stack(
                interleaved, devices=self.shard_devices
            )[self.shard_index]
        else:
            shard = ops.stack(interleaved, axis=0)

        # Matching sigma-permutation when fused SwiGLU+NVFP4 is enabled.
        # The stacked [2E, scale_m, scale_k] tensor splits to
        # [E, 2, scale_m, scale_k], then permute axes 1,2 → collapse to
        # [E, 2*scale_m, scale_k] with rows row-interleaved (g_0, u_0, ...).
        # This sits before NvMxf4f8Strategy.prepare_weight_scales lifts to the
        # 5D tcgen05 layout the kernel expects.
        if self._uses_fused_swiglu_layout():
            shard = shard.reshape([len(gate_scales), 2, -1, scale_k_dim])
            shard = ops.permute(shard, [0, 2, 1, 3])
            return shard.reshape([len(gate_scales), -1, scale_k_dim]).to(
                self.devices[0]
            )

        return shard.reshape([len(gate_scales), -1, scale_k_dim]).to(
            self.devices[0]
        )

    @property
    def down_proj_scales(self) -> TensorValue:
        """Returns stacked down-projection weight scales."""
        scales = [e.down_proj.weight_scale for e in self.experts]
        down_shared = (
            self.shared_experts.down_proj.weight_scale
            if self.has_shared_experts and self._shared_experts_use_quant
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

    @property
    def _uses_nvidia_block_scaled_ep_layout(self) -> bool:
        """Whether local expert inputs include NVIDIA scale offsets."""
        return self.quant_config is not None and (
            self.quant_config.is_nvfp4 or self.quant_config.is_mxfp8
        )

    def _can_fuse_swiglu_nvfp4(self) -> bool:
        """Whether the fused SwiGLU+NVFP4 grouped matmul kernel should fire.

        Gated on the NVFP4 :class:`QuantConfig` flag,
        ``gated_activation_fn is None`` (the kernel cannot run a custom
        activation), and an active expert-parallel batch manager. The
        ``MAX_DISABLE_FUSED_SWIGLU_NVFP4=1`` env-var kill-switch is read
        at :class:`QuantConfig` setup time (see
        ``max/python/max/pipelines/lib/quant.py``), which flips the flag
        so the model's ``gate_up_proj`` sigma-permutation stays consistent
        with the kernel choice.

        SM100 device-arch gating is handled by the kernel's own dispatch.
        TP-MoE would break the sigma-permuted layout, so a future TP-MoE
        consumer must update the sharding strategy before relaxing the EP
        check.
        """
        return (
            self.quant_config is not None
            and (self.quant_config.is_nvfp4 or self.quant_config.is_mxfp8)
            and self._uses_fused_swiglu_layout()
        )

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
        if self.gated_activation_fn is not None:
            raise ValueError(
                "Custom gated_activation_fn is not supported in the EP"
                " quantized path due to a specialized fused kernel."
            )
        strategy = self._strategy()
        nvfp4 = self._nvfp4_scales() if self._is_nvfp4 else None

        gate_up_scales, down_scales = strategy.prepare_weight_scales(
            self.gate_up_proj_scales, self.down_proj_scales, x.device
        )

        # For the MXFP4 preb EP path, the producers write the
        # grouped-matmul A-scale directly into the matmul's per-expert slot
        # layout, so the standalone preshuffle kernels are dropped.  `ep_wait`
        # does this for the up/gate proj (KS224) and `fused_silu` for the down
        # proj (KS64); both share the SAME graph-build-time `max_padded_M`
        # (single source of truth — the dispatch producer wrote the up-proj
        # scales with it, and the matmul reader MUST use the same constant).
        # Read the flag the EP forward driver already resolved via
        # `configure_ep_scale_fusion` (single source of truth) so the matmul
        # reader and the dispatch producer agree on the slot layout.
        mxfp4_ep_scale_fusion = bool(
            self._ep_batch_manager
            and self.ep_batch_manager.config.mxfp4_a_scales_preshuffled
        )
        mxfp4_ep_max_padded_m = (
            ep_mxfp4_max_padded_m(self.ep_batch_manager.config)
            if mxfp4_ep_scale_fusion
            else 0
        )
        # The up-proj reads its A-scale from the dispatched tokens, which
        # `ep_wait` wrote in slot layout when the fusion is on.
        up_a_scales_preshuffled = (
            isinstance(strategy, Mxfp4Strategy) and mxfp4_ep_scale_fusion
        )

        if self._can_fuse_swiglu_nvfp4():
            assert isinstance(strategy, NvMxf4f8Strategy)
            down_in, silu_scales = strategy.grouped_matmul_swiglu(
                self.gate_up_proj,
                gate_up_scales,
                expert_scales=nvfp4.gate_up_expert if nvfp4 else None,
                input_scales=nvfp4.down_input if nvfp4 else None,
                expert_inputs=expert_inputs,
                estimated_total_m=estimated_total_m,
                use_swigluoai=self.use_swigluoai,
                swiglu_alpha=self.swiglu_alpha,
                swiglu_limit=self.swiglu_limit,
            )
        else:
            if isinstance(strategy, Mxfp4Strategy) and not self.use_swigluoai:
                # MXFP4 EP fold: ep_wait writes the up-proj A-scale
                # (KS224) and fused_silu the down-proj A-scale (KS64) directly
                # into the grouped-matmul slot layout. This covers standard
                # SiLU only; OAI-clamped SwiGLU (e.g. gpt-oss) is excluded in
                # `configure_ep_scale_fusion` and handled by the generic path
                # below.
                gate_up = strategy.grouped_matmul(
                    self.gate_up_proj,
                    gate_up_scales,
                    expert_inputs=expert_inputs,
                    estimated_total_m=estimated_total_m,
                    # KS224: ep_wait wrote the up-proj A-scale in slot layout.
                    a_scales_preshuffled=up_a_scales_preshuffled,
                    a_scales_max_padded_m=mxfp4_ep_max_padded_m,
                )
                down_in, silu_scales = strategy.fused_silu_quantize(
                    gate_up,
                    input_scales=None,
                    expert_inputs=expert_inputs,
                    max_padded_M=mxfp4_ep_max_padded_m,
                )
            else:
                gate_up = strategy.grouped_matmul(
                    self.gate_up_proj,
                    gate_up_scales,
                    expert_scales=nvfp4.gate_up_expert if nvfp4 else None,
                    expert_inputs=expert_inputs,
                    estimated_total_m=estimated_total_m,
                )

                if self.use_swigluoai:
                    gate_up = self._swigluoai_activation(gate_up)
                    if self._uses_nvidia_block_scaled_ep_layout:
                        _, _, expert_start, scales_offset, expert_ids, _ = (
                            expert_inputs
                        )
                        down_in, silu_scales = strategy.grouped_quantize(
                            gate_up,
                            self._token_group_size,
                            nvfp4.down_input if nvfp4 else None,
                            expert_start,
                            scales_offset,
                            expert_ids,
                        )
                    else:
                        down_in, silu_scales = strategy.quantize(
                            gate_up, self._token_group_size
                        )
                else:
                    down_in, silu_scales = strategy.fused_silu_quantize(
                        gate_up,
                        input_scales=nvfp4.down_input if nvfp4 else None,
                        expert_inputs=expert_inputs,
                    )

        down_inputs = (down_in, silu_scales) + expert_inputs[2:]
        if isinstance(strategy, Mxfp4Strategy):
            return strategy.grouped_matmul(
                self.down_proj,
                down_scales,
                expert_inputs=down_inputs,
                estimated_total_m=estimated_total_m,
                # KS64: fused_silu wrote the down-proj A-scale in slot layout.
                a_scales_preshuffled=mxfp4_ep_max_padded_m > 0,
                # Reader slot stride MUST equal the constant the producer wrote
                # with (single source of truth) — not the runtime per-expert
                # max — or the matmul reads the wrong expert's scales.
                a_scales_max_padded_m=mxfp4_ep_max_padded_m,
            )
        return strategy.grouped_matmul(
            self.down_proj,
            down_scales,
            expert_scales=nvfp4.down_expert if nvfp4 else None,
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
            needs_scales_offset=self._uses_nvidia_block_scaled_ep_layout,
        )
        token_order, expert_start, restore_order, expert_ids, usage_stats = (
            create_indices_result[:5]
        )
        scales_offset = (
            create_indices_result[5]
            if self._uses_nvidia_block_scaled_ep_layout
            else None
        )

        if self.pre_expert_norm is not None:
            x = self.pre_expert_norm(x)

        permuted = ops.gather(
            x,
            ops.cast(token_order // self.num_experts_per_token, DType.int32),
            axis=0,
        )

        total_m = ops.shape_to_tensor(permuted.shape)[0].cast(DType.uint32)

        if self._uses_nvidia_block_scaled_ep_layout:
            assert scales_offset is not None
            permuted_quant, permuted_scales = strategy.grouped_quantize(
                permuted,
                self._token_group_size,
                nvfp4.gate_up_input if nvfp4 else None,
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

        if self._uses_nvidia_block_scaled_ep_layout:
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

        if self.gated_activation_fn is not None:
            gate_up = self.gated_activation_fn(gate_up, self.moe_dim)
        elif self.use_swigluoai:
            gate_up = self._swigluoai_activation(gate_up)
        else:
            gate_up = fused_silu(gate_up, expert_start)

        if self._uses_nvidia_block_scaled_ep_layout:
            assert scales_offset is not None
            gate_up_quant, gate_up_scales = strategy.grouped_quantize(
                gate_up,
                self._token_group_size,
                nvfp4.down_input if nvfp4 else None,
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
