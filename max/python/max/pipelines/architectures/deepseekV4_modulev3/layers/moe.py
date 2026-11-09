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
"""DeepSeek-V4 MoE on ModuleV3: gate, shared expert, W4A8 routed experts.

Op-for-op ports of the ModuleV2 implementation (``deepseekV4/layers/moe.py``
at dfc8109fc0c); the semantics (hash routing via ``tid2eid`` with
``gate.weight`` scores, bias for selection only, normalization after the
gather, the asymmetric SwiGLU clamp, the 128-to-32 activation scale repeat)
are documented there.

Tensor parallel as V2: attention-side activations arrive whole, each device
runs its run of whole experts over every token, and the float32 partial sums
are all-reduced before the (whole) shared expert is added.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from max import functional as F
from max._core.dialects import kgen
from max.driver import CPU
from max.dtype import DType
from max.graph import AlgebraicDim, DimLike
from max.nn import Module
from max.nn._graph_ops import _is_apple_gpu
from max.nn.common_layers import quant_ops
from max.nn.common_layers.mesh_axis import TP
from max.nn.quant_config import QuantConfig, ceildiv
from max.tensor import Tensor
from max.tensor.sharding import (
    AxisAssignment,
    DeviceMapping,
    NamedMapping,
    Partial,
    Replicated,
    Sharded,
    TensorLayout,
)
from max.tensor.sharding.action import PerShard

from ..model_config import DeepseekV4Config
from .kernels import moe_create_indices_with_slots, moe_router_group_limited
from .quantization import LINEAR_QUANT_BLOCK, linear_for

__all__ = [
    "DeepseekV4Gate",
    "DeepseekV4MoE",
    "DeepseekV4RoutedExperts",
    "DeepseekV4SharedExpert",
    "sqrt_softplus",
]


# Extracted from deepseekV4/layers/moe.py at dfc8109fc0c.
# e2m1 weights carry one e8m0 scale per 32 elements along K (OCP MXFP4), and
# the SM100 scale-factor atom covers 128 rows -- both fixed by the kernel.
FP4_WEIGHT_BLOCK = 32
SF_ROWS = 128


def sqrt_softplus(x: Tensor) -> Tensor:
    """``sqrt(softplus(x))`` in the overflow-free form."""
    softplus = F.elementwise_max(x, 0.0) + F.log(1.0 + F.exp(-F.abs(x)))
    return F.sqrt(softplus)


def _clamped_swiglu(gate: Tensor, up: Tensor, swiglu_limit: float) -> Tensor:
    """float32 ``silu(gate) * up``; ``up`` clamped both ways, ``gate`` above."""
    gate = gate.cast(DType.float32)
    up = up.cast(DType.float32)
    if swiglu_limit > 0:
        limit = float(swiglu_limit)
        up = F.elementwise_min(F.elementwise_max(up, -limit), limit)
        gate = F.elementwise_min(gate, limit)
    return F.silu(gate) * up


class DeepseekV4SharedExpert(Module[[Tensor], Tensor]):
    """The fp8 shared expert; the adapter stacks ``w1``/``w3`` into ``w13``."""

    def __init__(self, config: DeepseekV4Config) -> None:
        super().__init__()
        self.swiglu_limit = config.swiglu_limit
        self.intermediate = config.moe_intermediate_size
        hidden = config.hidden_size
        self.w13 = linear_for(config, hidden, 2 * self.intermediate)
        self.w2 = linear_for(config, self.intermediate, hidden)

    def forward(self, x: Tensor) -> Tensor:
        """Applies the clamped SwiGLU expert to ``[T, hidden]`` tokens."""
        gate_up = self.w13(x)
        width = int(gate_up.shape[-1])
        gate = gate_up[..., : self.intermediate]
        up = gate_up[..., self.intermediate : width]
        h = _clamped_swiglu(gate, up, self.swiglu_limit).cast(x.dtype)
        return self.w2(h)


class DeepseekV4Gate(Module[[Tensor, Tensor], tuple[Tensor, Tensor]]):
    """Expert selection and routing weights for one layer."""

    def __init__(self, config: DeepseekV4Config, layer_idx: int) -> None:
        super().__init__()
        self.is_hash_routed = config.layer_is_hash_routed(layer_idx)
        self.topk = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.route_scale = config.routed_scaling_factor
        # The fused router runs full-size expert counts on GPU; Metal's warp
        # top-k is wrong (GEX-3342).
        self.fused = (
            not self.is_hash_routed
            and config.devices[0].is_gpu()
            and not _is_apple_gpu()
            and self.n_routed_experts % 64 == 0
        )
        self.weight = Tensor.zeros(
            [config.n_routed_experts, config.hidden_size]
        )
        if self.is_hash_routed:
            # int64 in the checkpoint, whatever the reference declares.
            self.tid2eid = Tensor.zeros(
                [config.vocab_size, config.num_experts_per_tok],
                dtype=DType.int64,
            )
        else:
            self.bias = Tensor.zeros(
                [config.n_routed_experts], dtype=DType.float32
            )

    def forward(self, x: Tensor, token_ids: Tensor) -> tuple[Tensor, Tensor]:
        """``[T, d]`` and ``[T]`` -> weights and int32 indices, ``[T, k]``."""
        logits = x.cast(DType.float32) @ self.weight.cast(DType.float32).T
        scores = sqrt_softplus(logits)
        if self.fused:
            indices, weights = moe_router_group_limited(
                scores,
                self.bias,
                n_routed_experts=self.n_routed_experts,
                n_experts_per_tok=self.topk,
                n_groups=1,
                topk_group=1,
                norm_weights=True,
                routed_scaling_factor=self.route_scale,
            )
            return weights, indices
        if self.is_hash_routed:
            indices = F.gather(self.tid2eid, token_ids, axis=0)
        else:
            _, indices = F.top_k(scores + self.bias, self.topk, axis=-1)
        indices = indices.cast(DType.int32)
        weights = F.gather_nd(scores, indices.unsqueeze(-1), batch_dims=1)
        weights = weights / weights.sum(axis=-1)
        return weights * self.route_scale, indices


class _PackedExpertWeight(Module[..., None]):
    """``[E, N, K/2]`` packed e2m1 + ``[E, N, K/32]`` e8m0, stacked by expert."""

    def __init__(self, n_experts: int, out_dim: int, in_dim: int) -> None:
        super().__init__()
        if in_dim % SF_ROWS:
            raise ValueError(f"fp4 expert K={in_dim} is not a multiple of 128")
        self.weight = Tensor.zeros(
            [n_experts, out_dim, in_dim // 2], dtype=DType.uint8
        )
        self.weight_scale = Tensor.zeros(
            [n_experts, out_dim, in_dim // FP4_WEIGHT_BLOCK],
            dtype=DType.float8_e8m0fnu,
        )

    def shard_experts(self) -> None:
        """Whole experts per device, along axis 0 of both stacks."""
        mesh = self.weight.mesh
        experts = NamedMapping(mesh, (TP, None, None))
        self.weight = self.weight.to(experts)
        self.weight_scale = self.weight_scale.to(experts)


def _interleaved_scales(weight_scale: Tensor) -> Tensor:
    """``[E, N, K/32]`` -> ``[E, N/128, K/128, 32, 4, 4]``, the SF-atom layout."""
    n_experts, out_dim, blocks = (int(d) for d in weight_scale.shape)
    per_expert = weight_scale.split([1] * n_experts, axis=0)
    return F.stack(
        [
            quant_ops.block_scales_interleave(
                e.reshape([out_dim, blocks]), FP4_WEIGHT_BLOCK
            )
            for e in per_expert
        ],
        axis=0,
    )


def _quantize_rows_for_w4a8(
    x: Tensor, quant_config: QuantConfig, scale_slot: Tensor
) -> tuple[Tensor, Tensor]:
    """``act_quant(x, 128, "ue8m0")``, each scale repeated 4x for K/32."""
    width = int(x.shape[1])
    quantized, scales = quant_ops.quantize_dynamic_scaled_float8(
        x.cast(DType.bfloat16),
        quant_config.input_scale,
        quant_config.weight_scale,
        group_size_or_per_token=LINEAR_QUANT_BLOCK,
        out_type=DType.float8_e4m3fn,
        scales_type=DType.float8_e8m0fnu,
    )
    groups = width // LINEAR_QUANT_BLOCK
    repeat = LINEAR_QUANT_BLOCK // FP4_WEIGHT_BLOCK
    per_row = scales[:, 0 : x.shape[0]].transpose(0, 1)
    per_row = F.gather(per_row, scale_slot, axis=0)
    rows = scale_slot.shape[0]
    per_row = F.broadcast_to(
        per_row.unsqueeze(-1), [rows, groups, repeat]
    ).reshape([rows, groups * repeat])
    return quantized, quant_ops.block_scales_interleave(
        per_row, FP4_WEIGHT_BLOCK
    )


def _usage_stats(rows: DimLike, n_experts: int) -> Tensor:
    """Host ``[rows, n_experts]`` uint32: the grouped kernel's usage stats.

    Only the active-expert count is read; the max-tokens slot is an estimate
    the row count bounds.
    """
    return F.concat(
        [
            F.shape_to_tensor([rows]).cast(DType.uint32),
            F.constant(
                np.array([n_experts], np.uint32), DType.uint32, device=CPU()
            ),
        ],
        axis=0,
    )


@dataclass(frozen=True)
class _RoutedSpec:
    """The routed body's non-tensor arguments."""

    topk: int
    hidden: int
    inter: int
    swiglu_limit: float
    quant_config: QuantConfig


def _grouped(
    x: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    a_offsets: Tensor,
    scale_offsets: Tensor,
    scale_slot: Tensor,
    expert_ids: Tensor,
    quant_config: QuantConfig,
) -> Tensor:
    n_experts = int(weight.shape[0])
    quantized, a_scales = _quantize_rows_for_w4a8(x, quant_config, scale_slot)
    ones_f32 = F.constant(
        np.ones(n_experts, np.float32), DType.float32, device=x.device
    )
    return quant_ops.grouped_matmul_block_scaled(
        quantized,
        weight,
        a_scales,
        _interleaved_scales(weight_scale),
        a_offsets,
        scale_offsets,
        expert_ids,
        ones_f32,
        _usage_stats(x.shape[0], n_experts),
        out_type=DType.bfloat16,
    )


def _routed_body(
    x: Tensor,
    weights: Tensor,
    indices: Tensor,
    gate_up: Tensor,
    gate_up_scale: Tensor,
    down: Tensor,
    down_scale: Tensor,
    first: int,
    spec: _RoutedSpec,
) -> Tensor:
    """Sums each token's slots routed to the experts held on this device.

    Returns float32 ``[T, hidden]``. The local experts are
    ``[first, first + E)`` for ``E`` stacked experts.

    Every other slot -- another device's expert, or an id outside the expert
    range -- goes to the trailing group ``E`` and contributes zero.
    """
    tokens = x.shape[0]
    slots = tokens * spec.topk
    n_local = int(gate_up.shape[0])
    flat_idx = indices.reshape([slots])
    is_local = F.logical_and(
        F.greater_equal(flat_idx, first),
        F.greater(first + n_local, flat_idx),
    )
    router_idx = F.where(is_local, flat_idx - first, n_local)
    groups = n_local + 1
    padded = SF_ROWS * (
        ceildiv(slots, SF_ROWS)
        + AlgebraicDim.apply(kgen.POC.min, groups, slots)
    )
    (
        _order,
        a_offsets,
        restore,
        expert_ids,
        _usage,
        scale_offsets,
        slot_token,
        slot_weight,
        scale_slot,
    ) = moe_create_indices_with_slots(
        router_idx,
        weights.reshape([slots]).cast(DType.float32),
        groups,
        slot_stride=spec.topk,
        scale_rows=padded,
        # Ids are already groups in ``[0, n_local]``: identity mapping.
        expert_offset=0,
        local_experts=n_local,
    )
    a_offsets = a_offsets[0 : n_local + 1]
    scale_offsets = scale_offsets[0:n_local]
    expert_ids = expert_ids[0:n_local]

    def grouped(h: Tensor, weight: Tensor, scale: Tensor) -> Tensor:
        return _grouped(
            h,
            weight,
            scale,
            a_offsets,
            scale_offsets,
            scale_slot,
            expert_ids,
            spec.quant_config,
        )

    x_sorted = F.gather(x, slot_token, axis=0)
    gate_up_out = grouped(x_sorted, gate_up, gate_up_scale)
    gate, up = gate_up_out.split([spec.inter, spec.inter], axis=1)
    h = _clamped_swiglu(gate, up, spec.swiglu_limit)
    h = h * slot_weight.unsqueeze(-1)
    down_out = grouped(h, down, down_scale)

    out = F.gather(down_out, restore, axis=0).cast(DType.float32)
    # ``where``, not a multiply: masked rows hold whatever the GEMM left.
    out = F.where(is_local.unsqueeze(-1), out, 0.0)
    out = out.reshape([tokens, spec.topk, spec.hidden])
    return out.sum(axis=1).squeeze(1)


def _routed_rule(
    x: TensorLayout,
    weights: TensorLayout,
    indices: TensorLayout,
    gate_up: TensorLayout,
    gate_up_scale: TensorLayout,
    down: TensorLayout,
    down_scale: TensorLayout,
    first: object,
    spec: _RoutedSpec,
) -> list[AxisAssignment]:
    """Whole experts per device over every token: a partial sum out.

    Linear in the experts: the devices' masked sums add up to the sum over
    all of a token's slots. ``first`` is each device's own first expert.
    """
    whole, experts = Replicated(), Sharded(0)
    return [
        AxisAssignment(
            (whole, whole, whole, experts, experts, experts, experts),
            (Partial(),),
        )
    ]


_routed = F.functional(_routed_body, rule=_routed_rule)


class DeepseekV4RoutedExperts(Module[[Tensor, Tensor, Tensor], Tensor]):
    """The routed experts on the W4A8 grouped kernel, slots sorted by expert.

    Layout (``a_offsets`` CSR, 128-row scale tiles located by
    ``scale_offsets``, the trailing masked group): see the V2 class. Under
    tensor parallelism each device holds a run of whole experts, as in V2's
    ``shard_experts``.
    """

    def __init__(self, config: DeepseekV4Config) -> None:
        super().__init__()
        if config.quant_config is None:
            raise ValueError("native routed experts need config.quant_config")
        self.n_experts = config.n_routed_experts
        self.spec = _RoutedSpec(
            topk=config.num_experts_per_tok,
            hidden=config.hidden_size,
            inter=config.moe_intermediate_size,
            swiglu_limit=config.swiglu_limit,
            quant_config=config.quant_config,
        )
        self.gate_up_proj = _PackedExpertWeight(
            self.n_experts, 2 * self.spec.inter, self.spec.hidden
        )
        self.down_proj = _PackedExpertWeight(
            self.n_experts, self.spec.hidden, self.spec.inter
        )
        n = self.gate_up_proj.weight.mesh.num_devices
        if n > 1:
            if self.n_experts % n:
                raise ValueError(
                    f"{self.n_experts} experts do not split across {n} devices"
                )
            self.gate_up_proj.shard_experts()
            self.down_proj.shard_experts()

    def forward(self, x: Tensor, weights: Tensor, indices: Tensor) -> Tensor:
        """``[T, hidden]``, ``[T, k]``, ``[T, k]`` -> ``[T, hidden]``.

        ``x.dtype`` on one device. With the experts split, a float32 partial
        sum over the devices, as the reference's ``y`` before its all-reduce.
        """
        n = self.gate_up_proj.weight.mesh.num_devices
        local = self.n_experts // n
        out = _routed(
            x,
            weights,
            indices,
            self.gate_up_proj.weight,
            self.gate_up_proj.weight_scale,
            self.down_proj.weight,
            self.down_proj.weight_scale,
            PerShard(range(0, self.n_experts, local)),
            self.spec,
        )
        return out if n > 1 else out.cast(x.dtype)


class DeepseekV4MoE(Module[[Tensor, Tensor], Tensor]):
    """``num_experts_per_tok`` routed experts plus one shared expert.

    Native fp4 experts only: the dense every-expert path of the V2 module is
    not ported (it is 42x the routed work at 256 experts). The shared expert
    is whole on every device, as in V2.
    """

    def __init__(self, config: DeepseekV4Config, layer_idx: int) -> None:
        super().__init__()
        self.gate = DeepseekV4Gate(config, layer_idx)
        self.experts = DeepseekV4RoutedExperts(config)
        self.shared_experts = DeepseekV4SharedExpert(config)

    def forward(self, x: Tensor, token_ids: Tensor) -> Tensor:
        """``[T, hidden]`` and ``[T]`` in, ``[T, hidden]`` out."""
        weights, indices = self.gate(x, token_ids)
        routed = self.experts(x, weights, indices)
        # As V2: the routed sum is all-reduced in float32 before the shared
        # expert joins it; adding it first would count it once per device.
        routed = routed.to(DeviceMapping.replicated(routed.mesh))
        y = self.shared_experts(x).cast(DType.float32) + routed.cast(
            DType.float32
        )
        return y.cast(x.dtype)
