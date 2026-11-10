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
"""The DSpark drafter's MoE: its own gate, grouped MXFP4 experts, shared expert.

The drafter has 128 experts and top-3, so its gate is independent of the
trunk's router; it shares the reference's ``noaux_tc`` scoring.

The routed experts keep the reference's grouped MXFP4 path: weights are
``[E, N, K/2]`` uint8 with E8M0 scales per 32 (OCP MXFP4), activations
quantize to MXFP8 with one E8M0 per 32 and a tensor scale of one, and both
projections run the public grouped block-scaled kernels. The trunk's NVFP4
experts op cannot host them: its formats are E2M1-with-E4M3/16 plus
per-tensor f32 scales, not E8M0/32.

Under tensor parallelism the reference gives each rank a contiguous run of
whole experts (128 % mesh == 0). The expert weights stay
replicated for now: every rank routes every slot to its expert here, so the
trace is correct at any mesh size but pays 4x drafter-expert memory at TP4.
The upgrade path -- split the experts along axis 0 and mask each rank's
non-local slots -- waits on the rank-aware offsets the grouped kernel
needs.

Sources: ``dspark_moe.py`` of the V2 port (activation quant 42-81, experts
84-159, stage MoE 162-247), the grouped chain it inherits
(``deepseekV4/layers/moe.py:376-573``), and the torch oracle's ``route`` and
``moe``.
"""

from __future__ import annotations

import numpy as np
from max import functional as F
from max._core.dialects import kgen
from max.driver import CPU
from max.dtype import DType
from max.graph import AlgebraicDim, TensorValue
from max.nn import Module
from max.nn._graph_ops import (
    moe_create_indices_with_slots as _moe_create_indices_with_slots,
)
from max.nn.common_layers.quant_ops import (
    block_scales_interleave,
    grouped_matmul_block_scaled,
    grouped_quantize_dynamic_block_scaled,
)
from max.nn.quant_config import ceildiv
from max.tensor import Tensor
from max.tensor.sharding import (
    AxisAssignment,
    DeviceMesh,
    Replicated,
    Sharded,
    TensorLayout,
    Unknown,
)

from ..model_config import DeepseekV41Config
from .moe import SharedExpert
from .router import gate_logits, route_logits

__all__ = ["DSparkMoE"]


# A-scale tile height used by the scale-slot row bound.
_SF_ROWS = 128


def _create_indices_with_slots(
    topk_ids: TensorValue,
    slot_weights: TensorValue,
    num_groups: int,
    slot_stride: int,
) -> tuple[TensorValue, ...]:
    return _moe_create_indices_with_slots(
        topk_ids,
        slot_weights,
        num_groups,
        slot_stride=slot_stride,
        scale_rows=_SF_ROWS
        * (
            ceildiv(topk_ids.shape[0], _SF_ROWS)
            + AlgebraicDim.apply(kgen.POC.min, num_groups, topk_ids.shape[0])
        ),
    )


def _create_indices_with_slots_rule(
    topk_ids: TensorLayout,
    slot_weights: TensorLayout,
    num_groups: int,
    slot_stride: int,
) -> list[AxisAssignment]:
    # Output extents mix global token and expert-local sizes.
    return [
        AxisAssignment((Replicated(), Replicated()), (Replicated(),) * 9),
        AxisAssignment((Sharded(0), Sharded(0)), (Unknown(),) * 9),
    ]


# The experimental wrapper, not the raw ``max.nn.kernels`` function: the
# raw one reads ``TensorValue.device`` off its operands, which raises on a
# tensor distributed over a mesh, and carries no sharding rule for the
# planner.
moe_create_indices_with_slots = F.functional(
    _create_indices_with_slots, rule=_create_indices_with_slots_rule
)

# One E8M0 scale per 32 elements: MXFP8 activations, MXFP4 weights.
_ACTIVATION_QUANT_GROUP = 32


class DSparkGate(Module[[Tensor], tuple[Tensor, Tensor]]):
    """The drafter's scored top-k gate, at the drafter's expert counts.

    The reference's ``noaux_tc`` selection: ``sqrtsoftplus`` scores, the
    bias shifting selection only (never the returned weights), the gathered
    weights renormalized with the oracle's ``sum > 0`` floor and scaled by
    ``routed_scaling_factor``. Drafts are text, so the image-token bias
    variant does not apply.
    """

    def __init__(self, config: DeepseekV41Config) -> None:
        super().__init__()
        n_experts = config.dspark_n_routed_experts
        self.num_experts = n_experts
        self.num_experts_per_tok = config.dspark_num_experts_per_tok
        self.route_scale = config.routed_scaling_factor
        self.weight = Tensor.zeros(
            [n_experts, config.hidden_size], dtype=DType.bfloat16
        )
        self.bias = Tensor.zeros([n_experts], dtype=DType.float32)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Route tokens to their top-k experts."""
        logits = gate_logits(x, self.weight)
        ids, weights = route_logits(
            logits, self.bias, self.num_experts_per_tok, self.route_scale
        )
        return weights, ids


class _PackedExpertWeights(Module[[], None]):
    """One stacked MXFP4 expert projection, stacked by expert id.

    ``weight`` is ``[E, N, K/2]`` uint8 (two E2M1 values per byte) and
    ``weight_scale`` ``[E, N, K/32]`` E8M0 -- exactly the checkpoint's
    storage stacked by expert id, which the weight adapter builds. The
    grouped kernel reads the scales in their interleaved SF-atom layout,
    which :meth:`interleaved` produces.
    """

    weight: Tensor
    weight_scale: Tensor

    def __init__(self, n_experts: int, out_dim: int, in_dim: int) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.weight = Tensor.zeros(
            [n_experts, out_dim, in_dim // 2], dtype=DType.uint8
        )
        self.weight_scale = Tensor.zeros(
            [n_experts, out_dim, in_dim // _ACTIVATION_QUANT_GROUP],
            dtype=DType.float8_e8m0fnu,
        )

    def interleaved(self) -> Tensor:
        """The block scales in the grouped matmul's interleaved layout."""
        per_expert = F.split(self.weight_scale, [1] * self.n_experts, axis=0)
        return F.stack(
            [
                block_scales_interleave(
                    F.reshape(expert, [self.out_dim, self.in_dim // 32]),
                    sf_vector_size=32,
                )
                for expert in per_expert
            ],
            axis=0,
        )


def _ones(n: int, mesh: DeviceMesh) -> Tensor:
    # Keep this host scalar in the graph; ``F.full`` adds a broadcast launch.
    return F.constant(np.ones(n, np.float32), DType.float32, device=mesh)


def _activation_quantize(
    x: Tensor,
    row_offsets: Tensor,
    scales_offsets: Tensor,
    expert_ids: Tensor,
) -> tuple[Tensor, Tensor]:
    """Quantizes expert-sorted activations to MXFP8, one scale per 32 values.

    There is no per-expert static input scale, so the kernel's tensor-wise
    scale is one.

    Args:
        x: ``[total_tokens, K]``, one row per token slot sorted by expert.
        row_offsets: Cumulative token offsets per expert, uint32, ``[E + 1]``.
        scales_offsets: Per-expert scale-tile offset corrections, uint32,
            ``[E]``.
        expert_ids: Expert id mapping, ``[E]``.

    Returns:
        ``(quantized, a_scales)``: ``float8_e4m3fn[total_tokens, K]`` and
        scales in the layout ``grouped_matmul_block_scaled`` takes.
    """
    sf_tensor = _ones(int(expert_ids.shape[0]), x.mesh)
    return grouped_quantize_dynamic_block_scaled(
        x.cast(DType.bfloat16),
        row_offsets=row_offsets,
        scales_offsets=scales_offsets,
        expert_ids=expert_ids,
        sf_tensor=sf_tensor,
        sf_vector_size=_ACTIVATION_QUANT_GROUP,
        scales_type=DType.float8_e8m0fnu,
        out_type=DType.float8_e4m3fn,
    )


class DSparkRoutedExperts(Module[[Tensor, Tensor, Tensor], Tensor]):
    """One stage's routed experts on the grouped MXFP4 kernels.

    Tokens' ``k`` expert slots are sorted by expert and packed, one row per
    slot, and the two projections run as grouped GEMMs over the sorted rows.
    Only the A-side scale tiles are padded: the kernel reads each group's
    scales from 128-row tiles located by ``scales_offsets``, so the padding
    costs the offset arithmetic below, never extra quantize work. The
    routing weight scales the SwiGLU output before the down projection, as
    in the reference.
    """

    gate_up_proj: _PackedExpertWeights
    down_proj: _PackedExpertWeights

    def __init__(
        self,
        config: DeepseekV41Config,
        mesh: DeviceMesh,
    ) -> None:
        super().__init__()
        n_experts = config.dspark_n_routed_experts
        self.num_experts = n_experts
        self.topk = config.dspark_num_experts_per_tok
        self.hidden = config.hidden_size
        self.inter = config.moe_intermediate_size
        self.swiglu_limit = config.swiglu_limit
        self.gate_up_proj = _PackedExpertWeights(
            n_experts, 2 * config.moe_intermediate_size, config.hidden_size
        )
        self.down_proj = _PackedExpertWeights(
            n_experts, config.hidden_size, config.moe_intermediate_size
        )

    def _grouped(
        self,
        proj: _PackedExpertWeights,
        x: Tensor,
        a_offsets: Tensor,
        scale_offsets: Tensor,
        expert_ids: Tensor,
    ) -> Tensor:
        """One grouped projection over the expert-sorted slots."""
        quantized, a_scales = _activation_quantize(
            x, a_offsets, scale_offsets, expert_ids
        )
        ones = _ones(self.num_experts, x.mesh)
        # Host-side usage stats: the kernel only reads the active-expert
        # count here; the max-tokens slot is an estimate the row count
        # bounds. Both are host values.
        usage = F.concat(
            [
                F.shape_to_tensor([x.shape[0]]).cast(DType.uint32),
                F.full([1], self.num_experts, dtype=DType.uint32, device=CPU()),
            ],
            axis=0,
        )
        return grouped_matmul_block_scaled(
            quantized,
            proj.weight,
            a_scales,
            proj.interleaved(),
            a_offsets,
            scale_offsets,
            expert_ids,
            ones,
            usage,
            out_type=DType.bfloat16,
        )

    def _activate_and_down(
        self,
        gate_up: Tensor,
        slot_weight: Tensor,
        a_offsets: Tensor,
        scale_offsets: Tensor,
        expert_ids: Tensor,
    ) -> Tensor:
        """The clamped SwiGLU over ``gate_up``, then the down projection.

        Rows stay in sorted-slot order; the routing weight multiplies in
        before the down projection, as in the reference.
        """
        gate, up = F.split(gate_up, [self.inter, self.inter], axis=1)
        if self.swiglu_limit > 0:
            # Clamp before widening; otherwise the compiler materializes a separate cast.
            limit = float(self.swiglu_limit)
            # Asymmetric on purpose: the reference clamps ``up`` on both
            # sides but ``gate`` only from above.
            low = F.constant(-limit, dtype=gate_up.dtype, device=up.mesh)
            high = F.constant(limit, dtype=gate_up.dtype, device=up.mesh)
            up = F.min(F.max(up, low), high)
            gate = F.min(gate, high)
        h = F.silu(gate.cast(DType.float32)) * up.cast(DType.float32)
        h = h * slot_weight.unsqueeze(-1)
        return self._grouped(
            self.down_proj, h, a_offsets, scale_offsets, expert_ids
        )

    def forward(self, x: Tensor, weights: Tensor, indices: Tensor) -> Tensor:
        """Sum each token's selected expert outputs in float32."""
        tokens = x.shape[0]
        slots = tokens * self.topk

        # The grouped kernel expects int32 expert ids and row indices.
        (
            _order,
            a_offsets,
            restore,
            expert_ids,
            _usage,
            scale_offsets,
            slot_token,
            slot_weight,
            _scale_slot,
        ) = moe_create_indices_with_slots(
            F.reshape(indices, [slots]),
            F.reshape(weights, [slots]).cast(DType.float32),
            self.num_experts,
            self.topk,
        )
        x_sorted = F.gather(x, slot_token, axis=0)

        gate_up = self._grouped(
            self.gate_up_proj, x_sorted, a_offsets, scale_offsets, expert_ids
        )
        down = self._activate_and_down(
            gate_up, slot_weight, a_offsets, scale_offsets, expert_ids
        )
        # ``restore`` puts each token's ``k`` slot rows back in token order;
        # the padded rows are never picked.
        out = F.gather(down, restore, axis=0).cast(DType.float32)
        out = F.reshape(out, [tokens, self.topk, self.hidden])
        return F.squeeze(F.sum(out, axis=1), axis=1)


class DSparkMoE(Module[[Tensor], Tensor]):
    """One DSpark stage's FFN: gate, grouped MXFP4 experts, shared expert.

    The shared expert is the trunk's, at the same widths and clamp: only the
    routed experts and the gate are drafter-specific.
    """

    gate: DSparkGate
    experts: DSparkRoutedExperts
    shared_experts: SharedExpert

    def __init__(
        self,
        config: DeepseekV41Config,
        mesh: DeviceMesh,
        *,
        data_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.gate = DSparkGate(config)
        self.experts = DSparkRoutedExperts(config, mesh)
        self.shared_experts = SharedExpert(
            config.hidden_size,
            config.moe_intermediate_size,
            mesh=mesh,
            swiglu_limit=config.swiglu_limit,
            data_parallel=data_parallel,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Maps ``[T, hidden]`` states to the block's FFN output.

        The routed and shared outputs accumulate in float32, as in the
        reference's ``y``.

        An empty rank -- a DP replica with no rows this step, which the
        capture warmup deliberately produces -- must not reach the grouped
        kernels with ``T == 0``: they size each group's activation scales
        as ``ceildiv(rows, 128)`` at runtime, so an empty rank would build a
        zero-size TMA descriptor, which ``cuTensorMapEncodeTiled`` refuses
        (the same class the seven-site empty-rank guards cover, one level
        deeper: a kernel-side early return cannot fix it because the grouped
        matmuls build their own descriptors from those scales). One
        sacrificial zero row keeps every kernel's rows at one or more. The
        row is row-local through the gate, the routed experts (its ``k``
        slots form their own sorted-group entries and ``restore`` puts only
        real slots back) and the shared expert, so the surviving rows'
        values are unchanged and the slice drops it.
        """
        zero_row = F.constant(
            np.zeros([1, int(x.shape[1])], np.float32),
            DType.float32,
            device=x.mapping,
        ).cast(x.dtype)
        padded = F.concat([x, zero_row], axis=0)
        weights, indices = self.gate(padded)
        routed = self.experts(padded, weights, indices)
        y = self.shared_experts(padded).cast(DType.float32) + routed
        return F.rebind(y[:-1], [x.shape[0], y.shape[1]]).cast(x.dtype)
