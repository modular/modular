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

The drafter routes with its own expert counts -- 128 experts, 3 per token --
so its gate is this module rather than the trunk's router, which is fixed at
the trunk's 384/6 geometry. The gate math itself is the reference's
``noaux_tc`` scoring, shared with the trunk's router.

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

from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.nn import Module
from max.nn.common_layers.quant_ops import (
    block_scales_interleave,
    grouped_matmul_block_scaled,
    grouped_quantize_dynamic_block_scaled,
)
from max.nn.legacy.kernels import moe_create_indices as _moe_create_indices
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
from .router import route_logits

__all__ = ["DSparkMoE"]


def _moe_create_indices_rule(
    lhs: TensorLayout,
    num_local_experts: int,
    *,
    needs_scales_offset: bool = False,
) -> list[AxisAssignment]:
    """Returns the rows for ``moe_create_indices``.

    Replicated ids give whole-batch positions; sharded ids give local ones.
    """
    # Over a shard of the batch the outputs mix per-token and per-expert
    # extents that no one global tensor spans, so the row forgets instead.
    return [
        AxisAssignment(
            (Replicated(),), (Replicated(),) * (6 if needs_scales_offset else 5)
        ),
        AxisAssignment(
            (Sharded(0),), (Unknown(),) * (6 if needs_scales_offset else 5)
        ),
    ]


# The experimental wrapper, not the raw ``max.nn.kernels`` function: the
# raw one reads ``TensorValue.device`` off its operands, which raises on a
# tensor distributed over a mesh, and carries no sharding rule for the
# planner.
moe_create_indices = F.functional(
    _moe_create_indices, rule=_moe_create_indices_rule
)

# One E8M0 scale per 32 elements: MXFP8 activations, MXFP4 weights.
_ACTIVATION_QUANT_GROUP = 32
# The grouped kernel reads each expert's activation scales from 128-row
# tiles; the scale offsets below place each group's first tile.
_SF_ROWS = 128


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
        """Routes each token to its top-k experts.

        The widened bf16 gate GEMM, then the trunk router's scoring and
        selection kernel (:func:`~.router.route_logits`).

        Args:
            x: ``[T, hidden]`` hidden states.

        Returns:
            ``(weights, indices)``: ``[T, k]`` float32 and ``[T, k]`` int32.
        """
        logits = x.cast(DType.float32) @ self.weight.cast(DType.float32).T
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
    sf_tensor = F.full(
        expert_ids.shape, 1.0, dtype=DType.float32, device=x.mesh
    )
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
        ones = F.full(
            [self.num_experts], 1.0, dtype=DType.float32, device=x.mesh
        )
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
        gate = gate.cast(DType.float32)
        up = up.cast(DType.float32)
        if self.swiglu_limit > 0:
            limit = float(self.swiglu_limit)
            # Asymmetric on purpose: the reference clamps ``up`` on both
            # sides but ``gate`` only from above.
            low = F.constant(-limit, dtype=DType.float32, device=up.mesh)
            high = F.constant(limit, dtype=DType.float32, device=up.mesh)
            up = F.min(F.max(up, low), high)
            gate = F.min(gate, high)
        h = F.silu(gate) * up
        h = h * slot_weight.unsqueeze(-1)
        return self._grouped(
            self.down_proj, h, a_offsets, scale_offsets, expert_ids
        )

    def forward(self, x: Tensor, weights: Tensor, indices: Tensor) -> Tensor:
        """Sums each token's ``k`` expert outputs, in float32.

        Args:
            x: ``[T, hidden]`` tokens.
            weights: ``[T, k]`` float32 routing weights.
            indices: ``[T, k]`` int32 expert ids.

        Returns:
            ``[T, hidden]`` float32, the per-token sum over its experts.
        """
        tokens = x.shape[0]
        slots = tokens * self.topk
        groups = self.num_experts + 1

        order, start, restore, expert_ids, _usage = moe_create_indices(
            F.reshape(indices, [slots]), groups
        )
        # The route metadata is uint32 all the way through -- the kernel's
        # own ragged dtype -- so the offset arithmetic stays unsigned; only
        # the gather and permutation indices narrow to int32, the dtype
        # ``F.gather`` indexes with.
        order = order.cast(DType.int32)

        # Group g's scales occupy the tiles [aligned_start[g],
        # aligned_start[g + 1]) / 128: the exclusive prefix sum of the group
        # sizes rounded up to 128. The kernel finds that first tile as
        # start[g] // 128 plus the group's scale offset.
        counts = start[1 : groups + 1] - start[0:groups]
        aligned_counts = (counts + (_SF_ROWS - 1)) // _SF_ROWS * _SF_ROWS
        aligned_start = F.concat(
            [
                F.constant(0, dtype=DType.uint32, device=counts.mesh).reshape(
                    [1]
                ),
                F.cumsum(aligned_counts, axis=0),
            ],
            axis=0,
        )
        scale_offsets = (
            aligned_start[0:groups] // _SF_ROWS - start[0:groups] // _SF_ROWS
        )

        a_offsets = start[0 : self.num_experts + 1]
        scale_offsets = scale_offsets[0 : self.num_experts]
        expert_ids = expert_ids[0 : self.num_experts]

        slot_token = order // self.topk
        x_sorted = F.gather(x, slot_token, axis=0)
        slot_weight = F.gather(F.reshape(weights, [slots]), order, axis=0).cast(
            DType.float32
        )

        gate_up = self._grouped(
            self.gate_up_proj, x_sorted, a_offsets, scale_offsets, expert_ids
        )
        down = self._activate_and_down(
            gate_up, slot_weight, a_offsets, scale_offsets, expert_ids
        )
        # ``restore`` puts each token's ``k`` slot rows back in token order;
        # the padded rows are never picked.
        out = F.gather(down, restore.cast(DType.int32), axis=0).cast(
            DType.float32
        )
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
        padded = F.concat(
            [
                x,
                F.zeros([1, x.shape[1]], dtype=x.dtype, device=x.mapping),
            ],
            axis=0,
        )
        weights, indices = self.gate(padded)
        routed = self.experts(padded, weights, indices)
        y = self.shared_experts(padded).cast(DType.float32) + routed
        return F.rebind(y[:-1], [x.shape[0], y.shape[1]]).cast(x.dtype)
