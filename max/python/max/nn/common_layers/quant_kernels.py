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

"""Quantized kernels as ``Tensor`` ops, with their sharding rules.

Each kernel from :mod:`max.nn.legacy.kernels` is wrapped with
:func:`~max.functional.functional` and the rule that says how it
runs on sharded inputs. :mod:`.quant_ops` dispatches onto these, and the
on-the-fly quantizers in :mod:`.quant_tensor` call them directly; this module
imports neither, so both can depend on it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from max import functional as F
from max.dtype import DType
from max.graph import BufferType, TensorValue, ops
from max.nn.legacy.kernels import (
    BLOCKWISE_FP8_LANE_BYTES,
    amd_mega_ffn_workspace_words,
    preshuffle_block_scaled_b,
)
from max.nn.legacy.kernels import (
    block_scales_interleave as _block_scales_interleave,
)
from max.nn.legacy.kernels import (
    dynamic_block_scaled_matmul as _dynamic_block_scaled_matmul,
)
from max.nn.legacy.kernels import (
    dynamic_block_scaled_matmul_amd as _dynamic_block_scaled_matmul_amd,
)
from max.nn.legacy.kernels import (
    dynamic_scaled_matmul as _dynamic_scaled_matmul,
)
from max.nn.legacy.kernels import (
    grouped_dynamic_scaled_fp8_matmul as _grouped_dynamic_scaled_fp8_matmul,
)
from max.nn.legacy.kernels import (
    grouped_ffn_blockwise_fp8_amd as _grouped_ffn_blockwise_fp8_amd,
)
from max.nn.legacy.kernels import (
    grouped_matmul_block_scaled as _grouped_matmul_block_scaled,
)
from max.nn.legacy.kernels import (
    grouped_quantize_dynamic_block_scaled as _grouped_quantize_dynamic_block_scaled,
)
from max.nn.legacy.kernels import (
    moe_fp8_host_usage_stats as _moe_fp8_host_usage_stats,
)
from max.nn.legacy.kernels import (
    quantize_dynamic_block_scaled as _quantize_dynamic_block_scaled,
)
from max.nn.legacy.kernels import (
    quantize_dynamic_block_scaled_mxfp4 as _quantize_dynamic_block_scaled_mxfp4,
)
from max.nn.legacy.kernels import (
    quantize_dynamic_scaled_float8 as _quantize_dynamic_scaled_float8,
)
from max.nn.legacy.quant_config import InputScaleSpec, WeightScaleSpec
from max.tensor.sharding.action import AxisAssignment
from max.tensor.sharding.cost import P, R
from max.tensor.sharding.placements import Placement, Sharded
from max.tensor.sharding.types import TensorLayout


def _transpose2d_placement(p: Placement) -> Placement:
    """Map a rank-2 tensor placement to its transpose (swap axes 0 and 1).

    The activation block scales produced by
    :func:`quantize_dynamic_scaled_float8` are laid out transposed relative to
    the activations they describe (``[K / block_k, M]`` vs ``[M, K]``), so a
    shard of the activations on tensor axis ``a`` corresponds to a shard of the
    scales on axis ``1 - a``. :class:`~max.tensor.sharding.Replicated`
    and :class:`~max.tensor.sharding.Partial` placements are unchanged.
    """
    if isinstance(p, Sharded):
        return Sharded(axis=1 - p.axis)
    return p


def _follow_input_rows(
    x: TensorLayout,
    scales_placement: Callable[[Placement], Placement] | None = None,
) -> list[AxisAssignment]:
    """Returns rows whose results follow the rank-2 input's placement.

    Covers a replicated input (the common case), a contraction-sharded input
    (``o_proj`` under tensor parallelism), and a row-sharded input (sequence /
    data parallel). There is no partial-sum row: quantizing is not linear,
    so a partial input is reduced first.

    Args:
        x: The rank-2 input whose placement the results follow.
        scales_placement: The placement of the block scales a quantize op
            also returns, given the data's placement. ``None`` for an op
            with one result.
    """

    def outputs(p: Placement) -> tuple[Placement, ...]:
        return (p,) if scales_placement is None else (p, scales_placement(p))

    return [
        AxisAssignment((p,), outputs(p)) for p in (R, Sharded(0), Sharded(1))
    ]


def _quantize_rule(x: TensorLayout, *extras: Any) -> list[AxisAssignment]:
    """Sharding rule for dynamic FP8 activation quantization.

    The FP8 data output follows the input placement; the block-scale output
    is its transpose.
    """
    return _follow_input_rows(x, _transpose2d_placement)


def _scaled_matmul_rule(
    a: TensorLayout,
    b: TensorLayout,
    a_scales: TensorLayout,
    b_scales: TensorLayout,
    *extras: Any,
    **kwargs: Any,
) -> list[AxisAssignment]:
    """Sharding rule for the block-scaled FP8 matmul ``a @ b.T``.

    ``a`` is ``[M, K]`` and ``b`` (Linear-convention weight) is ``[N, K]``, so
    the output is ``[M, N]``. The activation scales ``a_scales`` are
    ``[K / block_k, M]`` (transposed) while the weight scales ``b_scales`` are
    ``[N / block_m, K / block_k]``. The rows mirror the bf16 matmul strategies:
    column-parallel weights (shard ``N``), row-parallel weights (shard the
    ``K`` contraction, producing a partial sum), and row-sharded activations.
    """
    return [
        AxisAssignment((R, R, R, R), (R,)),
        # Column-parallel: weight rows (N) sharded -> output columns (N).
        AxisAssignment((R, Sharded(0), R, Sharded(0)), (Sharded(1),)),
        # Row-parallel: contraction (K) sharded on every operand -> partial.
        AxisAssignment((Sharded(1), Sharded(1), Sharded(0), Sharded(1)), (P,)),
        # Row-sharded activations (sequence / data parallel) -> output rows.
        AxisAssignment((Sharded(0), R, Sharded(1), R), (Sharded(0),)),
    ]


def _grouped_scaled_matmul_rule(
    hidden_states: TensorLayout,
    weight: TensorLayout,
    a_scales: TensorLayout,
    b_scales: TensorLayout,
    expert_start_indices: TensorLayout,
    expert_ids: TensorLayout,
    expert_usage_stats_host: TensorLayout,
    input_scale_spec: InputScaleSpec,
    weight_scale_spec: WeightScaleSpec,
    out_type: DType = DType.bfloat16,
    static_grid_z: bool = False,
    decode_grid_m_cap: int = 0,
) -> list[AxisAssignment]:
    """Sharding rule for the FP8 block-scaled grouped (MoE) matmul."""
    return [
        AxisAssignment((R, R, R, R, R, R, R), (R,)),
        # Column-parallel: weight's N (out) axis sharded -> output's N axis.
        AxisAssignment((R, Sharded(1), R, Sharded(1), R, R, R), (Sharded(1),)),
        # Row-parallel: weight's K (contraction) axis sharded, matched by
        # hidden_states' K axis; a_scales is transposed relative to
        # hidden_states, so its matching axis is 0 -> partial sum.
        AxisAssignment(
            (Sharded(1), Sharded(2), Sharded(0), Sharded(2), R, R, R), (P,)
        ),
    ]


def _block_scaled_quantize_rule(
    input: TensorLayout,
    tensor_sf: TensorLayout | float = 1.0,
    sf_vector_size: int = 16,
    scales_type: DType = DType.float8_e4m3fn,
    out_type: DType = DType.uint8,
) -> list[AxisAssignment]:
    """Sharding rule for dynamic block quantization (NVFP4 or MXFP8)."""
    rows = _follow_input_rows(input, lambda placement: placement)
    if not isinstance(tensor_sf, TensorLayout):
        return rows
    return [
        AxisAssignment((*row.needed_inputs, R), row.outputs) for row in rows
    ]


def _mxfp4_quantize_rule(
    input: TensorLayout,
    scales_type: DType = DType.float8_e8m0fnu,
    out_type: DType = DType.uint8,
) -> list[AxisAssignment]:
    """Sharding rule for the dedicated MXFP4 quantize op.

    Simpler than :func:`_block_scaled_quantize_rule`: MXFP4 is pure block
    scaling, so there is no ``tensor_sf`` operand to append a row for.
    """
    return _follow_input_rows(input, lambda placement: placement)


def _nvfp4_interleave_rule(
    scales: TensorLayout, *extras: Any
) -> list[AxisAssignment]:
    """Sharding rule for ``block_scales_interleave`` (rank-2 -> rank-5)."""
    return _follow_input_rows(scales)


def _nvfp4_matmul_rule(
    a: TensorLayout,
    b: TensorLayout,
    a_scales: TensorLayout,
    b_scales: TensorLayout,
    tensor_sf: TensorLayout | float = 1.0,
    sf_vector_size: int = 16,
    out_type: DType = DType.bfloat16,
) -> list[AxisAssignment]:
    """Sharding rule for the block-scaled NVFP4 matmul ``a @ b.T``."""
    rows = [
        AxisAssignment((R, R, R, R), (R,)),
        # Column-parallel: weight rows (N) sharded -> output columns (N).
        AxisAssignment((R, Sharded(0), R, Sharded(0)), (Sharded(1),)),
        # Row-parallel: contraction (K) sharded on every operand -> partial.
        AxisAssignment((Sharded(1), Sharded(1), Sharded(1), Sharded(1)), (P,)),
        # Row-sharded activations (sequence / data parallel) -> output rows.
        AxisAssignment((Sharded(0), R, Sharded(0), R), (Sharded(0),)),
    ]
    if isinstance(tensor_sf, TensorLayout):
        rows = [
            AxisAssignment((*row.needed_inputs, R), row.outputs) for row in rows
        ]
    return rows


def _mx_matmul_amd_rule(
    a: TensorLayout,
    b: TensorLayout,
    a_scales: TensorLayout,
    b_scales: TensorLayout,
    out_type: DType = DType.bfloat16,
) -> list[AxisAssignment]:
    """Sharding rule for the AMD block-scaled MX matmul ``a @ b.T``.

    Same operand roles as :func:`_nvfp4_matmul_rule`. The AMD op takes no
    ``tensor_sf``, so the trailing-input variant that rule appends does not
    apply here.
    """
    return _nvfp4_matmul_rule(a, b, a_scales, b_scales)


def _nvfp4_grouped_quantize_rule(
    input: TensorLayout,
    row_offsets: TensorLayout,
    scales_offsets: TensorLayout,
    expert_ids: TensorLayout,
    sf_tensor: TensorLayout,
    *extras: Any,
    **kwargs: Any,
) -> list[AxisAssignment]:
    """Sharding rule for grouped NVFP4 activation quantization."""
    # The block scales are placed like the quantized data, so each row
    # names that placement twice: once for the data, once for the scales.
    return [
        AxisAssignment((R, R, R, R, R), (R, R)),
        AxisAssignment((Sharded(1), R, R, R, R), (Sharded(1), Sharded(1))),
    ]


def _nvfp4_grouped_matmul_rule(
    hidden_states: TensorLayout,
    weight: TensorLayout,
    a_scales: TensorLayout,
    b_scales: TensorLayout,
    expert_start_indices: TensorLayout,
    a_scale_offsets: TensorLayout,
    expert_ids: TensorLayout,
    expert_scales: TensorLayout,
    expert_usage_stats_host: TensorLayout,
    out_type: DType = DType.bfloat16,
    estimated_total_m: TensorLayout | None = None,
) -> list[AxisAssignment]:
    """Sharding rule for the NVFP4 block-scaled grouped (MoE) matmul."""
    layouts = (
        hidden_states,
        weight,
        a_scales,
        b_scales,
        expert_start_indices,
        a_scale_offsets,
        expert_ids,
        expert_scales,
        expert_usage_stats_host,
        *((estimated_total_m,) if estimated_total_m is not None else ()),
    )
    metadata = (R,) * (len(layouts) - 4)
    return [
        AxisAssignment((R, R, R, R, *metadata), (R,)),
        # Column-parallel: weight's N axis (1) -> output's N axis.
        AxisAssignment(
            (R, Sharded(1), R, Sharded(1), *metadata), (Sharded(1),)
        ),
        # Row-parallel: weight's K axis (2) matched by hidden/a_scales K -> P.
        AxisAssignment(
            (Sharded(1), Sharded(2), Sharded(1), Sharded(2), *metadata),
            (P,),
        ),
    ]


quantize_dynamic_scaled_float8 = F.functional(
    _quantize_dynamic_scaled_float8, rule=_quantize_rule
)
quantize_dynamic_block_scaled = F.functional(
    _quantize_dynamic_block_scaled, rule=_block_scaled_quantize_rule
)
quantize_dynamic_block_scaled_mxfp4 = F.functional(
    _quantize_dynamic_block_scaled_mxfp4, rule=_mxfp4_quantize_rule
)
dynamic_scaled_matmul = F.functional(
    _dynamic_scaled_matmul, rule=_scaled_matmul_rule
)
grouped_dynamic_scaled_fp8_matmul = F.functional(
    _grouped_dynamic_scaled_fp8_matmul, rule=_grouped_scaled_matmul_rule
)
block_scales_interleave = F.functional(
    _block_scales_interleave, rule=_nvfp4_interleave_rule
)
dynamic_block_scaled_matmul = F.functional(
    _dynamic_block_scaled_matmul, rule=_nvfp4_matmul_rule
)
dynamic_block_scaled_matmul_amd = F.functional(
    _dynamic_block_scaled_matmul_amd, rule=_mx_matmul_amd_rule
)
grouped_quantize_dynamic_block_scaled = F.functional(
    _grouped_quantize_dynamic_block_scaled, rule=_nvfp4_grouped_quantize_rule
)
grouped_matmul_block_scaled = F.functional(
    _grouped_matmul_block_scaled, rule=_nvfp4_grouped_matmul_rule
)


def _silu_quantize_fp8(
    x: TensorValue,
    row_offsets: TensorValue,
    input_spec: InputScaleSpec,
    weight_spec: WeightScaleSpec,
    block_k: int,
    swiglu_limit: float = 0.0,
) -> tuple[TensorValue, TensorValue]:
    """Gated SiLU of a ``[gate | up]`` grouped matmul output, FP8 block-quantized.

    The graph compiler fuses the activation into the row-bounded quantize, so
    the rows past ``row_offsets[-1]`` of the EP receive buffer are neither
    activated nor quantized.
    """
    moe_dim = int(x.shape[1]) // 2
    gate, up = x[:, :moe_dim], x[:, moe_dim:]
    if swiglu_limit:
        gate = ops.min(gate, swiglu_limit)
        up = ops.min(ops.max(up, -swiglu_limit), swiglu_limit)
    activated = ops.silu(gate) * up
    return _quantize_dynamic_scaled_float8(
        activated,
        input_spec,
        weight_spec,
        group_size_or_per_token=block_k,
        scales_type=DType.float32,
        out_type=DType.float8_e4m3fn,
        row_offsets=row_offsets,
    )


silu_quantize_fp8 = F.functional(_silu_quantize_fp8)
moe_fp8_host_usage_stats = F.functional(_moe_fp8_host_usage_stats)


def _grouped_ffn_blockwise_fp8_amd_graph(
    x: TensorValue,
    x_scales: TensorValue,
    gate_up: TensorValue,
    gate_up_scales: TensorValue,
    down: TensorValue,
    down_scales: TensorValue,
    expert_start_indices: TensorValue,
    expert_ids: TensorValue,
    usage_stats: TensorValue,
    estimated_total_m: TensorValue,
    mega_band_m: int,
    mega_max_rows: int,
    activation_fp32: bool,
    swiglu_limit: float,
) -> TensorValue:
    # The counters persist across executions: one buffer per call site,
    # zeroed once at load and never again.
    workspace = ops.buffer_create(
        BufferType(
            DType.uint32,
            shape=[
                amd_mega_ffn_workspace_words(
                    int(gate_up.shape[0]),
                    mega_max_rows,
                    int(x.shape[0]),
                    grids=2,
                )
            ],
            device=x.device,
        ),
        init_value=0,
    )
    return _grouped_ffn_blockwise_fp8_amd(
        x,
        gate_up,
        x_scales,
        gate_up_scales,
        down,
        down_scales,
        expert_start_indices,
        expert_ids,
        usage_stats,
        workspace,
        estimated_total_m,
        mega_band_m=mega_band_m,
        mega_max_rows=mega_max_rows,
        activation_fp32=activation_fp32,
        swiglu_limit=swiglu_limit,
    )


grouped_ffn_blockwise_fp8_amd = F.functional(
    _grouped_ffn_blockwise_fp8_amd_graph
)


def _preshuffle_blockwise_fp8_b(weight: TensorValue) -> TensorValue:
    return preshuffle_block_scaled_b(weight, BLOCKWISE_FP8_LANE_BYTES)


preshuffle_blockwise_fp8_b = F.functional(_preshuffle_blockwise_fp8_b)


__all__ = [
    "block_scales_interleave",
    "dynamic_block_scaled_matmul",
    "dynamic_block_scaled_matmul_amd",
    "dynamic_scaled_matmul",
    "grouped_dynamic_scaled_fp8_matmul",
    "grouped_ffn_blockwise_fp8_amd",
    "grouped_matmul_block_scaled",
    "grouped_quantize_dynamic_block_scaled",
    "moe_fp8_host_usage_stats",
    "preshuffle_blockwise_fp8_b",
    "quantize_dynamic_block_scaled",
    "quantize_dynamic_block_scaled_mxfp4",
    "quantize_dynamic_scaled_float8",
    "silu_quantize_fp8",
]
