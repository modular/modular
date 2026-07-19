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

"""Quantization-aware kernel dispatch."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeAlias

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn.common_layers.functional_kernels import (
    fused_silu,
    fused_silu_quantized,
    grouped_matmul_ragged,
)
from max.experimental.sharding import DeviceMesh, PlacementMapping
from max.experimental.sharding.action import Action, ActionSet, AxisAssignment
from max.experimental.sharding.cost import (
    P,
    R,
    build_action_set,
)
from max.experimental.sharding.placements import Placement, Sharded
from max.experimental.sharding.types import TensorLayout
from max.experimental.tensor import Tensor
from max.graph import TensorValue
from max.nn.kernels import (
    dynamic_scaled_matmul as _dynamic_scaled_matmul,
)
from max.nn.kernels import (
    grouped_dynamic_scaled_fp8_matmul as _grouped_dynamic_scaled_fp8_matmul,
)
from max.nn.kernels import (
    quantize_dynamic_scaled_float8 as _quantize_dynamic_scaled_float8,
)
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)

from .quant_tensor import FP8BlockTensor, all_fp8_block

QuantAwareTensor: TypeAlias = Tensor | FP8BlockTensor


def _transpose2d_placement(p: Placement) -> Placement:
    """Map a rank-2 tensor placement to its transpose (swap axes 0 and 1).

    The activation block scales produced by
    :func:`quantize_dynamic_scaled_float8` are laid out transposed relative to
    the activations they describe (``[K / block_k, M]`` vs ``[M, K]``), so a
    shard of the activations on tensor axis ``a`` corresponds to a shard of the
    scales on axis ``1 - a``. :class:`~max.experimental.sharding.Replicated`
    and :class:`~max.experimental.sharding.Partial` placements are unchanged.
    """
    if isinstance(p, Sharded):
        return Sharded(axis=1 - p.axis)
    return p


def _quantize_finalize(action: Action) -> Action:
    """Expand the picked output into ``(data, scales)`` mappings.

    The picker derives one output placement, which we use for the FP8 data
    output (same layout as the input). This restores the second, transposed
    mapping for the block-scale output (see :func:`_transpose2d_placement`).
    """
    (data_mapping,) = action.outputs
    scales_mapping = PlacementMapping(
        data_mapping.mesh,
        tuple(_transpose2d_placement(p) for p in data_mapping.placements),
    )
    return Action(inputs=action.inputs, outputs=(data_mapping, scales_mapping))


def _quantize_rule(x: TensorLayout, *extras: Any) -> ActionSet:
    """Sharding rule for dynamic FP8 activation quantization.

    The FP8 data output follows the input placement; the block-scale output is
    its transpose (handled by :func:`_quantize_finalize`). Covers a replicated
    input (the common case), a contraction-sharded input (``o_proj`` under
    tensor parallelism), and a row-sharded input (sequence / data parallel).
    """
    rows = [
        AxisAssignment((R,), R),
        AxisAssignment((Sharded(0),), Sharded(0)),
        AxisAssignment((Sharded(1),), Sharded(1)),
    ]
    return build_action_set(rows, layouts=(x,), finalize=_quantize_finalize)


def _scaled_matmul_rule(
    a: TensorLayout,
    b: TensorLayout,
    a_scales: TensorLayout,
    b_scales: TensorLayout,
    *extras: Any,
) -> ActionSet:
    """Sharding rule for the block-scaled FP8 matmul ``a @ b.T``.

    ``a`` is ``[M, K]`` and ``b`` (Linear-convention weight) is ``[N, K]``, so
    the output is ``[M, N]``. The activation scales ``a_scales`` are
    ``[K / block_k, M]`` (transposed) while the weight scales ``b_scales`` are
    ``[N / block_m, K / block_k]``. The rows mirror the bf16 matmul strategies:
    column-parallel weights (shard ``N``), row-parallel weights (shard the
    ``K`` contraction, producing a partial sum), and row-sharded activations.
    """
    layouts = (a, b, a_scales, b_scales)
    rows = [
        AxisAssignment((R, R, R, R), R),
        # Column-parallel: weight rows (N) sharded -> output columns (N).
        AxisAssignment((R, Sharded(0), R, Sharded(0)), Sharded(1)),
        # Row-parallel: contraction (K) sharded on every operand -> partial.
        AxisAssignment((Sharded(1), Sharded(1), Sharded(0), Sharded(1)), P),
        # Row-sharded activations (sequence / data parallel) -> output rows.
        AxisAssignment((Sharded(0), R, Sharded(1), R), Sharded(0)),
    ]
    return build_action_set(rows, layouts=layouts)


def _grouped_scaled_matmul_rule(
    hidden_states: TensorLayout,
    weight: TensorLayout,
    a_scales: TensorLayout,
    b_scales: TensorLayout,
    expert_start_indices: TensorLayout,
    expert_ids: TensorLayout,
    *unused_kwargs: Any,
) -> ActionSet:
    """Sharding rule for the FP8 block-scaled grouped (MoE) matmul."""
    layouts = (
        hidden_states,
        weight,
        a_scales,
        b_scales,
        expert_start_indices,
        expert_ids,
    )
    rows = [
        AxisAssignment((R, R, R, R, R, R), R),
        # Column-parallel: weight's N (out) axis sharded -> output's N axis.
        AxisAssignment((R, Sharded(1), R, Sharded(1), R, R), Sharded(1)),
        # Row-parallel: weight's K (contraction) axis sharded, matched by
        # hidden_states' K axis; a_scales is transposed relative to
        # hidden_states, so its matching axis is 0 -> partial sum.
        AxisAssignment(
            (Sharded(1), Sharded(2), Sharded(0), Sharded(2), R, R), P
        ),
    ]
    return build_action_set(rows, layouts=layouts)


# Wrap raw graph ops so they accept ``Tensor`` and run inside an
# ``ensure_context()``.
quantize_dynamic_scaled_float8 = F.functional(
    _quantize_dynamic_scaled_float8, rule=_quantize_rule
)
dynamic_scaled_matmul = F.functional(
    _dynamic_scaled_matmul, rule=_scaled_matmul_rule
)
grouped_dynamic_scaled_fp8_matmul = F.functional(
    _grouped_dynamic_scaled_fp8_matmul, rule=_grouped_scaled_matmul_rule
)


def is_block_quantized(quant_config: QuantConfig | None) -> bool:
    """Return ``True`` if ``quant_config`` selects FP8 block-scaled weights."""
    return quant_config is not None and quant_config.weight_scale.is_block


def quantized_weight(
    out_dim: int,
    in_dim: int,
    quant_config: QuantConfig | None,
) -> QuantAwareTensor:
    """Build a Linear-shaped ``[out_dim, in_dim]`` weight parameter.

    Returns an :class:`FP8BlockTensor` when ``quant_config`` requests FP8
    block scaling, otherwise a plain bf16 :class:`Tensor` (dtype follows the
    ambient :func:`~max.experimental.tensor.default_dtype`). Used for
    parameter declaration inside a module ``__init__`` under
    :func:`~max.experimental.functional.lazy`.
    """
    if quant_config and quant_config.format != QuantFormat.BLOCKSCALED_FP8:
        raise ValueError(
            f"Quant type {quant_config.format} is not yet supported."
        )
    if is_block_quantized(quant_config):
        assert quant_config is not None
        block_size = quant_config.weight_scale.block_size
        assert block_size is not None
        return FP8BlockTensor.zeros(
            (int(out_dim), int(in_dim)), block_size=block_size
        )
    return Tensor.zeros((int(out_dim), int(in_dim)))


def stack(items: list[QuantAwareTensor], axis: int = 0) -> QuantAwareTensor:
    """Stack a homogeneous bundle along ``axis``, dispatching on quant type.

    For FP8 items, both leaves (``data`` and ``weight_scale_inv``) are stacked and
    rewrapped in an :class:`FP8BlockTensor`; for plain tensors the list is
    stacked directly. Companion to :func:`concat_weights`.

    Args:
        items: Homogeneous list of :class:`QuantAwareTensor`s (all plain
            tensors, or all :class:`FP8BlockTensor`s).
        axis: Axis to stack along (a new dimension is inserted here).

    Returns:
        A single stacked :class:`QuantAwareTensor` of the same kind as
        ``items``.
    """
    first = items[0]
    if isinstance(first, FP8BlockTensor):
        assert all_fp8_block(items)
        return FP8BlockTensor(
            data=F.stack([w.data for w in items], axis=axis),
            weight_scale_inv=F.stack(
                [w.weight_scale_inv for w in items], axis=axis
            ),
            block_size=first.block_size,
        )
    return F.stack(list(items), axis=axis)


def combine_quant_per_device(
    items: list[QuantAwareTensor],
    combine: Callable[[list[Tensor]], list[Tensor]],
) -> list[QuantAwareTensor]:
    """Map a per-device leaf transform over a homogeneous bundle.

    ``combine`` merges all items' tensors for one leaf into a per-device list
    (one tensor per mesh device) — e.g. a TP shard-and-stack.  For FP8 input,
    ``combine`` is applied to the ``data`` and ``weight_scale_inv`` leaves
    independently and the per-device leaves are zipped back into one
    :class:`FP8BlockTensor` per device, so the FP8 invariant (``data`` and
    ``weight_scale_inv`` are each a single :class:`~max.experimental.tensor.Tensor`)
    is preserved without the caller transposing a struct-of-lists into a
    list-of-structs.  For plain tensors the per-device list is returned as-is.

    ``combine`` must be leaf-agnostic — read any per-leaf difference (e.g. the
    block-scale leaf's smaller trailing dim) off the leaf tensors' own shapes
    rather than branching on which leaf it is.

    Args:
        items: Homogeneous list of :class:`QuantAwareTensor`s (all plain
            tensors, or all :class:`FP8BlockTensor`s).
        combine: Callable that merges a list of leaf tensors into a per-device
            list of tensors.

    Returns:
        One :class:`QuantAwareTensor` per device. For FP8 input, each is an
        :class:`FP8BlockTensor` whose ``data``/``weight_scale_inv`` are that device's
        leaves; for plain tensors, the per-device list is returned directly.
    """
    first = items[0]
    if isinstance(first, FP8BlockTensor):
        assert all_fp8_block(items)
        data = combine([w.data for w in items])
        weight_scale_inv = combine([w.weight_scale_inv for w in items])
        return [
            FP8BlockTensor(
                data=d, weight_scale_inv=s, block_size=first.block_size
            )
            for d, s in zip(data, weight_scale_inv, strict=True)
        ]
    plain = [w for w in items if isinstance(w, Tensor)]
    result: list[QuantAwareTensor] = [*combine(plain)]
    return result


def stack_device_shards(
    shards: Sequence[QuantAwareTensor], axis: int, mesh: DeviceMesh
) -> QuantAwareTensor:
    """Reassembles a per-device weight-shard bundle into one ``Sharded`` tensor."""
    if len(shards) == 1:
        return shards[0]
    mapping = PlacementMapping(mesh, (Sharded(axis=axis),))
    first = shards[0]
    if isinstance(first, FP8BlockTensor):
        assert all_fp8_block(shards)
        return FP8BlockTensor(
            data=Tensor.from_shard_values(
                [TensorValue(s.data) for s in shards], mapping
            ),
            weight_scale_inv=Tensor.from_shard_values(
                [TensorValue(s.weight_scale_inv) for s in shards], mapping
            ),
            block_size=first.block_size,
        )
    return Tensor.from_shard_values(
        [TensorValue(s) for s in shards if isinstance(s, Tensor)], mapping
    )


def concat_weights(
    *weights: QuantAwareTensor, axis: int = 0
) -> QuantAwareTensor:
    """Concatenate weights along ``axis``, dispatching on the weight type."""
    if not weights:
        raise ValueError("concat_weights requires at least one tensor")
    if isinstance(weights[0], FP8BlockTensor):
        assert all_fp8_block(weights), (
            "concat_weights requires all weights to be FP8BlockTensor when "
            "the first is"
        )
        return concat_fp8_block(*weights, axis=axis)
    assert all(not isinstance(w, FP8BlockTensor) for w in weights)
    return F.concat(list(weights), axis=axis)


def _fp8_block_specs(
    weight_block: tuple[int, int],
    *,
    input_block: tuple[int, int] = (1, 128),
) -> tuple[InputScaleSpec, WeightScaleSpec]:
    """Standard FP8 block-scale specs for matmul/grouped-matmul kernels."""
    return (
        InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.DYNAMIC,
            dtype=DType.float32,
            block_size=input_block,
        ),
        WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float32,
            block_size=weight_block,
        ),
    )


def matmul(x: Tensor, weight: QuantAwareTensor) -> Tensor:
    """Matmul ``x @ weight.T`` dispatching on the weight type.

    ``weight`` follows the Linear convention: shape ``[out_dim, in_dim]``.

    - ``Tensor`` weight: regular bf16/float matmul.
    - :class:`FP8BlockTensor` weight: quantizes ``x`` to FP8 with
      ``(1, block_k)`` activation blocks, then runs the block-scaled FP8
      matmul kernel and returns bf16.
    """
    if isinstance(weight, FP8BlockTensor):
        return _matmul_fp8_block(x, weight)
    return x @ weight.T


def _matmul_fp8_block(x: Tensor, weight: FP8BlockTensor) -> Tensor:
    """Block-scaled FP8 matmul ``x @ weight.data.T`` with dynamic activation
    quantization.

    The activation block is ``(1, block_k)`` and the weight block is
    ``(block_m, block_k) = weight.block_size``. The kernel returns bf16.
    """
    block_m, block_k = weight.block_size
    input_spec, weight_spec = _fp8_block_specs(
        (block_m, block_k), input_block=(1, block_k)
    )

    x_fp8, x_scales = quantize_dynamic_scaled_float8(
        x,
        input_spec,
        weight_spec,
        group_size_or_per_token=block_k,
        scales_type=DType.float32,
        out_type=DType.float8_e4m3fn,
    )

    return dynamic_scaled_matmul(
        x_fp8,
        weight.data,
        x_scales,
        weight.weight_scale_inv,
        input_spec,
        weight_spec,
        out_type=DType.bfloat16,
    )


def grouped_matmul(
    x: QuantAwareTensor,
    weight: QuantAwareTensor,
    expert_start_indices: Tensor,
    expert_ids: Tensor,
    expert_usage_stats: Tensor,
    *,
    out_type: DType = DType.bfloat16,
) -> Tensor:
    """Grouped (MoE) matmul dispatching on the stacked-weight type.

    For a plain ``Tensor`` weight of shape ``[num_experts, N, K]``, this
    falls back to the standard ragged grouped matmul. For an
    :class:`FP8BlockTensor` weight, it runs the block-scaled FP8 grouped
    matmul kernel.

    Args:
        x: Ragged activations of shape ``[total_tokens, K]``.
        weight: Stacked expert weights, ``[num_experts, N, K]``.
        expert_start_indices: Ragged group offsets, ``uint32``.
        expert_ids: Per-group expert id, ``int32``.
        expert_usage_stats: ``[max_tokens_per_expert, num_active_experts]``
            device tensor. The bf16 fallback requires it on-device (the SM100
            kernel reads ``num_active_experts`` there); the FP8 branch copies it
            to CPU itself.
        out_type: Output dtype for the FP8 branch (bf16 by default).
    """
    if isinstance(weight, FP8BlockTensor):
        if isinstance(x, FP8BlockTensor):
            # Activations already block-quantized (e.g. by the EP FP8 dispatch
            # or a quantized SiLU); pass them through directly.
            x_fp8, x_scales = x.data, x.weight_scale_inv
        else:
            x_fp8, x_scales = _quantize_activation_fp8(x, weight.block_size)
        return _grouped_fp8_matmul(
            x_fp8,
            x_scales,
            weight,
            expert_start_indices,
            expert_ids,
            expert_usage_stats,
            out_type=out_type,
        )
    assert isinstance(x, Tensor)
    return grouped_matmul_ragged(
        x, weight, expert_start_indices, expert_ids, expert_usage_stats
    )


def _quantize_activation_fp8(
    x: Tensor, weight_block: tuple[int, int]
) -> tuple[Tensor, Tensor]:
    """Per-token FP8 block quantization of ``x`` matching ``weight_block``."""
    block_m, block_k = weight_block
    input_spec, weight_spec = _fp8_block_specs(
        (block_m, block_k), input_block=(1, block_k)
    )
    return quantize_dynamic_scaled_float8(
        x,
        input_spec,
        weight_spec,
        scales_type=DType.float32,
        out_type=DType.float8_e4m3fn,
    )


def _grouped_fp8_matmul(
    x_fp8: Tensor,
    x_scales: Tensor,
    weight: FP8BlockTensor,
    expert_start_indices: Tensor,
    expert_ids: Tensor,
    expert_usage_stats: Tensor,
    *,
    out_type: DType = DType.bfloat16,
) -> Tensor:
    """Block-scaled FP8 grouped matmul on already-quantized activations."""
    block_m, block_k = weight.block_size
    input_spec, weight_spec = _fp8_block_specs(
        (block_m, block_k), input_block=(1, block_k)
    )
    # This kernel reads the usage stats host-side, so copy to CPU here rather
    # than at the call site (the bf16 path needs them on-device).
    return grouped_dynamic_scaled_fp8_matmul(
        x_fp8,
        weight.data,
        x_scales,
        weight.weight_scale_inv,
        expert_start_indices,
        expert_ids,
        expert_usage_stats.to(CPU()),
        input_spec,
        weight_spec,
        out_type=out_type,
    )


def grouped_silu(
    x: Tensor,
    expert_start_indices: Tensor,
    out_weight: QuantAwareTensor | None = None,
    quant_config: QuantConfig | None = None,
) -> QuantAwareTensor:
    """SiLU-gate a grouped gate/up output."""
    if isinstance(out_weight, FP8BlockTensor):
        assert quant_config is not None
        _, block_k = out_weight.block_size
        data, weight_scale_inv = fused_silu_quantized(
            x, expert_start_indices, quant_config, DType.float8_e4m3fn
        )
        return FP8BlockTensor(
            data=data,
            weight_scale_inv=weight_scale_inv,
            block_size=(1, block_k),
        )
    return fused_silu(x, expert_start_indices)


def concat_fp8_block(*tensors: FP8BlockTensor, axis: int = 0) -> FP8BlockTensor:
    """Concatenate two or more :class:`FP8BlockTensor`s along ``axis``."""
    if not tensors:
        raise ValueError("concat_fp8_block requires at least one tensor")
    if axis != 0:
        raise ValueError(
            "FP8BlockTensor concat currently only supports axis=0 (row axis)"
        )
    block_size = tensors[0].block_size
    for q in tensors[1:]:
        if q.block_size != block_size:
            raise ValueError(
                "All FP8BlockTensors must have the same block_size to "
                f"concat; got {block_size} and {q.block_size}"
            )

    data = F.concat([q.data for q in tensors], axis=0)
    weight_scale_inv = F.concat([q.weight_scale_inv for q in tensors], axis=0)
    return FP8BlockTensor(
        data=data, weight_scale_inv=weight_scale_inv, block_size=block_size
    )
