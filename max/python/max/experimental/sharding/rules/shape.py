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

"""TensorLayout-based rules for shape-manipulation ops.

Each rule receives only the parameters it inspects or modifies.
"""

from __future__ import annotations

import builtins
from collections.abc import Iterable, Sequence
from typing import Literal

from max.experimental.sharding.mappings import DeviceMapping, PlacementMapping
from max.experimental.sharding.placements import Placement, Replicated, Sharded
from max.experimental.sharding.types import TensorLayout
from max.graph import Dim, DimLike, StaticDim

from ._common import (
    RuleSignature,
    reject_any_sharded,
    reject_sharded_axis,
    remap_sharded,
    resolve_partials_mapping,
)


def _localize_sizes(
    sizes: Sequence[DimLike],
    axis: int,
    ndim: int,
    placements: tuple[Placement, ...],
    mesh_shape: tuple[int, ...],
) -> list[Dim] | None:
    norm_axis = axis % ndim
    for mesh_ax, p in enumerate(placements):
        if isinstance(p, Sharded) and p.axis == norm_axis:
            return [Dim(s) // mesh_shape[mesh_ax] for s in sizes]
    return None


# ═══════════════════════════════════════════════════════════════════════
#  Simple passthrough / reject rules
# ═══════════════════════════════════════════════════════════════════════


def passthrough_rule(x: TensorLayout, linear: bool = False) -> RuleSignature:
    """Sharding rule for passthrough."""
    s = resolve_partials_mapping(x.mapping) if not linear else x.mapping
    return (s,), (s,)


def tile_rule(x: TensorLayout, repeats: Iterable[DimLike]) -> RuleSignature:
    """Sharding rule for tile."""
    return (x.mapping, repeats), (x.mapping,)


# ═══════════════════════════════════════════════════════════════════════
#  Gather / scatter family
# ═══════════════════════════════════════════════════════════════════════


def gather_rule(
    input: TensorLayout, indices: TensorLayout, axis: int
) -> RuleSignature:
    """Sharding rule for gather."""
    s = resolve_partials_mapping(input.mapping)
    si = resolve_partials_mapping(indices.mapping)
    reject_sharded_axis(s.to_placements(), axis % input.rank, "gather")
    return (s, si, axis), (s,)


def scatter_rule(
    input: TensorLayout,
    updates: TensorLayout,
    indices: TensorLayout,
    axis: int = -1,
) -> RuleSignature:
    """Sharding rule for scatter."""
    s = resolve_partials_mapping(input.mapping)
    su = resolve_partials_mapping(updates.mapping)
    si = resolve_partials_mapping(indices.mapping)
    reject_sharded_axis(s.to_placements(), axis % input.rank, "scatter")
    return (s, su, si, axis), (s,)


def scatter_nd_rule(
    input: TensorLayout, updates: TensorLayout, indices: TensorLayout
) -> RuleSignature:
    """Sharding rule for scatter nd."""
    s = resolve_partials_mapping(input.mapping)
    su = resolve_partials_mapping(updates.mapping)
    si = resolve_partials_mapping(indices.mapping)
    reject_any_sharded(s.to_placements(), "scatter_nd")
    return (s, su, si), (s,)


def scatter_add_rule(
    input: TensorLayout,
    updates: TensorLayout,
    indices: TensorLayout,
    axis: int = -1,
) -> RuleSignature:
    """Sharding rule for scatter add (also used by scatter_max/min/mul)."""
    s = resolve_partials_mapping(input.mapping)
    su = resolve_partials_mapping(updates.mapping)
    si = resolve_partials_mapping(indices.mapping)
    reject_sharded_axis(s.to_placements(), axis % input.rank, "scatter_add")
    return (s, su, si, axis), (s,)


def scatter_nd_add_rule(
    input: TensorLayout, updates: TensorLayout, indices: TensorLayout
) -> RuleSignature:
    """Sharding rule for scatter nd add (also used by scatter_nd_max/min/mul)."""
    s = resolve_partials_mapping(input.mapping)
    su = resolve_partials_mapping(updates.mapping)
    si = resolve_partials_mapping(indices.mapping)
    reject_any_sharded(s.to_placements(), "scatter_nd_add")
    return (s, su, si), (s,)


def masked_scatter_rule(
    input: TensorLayout,
    mask: TensorLayout,
    updates: TensorLayout,
    out_dim: DimLike,
) -> RuleSignature:
    """Sharding rule for masked scatter."""
    s = resolve_partials_mapping(input.mapping)
    sm = resolve_partials_mapping(mask.mapping)
    su = resolve_partials_mapping(updates.mapping)
    reject_any_sharded(s.to_placements(), "masked_scatter")
    return (s, sm, su, out_dim), (s,)


def gather_nd_rule(
    input: TensorLayout, indices: TensorLayout, batch_dims: int = 0
) -> RuleSignature:
    """Sharding rule for gather nd."""
    sx = resolve_partials_mapping(input.mapping)
    si = resolve_partials_mapping(indices.mapping)
    for p in sx.to_placements():
        if isinstance(p, Sharded) and p.axis >= batch_dims:
            raise ValueError(
                f"gather_nd: cannot gather from tensor sharded along "
                f"non-batch axis {p.axis} (batch_dims={batch_dims})."
            )
    return (sx, si, batch_dims), (sx,)


# ═══════════════════════════════════════════════════════════════════════
#  Single-tensor + axis/shape params
# ═══════════════════════════════════════════════════════════════════════


def chunk_rule(x: TensorLayout, chunks: int, axis: int = 0) -> RuleSignature:
    """Sharding rule for chunk."""
    s = resolve_partials_mapping(x.mapping)
    reject_sharded_axis(s.to_placements(), axis % x.rank, "chunk")
    return (s, chunks, axis), (s,)


def top_k_rule(input: TensorLayout, k: int, axis: int = -1) -> RuleSignature:
    """Sharding rule for top_k (also used by bottom_k)."""
    s = resolve_partials_mapping(input.mapping)
    reject_sharded_axis(s.to_placements(), axis % input.rank, "top_k")
    return (s, k, axis), (s,)


def argsort_rule(x: TensorLayout, ascending: bool = True) -> RuleSignature:
    """Sharding rule for argsort."""
    s = resolve_partials_mapping(x.mapping)
    reject_any_sharded(s.to_placements(), "argsort")
    return (s, ascending), (s,)


def nonzero_rule(x: TensorLayout, out_dim: DimLike) -> RuleSignature:
    """Sharding rule for nonzero."""
    s = resolve_partials_mapping(x.mapping)
    reject_any_sharded(s.to_placements(), "nonzero")
    return (s, out_dim), (s,)


def repeat_interleave_rule(
    x: TensorLayout,
    repeats: int | TensorLayout,
    axis: int | None = None,
    out_dim: DimLike | None = None,
) -> RuleSignature:
    """Sharding rule for repeat interleave."""
    s = resolve_partials_mapping(x.mapping)
    sp = s.to_placements()
    if axis is not None:
        reject_sharded_axis(sp, axis % x.rank, "repeat_interleave")
    else:
        reject_any_sharded(sp, "repeat_interleave (axis=None)")
    return (s, repeats, axis, out_dim), (s,)


def pad_rule(
    input: TensorLayout,
    paddings: Iterable[int],
    mode: Literal["constant", "reflect", "edge"] = "constant",
    value: TensorLayout | int | float = 0,
) -> RuleSignature:
    """Sharding rule for pad."""
    s = input.mapping
    pad_list = list(paddings)
    if pad_list:
        for p in s.to_placements():
            if isinstance(p, Sharded):
                idx = p.axis * 2
                if idx + 1 < len(pad_list) and (
                    pad_list[idx] != 0 or pad_list[idx + 1] != 0
                ):
                    raise ValueError(
                        f"pad: cannot pad along sharded axis {p.axis}."
                    )
    return (s, pad_list, mode, value), (s,)


def slice_tensor_rule(x: TensorLayout, indices: object = None) -> RuleSignature:
    """Sharding rule for slice_tensor."""
    s = x.mapping
    if indices is not None:
        for p in s.to_placements():
            if isinstance(p, Sharded) and _slice_modifies_axis(
                indices, p.axis, x.rank
            ):
                raise ValueError(
                    f"slice_tensor: cannot slice along sharded axis {p.axis}."
                )
    return (s, indices), (s,)


def _slice_modifies_axis(indices: object, axis: int, ndim: int) -> bool:
    if not isinstance(indices, (list, tuple)):
        return True
    expanded: list[object] = []
    for idx in indices:
        if idx is Ellipsis:
            n_explicit = builtins.sum(1 for i in indices if i is not Ellipsis)
            expanded.extend([slice(None)] * (ndim - n_explicit))
        else:
            expanded.append(idx)
    while len(expanded) < ndim:
        expanded.append(slice(None))
    if axis >= len(expanded):
        return False
    return not (
        isinstance(expanded[axis], slice) and expanded[axis] == slice(None)
    )


# ═══════════════════════════════════════════════════════════════════════
#  Axis remapping
# ═══════════════════════════════════════════════════════════════════════


def permute_rule(x: TensorLayout, dims: list[int]) -> RuleSignature:
    """Sharding rule for permute."""
    s = x.mapping
    mesh = x.mapping.mesh
    inv = [0] * len(dims)
    for i, d in enumerate(dims):
        inv[d] = i
    out_m = PlacementMapping(
        mesh, remap_sharded(s.to_placements(), lambda a: inv[a])
    )
    return (s, dims), (out_m,)


def transpose_rule(x: TensorLayout, axis_1: int, axis_2: int) -> RuleSignature:
    """Sharding rule for transpose."""
    s = x.mapping
    mesh = x.mapping.mesh
    a1, a2 = axis_1 % x.rank, axis_2 % x.rank
    swap = {a1: a2, a2: a1}
    out_m = PlacementMapping(
        mesh, remap_sharded(s.to_placements(), lambda a: swap.get(a, a))
    )
    return (s, axis_1, axis_2), (out_m,)


def unsqueeze_rule(x: TensorLayout, axis: int) -> RuleSignature:
    """Sharding rule for unsqueeze."""
    s = x.mapping
    mesh = x.mapping.mesh
    norm = axis if axis >= 0 else axis + x.rank + 1
    out_m = PlacementMapping(
        mesh,
        remap_sharded(s.to_placements(), lambda a: a + 1 if a >= norm else a),
    )
    return (s, axis), (out_m,)


def squeeze_rule(x: TensorLayout, axis: int) -> RuleSignature:
    """Sharding rule for squeeze."""
    s = x.mapping
    sp = s.to_placements()
    mesh = x.mapping.mesh
    norm = axis % x.rank
    reject_sharded_axis(sp, norm, "squeeze")
    out_m = PlacementMapping(
        mesh, remap_sharded(sp, lambda a: a - 1 if a > norm else a)
    )
    return (s, axis), (out_m,)


def flatten_rule(
    x: TensorLayout, start_dim: int = 0, end_dim: int = -1
) -> RuleSignature:
    """Sharding rule for flatten."""
    s = x.mapping
    sp = s.to_placements()
    mesh = x.mapping.mesh
    ndim = x.rank
    sd = start_dim if start_dim >= 0 else start_dim + ndim
    ed = end_dim if end_dim >= 0 else end_dim + ndim
    removed = ed - sd
    for p in sp:
        if isinstance(p, Sharded) and sd <= p.axis <= ed:
            raise ValueError(
                f"flatten: cannot flatten across sharded axis {p.axis} "
                f"(range [{sd}, {ed}])."
            )
    out_m = PlacementMapping(
        mesh, remap_sharded(sp, lambda a: a - removed if a > ed else a)
    )
    return (s, start_dim, end_dim), (out_m,)


# ═══════════════════════════════════════════════════════════════════════
#  Multi-input (concat, stack)
# ═══════════════════════════════════════════════════════════════════════


def same_placement_multi_input_rule(
    original_vals: Iterable[TensorLayout],
    axis: int = 0,
) -> RuleSignature:
    """Placement rule for ``ops.concat``.

    ``original_vals`` mirrors the upstream param name; it arrives as a list
    or tuple of per-input :class:`TensorLayout` after dispatcher mapping.
    """
    values = list(original_vals)
    if not values:
        raise ValueError("same_placement_multi_input_rule: no tensor inputs")

    mesh = values[0].mapping.mesh
    suggested = [t.mapping for t in values]
    first_p = suggested[0].to_placements()
    for sm in suggested[1:]:
        if sm.to_placements() != first_p:
            raise ValueError(
                f"All inputs must have the same placements. "
                f"Got {first_p} and {sm.to_placements()}."
            )
    out_m = PlacementMapping(mesh, first_p)
    result: list[DeviceMapping] | tuple[DeviceMapping, ...] = (
        tuple(suggested) if isinstance(original_vals, tuple) else suggested
    )
    return (result, axis), (out_m,)


def stack_rule(
    values: Iterable[TensorLayout],
    axis: int = 0,
) -> RuleSignature:
    """Placement rule for ``ops.stack``.

    ``values`` mirrors the upstream param name; it arrives as a list or tuple
    of per-input :class:`TensorLayout` after dispatcher mapping.
    """
    layouts = list(values)
    if not layouts:
        raise ValueError("stack_rule: no tensor inputs")

    mesh = layouts[0].mapping.mesh
    suggested = [t.mapping for t in layouts]
    first_p = suggested[0].to_placements()
    for sm in suggested[1:]:
        if sm.to_placements() != first_p:
            raise ValueError("stack: all inputs must have the same placements.")

    ndim = layouts[0].rank
    norm = axis if axis >= 0 else axis + ndim + 1
    out_p = remap_sharded(first_p, lambda a: a + 1 if a >= norm else a)
    out_m = PlacementMapping(mesh, out_p)
    result: list[DeviceMapping] | tuple[DeviceMapping, ...] = (
        tuple(suggested) if isinstance(values, tuple) else suggested
    )
    return (result, axis), (out_m,)


# ═══════════════════════════════════════════════════════════════════════
#  Shape-dependent: broadcast_to, split, reshape
# ═══════════════════════════════════════════════════════════════════════


def split_rule(
    x: TensorLayout,
    split_sizes: Sequence[DimLike],
    axis: int = 0,
) -> RuleSignature:
    """Sharding rule for split."""
    s = x.mapping
    sp = s.to_placements()
    mesh = x.mapping.mesh
    ndim = x.rank
    norm_axis = axis % ndim

    normalized = [Dim(sz) for sz in split_sizes]
    for mesh_ax, p in enumerate(sp):
        if isinstance(p, Sharded) and p.axis == norm_axis:
            n = mesh.mesh_shape[mesh_ax]
            for sz in normalized:
                if isinstance(sz, StaticDim) and sz.dim % n != 0:
                    raise ValueError(
                        f"split: split size {sz} along sharded axis "
                        f"{norm_axis} is not evenly divisible by {n}."
                    )
    local_sizes: Sequence[DimLike] = (
        _localize_sizes(normalized, axis, ndim, sp, mesh.mesh_shape)
        or normalized
    )

    out_m = PlacementMapping(mesh, sp)
    return (s, local_sizes, axis), (out_m,)


# ═══════════════════════════════════════════════════════════════════════
#  Outer product
# ═══════════════════════════════════════════════════════════════════════


def outer_rule(lhs: TensorLayout, rhs: TensorLayout) -> RuleSignature:
    """Sharding rule for outer."""
    sx = resolve_partials_mapping(lhs.mapping)
    sy = resolve_partials_mapping(rhs.mapping)
    mesh = lhs.mapping.mesh
    xp, yp = sx.to_placements(), sy.to_placements()

    out_p: list[Placement] = []
    for ax in builtins.range(mesh.ndim):
        px = xp[ax] if ax < len(xp) else Replicated()
        py = yp[ax] if ax < len(yp) else Replicated()
        if isinstance(px, Replicated) and isinstance(py, Replicated):
            out_p.append(Replicated())
        elif isinstance(px, Sharded) and isinstance(py, Replicated):
            out_p.append(Sharded(0))
        elif isinstance(px, Replicated) and isinstance(py, Sharded):
            out_p.append(Sharded(1))
        else:
            raise ValueError(
                f"outer: incompatible on mesh axis {ax}: x={px}, y={py}."
            )

    return (sx, sy), (PlacementMapping(mesh, tuple(out_p)),)
