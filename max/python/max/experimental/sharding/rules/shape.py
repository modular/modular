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
import math
from collections.abc import Iterable, Sequence
from typing import Literal

from max.experimental.sharding.mappings import DeviceMapping, PlacementMapping
from max.experimental.sharding.placements import Placement, Replicated, Sharded
from max.experimental.sharding.types import TensorLayout
from max.graph import Dim, DimLike, Shape, ShapeLike, StaticDim

from ._common import (
    RuleSignature,
    reject_any_sharded,
    reject_sharded_axis,
    remap_sharded,
    resolve_partials_mapping,
)


def _localize_shape(
    shape: Sequence[DimLike],
    placements: tuple[Placement, ...],
    mesh_shape: tuple[int, ...],
) -> Shape:
    local: list[Dim] = [Dim(d) for d in shape]
    for mesh_ax, p in enumerate(placements):
        if isinstance(p, Sharded) and p.axis < len(local):
            # Preserve the ``-1`` reshape-infer sentinel: MLIR would fold
            # ``Dim(-1) // n`` to ``0`` and silently corrupt the placeholder.
            if _is_minus_one(local[p.axis]):
                continue
            local[p.axis] //= mesh_shape[mesh_ax]
    return Shape(local)


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


def _is_minus_one(d: Dim) -> bool:
    return isinstance(d, StaticDim) and d.dim == -1


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


def broadcast_to_rule(
    x: TensorLayout,
    shape: TensorLayout | ShapeLike,
    out_dims: Iterable[DimLike] | None = None,
) -> RuleSignature:
    """Sharding rule for broadcast_to.

    A tensor-valued ``shape`` is unsupported by the sharding rule.
    """
    if isinstance(shape, TensorLayout):
        raise NotImplementedError(
            "broadcast_to sharding rule does not support a tensor-valued "
            "shape; pass a ShapeLike (list of DimLike)."
        )
    target_shape = list(shape)
    src = x.shape
    for i in builtins.range(1, builtins.min(len(src), len(target_shape)) + 1):
        s_dim, t_dim = src[-i], target_shape[-i]
        if s_dim != 1 and s_dim != t_dim:
            raise ValueError(
                f"broadcast_to: input dimension {-i} (size {s_dim}) "
                f"must be either 1 or equal to the target size {t_dim}."
            )
    s = x.mapping
    mesh = x.mapping.mesh
    local_shape = _localize_shape(
        target_shape, s.to_placements(), mesh.mesh_shape
    )
    return (s, local_shape, out_dims), (
        PlacementMapping(mesh, s.to_placements()),
    )


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


# ─── Reshape ──────────────────────────────────────────────────────────


# Helper to grab the start position (which is just the previous end position)
def _get_start(
    boundaries_list: list[tuple[int, int]], idx: int
) -> tuple[int, int]:
    return boundaries_list[idx - 1] if idx > 0 else (0, 1)


def _map_old_axis_to_new_axis(
    old_shape: Shape, new_shape: Shape
) -> dict[int, list[int]]:
    """Maps each old axis to the contiguous list of new axes it spans.

    Returns a dictionary where each old axis maps to the list of new axis
    indices whose cumulative-position range overlaps non-trivially with
    the old axis's range.

    - Length 0: the old axis collapsed (e.g., a size-1 axis with no
      non-trivial overlap in the new shape).
    - Length 1: clean 1-to-1 mapping or merged-into a single new axis.
    - Length > 1: the old axis splits across multiple new axes; the caller
      decides whether sharding (or any other per-axis property) can be
      preserved on a single new axis.
    """
    # Find the single -1 dimension (if any).
    if (has_negative := new_shape.count(Dim(-1))) > 1:
        raise ValueError("reshape(): at most one -1 dimension is allowed")

    old_static_total = math.prod(
        int(d) for d in old_shape if isinstance(d, StaticDim)
    )
    new_static_total = math.prod(
        int(d) for d in new_shape if isinstance(d, StaticDim) and int(d) != -1
    )

    old_dynamic_dims = set(d for d in old_shape if not isinstance(d, StaticDim))
    new_dynamic_dims = set(d for d in new_shape if not isinstance(d, StaticDim))

    # Compute absorbed static and dynamic dimensions and/or validate
    # input/output dimensions.
    if has_negative:
        if old_static_total % new_static_total != 0:
            raise ValueError(
                f"Invalid Reshape: Static dimensions in new shape ({new_shape}) must match those in the old shape ({old_shape})."
            )
        if new_dynamic_dims - old_dynamic_dims:
            raise ValueError(
                f"Invalid Reshape: Dynamic dimensions in new shape ({new_shape}) must match those in the old shape ({old_shape})."
            )

        absorbed_static_dims = old_static_total // new_static_total
        absorbed_dynamic_dims = len(old_dynamic_dims - new_dynamic_dims)
    else:
        if new_static_total != old_static_total:
            raise ValueError(
                f"Invalid Reshape: Static dimensions in new shape ({new_shape}) must match those in the old shape ({old_shape})."
            )
        if new_dynamic_dims != old_dynamic_dims:
            raise ValueError(
                f"Invalid Reshape: Dynamic dimensions in new shape ({new_shape}) must match those in the old shape ({old_shape})."
            )

        absorbed_static_dims = 0
        absorbed_dynamic_dims = 0

    def axis_boundaries(shape: Shape) -> list[tuple[int, int]]:
        boundaries: list[tuple[int, int]] = []
        dynamic_count = 0
        static_product = 1

        for dim in shape:
            # Update our running totals based on the type of dimension
            if dim == -1:
                dynamic_count += absorbed_dynamic_dims
                static_product *= absorbed_static_dims
            elif isinstance(dim, StaticDim):
                static_product *= int(dim)
            else:
                # SymbolicDim or AlgebraicDim: counts as one dynamic slot.
                dynamic_count += 1

            # Record where this axis ends
            boundaries.append((dynamic_count, static_product))

        return boundaries

    # Compute axis boundaries, then collect every new axis whose cumulative
    # range strictly overlaps each old axis's range. A length-1 list is the
    # clean 1-to-1 / merged case; a longer list is a split.
    old_boundaries = axis_boundaries(old_shape)
    new_boundaries = axis_boundaries(new_shape)

    axis_map: dict[int, list[int]] = {}
    for old_idx, old_end in enumerate(old_boundaries):
        old_start = _get_start(old_boundaries, old_idx)

        spanned: list[int] = []
        for new_idx, new_end in enumerate(new_boundaries):
            new_start = _get_start(new_boundaries, new_idx)

            # Strict overlap (positive-area intersection). Equality on a
            # single coordinate is treated as no overlap so that inserted
            # size-1 new axes and zero-width boundary touches don't get
            # spuriously attached to a neighbour.
            if new_start < old_end and old_start < new_end:
                spanned.append(new_idx)

        axis_map[old_idx] = spanned

    return axis_map


def reshape_rule(x: TensorLayout, shape: ShapeLike) -> RuleSignature:
    """Sharding rule for reshape.

    For each sharded axis the rule decides which new axis (if any) the
    sharding can land on:

    - **Clean map / merge** (one new axis): the sharding moves to that
      new axis.
    - **Pure split** (sharded old axis splits across several new axes,
      and those new axes contain no contributions from other old axes):
      the sharding lands on the leftmost candidate new axis ``k_new`` in
      the split such that all preceding split components are size 1 and
      ``new_shape[k_new] % mesh_size == 0``.
    - **Mixed split** (a new axis in the split also absorbs another old
      axis's contribution, e.g. via a ``-1`` that carries a dynamic dim):
      rejected, because the local reshape on each shard would not equal
      the corresponding slab of the global reshape (shard data would be
      interleaved across devices).
    - **No compatible candidate** (e.g., none of the new axes in a pure
      split is divisible by the mesh size): rejected.
    """
    device_mapping = x.mapping
    placements = device_mapping.to_placements()
    mesh = x.mapping.mesh

    old_shape = Shape(x.shape)
    new_shape = Shape(shape)
    out_placements = list(placements)
    has_sharded = any(isinstance(p, Sharded) for p in placements)
    if has_sharded:
        axis_map = _map_old_axis_to_new_axis(old_shape, new_shape)

        for i, p in enumerate(placements):
            if not isinstance(p, Sharded):
                continue
            new_axes = axis_map[p.axis]
            n = mesh.mesh_shape[i]

            if not new_axes:
                raise ValueError(
                    f"reshape: sharded axis {p.axis} of {old_shape} has no "
                    f"corresponding axis in {new_shape}; cannot place sharding."
                )
            if len(new_axes) == 1:
                out_placements[i] = Sharded(new_axes[0])
                continue

            # ── Split case ──────────────────────────────────────────────
            # Verify it's a "pure split": the spanned new axes contain
            # ONLY contributions from this old axis. The cleanest check:
            # the static product of spanned new axes equals the (static)
            # old axis size. A non-static spanned axis (-1 or symbolic)
            # implies the new axis is also absorbing other old-axis
            # contributions, which would interleave shard data.
            new_sizes = [new_shape[j] for j in new_axes]
            old_size = old_shape[p.axis]
            pure_split = (
                isinstance(old_size, StaticDim)
                and all(
                    isinstance(s, StaticDim) and int(s) > 0 for s in new_sizes
                )
                and math.prod(int(s) for s in new_sizes) == int(old_size)
            )
            if not pure_split:
                raise ValueError(
                    f"reshape: cannot preserve sharding on axis {p.axis} of "
                    f"{old_shape} -> {new_shape}: the split spans new axes "
                    f"{new_axes} (sizes {new_sizes}) which also absorb other "
                    f"axes' contributions (or include a -1 / dynamic dim). "
                    f"The local reshape on each shard would interleave shard "
                    f"data; allgather the sharded axis first."
                )

            # Pure split. Pick the leftmost candidate new axis k_new such
            # that (1) the product of preceding split components is 1
            # (otherwise their strides would interleave shards across
            # k_new), and (2) new_shape[k_new] % n == 0.
            chosen: int | None = None
            for k_new in new_axes:
                leading = math.prod(
                    int(new_shape[j]) for j in new_axes if j < k_new
                )
                if leading == 1 and int(new_shape[k_new]) % n == 0:
                    chosen = k_new
                    break
            if chosen is None:
                raise ValueError(
                    f"reshape: cannot preserve sharding on axis {p.axis} of "
                    f"{old_shape} -> {new_shape}: split into new axes "
                    f"{new_axes} of sizes "
                    f"{[int(new_shape[j]) for j in new_axes]}, but no "
                    f"candidate has all preceding split components == 1 and "
                    f"size divisible by mesh size {n}. Rearrange the new "
                    f"shape so a divisible factor sits leftmost in the split, "
                    f"or allgather the sharded axis first."
                )
            out_placements[i] = Sharded(chosen)

    placement_tuple = tuple(out_placements)
    local_shape = _localize_shape(new_shape, placement_tuple, mesh.mesh_shape)

    out_mapping = PlacementMapping(mesh, placement_tuple)
    return (device_mapping, local_shape), (out_mapping,)


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
