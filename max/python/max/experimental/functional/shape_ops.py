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

"""Shape-manipulation ops that don't use spmd dispatch."""

from __future__ import annotations

import builtins
import math
from collections.abc import Iterable, Sequence

from max.experimental.sharding.mappings import DeviceMapping, PlacementMapping
from max.experimental.sharding.mesh import DeviceMesh
from max.experimental.sharding.placements import Placement, Sharded
from max.experimental.sharding.types import TensorLayout
from max.experimental.tensor import Tensor
from max.graph import (
    Dim,
    DimLike,
    Shape,
    ShapeLike,
    StaticDim,
    SymbolicDim,
    TensorValue,
    ops,
)

from ..sharding.rules._common import RuleSignature, is_replicated
from ..sharding.shapes import (
    global_shape_from_local,
    sharded_symbolic_dim,
)
from .collective_ops import transfer_to
from .spmd_ops import functional, tensor_to_layout
from .utils import ensure_context


def _is_minus_one(d: Dim) -> bool:
    return isinstance(d, StaticDim) and d.dim == -1


def _localize_shape_for_device(
    shape: Sequence[DimLike],
    placements: tuple[Placement, ...],
    mesh: DeviceMesh,
    device_idx: int,
) -> Shape:
    """Maps a global shape to one device's per-shard form."""
    local: list[Dim] = [Dim(d) for d in shape]
    for mesh_ax, p in enumerate(placements):
        if not isinstance(p, Sharded) or p.axis >= len(local):
            continue
        # Preserve the ``-1`` reshape-infer sentinel: MLIR would fold
        # ``Dim(-1) // n`` to ``0`` and silently corrupt the placeholder.
        if _is_minus_one(local[p.axis]):
            continue
        dim = local[p.axis]
        if isinstance(dim, SymbolicDim):
            local[p.axis] = sharded_symbolic_dim(dim, mesh, mesh_ax, device_idx)
        else:
            local[p.axis] //= mesh.mesh_shape[mesh_ax]
    return Shape(local)


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


def _reshape_mappings(
    x: TensorLayout, shape: ShapeLike
) -> tuple[DeviceMapping, DeviceMapping]:
    """Return suggested arguments and output mappings for reshape.

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
    out_mapping = PlacementMapping(mesh, placement_tuple)

    return device_mapping, out_mapping


_single_device_reshape = functional(ops.reshape)


def reshape(x: Tensor, shape: ShapeLike) -> Tensor:
    """Reshape a tensor.

    See :func:`max.graph.ops.reshape` for op semantics.
    """
    if not x.is_distributed:
        return _single_device_reshape(x, shape)

    layout = tensor_to_layout(x)
    suggested_mapping, out_mapping = _reshape_mappings(layout, shape)
    redistributed = transfer_to(x, suggested_mapping)
    placements = out_mapping.to_placements()
    mesh = out_mapping.mesh
    target_shape = Shape(shape)

    with ensure_context():
        per_shard = [
            ops.reshape(
                TensorValue(redistributed.local_shards[i]),
                _localize_shape_for_device(target_shape, placements, mesh, i),
            )
            for i in builtins.range(mesh.num_devices)
        ]

        # We must compute the global shape (instead of just using `shape`)
        # because the user-provided `shape` can carry `-1` placeholders.
        global_shape = global_shape_from_local(
            [list(s.shape) for s in per_shard],
            mesh,
            placements,
            [redistributed],
        )
        return Tensor.from_shard_values(
            [TensorValue(s) for s in per_shard],
            out_mapping,
            global_shape=global_shape,
        )


# ─── Broadcast to ────────────────────────────────────────────────────────


def _broadcast_to_mappings(
    x: TensorLayout,
    shape: ShapeLike,
) -> tuple[DeviceMapping, DeviceMapping]:
    """Return suggested arguments and output mappings for broadcast_to."""
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

    # Note: The `shape` argument will be overwritten
    return s, PlacementMapping(mesh, s.to_placements())


def _pass_through_rule(x: TensorLayout, *args: object) -> RuleSignature:
    """Returns unchanged input mappings and output mapping set to the first input mapping."""
    return (x, *args), (x,)


_naive_broadcast_to = functional(ops.broadcast_to, rule=_pass_through_rule)


def broadcast_to(
    x: Tensor,
    shape: TensorValue | ShapeLike,
    out_dims: Iterable[DimLike] | None = None,
) -> Tensor:
    """Broadcast a tensor to a new shape.

    See :func:`max.graph.ops.broadcast_to` for op semantics.
    """
    if not x.is_distributed:
        return _naive_broadcast_to(x, shape, out_dims)

    if isinstance(shape, TensorValue):
        all_replicated = all(
            is_replicated(p) for p in x.mapping.to_placements()
        )
        if not all_replicated:
            raise ValueError(
                "Only fully replicated tensors can be broadcasted using `TensorValue`-type shape."
            )
        return _naive_broadcast_to(x, shape, out_dims)

    layout = tensor_to_layout(x)
    suggested_mapping, out_mapping = _broadcast_to_mappings(layout, shape)
    redistributed = transfer_to(x, suggested_mapping)
    placements = out_mapping.to_placements()
    mesh = out_mapping.mesh
    target_shape = Shape(shape)

    with ensure_context():
        per_shard = [
            ops.broadcast_to(
                TensorValue(redistributed.local_shards[i]),
                _localize_shape_for_device(target_shape, placements, mesh, i),
                None,  # out_dims is not used when shape is ShapeLike.
            )
            for i in builtins.range(mesh.num_devices)
        ]
        global_shape = global_shape_from_local(
            [list(s.shape) for s in per_shard],
            mesh,
            placements,
            [redistributed],
        )
        return Tensor.from_shard_values(
            [TensorValue(s) for s in per_shard],
            out_mapping,
            global_shape=global_shape,
        )
