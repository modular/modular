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

"""Shard shape math: computing per-device shapes from global shapes."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from max.graph import Dim, Shape, StaticDim, SymbolicDim
from max.graph.dim import DimLike

from .mesh import DeviceMesh
from .placements import Partial, Placement, Replicated, Sharded

if TYPE_CHECKING:
    from max.experimental.tensor import Tensor


def sharded_symbolic_dim(
    parent: SymbolicDim,
    mesh: DeviceMesh,
    mesh_axis: int,
    device_idx: int,
) -> SymbolicDim:
    """Returns the per-device symbolic dim name for a sharded ``parent``.

    The naming convention ``"{parent}_{axis_name}_{coord}"`` keeps each
    device's shard symbolically distinct. Without a per-device suffix
    every shard would share one ``SymbolicDim`` and the graph's
    ``same name = same size`` rule would force the runtime sizes to be
    equal — wrong for uneven data parallelism.

    Args:
        parent: The global symbolic dim being sharded.
        mesh: The device mesh.
        mesh_axis: The mesh axis index along which ``parent`` is sharded.
        device_idx: The flat device index in row-major order.

    Returns:
        ``SymbolicDim("{parent.name}_{axis_name}_{coord}")``, where
        ``coord`` is this device's coordinate along ``mesh_axis``.
    """
    coord = mesh.device_coord(device_idx, mesh_axis)
    return SymbolicDim(f"{parent.name}_{mesh.axis_names[mesh_axis]}_{coord}")


def shard_shape(
    global_shape: Sequence[DimLike],
    placements: Sequence[Placement],
    mesh_shape: Sequence[int],
) -> list[Dim]:
    """Computes the per-shard shape from a global shape and placements.

    For each ``Sharded(axis=k)`` placement on mesh axis ``i``, dimension
    ``k`` is divided by ``mesh_shape[i]``. :class:`Replicated` and
    :class:`Partial` placements leave the shape unchanged. Works with both
    static and symbolic dimensions.

    Args:
        global_shape: The global shape of the tensor.
        placements: One :class:`Placement` per mesh axis.
        mesh_shape: The size of each mesh axis.

    Returns:
        The per-shard dimensions after applying every sharded placement.
    """
    result = [Dim(d) for d in global_shape]
    for mesh_axis, p in enumerate(placements):
        if isinstance(p, Sharded):
            result[p.axis] = result[p.axis] // mesh_shape[mesh_axis]
    return result


def global_shape_from_local(
    local_shapes: Sequence[Sequence[Dim]],
    mesh: DeviceMesh,
    placements: Sequence[Placement],
    reference_tensors: Sequence[Tensor] = (),
) -> Shape:
    """Reconstruct a global shape from per-device shard shapes.

    Dispatches per dim subtype on each :class:`Sharded` axis:

    - :class:`~max.graph.StaticDim`: sum per-device shard sizes along
      the mesh axis. Handles uneven static splits exactly (e.g.,
      shards of sizes ``[4, 3]`` reconstruct to ``7``).
    - :class:`~max.graph.SymbolicDim`: look up the per-shard dim in a
      table built from *reference_tensors*' (per-shard → global)
      mappings. Falls back to multiplication if the dim isn't in the
      table — lossy, but the global symbolic identity is irrecoverable
      from a per-shard name without a reference.
    - All other dim subtypes (:class:`~max.graph.AlgebraicDim`, …):
      multiply by the mesh axis size, which round-trips the
      ``parent // mesh_axis_size`` forward direction.

    :class:`Replicated` and :class:`Partial` placements pass the dim
    through from the first shard.

    Args:
        local_shapes: One shape per device, in row-major mesh order.
        mesh: The device mesh.
        placements: One :class:`Placement` per mesh axis.
        reference_tensors: Optional tensors whose per-shard and global shapes
            are used to recover sharded :class:`SymbolicDim` global names.
            When omitted, symbolic dims fall back to multiplication.

    Returns:
        The reconstructed global tensor shape.

    Raises:
        ValueError: If ``len(local_shapes)`` does not match
            ``mesh.num_devices``.
        NotImplementedError: If ``placements`` contains a placement type
            other than :class:`Sharded`, :class:`Replicated`, or
            :class:`Partial`.
    """
    if len(local_shapes) != mesh.num_devices:
        raise ValueError(
            f"Expected {mesh.num_devices} local shapes for mesh {mesh}, "
            f"got {len(local_shapes)}."
        )

    # Build a map of symbolic dims to global dims from the reference tensors,
    # used to use consistent global dim names for sharded symbolic dims.
    sym_map: dict[SymbolicDim, Dim] = {}
    for t in reference_tensors:
        if not t.is_distributed:
            continue
        global_shape = t.shape
        for shard in t.local_shards:
            for ps, g in zip(shard.shape, global_shape, strict=True):
                if isinstance(ps, SymbolicDim):
                    sym_map.setdefault(ps, g)

    shape: list[Dim] = [Dim(d) for d in local_shapes[0]]
    for axis_idx, placement in enumerate(placements):
        if isinstance(placement, Sharded):
            tensor_axis = placement.axis
            mesh_axis_size = mesh.mesh_shape[axis_idx]
            stride = math.prod(mesh.mesh_shape[axis_idx + 1 :])
            first_dim = shape[tensor_axis]

            if isinstance(first_dim, StaticDim):
                # Walk one slice along the mesh axis (all other mesh
                # coords zero) and sum the per-shard sizes. Devices on
                # other slices are duplicates of this slice for sharding
                # purposes.
                total = 0
                fell_back = False
                for coord_along_axis in range(mesh_axis_size):
                    device_idx = coord_along_axis * stride
                    shard_dim = local_shapes[device_idx][tensor_axis]
                    if isinstance(shard_dim, StaticDim):
                        total += int(shard_dim)
                    else:
                        fell_back = True
                        break
                if fell_back:
                    shape[tensor_axis] = first_dim * mesh_axis_size
                else:
                    shape[tensor_axis] = StaticDim(total)
            elif isinstance(first_dim, SymbolicDim) and first_dim in sym_map:
                shape[tensor_axis] = sym_map[first_dim]
            else:
                shape[tensor_axis] = first_dim * mesh_axis_size
        elif isinstance(placement, (Replicated, Partial)):
            continue
        else:
            raise NotImplementedError(f"Unknown placement type: {placement}")
    return Shape(shape)


def _shard_sizes_along_axis(global_size: int, num_shards: int) -> list[int]:
    """Splits ``global_size`` across ``num_shards``, sizes differ by at most 1.

    The first ``global_size % num_shards`` shards get
    ``ceil(global_size / num_shards)`` elements; the rest get
    ``floor(global_size / num_shards)``.
    """
    base, remainder = divmod(global_size, num_shards)
    return [base + 1 if i < remainder else base for i in range(num_shards)]


def local_shard_shape_from_global(
    global_shape: Shape,
    mesh: DeviceMesh,
    placements: Sequence[Placement],
) -> list[Shape]:
    """Maps a global tensor shape to each device's local shard shape.

    For each :class:`Sharded` placement, the corresponding tensor dimension is
    split across ``mesh.mesh_shape[mesh_axis]`` ranks along that mesh axis.
    Dispatch on the parent dim subtype (mirrors
    :meth:`sharding.types.DistributedTensorType._local_shard_shape`):

    - :class:`~max.graph.StaticDim`: standard strided decomposition. When the
      parent size is not divisible by the mesh axis size, shard extents
      differ by at most one element.
    - :class:`~max.graph.SymbolicDim`: emit a fresh named symbolic dim for
        each device along the sharded mesh axis.
    - :class:`~max.graph.AlgebraicDim` (and any other non-static case):
      divide symbolically via ``parent // mesh_axis_size``. The resulting
      ``AlgebraicDim`` folds eagerly when operands are static.

    Device flat indices follow the same row-major order as
    :attr:`DeviceMesh.devices`.

    Args:
        global_shape: The global tensor shape.
        mesh: The device mesh.
        placements: One :class:`Placement` per mesh axis.

    Returns:
        One :class:`~max.graph.Shape` per device, in row-major mesh order.

    Raises:
        ValueError: If ``placements`` length does not match ``mesh.ndim`` or
            a sharded axis is out of range.
        NotImplementedError: If ``placements`` contains a placement type
            other than :class:`Sharded`, :class:`Replicated`, or
            :class:`Partial`.
    """
    if len(placements) != mesh.ndim:
        raise ValueError(
            f"Need one placement per mesh axis. Mesh has {mesh.ndim} axes, "
            f"got {len(placements)} placements."
        )

    local_dims = [list(global_shape) for _ in range(mesh.num_devices)]

    for mesh_axis, placement in enumerate(placements):
        if isinstance(placement, Sharded):
            tensor_axis = placement.axis
            if tensor_axis < 0:
                tensor_axis = len(global_shape) + tensor_axis
            if tensor_axis < 0 or tensor_axis >= len(global_shape):
                raise ValueError(
                    f"Sharded(axis={tensor_axis}) is out of range for tensor "
                    f"with rank {len(global_shape)}."
                )

            mesh_axis_size = mesh.mesh_shape[mesh_axis]

            for device_idx in range(mesh.num_devices):
                parent_dim = local_dims[device_idx][tensor_axis]
                if isinstance(parent_dim, StaticDim):
                    shard_sizes = _shard_sizes_along_axis(
                        int(parent_dim), mesh_axis_size
                    )
                    coord = mesh.device_coord(device_idx, mesh_axis)
                    local_dims[device_idx][tensor_axis] = StaticDim(
                        shard_sizes[coord]
                    )
                elif isinstance(parent_dim, SymbolicDim):
                    local_dims[device_idx][tensor_axis] = sharded_symbolic_dim(
                        parent_dim, mesh, mesh_axis, device_idx
                    )
                else:
                    local_dims[device_idx][tensor_axis] = (
                        parent_dim // mesh_axis_size
                    )
        elif isinstance(placement, (Replicated, Partial)):
            continue
        else:
            raise NotImplementedError(f"Unknown placement type: {placement}")

    return [Shape(dims) for dims in local_dims]
