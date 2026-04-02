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

from max.graph import Dim, Shape, StaticDim
from max.graph.dim import DimLike

from .mesh import DeviceMesh
from .placements import Partial, Placement, Replicated, Sharded


def shard_shape(
    global_shape: Sequence[DimLike],
    placements: Sequence[Placement],
    mesh_shape: Sequence[int],
) -> list[Dim]:
    """Computes the per-shard shape from a global shape and placements.

    For each ``Sharded(axis=k)`` placement on mesh axis ``i``, dimension
    ``k`` is divided by ``mesh_shape[i]``.  Replicated and Partial
    placements leave the shape unchanged.  Works with both static and
    symbolic dimensions.
    """
    result = [Dim(d) for d in global_shape]
    for mesh_axis, p in enumerate(placements):
        if isinstance(p, Sharded):
            result[p.axis] = result[p.axis] // mesh_shape[mesh_axis]
    return result


def global_shape_from_local(
    local_shape: Sequence[Dim],
    mesh: DeviceMesh,
    placements: Sequence[Placement],
) -> Shape:
    """Derives the global shape from one local shard's shape and placements."""
    shape = list(local_shape)
    for axis_idx, placement in enumerate(placements):
        if isinstance(placement, Sharded):
            shape[placement.axis] = (
                shape[placement.axis] * mesh.mesh_shape[axis_idx]
            )
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
    When a parent size is not divisible by the mesh axis size, shard extents
    differ by at most one element (standard strided decomposition).

    Device flat indices follow the same row-major order as
    :attr:`DeviceMesh.devices`.

    Args:
        global_shape: Logical global tensor shape (static sizes along sharded
            axes are required for uneven splits; symbolic dims are not
            supported yet for sharded axes).
        mesh: Device mesh.
        placements: One placement per mesh axis.

    Returns:
        One :class:`Shape` per device, in mesh device order.

    Raises:
        ValueError: If ``placements`` length does not match ``mesh.ndim`` or a
            sharded axis is out of range.
        NotImplementedError: If a sharded tensor dimension is not static.
    """
    if len(placements) != mesh.ndim:
        raise ValueError(
            f"Need one placement per mesh axis. Mesh has {mesh.ndim} axes, "
            f"got {len(placements)} placements."
        )

    for dim in global_shape:
        if not isinstance(dim, StaticDim):
            raise NotImplementedError(
                "Sharding from global shape requires static dimensions; got "
                f"{global_shape}."
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
            stride = math.prod(mesh.mesh_shape[mesh_axis + 1 :])

            for device_idx in range(mesh.num_devices):
                parent_dim = local_dims[device_idx][tensor_axis]
                shard_sizes = _shard_sizes_along_axis(
                    int(parent_dim), mesh_axis_size
                )
                coord_along_axis = (device_idx // stride) % mesh_axis_size
                local_dims[device_idx][tensor_axis] = StaticDim(
                    shard_sizes[coord_along_axis]
                )
        elif isinstance(placement, (Replicated, Partial)):
            continue
        else:
            raise NotImplementedError(f"Unknown placement type: {placement}")

    return [Shape(dims) for dims in local_dims]
