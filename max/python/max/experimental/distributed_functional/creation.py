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

"""Tensor creation ops with DeviceMapping support.

All creation ops accept ``device`` as either a single
:class:`~max.driver.Device` or a
:class:`~max.experimental.sharding.DeviceMapping`.  Constants are
created directly on each device with the correct per-shard shape.
"""

from __future__ import annotations

from max.driver import Device
from max.dtype import DType
from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    PlacementMapping,
    Replicated,
    local_shard_shape_from_global,
)
from max.experimental.tensor import Tensor, defaults
from max.graph import DeviceRef, Shape, ShapeLike, ops

from ._context_provider import functional
from .collectives import make_distributed


def _normalize_device(device: Device | DeviceMapping | None) -> DeviceMapping:
    """Normalizes a Device or None to a DeviceMapping."""
    if isinstance(device, DeviceMapping):
        return device
    if isinstance(device, Device):
        return PlacementMapping(DeviceMesh.single(device), (Replicated(),))
    # None — resolve default device
    _, resolved = defaults(None, None)
    return PlacementMapping(DeviceMesh.single(resolved), (Replicated(),))


@functional(linear=None)
def full(
    shape: ShapeLike,
    value: int | float,
    *,
    dtype: DType | None = None,
    device: Device | DeviceMapping | None = None,
) -> Tensor:
    """Creates a tensor filled with a constant value."""
    mapping = _normalize_device(device)
    resolved_dtype, _ = defaults(dtype, mapping.mesh.devices[0])
    placements = mapping.to_placements()
    mesh = mapping.mesh
    shard_shapes = local_shard_shape_from_global(Shape(shape), mesh, placements)
    shard_values = tuple(
        ops.broadcast_to(
            ops.constant(value, resolved_dtype, DeviceRef.from_device(d)),
            list(shard_shapes[i]),
        )
        for i, d in enumerate(mesh.devices)
    )
    return make_distributed(list(shard_values), mapping)


def ones(
    shape: ShapeLike,
    *,
    dtype: DType | None = None,
    device: Device | DeviceMapping | None = None,
) -> Tensor:
    """Creates a tensor filled with ones."""
    return full(shape, 1.0, dtype=dtype, device=device)


def zeros(
    shape: ShapeLike,
    *,
    dtype: DType | None = None,
    device: Device | DeviceMapping | None = None,
) -> Tensor:
    """Creates a tensor filled with zeros."""
    return full(shape, 0.0, dtype=dtype, device=device)
