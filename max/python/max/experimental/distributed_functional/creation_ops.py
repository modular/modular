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

"""Creation and random ops for distributed tensors.

``full``, ``ones``, ``zeros``, ``*_like``, ``uniform``, ``gaussian``.
"""

from __future__ import annotations

import builtins
import itertools
from typing import cast

from max.driver import Device, DLPackArray
from max.dtype import DType
from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    DistributedTensorType,
    PlacementMapping,
    Replicated,
    Sharded,
    local_shard_shape_from_global,
)
from max.experimental.tensor import Tensor, TensorType, defaults
from max.graph import (
    DeviceRef,
    DimLike,
    Shape,
    ShapeLike,
    TensorValue,
    TensorValueLike,
    ops,
)
from max.graph.ops.constant import NestedArray, Number

from .collective_ops import transfer_to
from .utils import ensure_context

# ═════════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════════


def _normalized_device(
    device: Device | DeviceMapping | DeviceRef | None,
) -> DeviceMapping:
    """Coerce any device specification to a ``DeviceMapping``."""
    if isinstance(device, DeviceMapping):
        return device
    if isinstance(device, DeviceRef):
        device = device.to_device()
    if isinstance(device, Device):
        return PlacementMapping(DeviceMesh.single(device), (Replicated(),))
    _, resolved = defaults(None, None)
    return PlacementMapping(DeviceMesh.single(resolved), (Replicated(),))


def _device_from_like(
    like: Tensor | TensorType | DistributedTensorType,
) -> Device | DeviceMapping:
    """Extract device or mapping from a tensor-like for ``*_like`` factories."""
    if isinstance(like, Tensor):
        if like.is_distributed:
            return PlacementMapping(like.mesh, like.placements)
        return like.device
    if isinstance(like, DistributedTensorType):
        return PlacementMapping(like.mesh, like.placements)
    return like.device.to_device()


# ═════════════════════════════════════════════════════════════════════════
#  Creation ops
# ═════════════════════════════════════════════════════════════════════════


def full(
    shape: ShapeLike,
    value: Number,
    *,
    dtype: DType | None = None,
    device: Device | DeviceMapping | DeviceRef | None = None,
) -> Tensor:
    """Create a tensor filled with *value*, optionally distributed across devices."""
    mapping = _normalized_device(device)
    mesh = mapping.mesh
    resolved_dtype, _ = defaults(dtype, mesh.devices[0])
    placements = mapping.to_placements()
    shard_shapes = local_shard_shape_from_global(Shape(shape), mesh, placements)
    with ensure_context():
        tvs = [
            ops.broadcast_to(
                ops.constant(
                    value,
                    resolved_dtype,
                    DeviceRef.from_device(mesh.devices[i]),
                ),
                list(shard_shapes[i]),
            )
            for i in builtins.range(mesh.num_devices)
        ]
        return Tensor.from_shard_values(
            [TensorValue(tv) for tv in tvs], mapping
        )


def ones(
    shape: ShapeLike,
    *,
    dtype: DType | None = None,
    device: Device | DeviceMapping | DeviceRef | None = None,
) -> Tensor:
    """Create an all-ones tensor, optionally distributed across devices."""
    return full(shape, 1.0, dtype=dtype, device=device)


def zeros(
    shape: ShapeLike,
    *,
    dtype: DType | None = None,
    device: Device | DeviceMapping | DeviceRef | None = None,
) -> Tensor:
    """Create an all-zeros tensor, optionally distributed across devices."""
    return full(shape, 0.0, dtype=dtype, device=device)


def full_like(
    like: Tensor | TensorType | DistributedTensorType, value: Number
) -> Tensor:
    """Create a tensor filled with *value*, matching the shape and dtype of *like*."""
    shape = [int(d) for d in like.shape]
    return full(shape, value, dtype=like.dtype, device=_device_from_like(like))


def ones_like(like: Tensor | TensorType | DistributedTensorType) -> Tensor:
    """Create an all-ones tensor matching the shape and dtype of *like*."""
    shape = [int(d) for d in like.shape]
    return full(shape, 1.0, dtype=like.dtype, device=_device_from_like(like))


def zeros_like(like: Tensor | TensorType | DistributedTensorType) -> Tensor:
    """Create an all-zeros tensor matching the shape and dtype of *like*."""
    shape = [int(d) for d in like.shape]
    return full(shape, 0.0, dtype=like.dtype, device=_device_from_like(like))


# ═════════════════════════════════════════════════════════════════════════
#  Random ops
# ═════════════════════════════════════════════════════════════════════════

# Each distributed random op gets a unique base seed so that successive
# ops (e.g. two calls to ``uniform()``) produce different streams.
# Within one op, shards in the same "group" (same Replicated axes) share
# a seed so they draw identical values; Sharded-axis shards get distinct
# seeds so each shard is independent.  The gap of 100 between base seeds
# leaves room for up to 100 devices per group without seed collision.
_BASE_SEED_COUNTER = 0


def _next_base_seed() -> int:
    """Returns a fresh base seed and advances the counter by 100."""
    global _BASE_SEED_COUNTER
    val = _BASE_SEED_COUNTER
    _BASE_SEED_COUNTER += 100
    return val


def _shard_group_ids(
    mesh_shape: tuple[int, ...],
    placements: tuple[Replicated | Sharded, ...],
) -> list[int]:
    """Assign a group ID to each device so Replicated axes share seeds."""
    ndim = len(mesh_shape)
    ids = []
    for coords in itertools.product(*(builtins.range(s) for s in mesh_shape)):
        group_id = 0
        stride = 1
        for ax in reversed(builtins.range(ndim)):
            if isinstance(placements[ax], Sharded):
                group_id += coords[ax] * stride
                stride *= mesh_shape[ax]
        ids.append(group_id)
    return ids


def _distributed_random_setup(
    shape: ShapeLike,
    mapping: DeviceMapping,
) -> tuple[DeviceMesh, list[Shape], list[int], int, int]:
    """Shared setup for distributed random ops: shapes, groups, seed."""
    mesh = mapping.mesh
    placements = mapping.to_placements()
    shard_shapes = local_shard_shape_from_global(Shape(shape), mesh, placements)
    # Random ops only support Replicated/Sharded — Partial is invalid.
    assert all(isinstance(p, (Replicated, Sharded)) for p in placements)
    group_ids = _shard_group_ids(
        mesh.mesh_shape, cast(tuple[Replicated | Sharded, ...], placements)
    )
    n_unique = builtins.max(group_ids) + 1
    base = _next_base_seed()
    return mesh, shard_shapes, group_ids, n_unique, base


def uniform(
    shape: ShapeLike = (),
    range: tuple[float, float] = (0, 1),
    *,
    dtype: DType | None = None,
    device: Device | DeviceMapping | DeviceRef | None = None,
) -> Tensor:
    """Sample from a uniform distribution over [range[0], range[1])."""
    mapping = _normalized_device(device)
    resolved_dtype, _ = defaults(dtype, mapping.mesh.devices[0])

    with ensure_context():
        if mapping.mesh.num_devices == 1:
            tt = TensorType(
                resolved_dtype,
                shape,
                DeviceRef.from_device(mapping.mesh.devices[0]),
            )
            return Tensor.from_graph_value(ops.random.uniform(tt, range=range))
        mesh, shard_shapes, group_ids, n_unique, base = (
            _distributed_random_setup(shape, mapping)
        )
        shard_values = []
        for i, d in enumerate(mesh.devices):
            tt = TensorType(
                resolved_dtype, shard_shapes[i], DeviceRef.from_device(d)
            )
            ops.random.set_seed(base + group_ids[i])
            shard_values.append(ops.random.uniform(tt, range=range))
        ops.random.set_seed(base + n_unique)
        return Tensor.from_shard_values(shard_values, mapping)


def gaussian(
    shape: ShapeLike = (),
    mean: float = 0.0,
    std: float = 1.0,
    *,
    dtype: DType | None = None,
    device: Device | DeviceMapping | DeviceRef | None = None,
) -> Tensor:
    """Sample from a Gaussian (normal) distribution with given *mean* and *std*."""
    mapping = _normalized_device(device)
    resolved_dtype, _ = defaults(dtype, mapping.mesh.devices[0])

    with ensure_context():
        if mapping.mesh.num_devices == 1:
            tt = TensorType(
                resolved_dtype,
                shape,
                DeviceRef.from_device(mapping.mesh.devices[0]),
            )
            return Tensor.from_graph_value(
                ops.random.gaussian(tt, mean=mean, std=std)
            )
        mesh, shard_shapes, group_ids, n_unique, base = (
            _distributed_random_setup(shape, mapping)
        )
        shard_values = []
        for i, d in enumerate(mesh.devices):
            tt = TensorType(
                resolved_dtype, shard_shapes[i], DeviceRef.from_device(d)
            )
            ops.random.set_seed(base + group_ids[i])
            shard_values.append(ops.random.gaussian(tt, mean=mean, std=std))
        ops.random.set_seed(base + n_unique)
        return Tensor.from_shard_values(shard_values, mapping)


normal = gaussian


def uniform_like(
    like: Tensor | TensorType | DistributedTensorType,
    range: tuple[float, float] = (0, 1),
) -> Tensor:
    """Sample uniform values matching the shape and dtype of *like*."""
    return uniform(
        like.shape,
        range=range,
        device=_device_from_like(like),
        dtype=like.dtype,
    )


def gaussian_like(
    like: Tensor | TensorType | DistributedTensorType,
    mean: float = 0.0,
    std: float = 1.0,
) -> Tensor:
    """Sample Gaussian values matching the shape and dtype of *like*."""
    return gaussian(
        like.shape,
        mean=mean,
        std=std,
        device=_device_from_like(like),
        dtype=like.dtype,
    )


normal_like = gaussian_like


# ═════════════════════════════════════════════════════════════════════════
#  Creation-like ops (replicated per-device, no tensor inputs)
# ═════════════════════════════════════════════════════════════════════════


def _validated_creation_mapping(
    device: Device | DeviceMapping | DeviceRef | None,
    op_name: str,
) -> DeviceMapping:
    """Return a validated DeviceMapping, rejecting Sharded placements."""
    mapping = _normalized_device(device)
    for p in mapping.to_placements():
        if isinstance(p, Sharded):
            raise ValueError(
                f"{op_name}: cannot create with Sharded placement. "
                f"Use Replicated and shard after creation."
            )
    return mapping


def hann_window(
    window_length: int,
    *,
    periodic: bool = True,
    dtype: DType | None = None,
    device: Device | DeviceMapping | DeviceRef | None = None,
) -> Tensor:
    """Create a Hann window of the given *window_length*."""
    mapping = _validated_creation_mapping(device, "hann_window")
    mesh = mapping.mesh
    resolved_dtype, _ = defaults(dtype, mesh.devices[0])
    with ensure_context():
        shard_values = [
            ops.hann_window(
                window_length,
                DeviceRef.from_device(d),
                periodic=periodic,
                dtype=resolved_dtype,
            )
            for d in mesh.devices
        ]
        return Tensor.from_shard_values(shard_values, mapping)


def range(
    start: TensorValueLike,
    stop: TensorValueLike,
    step: TensorValueLike = 1,
    out_dim: DimLike | None = None,
    *,
    dtype: DType | None = None,
    device: Device | DeviceMapping | DeviceRef | None = None,
) -> Tensor:
    """Create a 1-D tensor with values from *start* to *stop* (exclusive) by *step*."""
    mapping = _validated_creation_mapping(device, "range")
    mesh = mapping.mesh
    resolved_dtype, _ = defaults(dtype, mesh.devices[0])
    with ensure_context():
        shard_values = [
            ops.range(
                start,
                stop,
                step,
                out_dim,
                dtype=resolved_dtype,
                device=DeviceRef.from_device(d),
            )
            for d in mesh.devices
        ]
        return Tensor.from_shard_values(shard_values, mapping)


# Backward compat alias — callers use both F.arange and F.range.
arange = range


# ═════════════════════════════════════════════════════════════════════════
#  constant / constant_external
# ═════════════════════════════════════════════════════════════════════════


def constant(
    value: DLPackArray | NestedArray | Number,
    dtype: DType | None = None,
    device: Device | DeviceMapping | DeviceRef | None = None,
) -> Tensor:
    """Create a constant tensor from a scalar, nested list, or DLPack array.

    For DLPack arrays, the array's own dtype is preserved when ``dtype`` is
    ``None`` (matching ``ops.constant`` semantics).  For Python scalars and
    nested lists, ``dtype`` defaults to float32 on CPU / bfloat16 on
    accelerators.

    Inside a realization context, emits ``ops.constant`` per device.
    """
    mapping = _normalized_device(device)
    mesh = mapping.mesh
    # For DLPack-compatible arrays, pass dtype through as-is so that
    # ops.constant preserves the array's native dtype when dtype is None.
    if isinstance(value, DLPackArray):
        resolved_dtype = dtype
    else:
        resolved_dtype, _ = defaults(dtype, mesh.devices[0])
    with ensure_context():
        tvs = [
            ops.constant(value, resolved_dtype, DeviceRef.from_device(d))
            for d in mesh.devices
        ]
        return Tensor.from_shard_values(
            [TensorValue(tv) for tv in tvs], mapping
        )


def constant_external(
    name: str,
    type: TensorType,
    device: Device | DeviceMapping | DeviceRef | None = None,
) -> Tensor:
    """Create a constant tensor from external (weight) data.

    External constants are loaded at graph compile time. Supports
    distributed placement via DeviceMapping.
    """
    mapping = _normalized_device(device) if device is not None else None
    with ensure_context():
        tv = ops.constant_external(name, type)
        t = Tensor.from_graph_value(tv)
    if mapping is not None:
        return transfer_to(t, mapping)
    return t
