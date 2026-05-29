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
    """Creates a tensor filled with a single value.

    When ``device`` is a
    :class:`~max.experimental.sharding.DeviceMapping`, the result is
    distributed across that mesh according to its placements.

    Args:
        shape: The shape of the resulting tensor.
        value: The fill value.
        dtype: The data type of the tensor. Defaults to ``float32`` on
            CPU or ``bfloat16`` on accelerators.
        device: A single device or a
            :class:`~max.experimental.sharding.DeviceMapping` for
            distributed placement. Defaults to the current realization
            context's device.

    Returns:
        A tensor of the requested shape, dtype, and placement with
        every element set to ``value``.
    """
    mapping = _normalized_device(device)
    mesh = mapping.mesh
    resolved_dtype, _ = defaults(dtype, mesh.devices[0])
    placements = mapping.to_placements()
    global_shape = Shape(shape)
    shard_shapes = local_shard_shape_from_global(global_shape, mesh, placements)
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
            [TensorValue(tv) for tv in tvs],
            mapping,
            global_shape=global_shape,
        )


def ones(
    shape: ShapeLike,
    *,
    dtype: DType | None = None,
    device: Device | DeviceMapping | DeviceRef | None = None,
) -> Tensor:
    """Creates a tensor filled with ones.

    Args:
        shape: The shape of the resulting tensor.
        dtype: The data type. Defaults to ``float32`` on CPU or
            ``bfloat16`` on accelerators.
        device: A single device or a
            :class:`~max.experimental.sharding.DeviceMapping` for
            distributed placement.

    Returns:
        A tensor of the requested shape, dtype, and placement with every
        element set to ``1``.
    """
    return full(shape, 1.0, dtype=dtype, device=device)


def zeros(
    shape: ShapeLike,
    *,
    dtype: DType | None = None,
    device: Device | DeviceMapping | DeviceRef | None = None,
) -> Tensor:
    """Creates a tensor filled with zeros.

    Args:
        shape: The shape of the resulting tensor.
        dtype: The data type. Defaults to ``float32`` on CPU or
            ``bfloat16`` on accelerators.
        device: A single device or a
            :class:`~max.experimental.sharding.DeviceMapping` for
            distributed placement.

    Returns:
        A tensor of the requested shape, dtype, and placement with every
        element set to ``0``.
    """
    return full(shape, 0.0, dtype=dtype, device=device)


def full_like(
    like: Tensor | TensorType | DistributedTensorType, value: Number
) -> Tensor:
    """Creates a tensor filled with a single value, matching another tensor's shape and dtype.

    Args:
        like: The template tensor whose shape, dtype, and placement are
            copied.
        value: The fill value.

    Returns:
        A tensor matching the shape, dtype, and placement of ``like``,
        with every element set to ``value``.
    """
    return full(
        like.shape, value, dtype=like.dtype, device=_device_from_like(like)
    )


def ones_like(like: Tensor | TensorType | DistributedTensorType) -> Tensor:
    """Creates a tensor filled with ones, matching another tensor's shape and dtype.

    Args:
        like: The template tensor whose shape, dtype, and placement are
            copied.

    Returns:
        A tensor matching the shape, dtype, and placement of ``like``,
        with every element set to ``1``.
    """
    return full(
        like.shape, 1.0, dtype=like.dtype, device=_device_from_like(like)
    )


def zeros_like(like: Tensor | TensorType | DistributedTensorType) -> Tensor:
    """Creates a tensor filled with zeros, matching another tensor's shape and dtype.

    Args:
        like: The template tensor whose shape, dtype, and placement are
            copied.

    Returns:
        A tensor matching the shape, dtype, and placement of ``like``,
        with every element set to ``0``.
    """
    return full(
        like.shape, 0.0, dtype=like.dtype, device=_device_from_like(like)
    )


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
    """Samples values from a uniform distribution.

    When ``device`` is a
    :class:`~max.experimental.sharding.DeviceMapping`, each Sharded
    axis draws an independent stream while shards on Replicated axes
    draw identical values.

    Args:
        shape: The shape of the resulting tensor.
        range: A ``(low, high)`` pair giving the half-open interval to
            sample from. Defaults to ``(0, 1)``.
        dtype: The data type of the tensor.
        device: A single device or a
            :class:`~max.experimental.sharding.DeviceMapping` for
            distributed placement.

    Returns:
        A tensor of the requested shape, dtype, and placement with
        values sampled uniformly from ``[range[0], range[1])``.
    """
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
        return Tensor.from_shard_values(
            shard_values, mapping, global_shape=Shape(shape)
        )


def gaussian(
    shape: ShapeLike = (),
    mean: float = 0.0,
    std: float = 1.0,
    *,
    dtype: DType | None = None,
    device: Device | DeviceMapping | DeviceRef | None = None,
) -> Tensor:
    """Samples values from a Gaussian (normal) distribution.

    When ``device`` is a
    :class:`~max.experimental.sharding.DeviceMapping`, each Sharded
    axis draws an independent stream while shards on Replicated axes
    draw identical values.

    Args:
        shape: The shape of the resulting tensor.
        mean: The mean of the distribution. Defaults to ``0.0``.
        std: The standard deviation of the distribution. Defaults to
            ``1.0``.
        dtype: The data type of the tensor.
        device: A single device or a
            :class:`~max.experimental.sharding.DeviceMapping` for
            distributed placement.

    Returns:
        A tensor of the requested shape, dtype, and placement with
        values sampled from ``Normal(mean, std**2)``.
    """
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
        return Tensor.from_shard_values(
            shard_values, mapping, global_shape=Shape(shape)
        )


normal = gaussian


def uniform_like(
    like: Tensor | TensorType | DistributedTensorType,
    range: tuple[float, float] = (0, 1),
) -> Tensor:
    """Samples uniform values matching another tensor's shape and dtype.

    Args:
        like: The template tensor whose shape, dtype, and placement are
            copied.
        range: A ``(low, high)`` pair giving the half-open interval to
            sample from. Defaults to ``(0, 1)``.

    Returns:
        A tensor matching the shape, dtype, and placement of ``like``,
        with values sampled uniformly from ``[range[0], range[1])``.
    """
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
    """Samples Gaussian values matching another tensor's shape and dtype.

    Args:
        like: The template tensor whose shape, dtype, and placement are
            copied.
        mean: The mean of the distribution. Defaults to ``0.0``.
        std: The standard deviation of the distribution. Defaults to
            ``1.0``.

    Returns:
        A tensor matching the shape, dtype, and placement of ``like``,
        with values sampled from ``Normal(mean, std**2)``.
    """
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
    """Creates a Hann window of the given length.

    Args:
        window_length: The length of the window.
        periodic: When ``True``, returns a window suitable for use as a
            periodic function (matches NumPy's ``hann`` convention).
            When ``False``, returns a symmetric window.
        dtype: The data type of the resulting window.
        device: A single device or a
            :class:`~max.experimental.sharding.DeviceMapping`. Sharded
            placement is not supported.

    Returns:
        A 1-D tensor of length ``window_length`` containing the Hann
        window samples.
    """
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
    """Creates a 1-D tensor with values from a start, stop, and step.

    Args:
        start: The first value (inclusive).
        stop: The end value (exclusive).
        step: The increment between consecutive values. Defaults to ``1``.
        out_dim: The symbolic dimension for the output. Required when
            ``start`` / ``stop`` / ``step`` are dynamic and the output
            size cannot be inferred at graph build time.
        dtype: The data type of the resulting tensor.
        device: A single device or a
            :class:`~max.experimental.sharding.DeviceMapping`. Sharded
            placement is not supported.

    Returns:
        A 1-D tensor of values ``[start, start+step, start+2*step, ...]``
        up to but excluding ``stop``.
    """
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
    """Creates a constant tensor from a Python value, nested list, or DLPack array.

    For DLPack arrays, the array's own dtype is preserved when
    ``dtype`` is :obj:`None`. For Python scalars and nested lists,
    ``dtype`` defaults to ``float32`` on CPU or ``bfloat16`` on
    accelerators.

    Args:
        value: The constant value. Accepts a Python scalar, a nested
            list of numbers, or a DLPack-compatible array (NumPy,
            PyTorch, etc.).
        dtype: The data type of the resulting tensor. Defaults vary by
            input type as described above.
        device: A single device or a
            :class:`~max.experimental.sharding.DeviceMapping` for
            distributed placement.

    Returns:
        A tensor on the requested placement initialized from ``value``.
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
    """Creates a constant tensor from external (weight) data.

    External constants are loaded at compile time from the named
    weight rather than being inlined into the graph.

    Args:
        name: The external symbol name to load (typically a weight
            identifier).
        type: The :class:`~max.graph.TensorType` describing the
            constant's shape and dtype.
        device: A single device or a
            :class:`~max.experimental.sharding.DeviceMapping` for
            distributed placement.

    Returns:
        A tensor on the requested placement initialized from the
        external data.
    """
    mapping = _normalized_device(device) if device is not None else None
    with ensure_context():
        tv = ops.constant_external(name, type)
        t = Tensor.from_graph_value(tv)
    if mapping is not None:
        return transfer_to(t, mapping)
    return t
