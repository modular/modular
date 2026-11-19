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

"""Collective operations for distributed :class:`~max.tensor.Tensor` inputs.

Each collective runs over one or more mesh axes and is intended for use on
tensors that are sharded across a multi-device mesh. :func:`transfer_to`
is the universal entry point for moving a tensor between devices or
placements.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence

from max import _validation_hooks
from max.driver import Device
from max.dtype import DType
from max.graph import BufferValue, DeviceRef, TensorValue, ops
from max.graph.dim import Dim, StaticDim
from max.graph.ops.allreduce_rms_norm import gamma_dtype_supported
from max.graph.ops.slice_tensor import SliceIndex
from max.tensor import tensor as _experimental_tensor
from max.tensor.realization_context import (
    EagerRealizationContext,
    GraphRealizationContext,
    ensure_context,
)
from max.tensor.sharding import (
    DeviceMapping,
    DeviceMesh,
    Partial,
    Placement,
    Replicated,
    Sharded,
    ShardingError,
    Unknown,
)
from max.tensor.sharding.action import PerShard
from max.tensor.sharding.placements import even_shard_sizes
from max.tensor.tensor import Tensor

from .dispatch import _device_groups, call_on_mesh


def _even_chunk(
    value: TensorValue, axis: int, num_shards: int, index: int
) -> TensorValue:
    """Returns shard ``index`` of ``num_shards`` balanced shards along ``axis``.

    The larger shards come first, as the reduce-scatter kernel orders them.
    """
    axis %= value.rank
    dim = value.shape[axis]
    if num_shards == 1:
        return value
    indices: list[SliceIndex] = [slice(None)] * (axis + 1)
    if isinstance(dim, StaticDim):
        sizes = even_shard_sizes(int(dim), num_shards)
        start = sum(sizes[:index])
        indices[axis] = slice(start, start + sizes[index])
        return value[tuple(indices)]
    begin: Dim = Dim(0)
    for i in range(index):
        begin = begin + (dim + (num_shards - 1 - i)) // num_shards
    size = (dim + (num_shards - 1 - index)) // num_shards
    indices[axis] = (
        slice(
            ops.shape_to_tensor([begin]), ops.shape_to_tensor([begin + size]), 1
        ),
        size,
    )
    return ops.slice_tensor(value, indices)


def _split(value: TensorValue, axis: int, num_shards: int) -> list[TensorValue]:
    """Returns every balanced shard of ``value`` along ``axis``."""
    dim = value.shape[axis]
    if isinstance(dim, StaticDim):
        return list(
            ops.split(value, even_shard_sizes(int(dim), num_shards), axis=axis)
        )
    return [_even_chunk(value, axis, num_shards, i) for i in range(num_shards)]


def _mesh_axes(
    mesh: DeviceMesh, mesh_axis: int | str | Sequence[int | str]
) -> list[int]:
    names = (mesh_axis,) if isinstance(mesh_axis, (int, str)) else mesh_axis
    return [mesh._resolve_axis(name) for name in names]


def _placed(
    t: Tensor, axes: Sequence[int], placement: Placement
) -> DeviceMapping:
    """Returns ``t``'s mapping with ``placement`` on ``axes``."""
    placements = list(t.placements)
    for axis in axes:
        placements[axis] = placement
    return DeviceMapping(t.mesh, tuple(placements))


def _launch_axes(
    mesh: DeviceMesh,
    axes: Sequence[int],
    signal_buffers: PerShard[BufferValue | None],
) -> Sequence[int]:
    """Returns the mesh axes to launch a collective over.

    When every device is distinct and has a signal buffer, and the groups
    are runs of consecutive devices, one kernel launch over the whole mesh,
    with the kernel's ``group_size``, serves every group, as the Graph API
    emits it.
    """
    kernel_runs = (
        len(set(mesh.devices)) == mesh.num_devices
        and None not in signal_buffers.values
    )
    contiguous = all(
        other < min(axes)
        for other in range(mesh.ndim)
        if other not in axes and mesh.mesh_shape[other] > 1
    )
    if (
        kernel_runs
        and contiguous
        and 1 < mesh.axis_size(axes) < mesh.num_devices
    ):
        return range(mesh.ndim)
    return axes


def _signal_buffers(
    mesh: DeviceMesh, axes: Sequence[int]
) -> PerShard[BufferValue | None]:
    """Returns the signal buffer of each device of ``mesh``.

    A device without one, as on a mesh with fewer than two GPUs, gets
    ``None``. Eager execution adds buffers only when a group along ``axes``
    spans distinct devices: a group on one device exchanges nothing. Call it
    before tracing the group: each graph input added later costs more.
    """
    ctx = _experimental_tensor.current_realization_context()
    buffers = None
    spans_devices = any(
        len({mesh.devices[d] for d in group}) > 1
        for group in _device_groups(mesh, axes)
    )
    if isinstance(ctx, EagerRealizationContext) and spans_devices:
        buffers = ctx.ensure_signal_buffers(mesh)
    elif isinstance(ctx, GraphRealizationContext):
        buffers = ctx.signal_buffers
    by_device = {buffer.device: buffer for buffer in buffers or []}
    return PerShard(
        [by_device.get(DeviceRef.from_device(d)) for d in mesh.devices]
    )


def _kernel_signal_buffers(
    shards: list[Tensor], signal_buffers: list[BufferValue | None]
) -> list[BufferValue] | None:
    """Returns a group's signal buffers when a collective kernel can run it.

    A kernel runs over a group of distinct devices that each have a signal
    buffer. Otherwise, as on a simulated mesh or in a graph without signal
    buffers, this returns ``None`` and the group computes its result on its
    first device.

    Raises:
        ValueError: If the group repeats some devices but not all.
    """
    devices = {shard.device for shard in shards}
    if 1 < len(devices) < len(shards):
        raise ValueError(
            "A collective runs over a group of distinct devices, or of one "
            f"device repeated, got {[shard.device for shard in shards]}."
        )
    buffers = [buffer for buffer in signal_buffers if buffer is not None]
    if len(devices) == 1 or len(buffers) < len(shards):
        return None
    return buffers


def _on_devices(
    arg: list[Tensor] | list[TensorValue] | Tensor,
    devices: Sequence[DeviceRef],
) -> list[TensorValue]:
    """Returns a group's values on ``devices``, or one tensor copied to each."""
    values = [
        TensorValue(x)
        for x in (arg if isinstance(arg, list) else [arg] * len(devices))
    ]
    return [
        v if v.device == d else v.to(d)
        for v, d in zip(values, devices, strict=True)
    ]


def _check_1d(
    t: Tensor, placement: type[Placement] | Placement, name: str
) -> None:
    placed = (
        isinstance(t.placements[0], placement)
        if isinstance(placement, type)
        else t.placements[0] == placement
    )
    if t.mesh.ndim != 1 or not placed:
        raise ValueError(
            f"{name} needs a {getattr(placement, '__name__', placement)} tensor "
            f"on a one-dimensional mesh, got {t.mapping}."
        )


def _allreduce_sum_group(
    shards: list[Tensor],
    signal_buffers: list[BufferValue | None],
    group_size: int,
) -> list[TensorValue]:
    values = [TensorValue(s) for s in shards]
    buffers = _kernel_signal_buffers(shards, signal_buffers)
    if buffers is not None:
        return ops.allreduce.sum(values, buffers, group_size=group_size)
    devices = [v.device for v in values]
    first = _on_devices(values, devices[:1] * len(values))
    return _on_devices(
        [functools.reduce(ops.add, first)] * len(values), devices
    )


def allreduce_sum(
    t: Tensor, mesh_axis: int | str | Sequence[int | str] = 0
) -> Tensor:
    """All-reduces a tensor by summing its shards across mesh axes.

    Transitions the tensor's placement on ``mesh_axis`` from
    :class:`~max.tensor.sharding.Partial` to
    :class:`~max.tensor.sharding.Replicated`. Every device on
    ``mesh_axis`` ends up holding the sum of all inputs along it. With
    several mesh axes, one collective sums over every device they span: on
    a 2x2 mesh, ``mesh_axis=(0, 1)`` gives all 4 devices the sum of all 4
    inputs.

    Args:
        t: The input distributed tensor.
        mesh_axis: The mesh axis or axes along which to reduce, by index or
            name. Defaults to ``0``.

    Returns:
        A tensor with the same per-device values everywhere along
        ``mesh_axis``.
    """
    axes = _mesh_axes(t.mesh, mesh_axis)
    with ensure_context():
        signal_buffers = _signal_buffers(t.mesh, axes)
        return call_on_mesh(
            _allreduce_sum_group,
            t.mesh,
            _launch_axes(t.mesh, axes, signal_buffers),
            out_specs=_placed(t, axes, Replicated()),
        )(t, signal_buffers, t.mesh.axis_size(axes))


def _allgather_group(
    shards: list[Tensor],
    signal_buffers: list[BufferValue | None],
    tensor_axis: int,
    group_size: int,
) -> list[TensorValue]:
    values = [TensorValue(s) for s in shards]
    buffers = _kernel_signal_buffers(shards, signal_buffers)
    if buffers is not None:
        return ops.allgather(
            values, buffers, axis=tensor_axis, group_size=group_size
        )
    devices = [v.device for v in values]
    first = _on_devices(values, devices[:1] * len(values))
    return _on_devices([ops.concat(first, tensor_axis)] * len(values), devices)


def allgather(
    t: Tensor,
    tensor_axis: int = 0,
    mesh_axis: int | str | Sequence[int | str] = 0,
) -> Tensor:
    """All-gathers a tensor's shards along mesh axes.

    Transitions the tensor's placement on ``mesh_axis`` from
    :class:`~max.tensor.sharding.Sharded` to
    :class:`~max.tensor.sharding.Replicated`. Each device gathers
    the shards from its peers and concatenates them along ``tensor_axis``.
    With several mesh axes, one collective gathers the shards of every
    device they span, in row-major device order. For example, on a 2x2 mesh
    where ``Sharded(0)`` on both axes gives each of the 4 devices one
    quarter of the rows, ``mesh_axis=(0, 1)`` gives every device all rows.

    Args:
        t: The input distributed tensor.
        tensor_axis: The tensor axis along which the shards are concatenated.
        mesh_axis: The mesh axis or axes, by index or name, whose placement
            changes from Sharded to Replicated. Defaults to ``0``.

    Returns:
        A tensor with the full data replicated across ``mesh_axis``.
    """
    axes = _mesh_axes(t.mesh, mesh_axis)
    with ensure_context():
        signal_buffers = _signal_buffers(t.mesh, axes)
        return call_on_mesh(
            _allgather_group,
            t.mesh,
            _launch_axes(t.mesh, axes, signal_buffers),
            out_specs=_placed(t, axes, Replicated()),
        )(t, signal_buffers, tensor_axis, t.mesh.axis_size(axes))


def _reduce_scatter_group(
    shards: list[Tensor],
    signal_buffers: list[BufferValue | None],
    scatter_axis: int,
    group_size: int,
) -> list[TensorValue]:
    values = [TensorValue(s) for s in shards]
    buffers = _kernel_signal_buffers(shards, signal_buffers)
    if buffers is not None:
        return ops.reducescatter.sum(
            values, buffers, axis=scatter_axis, group_size=group_size
        )
    devices = [v.device for v in values]
    first = _on_devices(values, devices[:1] * len(values))
    return _on_devices(
        _split(functools.reduce(ops.add, first), scatter_axis, len(values)),
        devices,
    )


def reduce_scatter(
    t: Tensor,
    scatter_axis: int = 0,
    mesh_axis: int | str | Sequence[int | str] = 0,
) -> Tensor:
    """Reduces a tensor across mesh axes and scatters the result.

    Transitions the tensor's placement on ``mesh_axis`` from
    :class:`~max.tensor.sharding.Partial` to
    :class:`~max.tensor.sharding.Sharded`. Each device contributes
    to the sum and ends up with one shard of the reduced tensor along
    ``scatter_axis``. With several mesh axes, one collective sums over
    every device they span and gives each device one shard, in row-major
    device order: on a 2x2 mesh, ``mesh_axis=(0, 1)`` gives each of the 4
    devices one quarter of the summed rows. When the rows do not divide
    evenly, the first devices get one more row each.

    Args:
        t: The input distributed tensor.
        scatter_axis: The tensor axis along which the reduced result is
            sharded.
        mesh_axis: The mesh axis or axes, by index or name, whose placement
            changes from Partial to Sharded. Defaults to ``0``.

    Returns:
        A tensor with the reduced and re-sharded result.
    """
    axes = _mesh_axes(t.mesh, mesh_axis)
    with ensure_context():
        signal_buffers = _signal_buffers(t.mesh, axes)
        return call_on_mesh(
            _reduce_scatter_group,
            t.mesh,
            _launch_axes(t.mesh, axes, signal_buffers),
            out_specs=_placed(t, axes, Sharded(scatter_axis)),
        )(t, signal_buffers, scatter_axis, t.mesh.axis_size(axes))


def _all_to_all_group(
    shards: list[Tensor],
    signal_buffers: list[BufferValue | None],
    split_axis: int,
    concat_axis: int,
    group_size: int,
) -> list[TensorValue]:
    values = [TensorValue(s) for s in shards]
    for v in values:
        extent = v.shape[split_axis]
        if isinstance(extent, StaticDim) and extent.dim % group_size:
            raise ValueError(
                f"all_to_all: split_axis {split_axis} has extent {extent.dim},"
                f" not a multiple of the group size {group_size}."
            )
    buffers = _kernel_signal_buffers(shards, signal_buffers)
    if buffers is not None:
        return ops.all_to_all(
            values,
            buffers,
            split_axis=split_axis,
            concat_axis=concat_axis,
            group_size=group_size,
        )
    devices = [v.device for v in values]
    first = _on_devices(values, devices[:1] * len(values))
    if split_axis != concat_axis:
        swapped = _split(ops.concat(first, concat_axis), split_axis, len(first))
    else:
        chunks = [_split(v, split_axis, len(first)) for v in first]
        swapped = [
            ops.concat([chunk[dst] for chunk in chunks], concat_axis)
            for dst in range(len(first))
        ]
    return _on_devices(swapped, devices)


def all_to_all(
    t: Tensor,
    split_axis: int = 0,
    concat_axis: int = 0,
    mesh_axis: int | str | Sequence[int | str] = 0,
) -> Tensor:
    """Exchanges one chunk of a tensor between every pair of devices.

    Each device splits its shard into one chunk per device along
    ``mesh_axis`` (by ``split_axis``), sends chunk ``j`` to the group's
    device ``j``, and concatenates what it receives, in device order, along
    ``concat_axis``. A tensor :class:`~max.tensor.sharding.Sharded` along
    ``concat_axis`` on ``mesh_axis`` comes out sharded along ``split_axis``,
    in one launch and moving each element once. With both axes equal it is
    ``torch.distributed.all_to_all_single`` along that axis. With several
    mesh axes, one collective spans every device they span, in row-major
    device order.

    The result claims :class:`~max.tensor.sharding.Sharded` along
    ``split_axis`` on ``mesh_axis`` only when the input is sharded along a
    different ``concat_axis`` there. Otherwise, for example for per-rank
    partial results or with both axes equal, the shards are not slices of
    one tensor and the result is :class:`~max.tensor.sharding.Unknown` on
    ``mesh_axis``.

    Args:
        t: The input distributed tensor. Its shards must have the same shape
            within each group, with ``split_axis`` a multiple of the group
            size.
        split_axis: The tensor axis each shard is split along.
        concat_axis: The tensor axis the received chunks concatenate along.
        mesh_axis: The mesh axis or axes, by index or name, the exchange
            runs over. Defaults to ``0``.

    Returns:
        A tensor whose shards have ``split_axis`` divided by the group size
        and ``concat_axis`` multiplied by it.
    """
    axes = _mesh_axes(t.mesh, mesh_axis)
    split, concat = split_axis % t.rank, concat_axis % t.rank
    sliced = split != concat and all(
        isinstance(p := t.placements[a], Sharded) and p.axis % t.rank == concat
        for a in axes
    )
    with ensure_context():
        signal_buffers = _signal_buffers(t.mesh, axes)
        return call_on_mesh(
            _all_to_all_group,
            t.mesh,
            _launch_axes(t.mesh, axes, signal_buffers),
            out_specs=_placed(t, axes, Sharded(split) if sliced else Unknown()),
        )(t, signal_buffers, split, concat, t.mesh.axis_size(axes))


def distributed_broadcast(t: Tensor, mesh: DeviceMesh) -> Tensor:
    """Replicates a non-distributed tensor onto every device with one collective.

    Args:
        t: A non-distributed source tensor.
        mesh: The device mesh to replicate onto.

    Returns:
        A distributed tensor with :class:`Replicated` placement on every axis.
    """
    if t.is_distributed:
        raise RuntimeError(
            "distributed_broadcast takes a single-device tensor."
        )
    replicated = DeviceMapping(mesh, (Replicated(),) * mesh.ndim)
    devices = [DeviceRef.from_device(d) for d in mesh.devices]
    if len(set(devices)) < len(devices) or not all(d.is_gpu() for d in devices):
        return transfer_to(t, replicated)
    with ensure_context():
        signal_buffers = [
            buffer
            for buffer in _signal_buffers(mesh, range(mesh.ndim)).values
            if buffer is not None
        ]
        if len(signal_buffers) < len(devices):
            return transfer_to(t, replicated)
        return Tensor.from_shard_values(
            ops.distributed_broadcast(TensorValue(t), signal_buffers),
            replicated,
        )


def _reduce_scatter_rms_norm_group(
    shards: list[Tensor],
    gammas: list[Tensor] | Tensor,
    signal_buffers: list[BufferValue | None],
    epsilon: float,
    weight_offset: float,
) -> list[tuple[TensorValue, TensorValue]]:
    values = [TensorValue(s) for s in shards]
    devices = [v.device for v in values]
    gamma_values = _on_devices(gammas, devices)
    buffers = _kernel_signal_buffers(shards, signal_buffers)
    if buffers is not None:
        normed, summed = ops.reduce_scatter_rms_norm(
            values,
            buffers,
            gamma_values,
            epsilon,
            weight_offset=weight_offset,
        )
        return list(zip(normed, summed, strict=True))
    first = _on_devices(values, devices[:1] * len(values))
    summed = _split(functools.reduce(ops.add, first), 0, len(values))
    normed = [
        ops.rms_norm(
            s,
            gamma_values[0],
            epsilon,
            weight_offset,
            multiply_before_cast=True,
        )
        for s in summed
    ]
    return list(
        zip(
            _on_devices(normed, devices),
            _on_devices(summed, devices),
            strict=True,
        )
    )


def reduce_scatter_rms_norm(
    t: Tensor, gamma: Tensor, epsilon: float, weight_offset: float = 0.0
) -> tuple[Tensor, Tensor]:
    """Reduce-scatters a tensor along axis 0 and RMS-normalizes each shard.

    Fuses :func:`reduce_scatter` over every device of a one-dimensional
    mesh with an RMS norm of the result, in one collective launch (see
    :func:`max.graph.ops.reduce_scatter_rms_norm`). The norm multiplies by
    ``gamma`` before casting back to the input dtype.

    Args:
        t: A ``[rows, cols]`` bfloat16 tensor that is
            :class:`~max.tensor.sharding.Partial` on a one-dimensional
            mesh.
        gamma: The norm weight, replicated on the mesh of ``t``.
        epsilon: The norm epsilon.
        weight_offset: A constant added to ``gamma``.

    Returns:
        A pair ``(normed, summed)``, each
        :class:`~max.tensor.sharding.Sharded` along axis 0: the norm
        of the reduce-scatter sum, and the sum itself.
    """
    _check_1d(t, Partial, "reduce_scatter_rms_norm")
    sharded = DeviceMapping(t.mesh, (Sharded(0),))
    with ensure_context():
        return call_on_mesh(
            _reduce_scatter_rms_norm_group,
            t.mesh,
            (0,),
            out_specs=[sharded, sharded],
        )(
            t,
            gamma,
            _signal_buffers(t.mesh, range(t.mesh.ndim)),
            epsilon,
            weight_offset,
        )


def _rms_norm_shard(
    shard: Tensor,
    gamma: Tensor,
    epsilon: float,
    weight_offset: float,
    multiply_before_cast: bool,
) -> TensorValue:
    value = TensorValue(shard)
    (gamma_value,) = _on_devices(gamma, [value.device])
    return ops.rms_norm(
        value, gamma_value, epsilon, weight_offset, multiply_before_cast
    )


def _allreduce_rms_norm_group(
    shards: list[Tensor],
    gammas: list[Tensor] | Tensor,
    signal_buffers: list[BufferValue | None],
    epsilon: float,
    weight_offset: float,
    multiply_before_cast: bool,
) -> list[TensorValue]:
    values = [TensorValue(s) for s in shards]
    devices = [v.device for v in values]
    gamma_values = _on_devices(gammas, devices)
    buffers = _kernel_signal_buffers(shards, signal_buffers)
    if buffers is not None:
        normed, _ = ops.allreduce_rms_norm(
            values,
            buffers,
            gamma_values,
            epsilon,
            weight_offset=weight_offset,
            multiply_before_cast=multiply_before_cast,
        )
        return normed
    first = _on_devices(values, devices[:1] * len(values))
    normed_sum = ops.rms_norm(
        functools.reduce(ops.add, first),
        gamma_values[0],
        epsilon,
        weight_offset,
        multiply_before_cast,
    )
    return _on_devices([normed_sum] * len(values), devices)


def allreduce_rms_norm(
    t: Tensor,
    gamma: Tensor,
    epsilon: float,
    weight_offset: float = 0.0,
    multiply_before_cast: bool = False,
    mesh_axis: int | str | Sequence[int | str] = 0,
) -> Tensor:
    """All-reduces a Partial tensor and RMS-normalizes the sum in one launch.

    Fuses :func:`allreduce_sum` over every device of a one-dimensional mesh
    with an RMS norm of the reduced rows, in one collective launch (see
    :func:`max.graph.ops.allreduce_rms_norm`). The norm runs on the exact
    float32 peer sum, so it is not bit-identical to :func:`allreduce_sum`
    followed by :func:`max.graph.ops.rms_norm`, which reads the sum rounded
    back to bfloat16. On a float32 input there is no such rounding, but its
    peer sum and mean square may accumulate in a different order, so the two
    still differ by float32 rounding.

    Falls back to that two-launch form when the fused kernel cannot run:
    without signal buffers (single-device and simulated meshes), or when the
    input is not a 2-D bfloat16 or float32 tensor, or when ``gamma`` is
    neither the input dtype nor bfloat16/float16 on a float32 input.

    Args:
        t: A ``[rows, cols]`` tensor that is
            :class:`~max.tensor.sharding.Partial` on a one-dimensional
            mesh.
        gamma: The norm weight, replicated on the mesh of ``t``.
        epsilon: The norm epsilon.
        weight_offset: A constant added to ``gamma``.
        multiply_before_cast: The norm's gamma cast order; see
            :func:`max.graph.ops.rms_norm`.
        mesh_axis: The mesh axis or axes to reduce along, by index or name.
            Must resolve to the only axis of a one-dimensional mesh.

    Returns:
        The normed sum, :class:`~max.tensor.sharding.Replicated` on the mesh.
    """
    _check_1d(t, Partial, "allreduce_rms_norm")
    if _mesh_axes(t.mesh, mesh_axis) != [0]:
        raise ValueError(
            f"allreduce_rms_norm reduces axis 0, got {mesh_axis!r}."
        )
    replicated = DeviceMapping(t.mesh, (Replicated(),))
    with ensure_context():
        if (
            t.dtype not in (DType.bfloat16, DType.float32)
            or t.rank != 2
            or not gamma_dtype_supported(gamma.dtype, t.dtype)
        ):
            return call_on_mesh(_rms_norm_shard, t.mesh, out_specs=replicated)(
                allreduce_sum(t),
                gamma,
                epsilon,
                weight_offset,
                multiply_before_cast,
            )
        return call_on_mesh(
            _allreduce_rms_norm_group, t.mesh, (0,), out_specs=replicated
        )(
            t,
            gamma,
            _signal_buffers(t.mesh, range(t.mesh.ndim)),
            epsilon,
            weight_offset,
            multiply_before_cast,
        )


def _allreduce_post_norm_residual_rms_norm_group(
    shards: list[Tensor],
    residuals: list[Tensor] | Tensor,
    gammas: list[Tensor] | Tensor,
    gamma2s: list[Tensor] | Tensor,
    mixes: list[Tensor] | Tensor,
    signal_buffers: list[BufferValue | None],
    epsilon: float,
) -> list[tuple[TensorValue, TensorValue]]:
    values = [TensorValue(s) for s in shards]
    devices = [v.device for v in values]
    buffers = _kernel_signal_buffers(shards, signal_buffers)
    if buffers is None or not all(d.is_gpu() for d in devices):
        raise ValueError(
            "allreduce_post_norm_residual_rms_norm runs only on accelerators."
        )
    joined, normed = ops.allreduce_post_norm_residual_rms_norm(
        values,
        buffers,
        _on_devices(residuals, devices),
        _on_devices(gammas, devices),
        _on_devices(gamma2s, devices),
        _on_devices(mixes, devices),
        epsilon,
    )
    return list(zip(joined, normed, strict=True))


def allreduce_post_norm_residual_rms_norm(
    t: Tensor,
    residual: Tensor,
    gamma: Tensor,
    gamma2: Tensor,
    mix: Tensor,
    epsilon: float,
) -> tuple[Tensor, Tensor]:
    """All-reduces a branch and joins it to the residual between two norms.

    A transformer block with post-norms (the sandwich-norm layout) ends each
    branch with a norm, joins it to the residual stream with scalar weights
    and norms the result for the next branch. This does all of it in one
    launch per device:

    .. code-block:: text

        joined = mix[0] * rms_norm(allreduce(t), gamma) + mix[1] * residual
        normed = rms_norm(joined, gamma2)

    Each product and the sum round to bfloat16 on their own, as the unfused
    ops do; both norms multiply by their weight in float32 before the cast
    and take no weight offset. A Lamport kernel runs it (see
    :func:`max.graph.ops.allreduce_post_norm_residual_rms_norm`), in row
    chunks that each fit its 1 MiB workspace.

    Args:
        t: A ``[rows, cols]`` bfloat16 tensor that is
            :class:`~max.tensor.sharding.Partial` on a one-dimensional mesh
            of accelerators.
        residual: The residual stream, replicated on that mesh.
        gamma: The branch norm's weight, replicated on that mesh or a host
            tensor, which is copied to every device.
        gamma2: The next norm's weight, as ``gamma``.
        mix: ``[2]`` float32, as ``gamma``: the weight of the normed branch,
            then of the residual.
        epsilon: Both norms' epsilon.

    Returns:
        ``(joined, normed)``, each replicated.

    Raises:
        ValueError: If ``t`` is not a 2-D bfloat16 Partial tensor on a
            one-dimensional mesh, ``residual`` is not replicated on that
            mesh, a weight is distributed but not replicated on it, or the
            mesh has no accelerators.
    """
    name = "allreduce_post_norm_residual_rms_norm"
    _check_1d(t, Partial, name)
    if t.dtype != DType.bfloat16 or t.rank != 2:
        raise ValueError(
            f"{name} takes a 2-D bfloat16 tensor, got {t.dtype} of rank {t.rank}."
        )
    replicated = DeviceMapping(t.mesh, (Replicated(),))
    for operand_name, operand in (
        ("residual", residual),
        ("gamma", gamma),
        ("gamma2", gamma2),
        ("mix", mix),
    ):
        is_replicated = operand.mesh == t.mesh and operand.placements == (
            Replicated(),
        )
        if not is_replicated and (
            operand_name == "residual" or operand.is_distributed
        ):
            raise ValueError(
                f"{name} takes {operand_name} replicated on the mesh of its input, "
                f"got {operand.mapping}."
            )
    with ensure_context():
        return call_on_mesh(
            _allreduce_post_norm_residual_rms_norm_group,
            t.mesh,
            (0,),
            out_specs=[replicated, replicated],
        )(
            t,
            residual,
            gamma,
            gamma2,
            mix,
            _signal_buffers(t.mesh, range(t.mesh.ndim)),
            epsilon,
        )


def _allreduce_rms_norm_exit_concat_group(
    shards: list[Tensor],
    gate_ups: list[Tensor] | Tensor,
    gammas: list[Tensor] | Tensor,
    signal_buffers: list[BufferValue | None],
    epsilon: float,
    beta: float,
    linear_beta: float,
) -> list[TensorValue]:
    values = [TensorValue(s) for s in shards]
    devices = [v.device for v in values]
    gate_up_values = _on_devices(gate_ups, devices)
    gamma_values = _on_devices(gammas, devices)
    buffers = _kernel_signal_buffers(shards, signal_buffers)
    if buffers is not None:
        return ops.allreduce_rms_norm_exit_concat(
            values,
            buffers,
            gamma_values,
            gate_up_values,
            epsilon,
            beta,
            linear_beta,
        )
    cols = int(values[0].shape[1])
    if cols % len(values):
        raise ValueError(
            f"allreduce_rms_norm_exit_concat splits rows into {len(values)} column "
            f"shards, so cols ({cols}) must be a multiple of the device count."
        )
    width = cols // len(values)
    first = _on_devices(values, devices[:1] * len(values))
    normed = ops.rms_norm(
        functools.reduce(ops.add, first), gamma_values[0], epsilon, 0.0, False
    )
    latents = _on_devices(
        [
            normed[:, rank * width : (rank + 1) * width]
            for rank in range(len(values))
        ],
        devices,
    )
    outs = []
    for latent, gate_up in zip(latents, gate_up_values, strict=True):
        half = int(gate_up.shape[1]) // 2
        gate = ops.cast(gate_up[:, :half], DType.float32)
        up = ops.cast(gate_up[:, half:], DType.float32)
        if linear_beta != 0.0:
            up = linear_beta * ops.tanh(up / linear_beta)
        activated = beta * ops.tanh(gate / beta) * ops.sigmoid(gate) * up
        outs.append(
            ops.concat(
                [
                    ops.cast(latent, DType.bfloat16),
                    ops.cast(activated, DType.bfloat16),
                ],
                axis=1,
            )
        )
    return outs


def allreduce_rms_norm_exit_concat(
    t: Tensor,
    gate_up: Tensor,
    gamma: Tensor,
    *,
    epsilon: float,
    beta: float,
    linear_beta: float,
    mesh_axis: int | str | Sequence[int | str] = 0,
) -> Tensor:
    """All-reduces a Partial tensor, norms it and activates the exit in one launch.

    Fuses, for a tensor-parallel latent MoE's exit, the reduction of ``t``
    across the devices of a one-dimensional mesh, the RMS norm of every
    reduced row, this device's column shard of the normed row (columns
    ``[r * cols / N, (r + 1) * cols / N)`` rounded to bfloat16) and the
    SiTU-and-mul of ``gate_up`` -- the ``[normed latent | shared
    activation]`` the exit GEMM reads -- into one collective launch (see
    :func:`max.graph.ops.allreduce_rms_norm_exit_concat`). The norm reads
    the peer sum rounded to ``t``'s dtype, so it is the same arithmetic as
    :func:`allreduce_sum` followed by :func:`max.graph.ops.rms_norm` with its
    defaults (``weight_offset`` 0, ``multiply_before_cast`` False).

    Without signal buffers (single-device and simulated meshes) it stages the
    unfused chain instead: the sum and the norm on the first device, then
    each device's column shard beside its own activation.

    Args:
        t: A ``[rows, cols]`` float32 or bfloat16 tensor that is
            :class:`~max.tensor.sharding.Partial` on a one-dimensional
            mesh.
        gate_up: The gated shared projection, ``[rows, 2 * half]`` bfloat16,
            gate half first, distributed over the same mesh (column-sharded
            or replicated).
        gamma: The norm weight ``[cols]``, in ``t``'s dtype or
            bfloat16/float16 on a float32 ``t``, replicated on the mesh of
            ``t``.
        epsilon: The norm epsilon.
        beta: The SiTU activation's gate beta.
        linear_beta: The SiTU activation's up-half saturation beta; ``0.0``
            passes the up half through unchanged.
        mesh_axis: The mesh axis or axes to reduce along, by index or name.
            Must resolve to the only axis of a one-dimensional mesh.

    Returns:
        A ``[rows, cols / N + half]`` bfloat16 tensor,
        :class:`~max.tensor.sharding.Sharded` along axis 1 on the reduced
        axes: each device holds its own normed column shard followed by its
        own activated half.
    """
    name = "allreduce_rms_norm_exit_concat"
    _check_1d(t, Partial, name)
    if _mesh_axes(t.mesh, mesh_axis) != [0]:
        raise ValueError(f"{name} reduces axis 0, got {mesh_axis!r}.")
    if t.dtype not in (DType.float32, DType.bfloat16) or t.rank != 2:
        raise ValueError(
            f"{name} takes a 2-D float32 or bfloat16 tensor, got {t.dtype} {t.rank}-D."
        )
    if gate_up.dtype != DType.bfloat16 or gate_up.rank != 2:
        raise ValueError(
            f"{name} takes a 2-D bfloat16 gate_up, got {gate_up.dtype} {gate_up.rank}-D."
        )
    with ensure_context():
        return call_on_mesh(
            _allreduce_rms_norm_exit_concat_group,
            t.mesh,
            (0,),
            out_specs=DeviceMapping(t.mesh, (Sharded(1),)),
        )(
            t,
            gate_up,
            gamma,
            _signal_buffers(t.mesh, range(t.mesh.ndim)),
            epsilon,
            beta,
            linear_beta,
        )


def _allgather_rms_norm_group(
    shards: list[Tensor],
    gammas: list[Tensor] | Tensor,
    signal_buffers: list[BufferValue | None],
    axis: int,
    epsilon: float,
    weight_offset: float,
    round_offset_gamma: bool,
) -> list[tuple[TensorValue, TensorValue]]:
    values = [TensorValue(s) for s in shards]
    devices = [v.device for v in values]
    gamma_values = _on_devices(gammas, devices)
    buffers = _kernel_signal_buffers(shards, signal_buffers)
    if buffers is not None:
        normed, gathered = ops.allgather_rms_norm(
            values,
            buffers,
            gamma_values,
            epsilon=epsilon,
            weight_offset=weight_offset,
            axis=axis,
            round_offset_gamma=round_offset_gamma,
        )
        return list(zip(normed, gathered, strict=True))
    whole = ops.concat(_on_devices(values, devices[:1] * len(values)), axis)
    if round_offset_gamma:
        normed_whole = ops.rms_norm(
            whole,
            gamma_values[0] + weight_offset,
            epsilon,
            multiply_before_cast=True,
        )
    else:
        normed_whole = ops.rms_norm(
            whole,
            gamma_values[0],
            epsilon,
            weight_offset,
            multiply_before_cast=True,
        )
    return list(
        zip(
            _on_devices([normed_whole] * len(values), devices),
            _on_devices([whole] * len(values), devices),
            strict=True,
        )
    )


def allgather_rms_norm(
    t: Tensor,
    gamma: Tensor,
    epsilon: float,
    weight_offset: float = 0.0,
    round_offset_gamma: bool = False,
) -> tuple[Tensor, Tensor]:
    """All-gathers a 2D tensor and RMS-normalizes the result.

    Fuses :func:`allgather` over every device of a one-dimensional mesh with
    an RMS norm of the gathered rows, in one collective launch (see
    :func:`max.graph.ops.allgather_rms_norm`). The norm multiplies by
    ``gamma`` before casting back to the input dtype.

    Args:
        t: A ``[rows, cols]`` bfloat16 tensor that is
            :class:`~max.tensor.sharding.Sharded` along axis 0 or 1 on
            a one-dimensional mesh.
        gamma: The norm weight, replicated on the mesh of ``t``.
        epsilon: The norm epsilon.
        weight_offset: A constant added to ``gamma``.
        round_offset_gamma: Whether ``gamma + weight_offset`` is rounded to
            the input dtype before the multiply.

    Returns:
        A pair ``(normed, gathered)``, each
        :class:`~max.tensor.sharding.Replicated`: the norm of the
        gathered tensor, and the gathered tensor itself.
    """
    placement = t.placements[0] if t.mesh.ndim == 1 else None
    if t.rank != 2 or placement not in (Sharded(0), Sharded(1)):
        raise ValueError(
            "allgather_rms_norm needs a 2-D tensor sharded along axis 0 or 1 on "
            f"a one-dimensional mesh, got {t.mapping}."
        )
    assert isinstance(placement, Sharded)
    replicated = DeviceMapping(t.mesh, (Replicated(),))
    with ensure_context():
        return call_on_mesh(
            _allgather_rms_norm_group,
            t.mesh,
            (0,),
            out_specs=[replicated, replicated],
        )(
            t,
            gamma,
            _signal_buffers(t.mesh, range(t.mesh.ndim)),
            placement.axis,
            epsilon,
            weight_offset,
            round_offset_gamma,
        )


def _allgather_gated_residual_rms_norm_group(
    shards: list[Tensor],
    residuals: list[Tensor] | Tensor,
    post_weights: list[Tensor] | Tensor,
    gates: list[Tensor] | Tensor,
    norm_weights: list[Tensor] | Tensor,
    signal_buffers: list[BufferValue | None],
    post_eps: float,
    norm_eps: float,
    temperature: float,
    gate_eps: float,
    scale_branch: bool,
) -> list[tuple[TensorValue, TensorValue]]:
    values = [TensorValue(s) for s in shards]
    devices = [v.device for v in values]
    buffers = _kernel_signal_buffers(shards, signal_buffers)
    if buffers is None or not all(d.is_gpu() for d in devices):
        raise ValueError(
            "allgather_gated_residual_rms_norm runs only on accelerators."
        )
    stream, normed = ops.allgather_gated_residual_rms_norm(
        values,
        buffers,
        _on_devices(residuals, devices),
        _on_devices(post_weights, devices),
        _on_devices(gates, devices),
        _on_devices(norm_weights, devices),
        post_eps,
        norm_eps,
        temperature,
        gate_eps,
        scale_branch,
    )
    return list(zip(stream, normed, strict=True))


def allgather_gated_residual_rms_norm(
    branch: Tensor,
    residual: Tensor,
    post_weight: Tensor,
    gate: Tensor,
    norm_weight: Tensor,
    post_eps: float,
    norm_eps: float,
    temperature: float,
    gate_eps: float,
    scale_branch: bool,
) -> tuple[Tensor, Tensor]:
    """Gathers a branch's columns and joins it to the residual stream.

    See :func:`max.graph.ops.allgather_gated_residual_rms_norm`.

    Args:
        branch: A ``[rows, cols]`` bfloat16 tensor
            :class:`~max.tensor.sharding.Sharded` along axis 1 on a
            one-dimensional mesh of accelerators.
        residual: The residual stream, replicated on that mesh.
        post_weight: The branch's scale weight, replicated.
        gate: The residual gate's logits, replicated.
        norm_weight: The next pre-norm's weight, replicated.
        post_eps: The branch norm's epsilon.
        norm_eps: The next pre-norm's epsilon.
        temperature: The residual gate's temperature.
        gate_eps: The floor under the residual coefficient's root.
        scale_branch: Whether the branch is scaled by ``1 + post_weight``
            before its norm.

    Returns:
        ``(stream, normed)``, each replicated.
    """
    _check_1d(branch, Sharded(1), "allgather_gated_residual_rms_norm")
    replicated = DeviceMapping(branch.mesh, (Replicated(),))
    with ensure_context():
        return call_on_mesh(
            _allgather_gated_residual_rms_norm_group,
            branch.mesh,
            (0,),
            out_specs=[replicated, replicated],
        )(
            branch,
            residual,
            post_weight,
            gate,
            norm_weight,
            _signal_buffers(branch.mesh, range(branch.mesh.ndim)),
            post_eps,
            norm_eps,
            temperature,
            gate_eps,
            scale_branch,
        )


def _place_stack(values: Sequence[Tensor], target: DeviceMapping) -> Tensor:
    """Stacks non-distributed ``values`` on a new first axis, placed as ``target``.

    Each device receives only its part of the stack, so the host never
    builds the whole stack. When ``target`` splits the new axis, each device
    stacks only its own values. When it evenly splits one other axis, one
    ``ops.shard_and_stack`` call per group of devices slices every value,
    as the Graph API places stacked weights.
    """
    mesh = target.mesh
    rank = values[0].rank + 1
    groups: dict[int, list[int]] = {}
    for mesh_axis, p in enumerate(target.placements):
        if (tensor_axis := p.localized_axis()) is not None:
            groups.setdefault(tensor_axis % rank, []).append(mesh_axis)
        elif not isinstance(p, Replicated):
            raise ValueError(
                f"Cannot place a stack with placement {type(p).__name__}."
            )
    devices = [DeviceRef.from_device(device) for device in mesh.devices]
    with ensure_context():
        sources = [TensorValue(value) for value in values]
        shards: list[TensorValue] = []
        (axis, group), *others = groups.items() or [(0, [])]
        group_size = mesh.axis_size(group)
        dim = sources[0].shape[axis - 1] if axis > 0 else None
        if (
            group
            and not others
            and isinstance(dim, StaticDim)
            and int(dim) % group_size == 0
        ):
            # One ``shard_and_stack`` call per group of devices that share
            # their coordinates on the other mesh axes.
            by_group: dict[tuple[int, ...], list[int]] = {}
            for device_idx in range(mesh.num_devices):
                key = tuple(
                    mesh.device_coord(device_idx, a)
                    for a in range(mesh.ndim)
                    if a not in group
                )
                by_group.setdefault(key, []).append(device_idx)
            placed: dict[int, TensorValue] = {}
            for members in by_group.values():
                targets = [devices[i] for i in members]
                if axis == 1:
                    stacked = ops.shard_and_stack(sources, targets, axis=0)
                else:
                    host = [DeviceRef.CPU()] * len(members)
                    stacked = [
                        ops.transfer_to(shard, device)
                        for shard, device in zip(
                            ops.shard_and_stack(sources, host, axis=axis - 1),
                            targets,
                            strict=True,
                        )
                    ]
                placed.update(zip(members, stacked, strict=True))
            shards = [placed[i] for i in range(mesh.num_devices)]
        else:
            for device_idx, device in enumerate(devices):
                own = sources
                if 0 in groups:
                    sizes = even_shard_sizes(
                        len(sources),
                        mesh.axis_size(groups[0]),
                    )
                    index = mesh.device_coord(device_idx, groups[0])
                    start = sum(sizes[:index])
                    own = sources[start : start + sizes[index]]
                pieces = []
                for source in own:
                    for tensor_axis, split in groups.items():
                        if tensor_axis > 0:
                            source = _even_chunk(
                                source,
                                tensor_axis - 1,
                                mesh.axis_size(split),
                                mesh.device_coord(device_idx, split),
                            )
                    pieces.append(ops.transfer_to(source, device))
                shards.append(ops.stack(pieces))
        return Tensor.from_shard_values(shards, target)


def _nonnegative(mapping: DeviceMapping, rank: int) -> DeviceMapping:
    """Returns ``mapping`` with each Sharded axis counted from the start."""
    return DeviceMapping(
        mapping.mesh,
        tuple(
            Sharded(p.axis % rank) if isinstance(p, Sharded) else p
            for p in mapping.placements
        ),
    )


def _relabel(t: Tensor, placements: Sequence[Placement]) -> Tensor:
    """Returns ``t``'s shards under ``placements``, moving no data."""
    with ensure_context():
        return Tensor.from_shard_values(
            [TensorValue(s) for s in t.local_shards],
            DeviceMapping(t.mesh, tuple(placements)),
        )


def _scatter(t: Tensor, target: DeviceMapping) -> Tensor:
    """Places a single-device tensor at ``target``, each device its shard."""
    if any(isinstance(p, (Partial, Unknown)) for p in target.placements):
        raise ValueError(f"Cannot scatter a single-device tensor to {target}.")
    mesh, placements = target.mesh, _nonnegative(target, t.rank).placements
    for mesh_axis, p in enumerate(placements):
        if not isinstance(p, Sharded) or mesh.mesh_shape[mesh_axis] == 1:
            continue
        extent = t.shape[p.axis]
        if isinstance(extent, StaticDim) and extent.dim == 1:
            raise ShardingError(
                f"Cannot split tensor axis {p.axis}, of static extent 1, over"
                f" mesh axis {mesh.axis_names[mesh_axis]!r}. Use Replicated()"
                " on that mesh axis."
            )
    with ensure_context():
        value = TensorValue(t)
        shards = []
        for index, device in enumerate(mesh.devices):
            shard = value
            for tensor_axis in {
                p.axis for p in placements if isinstance(p, Sharded)
            }:
                axes = _splitting_axes(mesh, placements, tensor_axis)
                shard = _even_chunk(
                    shard,
                    tensor_axis,
                    mesh.axis_size(axes),
                    mesh.device_coord(index, axes),
                )
            shards.append(ops.transfer_to(shard, DeviceRef.from_device(device)))
        return Tensor.from_shard_values(shards, target)


def _splitting_axes(
    mesh: DeviceMesh, placements: Sequence[Placement], tensor_axis: int
) -> list[int]:
    """Returns the mesh axes, larger than one device, that split ``tensor_axis``."""
    return [
        axis
        for axis, p in enumerate(placements)
        if isinstance(p, Sharded)
        and p.axis == tensor_axis
        and mesh.mesh_shape[axis] > 1
    ]


def _common_prefix(source: list[int], target: list[int]) -> list[int]:
    """Returns the leading mesh axes both placements split a tensor axis by."""
    length = 0
    while (
        length < min(len(source), len(target))
        and source[length] == target[length]
    ):
        length += 1
    return source[:length]


def _local_split_group(
    copies: list[Tensor], tensor_axis: int
) -> list[TensorValue]:
    return [
        _even_chunk(TensorValue(c), tensor_axis, len(copies), i)
        for i, c in enumerate(copies)
    ]


def _keep_first_group(copies: list[Tensor]) -> list[TensorValue]:
    values = [TensorValue(c) for c in copies]
    return [values[0]] + [
        ops.broadcast_to(ops.constant(0, v.dtype, v.device), v.shape)
        for v in values[1:]
    ]


def transfer_to(
    t: Tensor, target: Device | DeviceMapping | DeviceRef
) -> Tensor:
    """Moves a tensor to a target device or device mapping.

    Handles every kind of placement transition: single-device transfers,
    scattering an unsharded tensor onto a mesh, redistributing across
    placements on the same mesh, and moving across meshes. Between two
    meshes of the same shape, each shard moves to the device at the same
    position; between meshes of different shapes, the tensor is gathered
    first.

    A tensor axis split over several mesh axes is one balanced split over
    every device they span, in row-major order. For example, on a 2x2 mesh,
    ``Sharded(0)`` on both axes splits 10 rows into 3, 3, 2, and 2 rows on
    devices 0 to 3. When the source and the target split a tensor axis
    over the same leading mesh axes, those pieces stay where they are, and
    only the other mesh axes gather or split.

    Args:
        t: The source tensor, distributed or single-device.
        target: A :class:`~max.driver.Device` to move to a single device,
            or a :class:`~max.tensor.sharding.DeviceMapping`
            describing the target mesh and placement.

    Returns:
        A tensor with the requested placement on the target device or mesh.

    Raises:
        ShardingError: If the move would change an
            :class:`~max.tensor.sharding.Unknown` placement, whose
            shards have no global value to preserve.
        NotImplementedError: If no supported collective performs the move.
    """
    if isinstance(target, DeviceRef):
        target = target.to_device()
    if isinstance(target, Device):
        target = DeviceMapping(DeviceMesh.single(target), (Replicated(),))
    source = t.mapping
    if source == target:
        return t
    if source.mesh.mesh_shape == target.mesh.mesh_shape:
        changes_unknown = any(
            old != new and Unknown in (type(old), type(new))
            for old, new in zip(
                source.placements, target.placements, strict=True
            )
        )
    else:
        changes_unknown = any(
            isinstance(p, Unknown)
            for p in (*source.placements, *target.placements)
        )
    if changes_unknown:
        raise ShardingError(
            f"transfer_to cannot move {source} to {target}: Unknown shards have no "
            "global value. Claim a placement with Tensor.rebind_mapping, or combine "
            "them with a collective."
        )

    if not t.is_distributed:
        if t.real and target.mesh.num_devices == 1:
            device = target.mesh.devices[0]
            if t.device == device:
                return t
            _validation_hooks.device_transfer("Tensor.to()", t, device)
            return Tensor(storage=t.driver_tensor.to(device))
        return _scatter(t, target)

    if source.mesh != target.mesh:
        if source.mesh.mesh_shape == target.mesh.mesh_shape:
            with ensure_context():
                moved = Tensor.from_shard_values(
                    [
                        ops.transfer_to(
                            TensorValue(s), DeviceRef.from_device(d)
                        )
                        for s, d in zip(
                            t.local_shards, target.mesh.devices, strict=True
                        )
                    ],
                    DeviceMapping(target.mesh, source.placements),
                )
            return transfer_to(moved, target)
        whole = transfer_to(
            t, DeviceMapping(source.mesh, (Replicated(),) * source.mesh.ndim)
        )
        return transfer_to(whole.local_shards[0], target)

    mesh, want = t.mesh, _nonnegative(target, t.rank).placements
    if t.placements != (counted := _nonnegative(t.mapping, t.rank).placements):
        t = _relabel(t, counted)
    # On a mesh axis of one device, every placement holds the same data.
    if any(
        mesh.mesh_shape[a] == 1 and t.placements[a] != want[a]
        for a in range(mesh.ndim)
    ):
        t = t.rebind_mapping(
            DeviceMapping(
                mesh,
                tuple(
                    w if mesh.mesh_shape[a] == 1 else p
                    for a, (p, w) in enumerate(
                        zip(t.placements, want, strict=True)
                    )
                ),
            )
        )

    # Partial: reduce-scatter into a target split, else all-reduce.
    for tensor_axis in {p.axis for p in want if isinstance(p, Sharded)}:
        have = _splitting_axes(mesh, t.placements, tensor_axis)
        kept = _common_prefix(have, _splitting_axes(mesh, want, tensor_axis))
        to_split = _splitting_axes(mesh, want, tensor_axis)[len(kept) :]
        if (
            to_split
            and have == kept
            and all(isinstance(t.placements[a], Partial) for a in to_split)
        ):
            t = reduce_scatter(t, tensor_axis, to_split)
    partial = [
        a
        for a in range(mesh.ndim)
        if isinstance(t.placements[a], Partial) and want[a] != t.placements[a]
    ]
    if partial:
        t = allreduce_sum(t, partial)

    # Sharded(a) to Sharded(b) over the same mesh axes, equal static shards.
    shapes = {tuple(s.shape) for s in t.local_shards}
    shape = next(iter(shapes))
    if len(shapes) == 1 and all(isinstance(d, StaticDim) for d in shape):
        swaps = [
            (p.axis, q.axis, axes)
            for p in set(t.placements)
            if isinstance(p, Sharded)
            and (axes := _splitting_axes(mesh, t.placements, p.axis))
            for q in set(want)
            if isinstance(q, Sharded)
            and q.axis != p.axis
            and _splitting_axes(mesh, want, q.axis) == axes
            and not _splitting_axes(mesh, t.placements, q.axis)
            and int(shape[q.axis]) % mesh.axis_size(axes) == 0
        ]
        if swaps:
            concat_axis, split_axis, axes = swaps[0]
            t = all_to_all(t, split_axis, concat_axis, axes)

    # Sharded: gather the axes the target does not keep, then split locally.
    for tensor_axis in {p.axis for p in t.placements if isinstance(p, Sharded)}:
        have = _splitting_axes(mesh, t.placements, tensor_axis)
        to_gather = have[
            len(
                _common_prefix(have, _splitting_axes(mesh, want, tensor_axis))
            ) :
        ]
        if to_gather:
            t = allgather(t, tensor_axis, to_gather)
    for tensor_axis in {p.axis for p in want if isinstance(p, Sharded)}:
        have = _splitting_axes(mesh, t.placements, tensor_axis)
        to_split = _splitting_axes(mesh, want, tensor_axis)[len(have) :]
        if to_split:
            t = call_on_mesh(
                _local_split_group,
                mesh,
                to_split,
                out_specs=_placed(t, to_split, Sharded(tensor_axis)),
            )(t, tensor_axis)

    # Replicated to Partial: one copy, zeros elsewhere.
    to_partial = [
        a
        for a in range(mesh.ndim)
        if isinstance(t.placements[a], Replicated)
        and isinstance(want[a], Partial)
    ]
    if to_partial:
        t = call_on_mesh(
            _keep_first_group,
            mesh,
            to_partial,
            out_specs=_placed(t, to_partial, Partial()),
        )(t)

    if t.placements != want:
        raise NotImplementedError(
            f"No transition from {t.placements} to {want}."
        )
    return (
        t
        if t.placements == target.placements
        else _relabel(t, target.placements)
    )
