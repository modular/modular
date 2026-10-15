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

"""Collective operations for distributed :class:`~max.experimental.tensor.Tensor` inputs.

Each collective runs over one or more mesh axes and is intended for use on
tensors that are sharded across a multi-device mesh. :func:`transfer_to`
is the universal entry point for moving a tensor between devices or
placements.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence

from max import _validation_hooks
from max.driver import Accelerator, Device
from max.experimental import tensor as _experimental_tensor
from max.experimental.realization_context import ensure_context
from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    Partial,
    Placement,
    Replicated,
    Sharded,
    ShardingError,
    Unknown,
)
from max.experimental.sharding.action import PerShard
from max.experimental.sharding.placements import even_shard_sizes
from max.experimental.tensor import Tensor
from max.graph import BufferValue, DeviceRef, TensorValue, ops
from max.graph.dim import Dim, StaticDim
from max.graph.ops.slice_tensor import SliceIndex

from .dispatch import call_on_mesh


def _signal_buffers(mesh: DeviceMesh) -> PerShard[BufferValue] | None:
    """Returns each device's signal buffer for a collective kernel on ``mesh``.

    Returns ``None`` when the kernel cannot run: with no realization context,
    on a mesh without accelerators, or on a mesh that repeats a device.
    """
    ctx = _experimental_tensor.current_realization_context(None)
    if (
        ctx is None
        or len(set(mesh.devices)) != mesh.num_devices
        or not any(isinstance(d, Accelerator) for d in mesh.devices)
    ):
        return None
    buffers = getattr(ctx, "signal_buffers", None)
    if buffers is None and hasattr(ctx, "ensure_signal_buffers"):
        # Returns None for fewer than two accelerators.
        buffers = ctx.ensure_signal_buffers(mesh)
    if buffers is None:
        return None
    # The context holds one buffer per device of the graph, in mesh order.
    return PerShard(buffers[: mesh.num_devices])


def _even_split_along_axis(
    sv: TensorValue, axis: int, n: int
) -> list[TensorValue]:
    """Splits ``sv`` into ``n`` load-balanced chunks along ``axis``.

    The larger chunks come first, as the reduce-scatter kernel orders them,
    so slicing a replicated tensor gives the same pieces as a reduce-scatter.
    """
    dim = sv.shape[axis]
    if isinstance(dim, StaticDim):
        return list(ops.split(sv, even_shard_sizes(int(dim), n), axis=axis))
    return [_even_chunk(sv, axis, n, index) for index in range(n)]


def _even_chunk(sv: TensorValue, axis: int, n: int, index: int) -> TensorValue:
    """Returns chunk ``index`` of ``_even_split_along_axis``."""
    axis %= sv.rank
    dim = sv.shape[axis]
    if n == 1:
        return sv
    indices: list[SliceIndex] = [slice(None)] * (axis + 1)
    if isinstance(dim, StaticDim):
        sizes = even_shard_sizes(int(dim), n)
        offset = sum(sizes[:index])
        indices[axis] = slice(offset, offset + sizes[index])
        return sv[tuple(indices)]
    start: Dim = Dim(0)
    for i in range(index):
        start = start + (dim + (n - 1 - i)) // n
    size = (dim + (n - 1 - index)) // n
    indices[axis] = (
        slice(
            ops.shape_to_tensor([start]),
            ops.shape_to_tensor([start + size]),
            1,
        ),
        size,
    )
    return ops.slice_tensor(sv, indices)


def _collective(
    t: Tensor,
    mesh_axis: int | str | Sequence[int | str],
    new_placement: Replicated | Sharded,
    kernel: Callable[..., list[TensorValue]],
    simulated: Callable[[list[TensorValue]], list[TensorValue]],
) -> Tensor:
    """Runs a collective on each group of devices along ``mesh_axis``.

    A group holds the devices that differ only in their coordinates along
    ``mesh_axis``. With several mesh axes, one group holds every device they
    span, in row-major order: on a 2x2 mesh, both axes form one group of 4
    devices.

    ``kernel`` synchronizes a group's devices through their signal buffers.
    Without signal buffers, it calls ``simulated`` instead, which computes
    the group's results on its first device.
    """

    def run(
        shards: list[Tensor], signal_buffers: list[BufferValue] | None
    ) -> list[TensorValue]:
        if signal_buffers is not None:
            return kernel(shards, signal_buffers)
        values = [TensorValue(shard) for shard in shards]
        home = values[0].device
        combined = simulated(
            [
                value if value.device == home else value.to(home)
                for value in values
            ]
        )
        return [
            result if value.device == home else result.to(value.device)
            for result, value in zip(combined, values, strict=True)
        ]

    mesh = t.mesh
    axes = [
        mesh._resolve_axis(axis)
        for axis in (
            (mesh_axis,) if isinstance(mesh_axis, (int, str)) else mesh_axis
        )
    ]
    placements = list(t.placements)
    for axis in axes:
        placements[axis] = new_placement
    with ensure_context():
        group_size = mesh.axis_size(axes)
        # A group of one device exchanges nothing, so it allocates no
        # signal buffers.
        signal_buffers = _signal_buffers(mesh) if group_size > 1 else None
        # The kernel splits its inputs into runs of ``group_size``
        # consecutive devices, so one launch serves every group when each
        # group is such a run.
        contiguous = all(
            other < min(axes)
            for other in range(mesh.ndim)
            if other not in axes and mesh.mesh_shape[other] > 1
        )
        if (
            signal_buffers is not None
            and contiguous
            and group_size < mesh.num_devices
        ):
            return call_on_mesh(
                lambda shards, buffers: kernel(
                    shards, buffers, group_size=group_size
                ),
                mesh,
                range(mesh.ndim),
                out_specs=DeviceMapping(mesh, tuple(placements)),
            )(t, signal_buffers)
        # Per-rank IR after the collective carries the honest algebraic
        # form (e.g. ``batch_dp_0 + batch_dp_1``); :attr:`Tensor.shape`
        # reads back the same dim on every rank and collapses the wrapper.
        return call_on_mesh(
            run,
            mesh,
            axes,
            out_specs=DeviceMapping(mesh, tuple(placements)),
        )(t, signal_buffers)


def allreduce_sum(
    t: Tensor, mesh_axis: int | str | Sequence[int | str] = 0
) -> Tensor:
    """All-reduces a tensor by summing its shards across mesh axes.

    Transitions the tensor's placement on ``mesh_axis`` from
    :class:`~max.experimental.sharding.Partial` to
    :class:`~max.experimental.sharding.Replicated`. Every device on
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
    return _collective(
        t,
        mesh_axis,
        Replicated(),
        kernel=ops.allreduce.sum,
        simulated=lambda shards: (
            [functools.reduce(ops.add, shards)] * len(shards)
        ),
    )


def allgather(
    t: Tensor,
    tensor_axis: int = 0,
    mesh_axis: int | str | Sequence[int | str] = 0,
) -> Tensor:
    """All-gathers a tensor's shards along mesh axes.

    Transitions the tensor's placement on ``mesh_axis`` from
    :class:`~max.experimental.sharding.Sharded` to
    :class:`~max.experimental.sharding.Replicated`. Each device gathers
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
    return _collective(
        t,
        mesh_axis,
        Replicated(),
        kernel=lambda shards, signal_buffers, group_size=None: ops.allgather(
            shards, signal_buffers, axis=tensor_axis, group_size=group_size
        ),
        simulated=lambda shards: (
            [ops.concat(shards, tensor_axis)] * len(shards)
        ),
    )


def reduce_scatter(
    t: Tensor,
    scatter_axis: int = 0,
    mesh_axis: int | str | Sequence[int | str] = 0,
) -> Tensor:
    """Reduces a tensor across mesh axes and scatters the result.

    Transitions the tensor's placement on ``mesh_axis`` from
    :class:`~max.experimental.sharding.Partial` to
    :class:`~max.experimental.sharding.Sharded`. Each device contributes
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
    return _collective(
        t,
        mesh_axis,
        Sharded(scatter_axis),
        kernel=lambda shards, signal_buffers, group_size=None: (
            ops.reducescatter.sum(
                shards, signal_buffers, axis=scatter_axis, group_size=group_size
            )
        ),
        simulated=lambda shards: _even_split_along_axis(
            functools.reduce(ops.add, shards), scatter_axis, len(shards)
        ),
    )


def _local_split(
    t: Tensor, mesh_axes: Sequence[int], target: Sharded
) -> Tensor:
    """``Replicated -> Sharded``: each device slices its local copy with no communication.

    With several mesh axes, the tensor axis is split once into one piece per
    device they span.
    """

    def split(copies: list[Tensor]) -> list[TensorValue]:
        return [
            _even_chunk(TensorValue(copy), target.axis, len(copies), index)
            for index, copy in enumerate(copies)
        ]

    placements = list(t.placements)
    for mesh_axis in mesh_axes:
        placements[mesh_axis] = target
    return call_on_mesh(
        split,
        t.mesh,
        mesh_axes,
        out_specs=DeviceMapping(t.mesh, tuple(placements)),
    )(t)


def _keep_one_copy(t: Tensor, mesh_axes: Sequence[int]) -> Tensor:
    """Moves ``t`` from Replicated to Partial along ``mesh_axes``.

    The first device along ``mesh_axes`` keeps its copy and the others hold
    zeros, with no communication, so the copies sum to the value exactly.
    """

    def keep_first(copies: list[Tensor]) -> list[TensorValue]:
        first, *others = (TensorValue(copy) for copy in copies)
        return [
            first,
            *(
                ops.broadcast_to(
                    ops.constant(0, other.dtype, other.device), other.shape
                )
                for other in others
            ),
        ]

    placements = list(t.placements)
    for mesh_axis in mesh_axes:
        placements[mesh_axis] = Partial()
    return call_on_mesh(
        keep_first,
        t.mesh,
        mesh_axes,
        out_specs=DeviceMapping(t.mesh, tuple(placements)),
    )(t)


def _scatter(t: Tensor, target: DeviceMapping) -> Tensor:
    """Distributes a non-distributed tensor across a mesh."""
    assert not t.is_distributed, "_scatter expects a non-distributed tensor"
    mesh = target.mesh
    placements = target.placements

    # Size-1 axes cannot host a shard.
    src_shape = t.shape
    for mesh_axis, p in enumerate(placements):
        ax = p.localized_axis()
        if ax is None and not isinstance(p, Replicated):
            raise ValueError(
                f"Cannot scatter with placement {type(p).__name__}; "
                "scatter requires Replicated or a placement that localizes "
                "a single tensor axis (override ``localized_axis()``)."
            )
        if ax is None or not 0 <= ax < len(src_shape):
            continue
        dim = src_shape[ax]
        if isinstance(dim, StaticDim) and dim.dim == 1:
            raise ShardingError(
                f"_scatter: placement {p!r} on mesh axis "
                f"{mesh.axis_names[mesh_axis]!r} targets tensor axis {ax} "
                f"with static extent 1; cannot split a size-1 axis. Use "
                f"Replicated() on this mesh axis instead."
            )

    groups: dict[int, list[int]] = {}
    for mesh_axis, p in enumerate(placements):
        if (tensor_axis := p.localized_axis()) is not None:
            groups.setdefault(tensor_axis % t.rank, []).append(mesh_axis)
    with ensure_context():
        tv = t.__tensorvalue__()
        # The host slices out each device's part, so a device receives only
        # its own part.
        shard_tvs = []
        for index, device in enumerate(mesh.devices):
            shard = tv
            for tensor_axis, group in groups.items():
                shard = _even_chunk(
                    shard,
                    tensor_axis,
                    mesh.axis_size(group),
                    mesh.device_coord(index, group),
                )
            shard_tvs.append(
                ops.transfer_to(shard, DeviceRef.from_device(device))
            )
        return Tensor.from_shard_values(
            shard_tvs,
            DeviceMapping(mesh, placements),
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


def distributed_broadcast(t: Tensor, mesh: DeviceMesh) -> Tensor:
    """Replicates a non-distributed tensor onto every device with one collective.

    Args:
        t: A non-distributed source tensor.
        mesh: The device mesh to replicate onto.

    Returns:
        A distributed tensor with :class:`Replicated` placement on every axis.
    """
    if t.mesh.num_devices > 1:
        raise RuntimeError(
            "`F.distributed_broadcast` requires the source tensor to be non-distributed."
        )
    replicated = DeviceMapping(mesh, (Replicated(),) * mesh.ndim)
    with ensure_context():
        signal_buffers = _signal_buffers(mesh)
        # One device and simulated meshes, such as in tests, have no signal
        # buffers.
        if signal_buffers is None:
            return transfer_to(t, replicated)
        shards = ops.distributed_broadcast(TensorValue(t), list(signal_buffers))
        return Tensor.from_shard_values(shards, replicated)


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
            or a :class:`~max.experimental.sharding.DeviceMapping`
            describing the target mesh and placement.

    Returns:
        A tensor with the requested placement on the target device or mesh.

    Raises:
        ShardingError: If the move would change an
            :class:`~max.experimental.sharding.Unknown` placement, whose
            shards have no global value to preserve.
        NotImplementedError: If no supported collective performs the move.
    """
    if isinstance(target, DeviceRef):
        target = target.to_device()
    if isinstance(target, Device):
        target = DeviceMapping(DeviceMesh.single(target), (Replicated(),))
    source = t.mapping
    if (
        t.is_distributed
        and source.mesh != target.mesh
        and source.mesh.mesh_shape == target.mesh.mesh_shape
    ):
        # Moving each shard to the same position needs no gather; the
        # placement then changes on the new mesh.
        with ensure_context():
            moved = Tensor.from_shard_values(
                [
                    ops.transfer_to(
                        TensorValue(shard), DeviceRef.from_device(device)
                    )
                    for shard, device in zip(
                        t.local_shards, target.mesh.devices, strict=True
                    )
                ],
                DeviceMapping(target.mesh, source.placements),
            )
        return transfer_to(moved, target)
    # Unknown shards have no global value to preserve, so only
    # Tensor.rebind_mapping can change an Unknown placement.
    if source != target and (
        any(
            isinstance(p, Unknown)
            for p in (*source.placements, *target.placements)
        )
        if source.mesh != target.mesh
        else any(
            old != new and Unknown in (type(old), type(new))
            for old, new in zip(
                source.placements, target.placements, strict=True
            )
        )
    ):
        raise ShardingError(
            f"transfer_to cannot move {source} to {target}: Unknown shards "
            "have no global value to preserve. Use Tensor.rebind_mapping to "
            "claim a placement, or a collective such as allreduce_sum to "
            "combine them."
        )

    if t.real and not t.is_distributed and target.mesh.num_devices == 1:
        mesh_device = target.mesh.devices[0]
        if t.device == mesh_device:
            return t
        _validation_hooks.device_transfer("Tensor.to()", t, mesh_device)
        return Tensor(storage=t.driver_tensor.to(mesh_device))

    target_p = target.placements

    if not t.is_distributed:
        return _scatter(t, target)

    # Cross-mesh: gather, transfer, scatter.
    if t.mesh != target.mesh:
        source_mesh = t.mesh
        target_mesh = target.mesh

        replicated_p = tuple(Replicated() for _ in range(source_mesh.ndim))
        if t.placements != replicated_p:
            t = transfer_to(t, DeviceMapping(source_mesh, replicated_p))

        single = t.local_shards[0]
        with ensure_context():
            if single.real:
                _validation_hooks.device_transfer(
                    "Tensor.to()", single, target_mesh.devices[0]
                )
                buf = single.driver_tensor.to(target_mesh.devices[0])
                single = Tensor(storage=buf)
            else:
                tv = ops.transfer_to(
                    single.__tensorvalue__(),
                    DeviceRef.from_device(target_mesh.devices[0]),
                )
                single = Tensor.from_graph_value(tv)

        if target_mesh.num_devices == 1:
            return single
        return _scatter(single, target)

    if t.placements == target_p:
        return t

    mesh = t.mesh
    # On a mesh axis of size 1, every placement describes the same data, so
    # the placement changes without communication.
    rebound = tuple(
        new if mesh.mesh_shape[ax] == 1 else old
        for ax, (old, new) in enumerate(
            zip(t.placements, target_p, strict=True)
        )
    )
    if rebound != t.placements:
        t = t.rebind_mapping(DeviceMapping(mesh, rebound))
        if t.placements == target_p:
            return t

    # Mesh axes of size 1 are left out, since they split nothing.
    def split_by(
        placements: Sequence[Placement], tensor_axis: int
    ) -> list[int]:
        return [
            ax
            for ax, p in enumerate(placements)
            if p == Sharded(tensor_axis) and mesh.mesh_shape[ax] > 1
        ]

    def shared_split_axes(tensor_axis: int) -> list[int]:
        """Returns the common leading mesh axes that split ``tensor_axis``."""
        have, want = (
            split_by(t.placements, tensor_axis),
            split_by(target_p, tensor_axis),
        )
        n = 0
        while n < min(len(have), len(want)) and have[n] == want[n]:
            n += 1
        return have[:n]

    # The mesh axes that split a tensor axis divide it in mesh order, so the
    # pieces of the axes the source and target share stay in place. The
    # source's other axes gather, and the target's other axes split each
    # piece further.
    targets = {p.axis for p in target_p if isinstance(p, Sharded)}
    to_replicated = [
        ax
        for ax in range(mesh.ndim)
        if isinstance(t.placements[ax], Partial)
        and isinstance(target_p[ax], Replicated)
    ]
    if to_replicated:
        t = allreduce_sum(t, mesh_axis=to_replicated)
    for tensor_axis in targets:
        shared = shared_split_axes(tensor_axis)
        to_split = split_by(target_p, tensor_axis)[len(shared) :]
        if (
            to_split
            and split_by(t.placements, tensor_axis) == shared
            and all(isinstance(t.placements[ax], Partial) for ax in to_split)
        ):
            t = reduce_scatter(t, scatter_axis=tensor_axis, mesh_axis=to_split)
    partial = [
        ax
        for ax in range(mesh.ndim)
        if isinstance(t.placements[ax], Partial)
        and t.placements[ax] != target_p[ax]
    ]
    if partial:
        t = allreduce_sum(t, mesh_axis=partial)

    for tensor_axis in {p.axis for p in t.placements if isinstance(p, Sharded)}:
        to_gather = split_by(t.placements, tensor_axis)[
            len(shared_split_axes(tensor_axis)) :
        ]
        if to_gather:
            t = allgather(t, tensor_axis=tensor_axis, mesh_axis=to_gather)

    for tensor_axis in targets:
        to_split = split_by(target_p, tensor_axis)[
            len(split_by(t.placements, tensor_axis)) :
        ]
        if to_split:
            t = _local_split(t, to_split, Sharded(tensor_axis))

    to_partial = [
        ax
        for ax in range(mesh.ndim)
        if isinstance(t.placements[ax], Replicated)
        and isinstance(target_p[ax], Partial)
    ]
    if to_partial:
        t = _keep_one_copy(t, to_partial)

    if t.placements != target_p:
        raise NotImplementedError(
            f"No transition from {t.placements} to {target_p}."
        )
    return t
