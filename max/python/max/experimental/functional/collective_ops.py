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

Each collective acts along a single mesh axis and is intended for use on
tensors that are sharded across a multi-device mesh. :func:`transfer_to`
is the universal entry point for moving a tensor between devices or
placements.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

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
from max.experimental.tensor import Tensor
from max.graph import BufferValue, DeviceRef, TensorValue, ops
from max.graph.dim import StaticDim
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


def _even_split_sizes(dim: int, n: int) -> list[int]:
    """Splits ``dim`` into ``n`` sizes that differ by at most 1."""
    base, rem = divmod(dim, n)
    return [base + (1 if i < rem else 0) for i in range(n)]


def _even_split_along_axis(
    sv: TensorValue, axis: int, n: int
) -> list[TensorValue]:
    """Splits ``sv`` into ``n`` load-balanced chunks along ``axis``."""
    dim = sv.shape[axis]
    if isinstance(dim, StaticDim):
        return list(ops.split(sv, _even_split_sizes(int(dim), n), axis=axis))

    rank_ndim = len(sv.shape)
    chunks: list[TensorValue] = []
    for i in range(n):
        start = (i * dim) // n
        stop = ((i + 1) * dim) // n
        size = stop - start
        start_tv = ops.shape_to_tensor([start])
        stop_tv = ops.shape_to_tensor([stop])
        indices: list[SliceIndex] = [slice(None)] * rank_ndim
        indices[axis] = (slice(start_tv, stop_tv, 1), size)
        chunks.append(ops.slice_tensor(sv, indices))
    return chunks


def _collective(
    t: Tensor,
    mesh_axis: int,
    new_placement: Replicated | Sharded,
    kernel: Callable[[list[Tensor], list[BufferValue]], list[TensorValue]],
    simulated: Callable[[list[TensorValue]], list[TensorValue]],
) -> Tensor:
    """Runs a collective on each group of devices along ``mesh_axis``.

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
    placements = list(t.placements)
    placements[mesh_axis] = new_placement
    with ensure_context():
        # A group of one device exchanges nothing, so it allocates no
        # signal buffers.
        signal_buffers = (
            _signal_buffers(mesh) if mesh.mesh_shape[mesh_axis] > 1 else None
        )
        # Per-rank IR after the collective carries the honest algebraic
        # form (e.g. ``batch_dp_0 + batch_dp_1``); :attr:`Tensor.shape`
        # reads back the same dim on every rank and collapses the wrapper.
        return call_on_mesh(
            run,
            mesh,
            (mesh_axis,),
            out_specs=DeviceMapping(mesh, tuple(placements)),
        )(t, signal_buffers)


def allreduce_sum(t: Tensor, mesh_axis: int = 0) -> Tensor:
    """All-reduces a tensor by summing its shards across a mesh axis.

    Transitions the tensor's placement on ``mesh_axis`` from
    :class:`~max.experimental.sharding.Partial` to
    :class:`~max.experimental.sharding.Replicated`. Every device on
    ``mesh_axis`` ends up holding the sum of all inputs along that axis.

    Args:
        t: The input distributed tensor.
        mesh_axis: The mesh axis along which to reduce.

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
    mesh_axis: int = 0,
) -> Tensor:
    """All-gathers a tensor's shards along a mesh axis.

    Transitions the tensor's placement on ``mesh_axis`` from
    :class:`~max.experimental.sharding.Sharded` to
    :class:`~max.experimental.sharding.Replicated`. Each device gathers
    the shards from its peers and concatenates them along ``tensor_axis``.

    Args:
        t: The input distributed tensor.
        tensor_axis: The tensor axis along which the shards are concatenated.
        mesh_axis: The mesh axis whose placement changes from Sharded to
            Replicated.

    Returns:
        A tensor with the full data replicated across ``mesh_axis``.
    """
    return _collective(
        t,
        mesh_axis,
        Replicated(),
        kernel=lambda shards, signal_buffers: ops.allgather(
            shards, signal_buffers, axis=tensor_axis
        ),
        simulated=lambda shards: (
            [ops.concat(shards, tensor_axis)] * len(shards)
        ),
    )


def reduce_scatter(
    t: Tensor,
    scatter_axis: int = 0,
    mesh_axis: int = 0,
) -> Tensor:
    """Reduces a tensor across a mesh axis and scatters the result.

    Transitions the tensor's placement on ``mesh_axis`` from
    :class:`~max.experimental.sharding.Partial` to
    :class:`~max.experimental.sharding.Sharded`. Each device contributes
    to the sum and ends up with one shard of the reduced tensor along
    ``scatter_axis``.

    Args:
        t: The input distributed tensor.
        scatter_axis: The tensor axis along which the reduced result is
            sharded.
        mesh_axis: The mesh axis whose placement changes from Partial to
            Sharded.

    Returns:
        A tensor with the reduced and re-sharded result.
    """
    return _collective(
        t,
        mesh_axis,
        Sharded(scatter_axis),
        kernel=lambda shards, signal_buffers: ops.reducescatter.sum(
            shards, signal_buffers, axis=scatter_axis
        ),
        # TODO(MXF-493): `_even_split_along_axis` splits uneven chunks and
        # returns the smallest chunks first, while the actual reduce-scatter
        # kernel splits the chunks from largest to smallest.
        simulated=lambda shards: _even_split_along_axis(
            functools.reduce(ops.add, shards), scatter_axis, len(shards)
        ),
    )


def _local_split(t: Tensor, mesh_axis: int, target: Sharded) -> Tensor:
    """``Replicated -> Sharded``: each device slices its local copy with no communication."""

    def split(copies: list[Tensor]) -> list[TensorValue]:
        return [
            _even_split_along_axis(TensorValue(copy), target.axis, len(copies))[
                index
            ]
            for index, copy in enumerate(copies)
        ]

    placements = list(t.placements)
    placements[mesh_axis] = target
    return call_on_mesh(
        split,
        t.mesh,
        (mesh_axis,),
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

    with ensure_context():
        tv = t.__tensorvalue__()

        shard_tvs = [tv]
        for mesh_axis in range(mesh.ndim):
            p = placements[mesh_axis]
            n = mesh.mesh_shape[mesh_axis]
            tensor_axis = p.localized_axis()
            if tensor_axis is not None:
                new_tvs: list[TensorValue] = []
                for sv in shard_tvs:
                    new_tvs.extend(_even_split_along_axis(sv, tensor_axis, n))
                shard_tvs = new_tvs
            elif isinstance(p, Replicated):
                shard_tvs = [sv for sv in shard_tvs for _ in range(n)]
            else:
                raise ValueError(
                    f"Cannot scatter with placement {type(p).__name__}; "
                    "scatter requires Replicated or a placement that localizes "
                    "a single tensor axis (override ``localized_axis()``)."
                )
        shard_tvs = [
            ops.transfer_to(sv, DeviceRef.from_device(mesh.devices[i]))
            for i, sv in enumerate(shard_tvs)
        ]
        return Tensor.from_shard_values(
            shard_tvs,
            DeviceMapping(mesh, placements),
        )


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
    placements on the same mesh, and gathering then re-distributing
    across different meshes.

    Args:
        t: The source tensor, distributed or single-device.
        target: A :class:`~max.driver.Device` to move to a single device,
            or a :class:`~max.experimental.sharding.DeviceMapping`
            describing the target mesh and placement.

    Returns:
        A tensor with the requested placement on the target device or mesh.
    """
    if isinstance(target, DeviceRef):
        target = target.to_device()
    if isinstance(target, Device):
        target = DeviceMapping(DeviceMesh.single(target), (Replicated(),))
    if t.mapping != target and any(
        isinstance(p, Unknown) for p in (*t.placements, *target.placements)
    ):
        raise ShardingError(
            f"transfer_to cannot move {t.mapping} to {target}: Unknown shards "
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

    # Phase-ordered redistribution: reduce Partials first, then unwind
    # Sharded to Replicated, then anything remaining (R -> S, etc.). This
    # avoids double-sharded transit on one tensor dim when a tensor axis
    # is sharded along multiple mesh axes.
    if t.placements == target_p:
        return t

    mesh = t.mesh

    # Phase 0: resolve Partials first (allreduce / reduce_scatter).
    for ax in range(mesh.ndim):
        cp, tp = t.placements[ax], target_p[ax]
        if isinstance(cp, Partial) and cp != tp:
            t = _axis_transition(t, cp, tp, mesh_axis=ax)

    # Phase 1: allgather Sharded to Replicated; reverse mesh-axis order
    # preserves element ordering when one tensor axis is sharded along
    # multiple mesh axes.
    for ax in reversed(range(mesh.ndim)):
        cp, tp = t.placements[ax], target_p[ax]
        if isinstance(cp, Sharded) and cp != tp:
            t = _axis_transition(t, cp, Replicated(), mesh_axis=ax)

    # Phase 2: anything still mismatched (now Replicated -> {Sharded, Partial}).
    for ax in range(mesh.ndim):
        cp, tp = t.placements[ax], target_p[ax]
        if cp != tp:
            t = _axis_transition(t, cp, tp, mesh_axis=ax)

    return t


def _axis_transition(
    t: Tensor, source: Placement, target: Placement, *, mesh_axis: int
) -> Tensor:
    """Inserts the collective for ``source -> target`` on one mesh axis."""
    if source == target:
        return t
    if isinstance(source, Replicated) and isinstance(target, Sharded):
        return _local_split(t, mesh_axis=mesh_axis, target=target)
    if isinstance(source, Sharded) and isinstance(target, Replicated):
        return allgather(
            t,
            tensor_axis=source.axis,
            mesh_axis=mesh_axis,
        )
    if isinstance(source, Sharded) and isinstance(target, Sharded):
        if source == target:
            return t
        if source.axis == target.axis:
            return t
        t = allgather(
            t,
            tensor_axis=source.axis,
            mesh_axis=mesh_axis,
        )
        return _local_split(t, mesh_axis=mesh_axis, target=target)
    if isinstance(source, Partial):
        if source.reduce_op.value in ("min", "max"):
            raise NotImplementedError(
                f"Partial({source.reduce_op}) redistribution is not supported."
            )
        if isinstance(target, Replicated):
            return allreduce_sum(t, mesh_axis=mesh_axis)
        if isinstance(target, Sharded):
            already_sharded = any(
                i != mesh_axis and p.localized_axis() == target.axis
                for i, p in enumerate(t.placements)
            )
            if already_sharded:
                return allreduce_sum(t, mesh_axis=mesh_axis)
            return reduce_scatter(
                t,
                mesh_axis=mesh_axis,
                scatter_axis=target.axis,
            )
        if isinstance(target, Partial):
            raise NotImplementedError(
                f"Partial({source.reduce_op}) -> "
                f"Partial({target.reduce_op}) redistribution is not "
                "supported."
            )
    # Custom-placement hook for subclasses.
    if hasattr(source, "materialize_to"):
        return source.materialize_to(t, target, mesh_axis=mesh_axis)
    if hasattr(target, "materialize_from"):
        return target.materialize_from(t, source, mesh_axis=mesh_axis)
    raise NotImplementedError(
        f"No transition {type(source).__name__} -> {type(target).__name__}"
    )
