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

from max.driver import Device
from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    Partial,
    PlacementMapping,
    Replicated,
    Sharded,
)
from max.experimental.tensor import Tensor
from max.graph import BufferValue, DeviceRef, TensorValue, ops

from .utils import (
    _devices_are_unique,
    _even_split_along_axis,
    _mesh_axis_groups,
    _signal_buffers,
    ensure_context,
    is_sharded_on,
)

# ═════════════════════════════════════════════════════════════════════════
#  Group-loop engine
# ═════════════════════════════════════════════════════════════════════════


def _apply_per_group(
    t: Tensor,
    mesh_axis: int,
    new_placement: Replicated | Sharded | Partial,
    *,
    hw_op: Callable[[list[TensorValue], list[BufferValue]], list[TensorValue]],
    sim_op: Callable[[list[TensorValue]], list[TensorValue]],
) -> Tensor:
    """Apply a collective operation per-group along a mesh axis.

    Two dispatch paths:

    1. **Hardware collective** (real multi-device, signal buffers available):
       uses ``hw_op`` with per-group signal buffers.
    2. **Simulated graph** (same-device mesh in graph context):
       uses ``sim_op`` with pairwise graph ops.
    """
    mesh = t.mesh
    groups = _mesh_axis_groups(mesh, mesh_axis)
    new_p = list(t.placements)
    new_p[mesh_axis] = new_placement
    new_placements = tuple(new_p)

    with ensure_context():
        shards = [s.__tensorvalue__() for s in t.local_shards]
        result = list(shards)

        use_hw = _devices_are_unique(shards) and len(groups[0]) > 1
        signal_bufs = _signal_buffers(mesh) if use_hw else None

        for group in groups:
            group_inputs = [shards[idx] for idx in group]
            if signal_bufs is not None:
                group_signals = [signal_bufs[idx] for idx in group]
                group_result = hw_op(group_inputs, group_signals)
            else:
                group_result = sim_op(group_inputs)
            for i, idx in enumerate(group):
                result[idx] = group_result[i]

        return Tensor.from_shard_values(
            result, PlacementMapping(mesh, new_placements)
        )


# ═════════════════════════════════════════════════════════════════════════
#  Primitive collectives (single mesh-axis, explicit parameters)
# ═════════════════════════════════════════════════════════════════════════


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
    return _apply_per_group(
        t,
        mesh_axis,
        Replicated(),
        hw_op=lambda inputs, sigs: ops.allreduce.sum(inputs, sigs),
        sim_op=lambda inputs: [functools.reduce(ops.add, inputs)] * len(inputs),
    )


def allgather(t: Tensor, tensor_axis: int = 0, mesh_axis: int = 0) -> Tensor:
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
    return _apply_per_group(
        t,
        mesh_axis,
        Replicated(),
        hw_op=lambda inputs, sigs: ops.allgather(
            inputs, sigs, axis=tensor_axis
        ),
        sim_op=lambda inputs: [ops.concat(inputs, tensor_axis)] * len(inputs),
    )


def reduce_scatter(
    t: Tensor, scatter_axis: int = 0, mesh_axis: int = 0
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
    t = allreduce_sum(t, mesh_axis=mesh_axis)
    return _local_split(t, mesh_axis=mesh_axis, tensor_axis=scatter_axis)


def _local_split(t: Tensor, mesh_axis: int, tensor_axis: int) -> Tensor:
    """Replicated → Sharded: each device slices its local copy (no communication)."""
    mesh = t.mesh
    n = mesh.mesh_shape[mesh_axis]
    groups = _mesh_axis_groups(mesh, mesh_axis)

    new_p = list(t.placements)
    new_p[mesh_axis] = Sharded(tensor_axis)
    new_placements = tuple(new_p)

    with ensure_context():
        shards: list[TensorValue] = [
            s.__tensorvalue__() for s in t.local_shards
        ]
        result: list[TensorValue] = list(shards)
        for group in groups:
            for rank_in_group, idx in enumerate(group):
                split_chunks = _even_split_along_axis(
                    shards[idx], tensor_axis, n
                )
                result[idx] = split_chunks[rank_in_group]
        return Tensor.from_shard_values(
            result, PlacementMapping(mesh, new_placements)
        )


# ═════════════════════════════════════════════════════════════════════════
#  Scatter — distribute a non-distributed tensor onto a mesh
# ═════════════════════════════════════════════════════════════════════════


def _scatter(t: Tensor, target: DeviceMapping) -> Tensor:
    """Distribute a non-distributed tensor across a mesh.

    Splits along Sharded axes and transfers each piece to its device.
    Replicated axes duplicate the data.  Partial placement is invalid
    for scatter (there is no "partial" source data).

    The loop iterates over mesh axes, expanding ``shard_tvs`` at each
    step.  Invariant: after processing axis *k*, ``len(shard_tvs)``
    equals ``product(mesh_shape[:k+1])``.
    """
    assert not t.is_distributed, "_scatter expects a non-distributed tensor"
    mesh = target.mesh
    placements = target.to_placements()

    with ensure_context():
        tv = t.__tensorvalue__()

        # Fast path: transfer fully-Replicated tensor using single
        # distributed_broadcast collective.
        if mesh.num_devices > 1 and all(
            isinstance(p, Replicated) for p in placements
        ):
            shard_tvs = _broadcast_replicated(tv, mesh)
            return Tensor.from_shard_values(
                shard_tvs, PlacementMapping(mesh, placements)
            )

        # General path: split along Sharded axes, duplicate along Replicated
        # axes, then transfer each piece to its target device.
        shard_tvs = [tv]
        for mesh_axis in range(mesh.ndim):
            p = placements[mesh_axis]
            n = mesh.mesh_shape[mesh_axis]
            if isinstance(p, Sharded):
                new_tvs: list[TensorValue] = []
                for sv in shard_tvs:
                    new_tvs.extend(_even_split_along_axis(sv, p.axis, n))
                shard_tvs = new_tvs
            elif isinstance(p, Replicated):
                shard_tvs = [sv for sv in shard_tvs for _ in range(n)]
            else:
                raise ValueError("Cannot scatter with Partial placement.")
        shard_tvs = [
            ops.transfer_to(sv, DeviceRef.from_device(mesh.devices[i]))
            for i, sv in enumerate(shard_tvs)
        ]
        return Tensor.from_shard_values(
            shard_tvs, PlacementMapping(mesh, placements)
        )


def _broadcast_replicated(
    tv: TensorValue, mesh: DeviceMesh
) -> list[TensorValue]:
    """Broadcast values to every device using one collective op."""
    signal_bufs = _signal_buffers(mesh)
    if signal_bufs is None:
        # If the device mesh is just CPUs, fall back to per-device transfers.
        return [
            ops.transfer_to(tv, DeviceRef.from_device(d)) for d in mesh.devices
        ]
    mesh_device_refs = [DeviceRef.from_device(d) for d in mesh.devices]
    if tv.device not in mesh_device_refs:
        tv = ops.transfer_to(tv, mesh_device_refs[0])
    return ops.distributed_broadcast(tv, signal_bufs)


# ═════════════════════════════════════════════════════════════════════════
#  transfer_to — the ONE public entry point
# ═════════════════════════════════════════════════════════════════════════


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
        target = PlacementMapping(DeviceMesh.single(target), (Replicated(),))

    # Short-circuit for single-device tensors.
    if t.real and not t.is_distributed and target.mesh.num_devices == 1:
        mesh_device = target.mesh.devices[0]
        if t.device == mesh_device:
            return t
        return Tensor(storage=t.driver_tensor.to(mesh_device))

    target_p = target.to_placements()

    # ── Non-distributed input: scatter onto target mesh ─────────────
    if not t.is_distributed:
        return _scatter(t, target)

    # ── Cross-mesh: gather → transfer → scatter ─────────────────────
    if t.mesh != target.mesh:
        source_mesh = t.mesh
        target_mesh = target.mesh

        # Resolve to fully Replicated on source mesh first.
        replicated_p = tuple(Replicated() for _ in range(source_mesh.ndim))
        if t.placements != replicated_p:
            t = transfer_to(t, PlacementMapping(source_mesh, replicated_p))

        # All shards are identical — take one and transfer.
        single = t.local_shards[0]
        with ensure_context():
            if single.real:
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

    # ── Same mesh: three-pass collective redistribution ──────────────
    if t.placements == target_p:
        return t

    mesh = t.mesh

    # Pass 1: resolve Partials.
    # Re-read t.placements[ax] each iteration — prior passes may change t.
    for ax in range(mesh.ndim):
        cp, tp = t.placements[ax], target_p[ax]
        if not isinstance(cp, Partial):
            continue
        if isinstance(tp, Partial):
            # Same reduce op → no-op; different reduce ops → unsupported.
            if cp == tp:
                continue
            raise NotImplementedError(
                f"Partial({cp.reduce_op}) → Partial({tp.reduce_op}) "
                "redistribution is not supported."
            )
        if isinstance(tp, Sharded) and not is_sharded_on(t, tp.axis, ax):
            t = reduce_scatter(t, scatter_axis=tp.axis, mesh_axis=ax)
        else:
            t = allreduce_sum(t, mesh_axis=ax)

    # Pass 2: gather Sharded axes that need to become Replicated or re-shard
    # to a different tensor axis.
    for ax in range(mesh.ndim):
        cp, tp = t.placements[ax], target_p[ax]
        if isinstance(cp, Sharded) and (
            isinstance(tp, Replicated)
            or (isinstance(tp, Sharded) and cp.axis != tp.axis)
        ):
            t = allgather(t, tensor_axis=cp.axis, mesh_axis=ax)

    # Pass 3: local split for Replicated → Sharded (zero communication).
    for ax in range(mesh.ndim):
        cp, tp = t.placements[ax], target_p[ax]
        if isinstance(cp, Replicated) and isinstance(tp, Sharded):
            t = _local_split(t, mesh_axis=ax, tensor_axis=tp.axis)

    return t
