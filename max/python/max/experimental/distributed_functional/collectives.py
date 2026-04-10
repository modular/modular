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

"""Collective communication ops for distributed Tensors.

Each op takes a distributed :class:`~max.experimental.tensor.Tensor`,
performs a collective across devices on a specified mesh axis, and returns
a distributed Tensor with updated placements.

The dispatch is **device-agnostic**: when signal buffers are available
(≥2 unique GPUs), ops use the hardware-accelerated ``max.graph.ops``
collectives.  Otherwise they fall back to simulated equivalents
(element-wise add / concat / split on the graph values) that work on
**any** device — CPU or GPU.  This means you can simulate N devices
on a single GPU for development and testing of fused GPU ops.

Placement transitions::

    all_reduce_sum      :  Partial      → Replicated
    all_gather          :  Sharded(k)   → Replicated
    reduce_scatter      :  Partial      → Sharded(dim)
    distributed_scatter :  pre-split chunks on root → Sharded (root-to-many)
    distributed_broadcast: single tensor on root → Replicated (root-to-all)
Multi-device multi-axis mesh strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The allreduce kernel indexes signal buffers by **absolute GPU device
ID** (``ctx.id()``), so sub-communicators with non-zero-based IDs
cannot use a smaller buffer array.  All multi-device collectives
therefore use **full-mesh signal buffers** with per-group masking or
post-processing:

- **allreduce**: zero-pad non-group shards → full allreduce → take
  group results.
- **allgather**: full allgather → split into per-device chunks →
  concatenate only group chunks.
- **reduce_scatter**: native ``ops.reducescatter.sum`` for single
  groups; falls back to grouped allreduce → split for multi-group.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Generator, Sequence
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

import numpy as np
from max.experimental import tensor
from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    Partial,
    PlacementMapping,
    Replicated,
    Sharded,
)
from max.graph import DeviceRef, TensorValue, ops

from ._context_provider import functional

if TYPE_CHECKING:
    from max.graph import BufferValue


# ─── Shared helpers (used by all dispatch modules) ──────────────────────


def to_shard_tvs(t: tensor.Tensor, n: int | None = None) -> list[TensorValue]:
    """Sharded -> per-shard TensorValues. Unsharded -> broadcast single TV."""
    if t.is_distributed:
        if t._state is None and t._storages is None:
            raise ValueError(
                "Distributed tensor has no graph values and no storage. "
                "All shards must be backed by valid graph values or "
                "realized storage."
            )
        return [s.__tensorvalue__() for s in t.local_shards]
    tv = t.__tensorvalue__()
    return [tv] * (n or 1)


def make_distributed(
    shard_values: Sequence[tensor.GraphValue],
    mapping: DeviceMapping,
) -> tensor.Tensor:
    """Wraps per-shard graph values into a distributed unrealized Tensor."""
    ctx = tensor.current_realization_context()
    return ctx.create_unrealized(
        tuple(shard_values),
        mapping=mapping,
    )


def _has_distributed(*args: Any) -> bool:
    """Returns True if any positional arg is a sharded Tensor."""
    return any(isinstance(a, tensor.Tensor) and a.is_distributed for a in args)


def _get_signal_buffers(
    mesh: DeviceMesh | None = None,
) -> list[BufferValue] | None:
    """Returns signal buffers from the current realization context, if any.

    When the active realization context has ``signal_buffers`` set,
    collective ops use hardware-accelerated multi-device communication.
    Otherwise they fall back to simulated equivalents that work on any
    device (CPU or GPU).

    Returns ``None`` (triggering the simulated path) when:
    - No realization context is active.
    - The mesh has fewer than 2 unique GPU devices.
    - The mesh has only CPU devices (even if signal buffers exist on
      the context from a prior GPU compilation).

    If *mesh* is provided and the context supports lazy creation
    (``ensure_signal_buffers``), signal buffers are created on-demand
    for multi-GPU meshes.
    """
    ctx = tensor.current_realization_context(None)
    if ctx is None:
        return None

    if hasattr(ctx, "signal_buffers") and ctx.signal_buffers is not None:
        bufs = ctx.signal_buffers
        if mesh is not None:
            from max.driver import Accelerator as _Acc

            if not any(isinstance(d, _Acc) for d in mesh.devices):
                # Signal buffers exist (compiled for GPU) but this mesh
                # has no GPUs — fall back to simulated collectives.
                return None
        return bufs

    if mesh is not None and hasattr(ctx, "ensure_signal_buffers"):
        return ctx.ensure_signal_buffers(mesh)

    return None


# ─── Mesh-axis grouping ──────────────────────────────────────────────────


def _mesh_axis_groups(mesh: DeviceMesh, mesh_axis: int) -> list[list[int]]:
    """Device-index groups that share coordinates on all axes except *mesh_axis*.

    For mesh ``(dp=2, tp=2)`` and ``mesh_axis=1`` (tp), returns
    ``[[0, 1], [2, 3]]``: each group is a set of devices that differ
    only in their tp coordinate.
    """
    axis_size = mesh.mesh_shape[mesh_axis]
    # Row-major stride for the target axis.
    stride = 1
    for k in range(mesh_axis + 1, len(mesh.mesh_shape)):
        stride *= mesh.mesh_shape[k]

    groups: list[list[int]] = []
    visited: set[int] = set()
    for base in range(mesh.num_devices):
        if base in visited:
            continue
        group = [base + i * stride for i in range(axis_size)]
        visited.update(group)
        groups.append(group)
    return groups


# ─── Simulated collective primitive ──────────────────────────────────────
# NOTE: This is intentionally part of the library, not the test harness.
# Simulated collectives are the fallback path when signal buffers are
# not available.  They work on ANY device (CPU or GPU) because they use
# only graph-level ops (add, concat, split).  This enables simulating
# N-device meshes on a single GPU for testing fused GPU-only kernels.


def _sim_collective(
    shards: list[TensorValue],
    combine: Callable[[TensorValue, TensorValue], TensorValue],
    mesh: DeviceMesh,
    mesh_axis: int,
) -> list[TensorValue]:
    """Apply *combine* pairwise within each mesh-axis group (simulated).

    This is the single workhorse behind simulated all-reduce
    (``combine=ops.add``), simulated all-gather
    (``combine=lambda a, b: ops.concat([a, b], axis)``), etc.
    """
    groups = _mesh_axis_groups(mesh, mesh_axis)
    result = list(shards)
    for group in groups:
        acc = shards[group[0]]
        for idx in group[1:]:
            acc = combine(acc, shards[idx])
        for idx in group:
            result[idx] = acc
    return result


# ─── Helpers ─────────────────────────────────────────────────────────────


def _resolve_mesh_axis(
    t: tensor.Tensor, mesh_axis: str | int | None
) -> tuple[DeviceMesh, int]:
    """Validate distributed tensor and resolve mesh axis."""
    if not t.is_distributed:
        raise ValueError("Collective ops require a distributed tensor.")
    mesh = t.mesh
    ax = mesh._resolve_axis(mesh_axis) if mesh_axis is not None else 0
    return mesh, ax


def _has_multi_device_buffers(mesh: DeviceMesh) -> bool:
    """True if signal buffers are available for real multi-device collectives.

    Returns True only when the mesh has ≥2 unique GPU devices and
    signal buffers can be created.  A mesh with multiple references to
    the same device (simulated multi-device on one CPU/GPU) returns
    False and the caller falls back to the simulated path, which works
    on any device including GPU.
    """
    return _get_signal_buffers(mesh) is not None


def _full_mesh_bufs(mesh: DeviceMesh) -> list[BufferValue]:
    """Get signal buffers for the full mesh, raising on failure."""
    bufs = _get_signal_buffers(mesh)
    if bufs is None:
        raise RuntimeError(
            "Failed to create signal buffers for mesh — "
            "expected multi-GPU mesh."
        )
    return bufs


def _zero_like(s: TensorValue) -> TensorValue:
    """Create a zero tensor with the same shape, dtype, and device as *s*."""
    shape = [int(d) for d in s.shape]
    np_dt = s.dtype.to_numpy()
    return ops.constant(
        np.zeros(shape, dtype=np_dt), dtype=s.dtype, device=s.device
    )


# ─── Multi-device collective primitives ──────────────────────────────────
#
# Each function handles *any* mesh rank uniformly.  For multi-axis meshes,
# we use full-mesh ops with per-group masking / post-processing (because
# the kernel indexes signal buffers by absolute device ID).


def _multi_device_allreduce(
    shards: list[TensorValue],
    mesh: DeviceMesh,
    mesh_axis: int,
) -> list[TensorValue]:
    """Grouped allreduce: zero-pad non-group → full allreduce → take group.

    For a single group (1D mesh or axis spanning all devices), no zeros
    are created and this reduces to a single full-mesh allreduce.
    """
    groups = _mesh_axis_groups(mesh, mesh_axis)
    bufs = _full_mesh_bufs(mesh)
    result = list(shards)
    for group in groups:
        group_set = set(group)
        masked = [
            shards[i] if i in group_set else _zero_like(shards[i])
            for i in range(len(shards))
        ]
        reduced = ops.allreduce.sum(masked, bufs)
        for idx in group:
            result[idx] = reduced[idx]
    return result


def _multi_device_allgather(
    shards: list[TensorValue],
    mesh: DeviceMesh,
    mesh_axis: int,
    tensor_axis: int,
) -> list[TensorValue]:
    """Grouped allgather: full allgather → split → keep group chunks.

    For a single group this is just a direct full-mesh allgather.
    """
    groups = _mesh_axis_groups(mesh, mesh_axis)
    bufs = _full_mesh_bufs(mesh)
    full = ops.allgather(shards, bufs, axis=tensor_axis)

    if len(groups) == 1:
        return full

    # Multi-group: each device got ALL shards concatenated.  Split into
    # per-device chunks and keep only the chunks from this device's group.
    num_devices = len(shards)
    shard_dim = int(shards[0].shape[tensor_axis])
    result = list(full)
    for group in groups:
        for idx in group:
            chunks = ops.split(
                full[idx], [shard_dim] * num_devices, axis=tensor_axis
            )
            result[idx] = ops.concat([chunks[i] for i in group], tensor_axis)
    return result


def _multi_device_reduce_scatter(
    shards: list[TensorValue],
    mesh: DeviceMesh,
    mesh_axis: int,
    scatter_axis: int,
) -> list[TensorValue]:
    """Grouped reduce-scatter using ``ops.reducescatter.sum``.

    For a single group (1D mesh or axis spanning all devices), this is a
    direct call to the hardware-accelerated reduce-scatter kernel.
    For multi-group meshes, falls back to grouped allreduce + split
    (the kernel indexes buffers by absolute device ID so
    sub-communicators need masking).
    """
    groups = _mesh_axis_groups(mesh, mesh_axis)
    bufs = _full_mesh_bufs(mesh)

    if len(groups) == 1:
        # Single group — use the native reduce-scatter op directly.
        return ops.reducescatter.sum(shards, bufs, axis=scatter_axis)

    # Multi-group: no native sub-communicator support yet, so fall back
    # to grouped allreduce + per-device split.
    reduced = _multi_device_allreduce(shards, mesh, mesh_axis)
    group_size = mesh.mesh_shape[mesh_axis]
    chunk_size = int(reduced[0].shape[scatter_axis]) // group_size
    result = list(reduced)
    for group in groups:
        for i, idx in enumerate(group):
            chunks = ops.split(
                reduced[idx], [chunk_size] * group_size, axis=scatter_axis
            )
            result[idx] = chunks[i]
    return result


# ─── Public collective ops ───────────────────────────────────────────────


def _all_reduce_sum(
    t: tensor.Tensor, mesh_axis: str | int | None = None
) -> tensor.Tensor:
    """All-reduce-sum: ``Partial → Replicated``.

    Sums partial results across devices on the specified mesh axis.

    Args:
        t: A distributed tensor (typically with Partial placement).
        mesh_axis: Mesh axis to reduce over (name or index).
            Can be omitted for 1D meshes.

    Example::

        out = hidden @ W_down              # → Partial(SUM)
        out = F.all_reduce_sum(out, "tp")  # → Replicated
    """
    mesh, ax = _resolve_mesh_axis(t, mesh_axis)
    p = t.placements[ax]
    if isinstance(p, Replicated):
        return t  # Already replicated — nothing to reduce.
    if not isinstance(p, Partial):
        raise ValueError(
            f"all_reduce_sum requires Partial placement on mesh axis {ax}, "
            f"but got {type(p).__name__}. "
            f"Use all_gather for Sharded tensors."
        )
    shards = to_shard_tvs(t)

    if _has_multi_device_buffers(mesh):
        reduced = _multi_device_allreduce(shards, mesh, ax)
    else:
        reduced = _sim_collective(shards, ops.add, mesh, ax)

    new_placements = list(t.placements)
    new_placements[ax] = Replicated()
    return make_distributed(
        reduced, PlacementMapping(mesh, tuple(new_placements))
    )


def _all_gather(
    t: tensor.Tensor,
    tensor_axis: int,
    mesh_axis: str | int | None = None,
) -> tensor.Tensor:
    """All-gather: ``Sharded(k) → Replicated``.

    Concatenates shards across devices on the specified mesh axis.

    Args:
        t: A distributed tensor (typically with Sharded placement).
        tensor_axis: Tensor dimension to concatenate along.
        mesh_axis: Mesh axis to gather over (name or index).
            Can be omitted for 1D meshes.

    Example::

        out = x @ W_col                                        # → Sharded(1)
        out = F.all_gather(out, tensor_axis=1, mesh_axis="tp") # → Replicated
    """
    mesh, ax = _resolve_mesh_axis(t, mesh_axis)
    p = t.placements[ax]
    if isinstance(p, Replicated):
        return t  # Already replicated — nothing to gather.
    if not isinstance(p, Sharded):
        raise ValueError(
            f"all_gather requires Sharded placement on mesh axis {ax}, "
            f"but got {type(p).__name__}. "
            f"Use all_reduce_sum for Partial tensors."
        )
    shards = to_shard_tvs(t)

    if _has_multi_device_buffers(mesh):
        gathered = _multi_device_allgather(shards, mesh, ax, tensor_axis)
    else:
        gathered = _sim_collective(
            shards, lambda a, b: ops.concat([a, b], tensor_axis), mesh, ax
        )

    new_placements = list(t.placements)
    new_placements[ax] = Replicated()
    return make_distributed(
        gathered, PlacementMapping(mesh, tuple(new_placements))
    )


def _reduce_scatter(
    t: tensor.Tensor,
    scatter_axis: int,
    mesh_axis: str | int | None = None,
) -> tensor.Tensor:
    """Reduce-scatter: ``Partial → Sharded(scatter_axis)``.

    Sums partial results and scatters the result so each device keeps
    only its slice along ``scatter_axis``.  This is more efficient than
    ``all_reduce_sum`` followed by slicing when the next op wants
    sharded input (e.g. col-TP → row-TP pipeline).

    Args:
        t: A distributed tensor with Partial placement.
        scatter_axis: Tensor dimension to scatter the reduced result along.
        mesh_axis: Mesh axis to reduce-scatter over (name or index).
            Can be omitted for 1D meshes.

    Example::

        out = hidden @ W_down                             # → Partial(SUM)
        out = F.reduce_scatter(out, scatter_axis=1, mesh_axis="tp")
        # → Sharded(1), each device has its slice of the reduced result
    """
    mesh, ax = _resolve_mesh_axis(t, mesh_axis)
    p = t.placements[ax]
    if not isinstance(p, Partial):
        raise ValueError(
            f"reduce_scatter requires Partial placement on mesh axis {ax}, "
            f"but got {type(p).__name__}. "
            f"Reduce-scatter sums partial results then scatters."
        )
    shards = to_shard_tvs(t)

    if _has_multi_device_buffers(mesh):
        scattered = _multi_device_reduce_scatter(shards, mesh, ax, scatter_axis)
    else:
        # Simulated: all-reduce within group, then split.
        reduced = _sim_collective(shards, ops.add, mesh, ax)
        group_size = mesh.mesh_shape[ax]
        groups = _mesh_axis_groups(mesh, ax)
        scattered = list(reduced)
        for group in groups:
            full_val = reduced[group[0]]
            chunks = ops.split(
                full_val,
                [int(full_val.shape[scatter_axis]) // group_size] * group_size,
                axis=scatter_axis,
            )
            for i, idx in enumerate(group):
                scattered[idx] = chunks[i]

    new_placements = list(t.placements)
    new_placements[ax] = Sharded(scatter_axis)
    return make_distributed(
        scattered, PlacementMapping(mesh, tuple(new_placements))
    )


def _has_partial(t: tensor.Tensor) -> bool:
    """Returns True if any placement on the tensor is Partial."""
    if not t.is_distributed:
        return False
    return any(isinstance(p, Partial) for p in t.placements)


# ─── Partial resolution policy ────────────────────────────────────────

_AUTO_REDUCE_PARTIAL: ContextVar[bool] = ContextVar(
    "_AUTO_REDUCE_PARTIAL", default=True
)


@contextlib.contextmanager
def auto_reduce_partial(enable: bool = True) -> Generator[None]:
    """Control whether Partial placements are auto-reduced.

    When enabled (default), ops with Partial inputs that cannot pass
    through automatically insert ``all_reduce_sum``.  When disabled,
    a ``ValueError`` is raised instead.
    """
    token = _AUTO_REDUCE_PARTIAL.set(enable)
    try:
        yield
    finally:
        _AUTO_REDUCE_PARTIAL.reset(token)


def _resolve_partials(
    t: tensor.Tensor, *, check_policy: bool = True
) -> tensor.Tensor:
    """Resolve all Partial placements via all-reduce-sum.

    Each Partial mesh axis is reduced to Replicated; non-Partial axes
    are left unchanged.  If ``check_policy`` is True and
    ``auto_reduce_partial`` is disabled, raises instead of reducing.

    Args:
        t: A distributed tensor, possibly with Partial placements.
        check_policy: Whether to respect the ``auto_reduce_partial`` flag.

    Returns:
        The tensor with all Partial axes reduced to Replicated.
    """
    if not t.is_distributed:
        return t
    if check_policy and not _AUTO_REDUCE_PARTIAL.get():
        raise ValueError(
            "Partial placement requires explicit all_reduce_sum. "
            "Use resolve_partials() or enable auto_reduce_partial."
        )
    result = t
    for ax, p in enumerate(t.placements):
        if isinstance(p, Partial):
            result = _all_reduce_sum(result, mesh_axis=ax)
    return result


# ─── Shard (distribute) ─────────────────────────────────────────────────


def _shard(
    t: tensor.Tensor,
    mapping: DeviceMapping,
) -> tensor.Tensor:
    """Distributes an unsharded tensor across a device mesh.

    Splits the tensor along sharded axes using graph ops and transfers
    each shard to its target device.  For single-device mappings, just
    transfers to that device.  Works on both realized and unrealized
    tensors since it operates at the TensorValue level.

    Args:
        t: The source (unsharded) tensor.
        mapping: The target device mapping.

    Returns:
        A distributed tensor with one shard per mesh device.
    """
    mesh = mapping.mesh
    if mesh.num_devices == 1:
        return t.to(mesh.devices[0])

    placements = mapping.to_placements()

    # Get the TensorValue — works for both realized and unrealized.
    tv = TensorValue(t)

    # Build per-shard TensorValues by iterating over mesh axes.
    # Each axis either chunks (Sharded) or duplicates (Replicated)
    # the current list of shards.
    shard_tvs: list[TensorValue] = [tv]
    for mesh_axis in range(mesh.ndim):
        p = placements[mesh_axis]
        n = mesh.mesh_shape[mesh_axis]
        if isinstance(p, Partial):
            raise ValueError("Cannot distribute with Partial placement.")
        if isinstance(p, Sharded):

            def _split_even(
                sv: TensorValue, n: int, axis: int
            ) -> list[TensorValue]:
                dim = int(sv.shape[axis])
                base, rem = divmod(dim, n)
                sizes = [base + (1 if i < rem else 0) for i in range(n)]
                return ops.split(sv, sizes, axis=axis)

            shard_tvs = [
                chunk
                for sv in shard_tvs
                for chunk in _split_even(sv, n, p.axis)
            ]
        else:
            shard_tvs = [sv for sv in shard_tvs for _ in range(n)]

    # Transfer each shard to its target device.
    shard_tvs = [
        ops.transfer_to(sv, DeviceRef.from_device(mesh.devices[i]))
        for i, sv in enumerate(shard_tvs)
    ]

    return make_distributed(shard_tvs, mapping)


def _distributed_scatter(
    chunks: Sequence[tensor.Tensor],
    mapping: DeviceMapping,
) -> tensor.Tensor:
    """Scatter pre-split chunks from root to mesh devices.

    Thin wrapper around ``ops.distributed_scatter`` that handles signal
    buffer management and wraps the result as a distributed Tensor.
    The caller is responsible for splitting the source data into chunks.

    Use this for root-to-many fan-out where data lives on one device
    and must be distributed (e.g., input tokens from root to DP replicas).

    Args:
        chunks: Pre-split tensor chunks, all on the root device. One per
            mesh device.
        mapping: The target device mapping describing distribution.

    Returns:
        A distributed tensor with one shard per mesh device.
    """
    mesh = mapping.mesh
    tvs = [TensorValue(c) for c in chunks]

    if _has_multi_device_buffers(mesh):
        bufs = _full_mesh_bufs(mesh)
        # Ensure chunks are on the root (first) GPU device — callers may
        # pass CPU tensors created from numpy.
        root_dev = DeviceRef.from_device(mesh.devices[0])
        tvs = [ops.transfer_to(tv, root_dev) for tv in tvs]
        results = ops.distributed_scatter(tvs, bufs)
        return make_distributed(results, mapping)

    # Simulated: transfer each chunk to its target device.
    results = [
        ops.transfer_to(tv, DeviceRef.from_device(mesh.devices[i]))
        for i, tv in enumerate(tvs)
    ]
    return make_distributed(results, mapping)


def _distributed_broadcast(
    t: tensor.Tensor,
    mapping: DeviceMapping,
) -> tensor.Tensor:
    """Broadcast a tensor from root to all mesh devices.

    Thin wrapper around ``ops.distributed_broadcast`` that handles signal
    buffer management and wraps the result as a distributed Tensor.

    The root device is inferred from the input tensor's device. On a
    simulated mesh (single GPU or CPU), falls back to ``transfer_to``.

    Args:
        t: Input tensor on the root device.
        mapping: The target device mapping (should use Replicated placement).

    Returns:
        A distributed tensor with one copy per mesh device.
    """
    mesh = mapping.mesh
    tv = TensorValue(t)

    if _has_multi_device_buffers(mesh):
        bufs = _full_mesh_bufs(mesh)
        # If the input is not on a mesh device (e.g. CPU tensor from
        # numpy), transfer it to the first GPU as the root.
        mesh_devs = [DeviceRef.from_device(d) for d in mesh.devices]
        if tv.device not in mesh_devs:
            raise ValueError(
                f"broadcast input tensor device {tv.device} is not in "
                f"the mesh devices: {mesh_devs}"
            )
        results = ops.distributed_broadcast(tv, bufs)
        return make_distributed(results, mapping)

    # Simulated: transfer input to each target device.
    results = [
        ops.transfer_to(tv, DeviceRef.from_device(mesh.devices[i]))
        for i in range(len(mesh.devices))
    ]
    return make_distributed(results, mapping)


# ─── Public API (wrapped for context, no Partial resolution) ──────────────

all_reduce_sum = functional(_all_reduce_sum, linear=None)
all_gather = functional(_all_gather, linear=None)
reduce_scatter = functional(_reduce_scatter, linear=None)
resolve_partials = functional(_resolve_partials, linear=None)
shard = functional(_shard, linear=None)
distributed_scatter = functional(_distributed_scatter, linear=None)
distributed_broadcast = functional(_distributed_broadcast, linear=None)


# ─── Materialization helpers (re-exported from _utils) ────────────────
# These live in _utils.py but are re-exported here for backward
# compatibility — many consumers import them from collectives.

from ._utils import materialize as materialize
from ._utils import to_numpy as to_numpy
