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

"""Provides :func:`call_on_mesh`, which runs a function on each mesh device.

Functional ops, collectives, creation ops and per-device model code all run
through it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from max import tree
from max.driver import CPU
from max.experimental.realization_context import ensure_context
from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    ShardingError,
    Unknown,
)
from max.experimental.sharding.action import PerShard
from max.experimental.sharding.per_shard_dim import PerShardDim
from max.experimental.tensor import Tensor, default_device
from max.graph import BufferValue, Dim, Shape, TensorValue


def call_on_mesh(
    fn: Callable[..., Any],
    mesh: DeviceMesh,
    mesh_axes: Sequence[str | int] = (),
    *,
    out_specs: DeviceMapping | Sequence[DeviceMapping] | None = None,
) -> Callable[..., Any]:
    """Wraps ``fn`` to run on each device of ``mesh`` or on groups of devices.

    Use it for code that a sharding rule cannot express, such as a
    collective kernel or model code that differs per device. It is similar
    to JAX ``shard_map`` and PyTorch ``local_map``.

    By default, ``fn`` runs once per device, with that device as the default
    device. It receives the device's part of each argument:

    - A distributed tensor becomes the device's shard.
    - A :class:`~max.experimental.sharding.action.PerShard` becomes the
      device's entry.
    - A shape or dim that differs per device becomes the device's shape or
      size.
    - A single-device tensor is passed whole to every device. On a mesh of
      several devices, it must be on the host.
    - Any other argument is passed unchanged.

    Arguments can be nested in lists, tuples and dicts. When ``fn`` writes to
    a shard, for example with a buffer store, the distributed tensor holds
    the write.

    With ``mesh_axes``, the devices are grouped along those axes and ``fn``
    runs once per group, like a collective kernel. It receives each
    per-device argument as a list with one entry per device of the group,
    and returns a list with one result per device of the group. For example,
    on a 2x2 mesh with axes ``("dp", "tp")``, grouping along ``"tp"`` makes
    the groups of devices ``[0, 1]`` and ``[2, 3]``.

    .. skip: next

    .. code-block:: python

        from max.experimental import functional as F
        from max.experimental.sharding import DeviceMapping, Replicated
        from max.experimental.sharding.action import PerShard
        from max.graph import ops

        # Sums the shards of x over the devices along "tp". ops.allreduce.sum
        # takes a list of tensors and a list of signal buffers, one per device.
        summed = F.call_on_mesh(
            ops.allreduce.sum,
            x.mesh,
            ("tp",),
            out_specs=DeviceMapping(x.mesh, (Replicated(),)),
        )(x, PerShard(signal_buffers))

    Args:
        fn: A function of single-device values.
        mesh: The mesh to run on.
        mesh_axes: The mesh axes to group the devices along, by name or
            index. Defaults to no axes, which runs ``fn`` once per device.
        out_specs: The :class:`~max.experimental.sharding.DeviceMapping` of
            every tensor output of ``fn``. Can be a single mapping, or a list
            with the same length as the number of tensor outputs. Defaults to
            :class:`~max.experimental.sharding.Unknown` on every mesh axis,
            which claims nothing about how the per-device outputs relate.

    Returns:
        A callable that wraps ``fn`` and applies it on every device of the
        mesh, or on every group along ``mesh_axes``. It takes ``fn``'s
        arguments and returns ``fn``'s outputs as distributed tensors.

    Raises:
        ShardingError: If an argument or result does not fit the mesh.
            This includes when a per-device argument does not have one entry
            per device, a distributed argument is on a mesh of another shape
            or with other axis names, a single-device argument is on an
            accelerator while the mesh has several devices, a group returns a
            different number of results than it has devices, or
            ``out_specs`` lists a different number of mappings than ``fn``
            has tensor outputs.
    """
    group_axes = tuple(mesh._resolve_axis(axis) for axis in mesh_axes)

    def call(*args: Any, **kwargs: Any) -> Any:
        with ensure_context():
            flat_args, structure = tree.flatten(
                (args, kwargs), leaf=_differs_per_device
            )
            shards_by_tensor: dict[int, tuple[Tensor, ...]] = {}
            per_device_args = [
                _per_device_values(arg, mesh, shards_by_tensor)
                for arg in flat_args
            ]
            results: list[Any] = [None] * mesh.num_devices
            for group in _device_groups(mesh, group_axes):
                flat_group_args = []
                for arg, values in zip(flat_args, per_device_args, strict=True):
                    if values is None:
                        # The same on every device, such as a host tensor.
                        flat_group_args.append(arg)
                    elif group_axes:
                        # fn sees the whole group, one entry per device.
                        flat_group_args.append([values[d] for d in group])
                    else:
                        # Each device is its own group.
                        flat_group_args.append(values[group[0]])
                group_args, group_kwargs = tree.unflatten(
                    structure, flat_group_args
                )
                if group_axes:
                    group_results = fn(*group_args, **group_kwargs)
                    if len(group_results) != len(group):
                        raise ShardingError(
                            f"A group of {len(group)} devices returned "
                            f"{len(group_results)} results."
                        )
                else:
                    # Single-device code creates its tensors on the device
                    # it runs for.
                    with default_device(mesh.devices[group[0]]):
                        group_results = [fn(*group_args, **group_kwargs)]
                for device, result in zip(group, group_results, strict=True):
                    results[device] = result
            for arg, values in zip(flat_args, per_device_args, strict=True):
                if isinstance(arg, Tensor) and values is not None:
                    _copy_state_from_shards(arg, values)
            return _join(results, mesh, out_specs)

    return call


def _device_groups(
    mesh: DeviceMesh, group_axes: Sequence[int]
) -> list[list[int]]:
    """Returns the groups of devices along ``group_axes``.

    A group holds the devices whose coordinates differ only along
    ``group_axes``, in row-major order.
    """
    other_axes = [a for a in range(mesh.ndim) if a not in group_axes]
    groups: dict[tuple[int, ...], list[int]] = {}
    for device in range(mesh.num_devices):
        coordinates = tuple(mesh.device_coord(device, a) for a in other_axes)
        groups.setdefault(coordinates, []).append(device)
    return list(groups.values())


def _differs_per_device(value: Any) -> bool:
    """Returns whether ``value`` may differ from one device to the next."""
    return isinstance(value, (Tensor, PerShard, PerShardDim, Shape))


def _per_device_values(
    value: Any,
    mesh: DeviceMesh,
    shards_by_tensor: dict[int, tuple[Tensor, ...]],
) -> Sequence[Any] | None:
    """Returns ``value`` as each device of ``mesh`` sees it, one per device.

    Returns ``None`` for a value that every device reads as it is: a shape
    with no per-device dims, or a single-device tensor on the host (CPU) when
    the mesh has several devices. On a one-device mesh, a single-device
    tensor returns ``(value,)``.
    """
    if isinstance(value, Shape):
        sizes = [
            _per_device_values(Dim(dim), mesh, shards_by_tensor)
            for dim in value
        ]
        if all(per_device is None for per_device in sizes):
            return None
        return [
            Shape(
                dim if per_device is None else per_device[device]
                for dim, per_device in zip(value, sizes, strict=True)
            )
            for device in range(mesh.num_devices)
        ]
    if isinstance(value, (PerShard, PerShardDim)):
        entries = (
            value.values if isinstance(value, PerShard) else value.per_shard
        )
        if len(entries) != mesh.num_devices:
            raise ShardingError(
                f"{value!r} holds {len(entries)} values for the "
                f"{mesh.num_devices} devices of {mesh}."
            )
        return entries
    if not isinstance(value, Tensor):
        return None
    if not value.is_distributed:
        if mesh.num_devices == 1:
            return (value,)
        # Only a host value is readable by every device.
        if not isinstance(value.device, CPU):
            raise ShardingError(
                f"A tensor on {value.device} alone cannot be read by every "
                f"device of {mesh}. Place it on the mesh with .to(), or keep "
                "it on the host."
            )
        return None
    if (value.mesh.mesh_shape, value.mesh.axis_names) != (
        mesh.mesh_shape,
        mesh.axis_names,
    ):
        # Device i of every input pairs with device i of the others.
        raise ShardingError(
            "call_on_mesh needs all distributed inputs on meshes of one "
            f"shape and axis names, got {mesh} and {value.mesh}."
        )
    if id(value) not in shards_by_tensor:
        shards_by_tensor[id(value)] = value.local_shards
    return shards_by_tensor[id(value)]


def _copy_state_from_shards(tensor: Tensor, shards: Sequence[Tensor]) -> None:
    """Copies the current value of each of ``shards`` into ``tensor``'s state.

    A write through a shard, such as a buffer store, gives only the shard a
    new value.
    """
    if tensor._state is None:
        return
    values = tuple(
        old if shard._state is None else shard._state.value
        for shard, old in zip(shards, tensor._state.values, strict=True)
    )
    if any(
        new is not old
        for new, old in zip(values, tensor._state.values, strict=True)
    ):
        tensor._state = type(tensor._state)(values, tensor._state.ctx)


def _is_result(value: Any) -> bool:
    """Returns whether ``value`` is a tensor output of ``fn`` on one device."""
    return isinstance(value, (Tensor, TensorValue, BufferValue))


def _join(
    results: Sequence[tree.Tree[Tensor | TensorValue | BufferValue]],
    mesh: DeviceMesh,
    out_specs: DeviceMapping | Sequence[DeviceMapping] | None,
) -> tree.Tree[Tensor]:
    """Builds one distributed tensor from each tensor output's device values.

    ``results`` holds ``fn``'s output on each device, all of one structure.
    """
    first, structure = tree.flatten(results[0], leaf=_is_result)
    flat_results = [first] + [
        structure.flatten_up_to(result) for result in results[1:]
    ]
    if out_specs is None:
        out_specs = DeviceMapping(mesh, (Unknown(),) * mesh.ndim)
    if isinstance(out_specs, DeviceMapping):
        mappings = [out_specs] * len(first)
    elif len(out_specs) == len(first):
        mappings = list(out_specs)
    else:
        raise ShardingError(
            f"out_specs names {len(out_specs)} mappings for "
            f"{len(first)} tensor results."
        )
    joined = []
    for index, mapping in enumerate(mappings):
        shards = [
            value
            if isinstance(value, Tensor)
            else Tensor.from_graph_value(value)
            for value in (values[index] for values in flat_results)
        ]
        devices = tuple(shard.device for shard in shards)
        if devices != mapping.mesh.devices:
            # A result can live on other devices than its mesh, for example
            # a count the kernel writes to host memory.
            mapping = DeviceMapping(
                DeviceMesh(devices, mesh.mesh_shape, mesh.axis_names),
                mapping.placements,
            )
        joined.append(_from_shards(shards, mapping))
    return tree.unflatten(structure, joined)


def _from_shards(shards: Sequence[Tensor], mapping: DeviceMapping) -> Tensor:
    """Returns the distributed tensor made of ``shards``, moving no data."""
    if all(shard.real for shard in shards):
        return Tensor._from_shards(
            tuple(shard.driver_tensor for shard in shards), mapping
        )
    return Tensor.from_shard_values(
        [_graph_value(shard) for shard in shards], mapping
    )


def _graph_value(tensor: Tensor) -> TensorValue | BufferValue:
    """Returns ``tensor``'s value in the current graph.

    A buffer, such as a KV cache's blocks, stays a buffer so that ops can
    write to it.
    """
    if isinstance(tensor._backing_value, BufferValue):
        return BufferValue(tensor)
    return TensorValue(tensor)
