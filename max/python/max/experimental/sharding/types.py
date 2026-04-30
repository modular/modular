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

"""Distributed type descriptors for graph compilation.

* :class:`DistributedTensorType`: symbolic type for a tensor distributed
  across a device mesh. Analogous to :class:`~max.graph.TensorType`.
* :class:`DistributedBufferType`: symbolic type for a mutable buffer
  distributed across a device mesh. Analogous to
  :class:`~max.graph.BufferType`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from max.dtype import DType
from max.graph import (
    BufferType,
    DeviceRef,
    Shape,
    StaticDim,
    SymbolicDim,
    TensorType,
)
from max.graph.shape import ShapeLike

from .mappings import DeviceMapping, NamedMapping
from .mesh import DeviceMesh
from .placements import Placement, Sharded

T = TypeVar("T", TensorType, BufferType)


class DistributedType(Generic[T], ABC):
    """Shared state and shard-shape logic for distributed type descriptors.

    Not intended for direct use. See :class:`DistributedTensorType` and
    :class:`DistributedBufferType`.
    """

    dtype: DType
    shape: Shape
    mesh: DeviceMesh
    placements: tuple[Placement, ...]

    def __init__(
        self,
        dtype: DType,
        shape: ShapeLike,
        mesh: DeviceMesh,
        placements: Sequence[Placement],
    ) -> None:
        self.dtype = dtype
        self.shape = Shape(shape)
        self.mesh = mesh
        self.placements = tuple(placements)

        if len(self.placements) != mesh.ndim:
            raise ValueError(
                f"Need one placement per mesh axis. Mesh has {mesh.ndim} "
                f"axes, got {len(self.placements)} placements."
            )
        for placement in self.placements:
            if isinstance(placement, Sharded):
                axis = placement.axis
                if axis < 0 or axis >= len(self.shape):
                    raise ValueError(
                        f"Sharded(axis={axis}) is out of range for tensor "
                        f"with rank {len(self.shape)}."
                    )

    def _local_shard_shape(self) -> Shape:
        """Computes the local shard shape from the global shape.

        Static dims are divided evenly. Symbolic dims produce a new
        :class:`~max.graph.SymbolicDim` named ``"{original}_{axis_name}"``.
        Algebraic dims use ``dim // mesh_axis_size``.
        """
        dims = list(self.shape)
        for mesh_axis, placement in enumerate(self.placements):
            if not isinstance(placement, Sharded):
                continue
            tensor_axis = placement.axis
            dim = dims[tensor_axis]
            mesh_axis_size = self.mesh.mesh_shape[mesh_axis]
            axis_name = self.mesh.axis_names[mesh_axis]

            if isinstance(dim, StaticDim):
                size = int(dim)
                if size % mesh_axis_size != 0:
                    raise ValueError(
                        f"Static dimension {size} at axis {tensor_axis} is "
                        f"not evenly divisible by mesh axis "
                        f"{axis_name!r} (size {mesh_axis_size})."
                    )
                dims[tensor_axis] = StaticDim(size // mesh_axis_size)
            elif isinstance(dim, SymbolicDim):
                dims[tensor_axis] = SymbolicDim(f"{dim.name}_{axis_name}")
            else:
                dims[tensor_axis] = dim // mesh_axis_size
        return Shape(dims)

    @property
    def rank(self) -> int:
        """The rank (number of dimensions) of the global tensor."""
        return len(self.shape)

    @property
    @abstractmethod
    def local_types(self) -> list[T]:
        """The per-device types in mesh order."""
        ...


class DistributedTensorType(DistributedType[TensorType]):
    """A symbolic type for a tensor distributed across a device mesh.

    Analogous to :class:`~max.graph.TensorType` for single-device tensors.
    Derives per-device :class:`~max.graph.TensorType` objects via
    :attr:`local_types`.

    When a :class:`~max.graph.SymbolicDim` is sharded along a mesh axis, the
    local shard dimension becomes a new ``SymbolicDim`` named
    ``"{original}_{axis_name}"``. This keeps symbolic names short and
    debuggable while ensuring that sharding the same global dim on different
    axes produces distinct names.

    Args:
        dtype: Element data type.
        shape: Global (logical) shape.
        mesh: The device mesh.
        placements: One :class:`Placement` per mesh axis.
    """

    @property
    def local_types(self) -> list[TensorType]:
        """The per-device :class:`~max.graph.TensorType` objects in mesh order."""
        local_shape = self._local_shard_shape()
        return [
            TensorType(self.dtype, local_shape, DeviceRef.from_device(device))
            for device in self.mesh.devices
        ]

    def __repr__(self) -> str:
        placement_str = ", ".join(repr(p) for p in self.placements)
        shape_str = ", ".join(str(d) for d in self.shape)
        return (
            f"DistributedTensorType(dtype={self.dtype}, shape=[{shape_str}], "
            f"mesh={self.mesh}, placements=[{placement_str}])"
        )


class DistributedBufferType(DistributedType[BufferType]):
    """A symbolic type for a mutable buffer distributed across a device mesh.

    Analogous to :class:`~max.graph.BufferType` for single-device buffers.
    Derives per-device :class:`~max.graph.BufferType` objects via
    :attr:`local_types`.

    Args:
        dtype: Element data type.
        shape: Global (logical) shape.
        mesh: The device mesh.
        placements: One :class:`Placement` per mesh axis.
    """

    @property
    def local_types(self) -> list[BufferType]:
        """The per-device :class:`~max.graph.BufferType` objects in mesh order."""
        local_shape = self._local_shard_shape()
        return [
            BufferType(self.dtype, local_shape, DeviceRef.from_device(device))
            for device in self.mesh.devices
        ]

    def __repr__(self) -> str:
        placement_str = ", ".join(repr(p) for p in self.placements)
        shape_str = ", ".join(str(d) for d in self.shape)
        return (
            f"DistributedBufferType(dtype={self.dtype}, shape=[{shape_str}], "
            f"mesh={self.mesh}, placements=[{placement_str}])"
        )


# ═════════════════════════════════════════════════════════════════════════
#  TensorLayout — metadata snapshot for rule evaluation
# ═════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class TensorLayout(DeviceMapping):
    """Metadata snapshot of a distributed tensor for rule evaluation.

    Bundles the tensor's dtype, shape, and distribution mapping. The mapping
    stays abstract (:class:`DeviceMapping`) so rules work with any concrete
    mapping type, such as :class:`PlacementMapping` or :class:`NamedMapping`.

    The shape is a :class:`~max.graph.Shape` (``list[Dim]``), supporting both
    static and symbolic dimensions for graph compilation compatibility.

    This class implements DeviceMapping, so sharding rules can
    return input TensorLayouts directly.
    """

    dtype: DType
    """The element data type of the tensor."""

    shape: Shape
    """The global shape of the tensor."""

    mapping: DeviceMapping
    """The distribution mapping over the device mesh."""

    def __init__(
        self, dtype: DType, shape: ShapeLike, mapping: DeviceMapping
    ) -> None:
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "shape", Shape(shape))
        object.__setattr__(self, "mapping", mapping)

    @property
    def rank(self) -> int:
        """The number of dimensions."""
        return len(self.shape)

    @property
    def mesh(self) -> DeviceMesh:
        """The device mesh derived from the mapping."""
        return self.mapping.mesh

    @property
    def is_fully_resolved(self) -> bool:
        """Whether this spec can be used in eager dispatch.

        Returns ``False`` if the spec contains compiler-only annotations
        (e.g. priorities) that cannot be resolved without a compiler.
        """
        return self.mapping.is_fully_resolved

    @property
    def is_fully_replicated(self) -> bool:
        """Whether every device holds a complete copy of the tensor.

        Returns ``True`` if no dimension is sharded and there are no
        pending reductions.
        """
        return self.mapping.is_fully_replicated

    def to_placements(self) -> tuple[Placement, ...]:
        """Converts to mesh-axis-indexed placements for eager dispatch."""
        return self.mapping.to_placements()

    def to_named_sharding(self, tensor_rank: int) -> NamedMapping:
        """Converts to tensor-dim-indexed spec for compiler lowering."""
        return self.mapping.to_named_sharding(tensor_rank)

    def __repr__(self) -> str:
        shape_str = ", ".join(str(d) for d in self.shape)
        return (
            f"TensorLayout(dtype={self.dtype}, shape=[{shape_str}], "
            f"mapping={self.mapping})"
        )
