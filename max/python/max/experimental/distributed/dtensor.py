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

"""Distributed tensor — a tensor whose data is spread across a device mesh.

A `DTensor` wraps per-device `Tensor` shards together with a `DeviceMesh`
and a sequence of `Placement` descriptors (one per mesh axis) that describe
how the global logical tensor maps to local shards.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NoReturn

from max import driver, graph
from max.driver import Device, DLPackArray
from max.dtype import DType
from max.experimental.tensor import RealizationState, Tensor, defaults
from max.graph import Dim
from max.graph.ops.constant import NestedArray, Number

from .device_mesh import DeviceMesh
from .placement import Partial, Placement, Replicated, Sharded
from .shape import global_shape


class DTensor(Tensor):
    """A tensor distributed across devices according to a mesh and placements.

    `DTensor` is the central abstraction for multi-device tensor parallelism.
    It pairs per-device `Tensor` shards with a `DeviceMesh` and `Placement`
    descriptors that record *how* the global tensor maps to local memory. This
    lets downstream code reason about distribution without inspecting raw shards.
    """

    _global_shape: graph.Shape
    _local_shards: tuple[Tensor, ...]
    _mesh: DeviceMesh
    _placements: tuple[Placement, ...]

    def __init__(
        self,
        data: DLPackArray | NestedArray | Number | None = None,
        *,
        dtype: DType | None = None,
        device: Device | None = None,
        storage: driver.Buffer | None = None,
        state: RealizationState | None = None,
    ) -> None:
        shard0 = Tensor(
            data, dtype=dtype, device=device, storage=storage, state=state
        )
        self._local_shards = (shard0,)
        self._mesh = DeviceMesh((shard0.device,), (1,), ("dim0",))
        self._placements = (Replicated(),)
        self._global_shape = shard0.shape

    @property
    def shape(self) -> graph.Shape:
        """The global (logical) shape of the distributed tensor."""
        return self._global_shape

    @property
    def dtype(self) -> DType:
        """The element dtype, consistent across all shards."""
        return self._local_shards[0].dtype

    @property
    def device(self) -> Device:
        """Gets the device where the tensor is stored."""
        if len(self._local_shards) == 1:
            return self._local_shards[0].device
        raise ValueError(
            "DTensor does not have a single device. Use the mesh to get the "
            "device of each shard."
        )

    @property
    def mesh(self) -> DeviceMesh:
        """The device mesh this tensor is distributed over."""
        return self._mesh

    @property
    def placements(self) -> tuple[Placement, ...]:
        """Per-axis placement descriptors (one per mesh dimension)."""
        return self._placements

    @property
    def local_shards(self) -> tuple[Tensor, ...]:
        """The per-device tensor shards in mesh row-major order."""
        return self._local_shards

    @property
    def real(self) -> bool:
        """Returns `True` if all shards are realized."""
        return all(shard.real for shard in self._local_shards)

    def _raise_no_op_dispatch(self, op_name: str) -> NoReturn:
        raise NotImplementedError(
            f"DTensor does not support '{op_name}' yet. "
            "Op dispatch for distributed tensors is not implemented. "
            "Use DTensor.finalize() to get a local Tensor first, or "
            "operate on individual shards via DTensor.local_shards."
        )

    def __tensorvalue__(self) -> graph.TensorValue:
        self._raise_no_op_dispatch("__tensorvalue__")

    def __buffervalue__(self) -> graph.BufferValue:
        self._raise_no_op_dispatch("__buffervalue__")

    @staticmethod
    def distributed_ones(
        shape: Sequence[int],
        mesh: DeviceMesh,
        placements: Sequence[Placement],
        *,
        dtype: DType | None = None,
    ) -> DTensor:
        """Creates a distributed tensor filled with ones.

        Allocates one shard per device. For `Sharded(axis)` placements, each
        shard covers its slice of the global shape along that axis. For
        `Replicated` placements, every shard holds the full extent.

        Args:
            shape: The global (logical) shape of the tensor.
            mesh: The device mesh.
            placements: One `Placement` per mesh axis.
            dtype: Element dtype. If not specified, defaults to
                :obj:`DType.float32` for CPU devices and :obj:`DType.bfloat16`
                for accelerator devices.

        Returns:
            DTensor: A distributed tensor filled with ones with the specified
            shape, dtype, mesh, and placements.
        """
        return DTensor._create_uniform(
            shape, mesh, placements, value=1, dtype=dtype
        )

    @staticmethod
    def distributed_zeros(
        shape: Sequence[int],
        mesh: DeviceMesh,
        placements: Sequence[Placement],
        *,
        dtype: DType | None = None,
    ) -> DTensor:
        """Creates a distributed tensor filled with zeros.

        Allocates one shard per device. For `Sharded(axis)` placements, each
        shard covers its slice of the global shape along that axis. For
        `Replicated` placements, every shard holds the full extent.

        Args:
            shape: The global (logical) shape of the tensor.
            mesh: The device mesh.
            placements: One `Placement` per mesh axis.
            dtype: Element dtype. If not specified, defaults to
                :obj:`DType.float32` for CPU devices and :obj:`DType.bfloat16`
                for accelerator devices.

        Returns:
            DTensor: A distributed tensor filled with zeros with the specified
            shape, dtype, mesh, and placements.
        """
        return DTensor._create_uniform(
            shape, mesh, placements, value=0, dtype=dtype
        )

    @staticmethod
    def distributed_full(
        shape: Sequence[int],
        value: int | float,
        mesh: DeviceMesh,
        placements: Sequence[Placement],
        *,
        dtype: DType | None = None,
    ) -> DTensor:
        """Creates a distributed tensor filled with a scalar value.

        Args:
            shape: The global (logical) shape of the tensor.
            value: The fill value.
            mesh: The device mesh.
            placements: One `Placement` per mesh axis.
            dtype: Element dtype. If not specified, defaults to
                :obj:`DType.float32` for CPU devices and :obj:`DType.bfloat16`
                for accelerator devices.

        Returns:
            DTensor: A distributed tensor filled with the specified value with
            the specified shape, dtype, mesh, and placements.
        """
        return DTensor._create_uniform(
            shape, mesh, placements, value=value, dtype=dtype
        )

    def redistribute(
        self,
        *,
        mesh: DeviceMesh | None = None,
        placements: Sequence[Placement] | None = None,
    ) -> DTensor:
        """Re-distributes this tensor to a new mesh and/or placements.

        Requires collective communication (allgather, allreduce, etc.) which
        is not yet implemented.

        Raises:
            NotImplementedError: Always, until collective ops are available.
        """
        raise NotImplementedError(
            "DTensor.redistribute() requires collective communication ops "
            "(allgather, allreduce, reduce_scatter) which are not yet "
            "implemented."
        )

    def finalize(self) -> Tensor:
        """Materializes the full tensor on the first device in the mesh.

        For `Replicated` tensors, returns the first shard directly. For
        `Sharded` or `Partial` tensors, requires collective communication
        which is not yet implemented.

        Raises:
            NotImplementedError: When the tensor has non-replicated placements
                that would require collective communication.
        """
        if all(isinstance(p, Replicated) for p in self._placements):
            return self._local_shards[0]
        raise NotImplementedError(
            "DTensor.finalize() for non-replicated placements requires "
            "collective communication ops which are not yet implemented. "
            f"Current placements: {list(self._placements)}"
        )

    @staticmethod
    def _create_uniform(
        shape: Sequence[int],
        mesh: DeviceMesh,
        placements: Sequence[Placement],
        *,
        value: int | float,
        dtype: DType | None = None,
    ) -> DTensor:
        """Creates a DTensor where every element has the same scalar value.

        Computes the local shard shape from the global shape and placements,
        then creates one `Tensor` per device.
        """
        if len(placements) != mesh.ndim:
            raise ValueError(
                f"Need one placement per mesh axis. Mesh has {mesh.ndim} "
                f"axes, got {len(placements)} placements."
            )

        for placement in placements:
            if isinstance(placement, Partial):
                raise ValueError(
                    "Cannot use Partial placement with factory methods "
                    "(ones/zeros/full). Partial placement means each device "
                    "holds a partial result — filling every shard with the "
                    "same value would give an incorrect global result after "
                    "reduction."
                )

        resolved_dtype, _ = defaults(dtype, mesh.devices[0])
        local_shape = list(shape)

        for mesh_axis, placement in enumerate(placements):
            if isinstance(placement, Sharded):
                axis = placement.axis
                if axis < 0 or axis >= len(shape):
                    raise ValueError(
                        f"Sharded(axis={axis}) is out of range for tensor "
                        f"with {len(shape)} dimensions."
                    )
                mesh_axis_size = mesh.mesh_shape[mesh_axis]
                if local_shape[axis] % mesh_axis_size != 0:
                    raise ValueError(
                        f"Axis {axis} (size {local_shape[axis]}) is not "
                        f"evenly divisible by mesh axis "
                        f"{mesh.axis_names[mesh_axis]!r} "
                        f"(size {mesh_axis_size})."
                    )
                local_shape[axis] //= mesh_axis_size

        global_dims = tuple(Dim(d) for d in shape)
        shards: list[Tensor] = []
        for device in mesh.devices:
            shard = Tensor.full(
                local_shape,
                value=value,
                dtype=resolved_dtype,
                device=device,
            )
            shards.append(shard)

        return DTensor._from_local_shards(
            shards, mesh, placements, shape=global_dims
        )

    @classmethod
    def _from_local_shards(
        cls,
        shards: Sequence[Tensor],
        mesh: DeviceMesh,
        placements: Sequence[Placement],
        shape: graph.ShapeLike | None = None,
    ) -> DTensor:
        """Wraps already-sharded Tensors into a single DTensor."""
        # This is a private method, so assume that the arguments are valid.
        instance = object.__new__(cls)
        instance._local_shards = tuple(shards)
        instance._mesh = mesh
        instance._placements = tuple(placements)
        instance._global_shape = (
            graph.Shape(shape)
            if shape
            else global_shape(
                [Dim(d) for d in shards[0].shape], mesh, placements
            )
        )
        return instance

    def __repr__(self) -> str:
        placement_str = ", ".join(repr(p) for p in self._placements)
        shape_str = ", ".join(str(d) for d in self._global_shape)
        return (
            f"DTensor(shape=[{shape_str}], dtype={self.dtype}, "
            f"mesh={self._mesh}, placements=[{placement_str}])"
        )
