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

"""Placement types describing how tensor data is distributed across one mesh axis."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from max.graph.dim import AlgebraicDim, Dim, DimLike, StaticDim, SymbolicDim
from max.graph.shape import Shape

from .mesh import DeviceMesh
from .per_shard_dim import (
    is_per_shard_dim,
    local_shape_at,
    make_per_shard_dim,
)


class Collective(Enum):
    """The collectives the cost model understands.

    A :meth:`Placement.transition_to` return value must be one of these.
    Custom :class:`Placement` subclasses that return anything else are
    reported as infeasible by the cost model.
    """

    NOOP = "noop"
    LOCAL_SLICE = "local_slice"
    KEEP_ONE_COPY = "keep_one_copy"
    ALLGATHER = "allgather"
    ALL_TO_ALL = "all_to_all"
    ALLREDUCE = "allreduce"
    REDUCE_SCATTER = "reduce_scatter"
    INFEASIBLE = "infeasible"


class ShardingError(RuntimeError):
    """Raised when a sharding rule cannot place a call.

    Covers a rule that declines every placement, operands on different
    meshes, a malformed rule declaration, a target that would split a
    size-1 axis, a transfer that would change an :class:`Unknown` placement,
    and an op the active :func:`auto_reshard` policy refuses to move.
    """


Transition = tuple[type["Placement"], type["Placement"]]
"""A placement change on one mesh axis, as ``(from, to)`` placement types.

``(Partial, Replicated)`` is an allreduce, ``(Sharded, Replicated)`` an
allgather, ``(Replicated, Sharded)`` a local slice, ``(Sharded, Sharded)``
a re-shard along another tensor axis, and ``(Partial, Sharded)`` a
reduce-scatter. Moving an input from another mesh is not a transition.
"""


def even_shard_sizes(global_size: int, num_shards: int) -> list[int]:
    """Returns the sizes of ``num_shards`` balanced pieces of ``global_size``.

    The sizes differ by at most 1, and the larger ones come first, as the
    reduce-scatter kernel orders them, so slicing a copy gives the same
    pieces as a reduce-scatter. For example, ``even_shard_sizes(10, 4)`` is
    ``[3, 3, 2, 2]``.

    A ``global_size`` below ``num_shards`` leaves the trailing pieces at 0,
    as ``even_shard_sizes(2, 4)`` is ``[1, 1, 0, 0]``. Check the axis size
    before assuming every piece holds data; a tensor cannot be created with
    an empty shard.

    Args:
        global_size: The size to split. A ``-1`` reshape wildcard stays
            ``-1`` in every piece.
        num_shards: The number of pieces.

    Returns:
        The size of each piece, in order.
    """
    if global_size == -1:
        return [-1] * num_shards
    base, remainder = divmod(global_size, num_shards)
    return [base + 1 if i < remainder else base for i in range(num_shards)]


class Placement(ABC):
    """Abstract base for all placement types.

    Each placement describes what a single mesh axis does to a tensor:
    ``Replicated`` (full copy), ``Sharded`` (split along a tensor dim),
    ``Partial`` (partial result needing reduction), or ``Unknown`` (per-device
    values with no known relation).
    """

    @abstractmethod
    def __repr__(self) -> str: ...

    def localized_axis(self) -> int | None:
        """Tensor axis this placement localizes, or ``None`` if it doesn't."""
        return None

    def transition_to(self, other: Placement) -> Collective:
        """Returns the :class:`Collective` for the move ``self -> other``.

        The base returns :attr:`Collective.NOOP` when the two placements are
        equal and :attr:`Collective.INFEASIBLE` otherwise, so a placement
        subclass that does not override this can never be moved by the picker.

        Args:
            other: The placement this one moves to.

        Returns:
            The :class:`Collective` that performs the move.
        """
        return Collective.NOOP if self == other else Collective.INFEASIBLE

    def local_dim(
        self,
        parent: DimLike,
        mesh: DeviceMesh,
        mesh_axis: int | Sequence[int],
        *,
        allow_symbolic_mint: bool = True,
    ) -> Dim:
        """Returns the per-shard local cells of ``parent`` along mesh axes.

        The default returns ``parent`` unchanged; ``Sharded`` overrides to
        split. A wrapper ``parent`` is passed through verbatim.

        Args:
            parent: The global dim to localize.
            mesh: The device mesh.
            mesh_axis: The mesh axis that splits ``parent``, or every mesh
                axis that splits it.
            allow_symbolic_mint: When ``False``, a bare :class:`SymbolicDim`
                raises :class:`ShardingError` instead of minting one symbol per
                shard. MAX callers use the default.
        """
        return Dim(parent)


@dataclass(frozen=True)
class Replicated(Placement):
    """Every device on this mesh axis holds the same copy of the data."""

    def __repr__(self) -> str:
        return "Replicated()"

    def transition_to(self, other: Placement) -> Collective:
        """Returns the :class:`Collective` for the move to another placement.

        :class:`Sharded` is a free local split, and :class:`Partial` keeps one
        copy and zeros the others.

        Args:
            other: The placement this one moves to.

        Returns:
            The :class:`Collective` that performs the move.
        """
        if self == other:
            return Collective.NOOP
        if isinstance(other, Sharded):
            return Collective.LOCAL_SLICE
        if isinstance(other, Partial):
            return Collective.KEEP_ONE_COPY
        return super().transition_to(other)


@dataclass(frozen=True)
class Unknown(Placement):
    """Per-device values on this mesh axis with no known relation.

    The devices may hold different values, or happen to hold equal ones; only
    a claim with :meth:`~max.experimental.tensor.Tensor.rebind_mapping` or a
    collective establishes a relation. An op takes an ``Unknown`` operand
    only through a sharding rule row that names it.
    """

    def __repr__(self) -> str:
        return "Unknown()"


@dataclass(frozen=True)
class Sharded(Placement):
    """Every device on this mesh axis holds a slice along ``axis``.

    Args:
        axis: The tensor axis along which data is split.
    """

    axis: int

    def __repr__(self) -> str:
        return f"Sharded(axis={self.axis})"

    def localized_axis(self) -> int | None:
        """Returns the tensor axis this Sharded localizes."""
        return self.axis

    def transition_to(self, other: Placement) -> Collective:
        """Returns the :class:`Collective` for the move to another placement.

        :class:`Replicated` takes an allgather. A different :class:`Sharded`
        axis returns :attr:`Collective.ALL_TO_ALL`, a cost and not a kernel:
        :func:`transfer_to` allgathers and then splits locally, so the whole
        tensor crosses the mesh. :class:`Partial` has no redistribution and
        reports :attr:`Collective.INFEASIBLE`.

        Args:
            other: The placement this one moves to.

        Returns:
            The :class:`Collective` that performs the move.
        """
        if self == other:
            return Collective.NOOP
        if isinstance(other, Replicated):
            return Collective.ALLGATHER
        if isinstance(other, Sharded):
            return Collective.ALL_TO_ALL
        return super().transition_to(other)

    def local_dim(
        self,
        parent: DimLike,
        mesh: DeviceMesh,
        mesh_axis: int | Sequence[int],
        *,
        allow_symbolic_mint: bool = True,
    ) -> Dim:
        """Splits ``parent`` along ``mesh_axis`` into per-shard cells.

        ``parent`` is split once into ``mesh.axis_size(mesh_axis)`` balanced
        pieces, and the result holds one cell per device of the mesh, in
        row-major device order. A mesh axis that does not span the whole mesh
        gives the same piece to every device that shares its coordinate along
        the splitting axes, and ``make_per_shard_dim`` collapses equal cells to
        a plain dim. For example, 10 rows split over both axes of a 2x2 mesh
        give devices 0 to 3 sizes 3, 3, 2, and 2.

        ``StaticDim`` parents use uneven divmod; ``SymbolicDim`` parents
        mint fresh per-shard cells; ``AlgebraicDim`` parents raise.
        Wrapper parents pass through verbatim. When
        ``allow_symbolic_mint`` is ``False``, a bare ``SymbolicDim``
        raises instead of minting.

        Args:
            parent: The global dim to localize.
            mesh: The device mesh.
            mesh_axis: The mesh axis that splits ``parent``, or every mesh
                axis that splits it.
            allow_symbolic_mint: When ``False``, a bare ``SymbolicDim``
                raises instead of minting per-shard cells. Defaults to
                ``True``.

        Raises:
            ShardingError: If ``parent`` is an ``AlgebraicDim``, or a bare
                ``SymbolicDim`` while ``allow_symbolic_mint`` is ``False``.
        """
        if is_per_shard_dim(parent):
            assert isinstance(parent, Dim)
            return parent
        parent_dim = Dim(parent)
        axes = sorted(
            {mesh_axis} if isinstance(mesh_axis, int) else set(mesh_axis)
        )
        if isinstance(parent_dim, StaticDim):
            sizes = even_shard_sizes(int(parent_dim), mesh.axis_size(axes))
            cells: tuple[Dim, ...] = tuple(
                StaticDim(sizes[mesh.device_coord(device_idx, axes)])
                for device_idx in range(mesh.num_devices)
            )
        elif isinstance(parent_dim, SymbolicDim):
            if not allow_symbolic_mint:
                raise ShardingError(
                    f"Sharded.local_dim cannot mint per-shard cells from "
                    f"bare symbolic {parent_dim!r}. Thread a per-shard "
                    "wrapper (``tensor.shape[k]`` from a sharded input) "
                    "into the target shape, or pin the axis Replicated "
                    "upstream."
                )
            cells = tuple(
                SymbolicDim(
                    parent_dim.name
                    + "".join(
                        f"_{mesh.axis_names[axis]}_"
                        f"{mesh.device_coord(device_idx, axis)}"
                        for axis in axes
                    )
                )
                for device_idx in range(mesh.num_devices)
            )
        elif isinstance(parent_dim, AlgebraicDim):
            raise ShardingError(
                f"Sharded.local_dim cannot split algebraic {parent_dim!r}: "
                "shard the underlying operand instead, or thread a "
                "per-shard wrapper (``tensor.shape[k]`` from a sharded "
                "input) into the target shape."
            )
        else:
            cells = (parent_dim // mesh.axis_size(axes),) * mesh.num_devices
        return make_per_shard_dim(cells)


@dataclass(frozen=True)
class Partial(Placement):
    """Every device holds a part of the value, and the value is their sum."""

    def __repr__(self) -> str:
        return "Partial()"

    def transition_to(self, other: Placement) -> Collective:
        """Returns the :class:`Collective` that resolves the partial sum.

        :class:`Replicated` takes the allreduce, and :class:`Sharded` takes
        the reduce-scatter.

        Args:
            other: The placement this one moves to.

        Returns:
            The :class:`Collective` that performs the move.
        """
        if self == other:
            return Collective.NOOP
        if isinstance(other, Replicated):
            return Collective.ALLREDUCE
        if isinstance(other, Sharded):
            return Collective.REDUCE_SCATTER
        return super().transition_to(other)


def local_shard_shape_from_global(
    global_shape: Sequence[DimLike],
    mesh: DeviceMesh,
    placements: Sequence[Placement],
) -> list[Shape]:
    """Returns one :class:`Shape` per device, in row-major mesh order.

    Calls :meth:`Placement.local_dim` once per tensor axis, with every mesh
    axis that splits it, then takes each device's size. An axis whose static
    extent is smaller than the mesh axes splitting it yields shapes with a 0
    extent; creating a tensor with such a mapping raises
    :class:`ShardingError`, so check the extent first.

    Args:
        global_shape: The global shape to split, one entry per tensor axis.
        mesh: The device mesh.
        placements: One placement per mesh axis.

    Returns:
        One local shape per device, in mesh row-major order.

    Raises:
        ValueError: If ``placements`` length differs from ``mesh.ndim`` or
            any placement localizes an axis out of range for ``global_shape``.
    """
    global_list: list[Dim] = [Dim(d) for d in global_shape]
    rank = len(global_list)
    if len(placements) != mesh.ndim:
        raise ValueError(
            f"Need one placement per mesh axis. Mesh has {mesh.ndim} axes, "
            f"got {len(placements)} placements."
        )
    norm_axes: list[int | None] = []
    for p in placements:
        ax = p.localized_axis()
        if ax is None:
            norm_axes.append(None)
            continue
        if ax < 0:
            ax += rank
        if ax < 0 or ax >= rank:
            raise ValueError(
                f"{p!r} localizes axis {p.localized_axis()}, out of range "
                f"for tensor with rank {rank}."
            )
        norm_axes.append(ax)
    wrapped: list[Dim] = []
    for tensor_axis, dim in enumerate(global_list):
        mesh_axes = [a for a in range(mesh.ndim) if norm_axes[a] == tensor_axis]
        wrapped.append(
            placements[mesh_axes[0]].local_dim(dim, mesh, mesh_axes)
            if mesh_axes
            else dim
        )
    return [local_shape_at(wrapped, r) for r in range(mesh.num_devices)]


ALL_TRANSITIONS: frozenset[Transition] = frozenset(
    {
        (Replicated, Sharded),
        (Sharded, Replicated),
        (Sharded, Sharded),
        (Partial, Replicated),
        (Partial, Sharded),
    }
)
"""Every transition the picker can plan with."""

DEFAULT_TRANSITIONS: frozenset[Transition] = ALL_TRANSITIONS - {
    (Partial, Sharded)
}
"""Every transition except ``(Partial, Sharded)``, so a partial sum resolves
to Replicated by allreduce; :data:`ALL_TRANSITIONS` also allows reduce-scatter.
"""
