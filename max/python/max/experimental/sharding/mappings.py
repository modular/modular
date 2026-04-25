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

"""Sharding specifications: DeviceMapping, PlacementMapping, NamedMapping.

Two concrete implementations of :class:`DeviceMapping`:

* :class:`PlacementMapping` — mesh-axis-indexed (PyTorch DTensor style).
* :class:`NamedMapping` — tensor-dim-indexed (JAX PartitionSpec style).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .mesh import DeviceMesh
from .placements import Partial, Placement, Replicated, Sharded

# ═════════════════════════════════════════════════════════════════════════
#  Conversion error
# ═════════════════════════════════════════════════════════════════════════


class ConversionError(Exception):
    """Raised when a sharding spec conversion would lose information."""


# Type for a single entry in a NamedMapping spec.
# - str: mesh axis name sharding this tensor dim
# - tuple[str, ...]: multiple mesh axes sharding this tensor dim (multi-axis)
# - None: this tensor dim is replicated
SpecEntry = str | tuple[str, ...] | None


# ═════════════════════════════════════════════════════════════════════════
#  Abstract base
# ═════════════════════════════════════════════════════════════════════════


class DeviceMapping(ABC):
    """Abstract base for all sharding specifications.

    A ``DeviceMapping`` pairs a :class:`DeviceMesh` with a description of how
    tensor data is distributed across that mesh.  Two concrete implementations
    exist:

    * :class:`PlacementMapping` — mesh-axis-indexed, for eager per-op dispatch.
    * :class:`NamedMapping` — tensor-dim-indexed, for future full-graph
      sharding search (e.g. a Python-level transform over an op trace).
    """

    @property
    @abstractmethod
    def mesh(self) -> DeviceMesh:
        """The device mesh this sharding is defined over."""
        ...

    @property
    @abstractmethod
    def is_fully_resolved(self) -> bool:
        """Whether this spec can be used in eager dispatch.

        Returns ``False`` if the spec contains compiler-only annotations
        (e.g. priorities) that cannot be resolved without a compiler.
        """
        ...

    @property
    @abstractmethod
    def is_fully_replicated(self) -> bool:
        """Whether every device holds a complete copy of the tensor.

        Returns ``True`` if no dimension is sharded and there are no
        pending reductions.
        """
        ...

    @abstractmethod
    def to_placements(self) -> tuple[Placement, ...]:
        """Converts to mesh-axis-indexed placements for eager dispatch.

        Returns one :class:`Placement` per mesh axis.

        Raises:
            ConversionError: If the spec contains features that cannot be
                represented as placements (e.g. priorities or custom
                placement types without a standard equivalent).
        """
        ...

    @abstractmethod
    def to_named_sharding(self, tensor_rank: int) -> NamedMapping:
        """Converts to tensor-dim-indexed spec for compiler lowering.

        Args:
            tensor_rank: Number of dimensions in the tensor.  Required
                because the spec must have one entry per tensor dim.

        Raises:
            ConversionError: If the spec contains custom placements that
                have no ``NamedMapping`` equivalent.
        """
        ...


# ═════════════════════════════════════════════════════════════════════════
#  PlacementMapping
# ═════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PlacementMapping(DeviceMapping):
    """Mesh-axis-indexed sharding (PyTorch DTensor style).

    Stores one :class:`Placement` per mesh axis.  Each placement describes
    what that mesh axis does to the tensor: ``Shard(dim)``, ``Replicate()``,
    or ``Partial(op)``.

    This is always fully resolved and can be used directly in eager dispatch.

    Args:
        _mesh: The device mesh.
        _placements: One placement per mesh axis.
    """

    _mesh: DeviceMesh
    _placements: tuple[Placement, ...]

    def __post_init__(self) -> None:
        if len(self._placements) != self._mesh.ndim:
            raise ValueError(
                f"Need one placement per mesh axis ({self._mesh.ndim}), "
                f"got {len(self._placements)}."
            )

    @property
    def mesh(self) -> DeviceMesh:
        """The device mesh this sharding is defined over."""
        return self._mesh

    @property
    def placements(self) -> tuple[Placement, ...]:
        """The raw placement tuple (one per mesh axis)."""
        return self._placements

    @property
    def is_fully_resolved(self) -> bool:
        """Returns True; placement mappings are always fully concrete."""
        return True  # Placements are always concrete.

    @property
    def is_fully_replicated(self) -> bool:
        """Returns True if every mesh axis placement is Replicated."""
        return all(isinstance(p, Replicated) for p in self._placements)

    def to_placements(self) -> tuple[Placement, ...]:
        """Returns the stored placement tuple directly."""
        return self._placements

    def to_named_sharding(self, tensor_rank: int) -> NamedMapping:
        """Converts to a tensor-dim-indexed NamedMapping.

        Raises:
            ConversionError: If any placement is not one of the standard
                types (``Replicated``, ``Sharded``, ``Partial``).
        """
        spec: list[str | tuple[str, ...] | None] = [None] * tensor_rank
        unreduced: set[str] = set()

        for ax, p in enumerate(self._placements):
            axis_name = self._mesh.axis_names[ax]

            if isinstance(p, Replicated):
                continue

            elif isinstance(p, Sharded):
                if p.axis < 0 or p.axis >= tensor_rank:
                    raise ConversionError(
                        f"Sharded(axis={p.axis}) is out of range for "
                        f"tensor with rank {tensor_rank}."
                    )
                existing = spec[p.axis]
                if existing is None:
                    spec[p.axis] = axis_name
                elif isinstance(existing, str):
                    # Multi-axis sharding on the same tensor dim.
                    spec[p.axis] = (existing, axis_name)
                else:
                    assert isinstance(existing, tuple)
                    spec[p.axis] = (*existing, axis_name)

            elif isinstance(p, Partial):
                unreduced.add(axis_name)

            else:
                raise ConversionError(
                    f"Placement {p!r} on mesh axis {axis_name!r} has no "
                    f"NamedMapping equivalent. Use the compiler path "
                    f"directly or redistribute to standard placements first."
                )

        return NamedMapping(
            _mesh=self._mesh,
            _spec=tuple(spec),
            _unreduced=frozenset(unreduced),
        )

    def __repr__(self) -> str:
        placement_str = ", ".join(repr(p) for p in self._placements)
        return f"PlacementMapping({self._mesh}, [{placement_str}])"


# ═════════════════════════════════════════════════════════════════════════
#  NamedMapping
# ═════════════════════════════════════════════════════════════════════════
#
# Why both PlacementMapping and NamedMapping?
#
# Currently we focus on eager op-by-op dispatch via PlacementMapping
# (PyTorch DTensor style).  NamedMapping (JAX PartitionSpec style) exists
# for two reasons:
#
#   1. Some users prefer JAX-style named-axis annotations, and for common
#      cases there is a 1:1 correspondence between the two styles.
#
#   2. If we later lower sharding annotations to a compiler pass like
#      Shardy (an MLIR dialect used in XLA), placement-style types are
#      too liberal.  Shardy-like systems that solve sharding to a fixed
#      point need the stricter constraints that NamedMapping provides.


@dataclass(frozen=True)
class NamedMapping(DeviceMapping):
    """Tensor-dimension-indexed sharding (JAX PartitionSpec style).

    Each entry in ``_spec`` corresponds to a tensor dimension and names the
    mesh axis (or axes) that shard it:

    * ``"dp"`` — shard this tensor dim across mesh axis ``"dp"``.
    * ``("dp", "tp")`` — shard across both axes (multi-axis).
    * ``None`` — this tensor dim is replicated.
    Additionally:

    * ``_unreduced`` names mesh axes with pending reductions (analogous to
      ``Partial`` in the placement world).  Contracting a sharded dimension
      produces an unreduced result that needs a collective reduction.
    * ``_priorities`` assigns per-dimension propagation priority for the
      compiler (e.g. batch parallelism at priority 0, tensor parallelism
      at priority 1).  Compiler-only — cannot be used in eager mode.
    * ``_memory_kind`` specifies the memory tier for this tensor's shards
      (e.g. ``"device"``, ``"pinned_host"``).  Mirrors JAX's
      ``NamedMapping.memory_kind``.

    Args:
        _mesh: The device mesh.
        _spec: One entry per tensor dimension.
        _unreduced: Mesh axes with pending reductions.
        _priorities: Per-dimension propagation priority (compiler-only).
        _memory_kind: Memory tier for shard placement (e.g. ``"device"``).
    """

    _mesh: DeviceMesh
    _spec: tuple[SpecEntry, ...] = ()
    _unreduced: frozenset[str] = frozenset()
    _priorities: tuple[int | None, ...] = ()
    _memory_kind: str | None = None
    _original_spec: tuple[SpecEntry, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "_original_spec", self._spec)

        # Resolve spec entries that reference axis names not present in the
        # mesh.  For example, a model defined with ("dp", "tp") placed on a
        # TP-only mesh will have "dp" silently replaced by None (Replicated).
        # This enables reuse of the same NamedMapping across different meshes.
        mesh_axes = set(self._mesh.axis_names)
        needs_resolve = False
        for entry in self._spec:
            if entry is None:
                continue
            axes = (entry,) if isinstance(entry, str) else entry
            for a in axes:
                if a not in mesh_axes:
                    needs_resolve = True
                    break
        if not needs_resolve and self._unreduced - mesh_axes:
            needs_resolve = True
        if needs_resolve:
            resolved = self._resolve(self._mesh)
            object.__setattr__(self, "_spec", resolved._spec)
            object.__setattr__(self, "_unreduced", resolved._unreduced)

        # Validate priorities length if provided.
        if self._priorities and len(self._priorities) != len(self._spec):
            raise ValueError(
                f"priorities length ({len(self._priorities)}) must match "
                f"spec length ({len(self._spec)})."
            )

        # Check that no mesh axis is used for both sharding and unreduced.
        sharding_axes: set[str] = set()
        for entry in self._spec:
            if entry is None:
                continue
            axes = (entry,) if isinstance(entry, str) else entry
            sharding_axes.update(axes)
        overlap = sharding_axes & self._unreduced
        if overlap:
            raise ValueError(
                f"Mesh axes {overlap} are used for both sharding and "
                f"unreduced. A mesh axis cannot do both simultaneously."
            )

    @property
    def mesh(self) -> DeviceMesh:
        """The device mesh this sharding is defined over."""
        return self._mesh

    @property
    def spec(self) -> tuple[SpecEntry, ...]:
        """The raw spec tuple (one entry per tensor dim)."""
        return self._spec

    @property
    def original_spec(self) -> tuple[SpecEntry, ...]:
        """The caller-supplied spec before mesh resolution."""
        assert self._original_spec is not None
        return self._original_spec

    @property
    def unreduced(self) -> frozenset[str]:
        """Mesh axes with pending reductions."""
        return self._unreduced

    @property
    def priorities(self) -> tuple[int | None, ...]:
        """Per-dimension propagation priorities (compiler-only)."""
        return self._priorities

    @property
    def memory_kind(self) -> str | None:
        """Memory tier for shard placement (e.g. ``"device"``)."""
        return self._memory_kind

    @property
    def is_fully_replicated(self) -> bool:
        """Returns True if no dimension is sharded and there are no pending reductions."""
        has_sharding = any(e is not None for e in self._spec)
        return not has_sharding and not self._unreduced

    @property
    def is_fully_resolved(self) -> bool:
        """Returns True if every dimension has a concrete sharding decision."""
        return not bool(self._priorities)

    def _resolve(self, mesh: DeviceMesh) -> NamedMapping:
        """Binds this spec to a mesh, dropping axes not present in the mesh.

        Axis names in the spec that don't exist in the mesh are treated as
        ``None`` (Replicated) — this enables graceful degradation when a
        model defined for ``("dp", "tp")`` is placed on a TP-only mesh.

        Args:
            mesh: The device mesh to resolve against.

        Returns:
            A new ``NamedMapping`` with the mesh set and unknown axes
            replaced by ``None``.
        """
        mesh_axes = set(mesh.axis_names)

        def _filter_entry(entry: SpecEntry) -> SpecEntry:
            if entry is None:
                return None
            if isinstance(entry, str):
                return entry if entry in mesh_axes else None
            # Multi-axis tuple: keep only axes present in mesh.
            kept = tuple(a for a in entry if a in mesh_axes)
            if not kept:
                return None
            return kept[0] if len(kept) == 1 else kept

        resolved_spec = tuple(_filter_entry(e) for e in self._spec)
        resolved_unreduced = self._unreduced & mesh_axes

        return NamedMapping(
            _mesh=mesh,
            _spec=resolved_spec,
            _unreduced=frozenset(resolved_unreduced),
            _priorities=self._priorities,
            _memory_kind=self._memory_kind,
        )

    @classmethod
    def from_spec(
        cls,
        spec: tuple[SpecEntry, ...] = (),
        mesh: DeviceMesh | None = None,
        *,
        _unreduced: frozenset[str] = frozenset(),
        _priorities: tuple[int | None, ...] = (),
        _memory_kind: str | None = None,
    ) -> NamedMapping:
        """Creates a NamedMapping, resolving mesh from context if needed.

        This is the preferred constructor when the mesh may come from an
        ambient context rather than being passed explicitly.

        Args:
            spec: One entry per tensor dimension.
            mesh: The device mesh. If ``None``, falls back to
                ``DeviceMesh.default()``.
            _unreduced: Mesh axes with pending reductions.
            _priorities: Per-dimension propagation priority (compiler-only).
            _memory_kind: Memory tier for shard placement.
        """
        if mesh is None:
            mesh = DeviceMesh.default()
        return cls(
            _mesh=mesh,
            _spec=spec,
            _unreduced=_unreduced,
            _priorities=_priorities,
            _memory_kind=_memory_kind,
        )

    def to_placements(self) -> tuple[Placement, ...]:
        """Converts to mesh-axis-indexed placements for eager dispatch."""
        if not self.is_fully_resolved:
            if self._priorities:
                raise ConversionError(
                    f"NamedMapping has priorities {self._priorities} "
                    f"which are compiler-only annotations. Remove "
                    f"priorities for eager execution."
                )

        placements: list[Placement] = [Replicated()] * self._mesh.ndim

        for tensor_dim, entry in enumerate(self._spec):
            if entry is None:
                continue
            axes = (entry,) if isinstance(entry, str) else entry
            assert isinstance(axes, tuple)
            for axis_name in axes:
                ax = self._mesh.axis_names.index(axis_name)
                if not isinstance(placements[ax], Replicated):
                    raise ConversionError(
                        f"Mesh axis {axis_name!r} is already assigned to "
                        f"a different tensor dimension. Each mesh axis can "
                        f"shard at most one tensor dimension in the "
                        f"placement representation."
                    )
                placements[ax] = Sharded(tensor_dim)

        for axis_name in self._unreduced:
            ax = self._mesh.axis_names.index(axis_name)
            if not isinstance(placements[ax], Replicated):
                raise ConversionError(
                    f"Mesh axis {axis_name!r} is both unreduced and used "
                    f"for sharding. This is not representable as a single "
                    f"Placement per mesh axis."
                )
            placements[ax] = Partial()

        return tuple(placements)

    def to_named_sharding(self, tensor_rank: int) -> NamedMapping:
        """Returns self since this is already a NamedMapping."""
        return self

    def __repr__(self) -> str:
        entries = []
        for e in self._spec:
            if e is None:
                entries.append("None")
            elif isinstance(e, str):
                entries.append(repr(e))
            else:
                entries.append(repr(e))
        spec_str = ", ".join(entries)
        mesh_str = repr(self._mesh)
        parts = [f"{mesh_str}, ({spec_str})"]
        if self._unreduced:
            parts.append(f"unreduced={set(self._unreduced)!r}")
        if self._priorities:
            parts.append(f"priorities={self._priorities!r}")
        if self._memory_kind:
            parts.append(f"memory_kind={self._memory_kind!r}")
        return f"NamedMapping({', '.join(parts)})"
