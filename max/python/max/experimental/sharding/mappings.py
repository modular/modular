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

"""Tensor-to-mesh bindings: :class:`DeviceMapping` and JAX-style :class:`NamedMapping` sugar."""

from __future__ import annotations

import contextlib
from collections.abc import Generator, Iterable
from contextvars import ContextVar
from dataclasses import dataclass

from .mesh import DeviceMesh
from .placements import (
    Partial,
    Placement,
    Replicated,
    Sharded,
    resolve_partials,
)


class ConversionError(Exception):
    """Raised when a sharding spec conversion would lose information."""


SpecEntry = str | tuple[str, ...] | None
"""One entry in a named spec: a mesh axis name, a tuple of names
(multi-axis sharding), or ``None`` for replicated."""


_DEFAULT_MESH: ContextVar[DeviceMesh | None] = ContextVar(
    "_DEFAULT_MESH", default=None
)
"""Ambient mesh consulted at mapping construction time.

When a :class:`DeviceMapping` or :class:`NamedMapping` is built with the
trivial singleton mesh (``DeviceMesh.single(...)``, axis name ``"_"``)
and this context var is set, construction swaps the mesh for the
ambient one before resolving placements. Layers that declare sharding
intent on a per-parameter singleton (for example
``NamedMapping(self.weight.mesh, (None, "TP"))`` in
``RowParallelLinear``) materialize the intent the moment the model is
constructed inside a :func:`default_mesh` block, with no deferred
resolution machinery.
"""


@contextlib.contextmanager
def default_mesh(mesh: DeviceMesh) -> Generator[DeviceMesh, None, None]:
    """Sets the ambient mesh for mapping construction inside the block.

    See :data:`_DEFAULT_MESH`.

    .. code-block:: python

        from max.experimental.sharding import DeviceMesh, default_mesh

        mesh = DeviceMesh((Accelerator(),) * 4, (4,), ("tp",))
        with default_mesh(mesh):
            model = MyModel(config)   # layer mappings resolve against mesh
    """
    token = _DEFAULT_MESH.set(mesh)
    try:
        yield mesh
    finally:
        _DEFAULT_MESH.reset(token)


def _resolve_construction_mesh(mesh: DeviceMesh) -> DeviceMesh:
    """Returns the ambient mesh when ``mesh`` is the singleton placeholder.

    Returns ``mesh`` unchanged when no ambient mesh is set or when ``mesh``
    is already a real (non-singleton) mesh.
    """
    active = _DEFAULT_MESH.get()
    if active is None:
        return mesh
    if mesh.num_devices == 1 and mesh.axis_names == ("_",):
        return active
    return mesh


@dataclass(frozen=True)
class DeviceMapping:
    """How a tensor is distributed across a device mesh.

    Args:
        mesh: The device mesh.
        placements: One :class:`Placement` per mesh axis, in mesh-axis
            order.
    """

    mesh: DeviceMesh
    placements: tuple[Placement, ...]

    def __post_init__(self) -> None:
        resolved = _resolve_construction_mesh(self.mesh)
        if resolved is not self.mesh and len(self.placements) == resolved.ndim:
            object.__setattr__(self, "mesh", resolved)
        if len(self.placements) != self.mesh.ndim:
            raise ValueError(
                f"Need one placement per mesh axis ({self.mesh.ndim}), "
                f"got {len(self.placements)}."
            )

    def to_placements(self) -> tuple[Placement, ...]:
        """Back-compat alias for :attr:`placements`."""
        return self.placements

    @property
    def is_fully_replicated(self) -> bool:
        """``True`` when every mesh axis is :class:`Replicated`."""
        return all(isinstance(p, Replicated) for p in self.placements)

    def __repr__(self) -> str:
        placement_str = ", ".join(repr(p) for p in self.placements)
        return f"{type(self).__name__}({self.mesh}, [{placement_str}])"


PlacementMapping = DeviceMapping
"""Back-compat alias for :class:`DeviceMapping`."""


def is_fully_replicated(mapping: DeviceMapping) -> bool:
    """``True`` when every mesh axis is :class:`Replicated`."""
    return all(isinstance(p, Replicated) for p in mapping.placements)


class NamedMapping(DeviceMapping):
    """Builds a :class:`DeviceMapping` from a JAX-style named spec.

    Each spec entry corresponds to a tensor dim and names the mesh
    axis that shards it (or ``None`` for replicated). Mesh-axis names
    not present on the target mesh resolve to :class:`Replicated`.
    ``unreduced`` names mesh axes carrying a pending reduction; each
    becomes a :class:`Partial` placement. After construction this is
    a regular :class:`DeviceMapping`; the spec is forgotten.

    Args:
        mesh: The target device mesh.
        spec: One entry per tensor dimension.
        unreduced: Mesh axes carrying pending reductions.
    """

    def __init__(
        self,
        mesh: DeviceMesh,
        spec: tuple[SpecEntry, ...] = (),
        *,
        unreduced: Iterable[str] = (),
    ) -> None:
        mesh = _resolve_construction_mesh(mesh)
        unreduced_t = tuple(unreduced)
        placements = _spec_to_placements(mesh, spec, unreduced_t)
        object.__setattr__(self, "mesh", mesh)
        object.__setattr__(self, "placements", placements)
        object.__setattr__(self, "_original_spec", tuple(spec))
        object.__setattr__(self, "_original_unreduced", unreduced_t)

    @property
    def original_spec(self) -> tuple[SpecEntry, ...]:
        """Returns the caller-supplied spec before mesh resolution.

        Preserved so :meth:`_resolve` can rebind this mapping against
        another mesh.
        """
        return self._original_spec  # type: ignore[attr-defined]

    @property
    def original_unreduced(self) -> tuple[str, ...]:
        """The caller-supplied unreduced axes, preserved for re-resolution."""
        return self._original_unreduced  # type: ignore[attr-defined]

    def _resolve(self, mesh: DeviceMesh) -> NamedMapping:
        """Rebinds this mapping's original spec against ``mesh``.

        Axis names in the spec that are not present in ``mesh`` are
        treated as :class:`Replicated`, matching the construction-time
        graceful-degradation behavior. Use this when a tensor with a
        stub-mesh :class:`NamedMapping` is being moved onto its real
        target mesh.
        """
        return NamedMapping(
            mesh, self.original_spec, unreduced=self.original_unreduced
        )

    def __repr__(self) -> str:
        """Renders the underlying placements in JAX-style named form."""
        names = self.mesh.axis_names
        spec_parts: list[str] = []
        unreduced: list[str] = []
        mesh_names_by_tensor_dim: dict[int, list[str]] = {}
        for axis, p in enumerate(self.placements):
            if isinstance(p, Sharded):
                mesh_names_by_tensor_dim.setdefault(p.axis, []).append(
                    names[axis]
                )
            elif isinstance(p, Partial):
                unreduced.append(names[axis])
        max_tensor_dim = max(mesh_names_by_tensor_dim, default=-1) + 1
        for tensor_dim in range(max_tensor_dim):
            mesh_names = mesh_names_by_tensor_dim.get(tensor_dim, [])
            if not mesh_names:
                spec_parts.append("None")
            elif len(mesh_names) == 1:
                spec_parts.append(repr(mesh_names[0]))
            else:
                spec_parts.append(
                    "(" + ", ".join(repr(n) for n in mesh_names) + ")"
                )
        spec_str = ", ".join(spec_parts) if spec_parts else ""
        parts = [repr(self.mesh), f"({spec_str})"]
        if unreduced:
            parts.append(
                f"unreduced={{{', '.join(repr(n) for n in unreduced)}}}"
            )
        return f"NamedMapping({', '.join(parts)})"


def _spec_to_placements(
    mesh: DeviceMesh,
    spec: tuple[SpecEntry, ...],
    unreduced: Iterable[str],
) -> tuple[Placement, ...]:
    """Resolves a JAX-style spec into mesh-axis-indexed placements."""
    mesh_axes = set(mesh.axis_names)
    placements: list[Placement] = [Replicated() for _ in range(mesh.ndim)]

    for tensor_dim, entry in enumerate(spec):
        if entry is None:
            continue
        names = (entry,) if isinstance(entry, str) else entry
        for name in names:
            if name not in mesh_axes:
                continue
            ax = mesh.axis_names.index(name)
            if not isinstance(placements[ax], Replicated):
                raise ConversionError(
                    f"Mesh axis {name!r} is already assigned to another "
                    f"tensor dimension; each mesh axis can shard at most one."
                )
            placements[ax] = Sharded(tensor_dim)

    for name in unreduced:
        if name not in mesh_axes:
            continue
        ax = mesh.axis_names.index(name)
        if not isinstance(placements[ax], Replicated):
            raise ConversionError(
                f"Mesh axis {name!r} is both unreduced and used for sharding."
            )
        placements[ax] = Partial()

    return tuple(placements)


def resolve_partials_mapping(current: DeviceMapping) -> DeviceMapping:
    """Resolves any :class:`Partial` placements on ``current`` to :class:`Replicated`."""
    new_p = resolve_partials(current.placements)
    if new_p == current.placements:
        return current
    return DeviceMapping(current.mesh, new_p)


def replicate_axes(
    current: DeviceMapping, tensor_axes: Iterable[int]
) -> DeviceMapping:
    """Forces :class:`Replicated` on any mesh axis currently sharding ``tensor_axes``.

    :class:`Partial` placements are preserved; compose with
    :func:`resolve_partials_mapping` when nonlinear semantics also apply.
    """
    bad = frozenset(tensor_axes)
    new_p = tuple(
        Replicated() if isinstance(p, Sharded) and p.axis in bad else p
        for p in current.placements
    )
    if new_p == current.placements:
        return current
    return DeviceMapping(current.mesh, new_p)


def replicate_all(current: DeviceMapping) -> DeviceMapping:
    """Forces fully-Replicated placement on every mesh axis."""
    replicated = tuple(Replicated() for _ in range(current.mesh.ndim))
    if current.placements == replicated:
        return current
    return DeviceMapping(current.mesh, replicated)
