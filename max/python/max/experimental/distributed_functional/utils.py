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

"""Shared utilities for the distributed functional module.

Helpers used by ``collective_ops``, ``spmd_ops``, and ``creation_ops``.
This is the leaf of the internal dependency graph — no imports from
sibling modules (``collective_ops``, ``spmd_ops``, ``creation_ops``).
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Generator

from max.driver import Accelerator, Buffer
from max.experimental import realization_context as rc
from max.experimental import tensor
from max.experimental.sharding import (
    DeviceMesh,
    PlacementMapping,
    Replicated,
    Sharded,
    TensorLayout,
)
from max.experimental.tensor import Tensor
from max.graph import BufferValue, Graph, TensorValue

# ═════════════════════════════════════════════════════════════════════════
#  Graph context + realization context
# ═════════════════════════════════════════════════════════════════════════


def in_graph_context() -> bool:
    """Return True when executing inside a ``Graph.current`` context."""
    try:
        _ = Graph.current
    except LookupError:
        return False
    else:
        return True


@contextlib.contextmanager
def ensure_context() -> Generator[None]:
    """Ensure a realization context exists for Tensor <-> TensorValue conversion.

    Three execution contexts are supported:

    * **Eager** (``EagerRealizationContext``) — created automatically when no
      context is active and we are *not* inside a ``Graph``.  On exit,
      ``realize_all()`` is called so that all symbolic graph ops executed
      within the block are compiled and realized into concrete tensors.
    * **Lazy** (``LazyRealizationContext``) — set externally via
      ``with lazy():``.  When already active this function re-uses it;
      tensors remain unrealized until explicitly awaited.
    * **Graph** (``GraphRealizationContext``) — created automatically when
      we are inside a ``Graph.current`` context.  Tensors stay symbolic.

    If a context of *any* kind already exists, it is re-used as-is.
    """
    if tensor.current_realization_context(None) is not None:
        yield
        return
    ctx: rc.EagerRealizationContext | rc.GraphRealizationContext = (
        rc.GraphRealizationContext(Graph.current)
        if in_graph_context()
        else rc.EagerRealizationContext()
    )
    with ctx, tensor.realization_context(ctx):
        yield


# ═════════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════════


def tensor_to_layout(t: Tensor) -> TensorLayout:
    """Convert a ``Tensor`` to a ``TensorLayout`` for sharding rule evaluation."""
    if t.is_distributed:
        return TensorLayout(
            t.dtype, t.shape, PlacementMapping(t.mesh, t.placements)
        )
    return TensorLayout(
        t.dtype,
        t.shape,
        PlacementMapping(DeviceMesh.single(t.device), (Replicated(),)),
    )


def _devices_are_unique(shards: list[TensorValue]) -> bool:
    """True when all shards live on distinct physical devices."""
    devices = [str(s.device) for s in shards]
    return len(set(devices)) == len(devices)


def _signal_buffers(
    mesh: DeviceMesh,
) -> list[BufferValue] | None:
    """Get signal buffers from the current realization context, if available.

    Checks two sources:

    1. ``ctx.signal_buffers`` — pre-created buffers from ``module.compile()``
       (:class:`GraphRealizationContext`). Returns ``None`` if the mesh
       contains only CPUs.
    2. ``ctx.ensure_signal_buffers(mesh)`` — lazy creation for
       :class:`EagerRealizationContext` on multi-GPU meshes.
    """
    ctx = tensor.current_realization_context(None)
    if ctx is None:
        return None

    # Path 1: pre-set signal buffers (from module.compile / GraphRealizationContext)
    if hasattr(ctx, "signal_buffers") and ctx.signal_buffers is not None:
        if not any(isinstance(d, Accelerator) for d in mesh.devices):
            return None  # CPU-only mesh — fall back to simulated
        return ctx.signal_buffers

    # Path 2: lazy creation (EagerRealizationContext)
    if hasattr(ctx, "ensure_signal_buffers"):
        return ctx.ensure_signal_buffers(mesh)

    return None


def _mesh_axis_groups(mesh: DeviceMesh, mesh_axis: int) -> list[list[int]]:
    """Partition device indices into groups that communicate along *mesh_axis*."""
    axis_size = mesh.mesh_shape[mesh_axis]
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


def _even_split_sizes(dim: int, n: int) -> list[int]:
    """Split *dim* into *n* sizes that differ by at most 1."""
    base, rem = divmod(dim, n)
    return [base + (1 if i < rem else 0) for i in range(n)]


# ═════════════════════════════════════════════════════════════════════════
#  Tensor conversion and introspection
# ═════════════════════════════════════════════════════════════════════════


def to_tensor(value: Buffer | Tensor | TensorValue) -> Tensor:
    """Convert a single graph/buffer result to a Tensor."""
    if isinstance(value, Tensor):
        return value
    if isinstance(value, Buffer):
        return Tensor(storage=value)
    return Tensor.from_graph_value(value)


def to_tensors(values: object) -> object:
    """Convert graph op results to Tensors, preserving container type.

    - ``None`` → ``None``
    - Single ``Buffer``/``TensorValue``/``Tensor`` → ``Tensor``
    - ``list``/``tuple`` of the above → same container of ``Tensor``
    - Anything else passes through unchanged.
    """
    if values is None:
        return None
    if isinstance(values, (Buffer, Tensor, TensorValue)):
        return to_tensor(values)
    if isinstance(values, (list, tuple)):
        return type(values)(to_tensor(v) for v in values)
    return values


def map_tensors(
    fn: Callable[[Tensor], object], args: tuple[object, ...]
) -> tuple[object, ...]:
    """Apply *fn* to every :class:`Tensor` leaf in *args*.

    Recurses into ``list`` and ``tuple`` containers; everything else
    (scalars, strings, ``None``, …) passes through unchanged.
    """

    def _walk(x: object) -> object:
        if isinstance(x, Tensor):
            return fn(x)
        if isinstance(x, list):
            return [_walk(v) for v in x]
        if isinstance(x, tuple):
            return tuple(_walk(v) for v in x)
        return x

    return tuple(_walk(a) for a in args)


def any_distributed(args: tuple[object, ...]) -> bool:
    """True if any :class:`Tensor` in *args* is distributed (multi-device).

    Checks top-level args and one level of list/tuple nesting.
    """
    for a in args:
        if isinstance(a, Tensor) and a.is_distributed:
            return True
        if isinstance(a, (list, tuple)):
            for item in a:
                if isinstance(item, Tensor) and item.is_distributed:
                    return True
    return False


def is_sharded_on(t: Tensor, tensor_axis: int, exclude_mesh_axis: int) -> bool:
    """True if *tensor_axis* is already Sharded on some mesh axis other than *exclude*."""
    for ax, p in enumerate(t.placements):
        if ax == exclude_mesh_axis:
            continue
        if isinstance(p, Sharded) and p.axis == tensor_axis:
            return True
    return False


# ═════════════════════════════════════════════════════════════════════════
#  Lazy evaluation context
# ═════════════════════════════════════════════════════════════════════════


@contextlib.contextmanager
def lazy() -> Generator[None]:
    """Context manager for lazy (deferred) tensor evaluation.

    Within this context, tensor operations are recorded but not executed.
    Tensors remain unrealized until explicitly awaited via
    ``await tensor.realize`` or until their values are needed.

    Example::

        from max.experimental import functional as F
        with F.lazy():
            a = Tensor.zeros([5, 5])
            b = a + 1
        # b is unrealized — no compilation has happened yet
        await b.realize
    """
    with rc.LazyRealizationContext() as ctx, tensor.realization_context(ctx):
        yield
