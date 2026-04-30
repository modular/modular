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

"""Shared helpers for TensorLayout-based placement rules.

Rule signature::

    def my_rule(x: TensorLayout, ...) -> RuleSignature

where RuleSignature = tuple[tuple[...], tuple[DeviceMapping, ...]]
  - First tuple: suggested args (DeviceMappings at tensor positions,
    possibly modified values at non-tensor positions).
  - Second tuple: output DeviceMappings.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeGuard

from max.experimental.sharding.mappings import DeviceMapping, PlacementMapping
from max.experimental.sharding.mesh import DeviceMesh
from max.experimental.sharding.placements import (
    Partial,
    Placement,
    Replicated,
    Sharded,
)

# The canonical rule return type — pair of tuples:
#   (suggested_args, output_mappings)
#
# - suggested_args:  tuple with DeviceMappings at tensor positions and
#                    (possibly modified) values at non-tensor positions.
# - output_mappings: tuple of DeviceMappings for outputs.
#
# Non-tensor positions carry passthrough values that the dispatch
# engine forwards unchanged (axis indices, shape tuples, padding
# configs, stride/dilation ints, layout enums, etc.).
RuleSignature = tuple[tuple[object, ...], tuple[DeviceMapping, ...]]


# ─── Validation helpers ──────────────────────────────────────────────


def reject_sharded_axis(
    placements: tuple[Placement, ...], axis: int, op_name: str
) -> None:
    """Raises ValueError if any placement is Sharded on the given axis."""
    for p in placements:
        if isinstance(p, Sharded) and p.axis == axis:
            raise ValueError(
                f"{op_name}: cannot operate along sharded axis {axis}."
            )


def reject_any_sharded(placements: tuple[Placement, ...], op_name: str) -> None:
    """Raises ValueError if any placement is Sharded."""
    for p in placements:
        if isinstance(p, Sharded):
            raise ValueError(
                f"{op_name}: cannot operate on a tensor sharded "
                f"along axis {p.axis}. Gather first."
            )


# ─── Placement predicates ──────────────────────────────────────────


def is_replicated(p: Placement) -> TypeGuard[Replicated]:
    """Check if a placement is Replicated (narrows type for mypy)."""
    return isinstance(p, Replicated)


def is_sharded(p: Placement, axis: int | None = None) -> TypeGuard[Sharded]:
    """Check if a placement is Sharded, optionally on a specific axis."""
    return isinstance(p, Sharded) and (axis is None or p.axis == axis)


def is_partial(p: Placement) -> TypeGuard[Partial]:
    """Check if a placement is Partial (narrows type for mypy)."""
    return isinstance(p, Partial)


# ─── Placement transform helpers ────────────────────────────────────


def remap_sharded(
    placements: tuple[Placement, ...],
    fn: Callable[[int], int],
) -> tuple[Placement, ...]:
    """Apply a mapping function to all Sharded axis indices.

    Used by shape-manipulation rules to adjust shard axes after
    reordering.  For example, transpose swaps two axes::

        remap_sharded(placements, lambda a: {0: 1, 1: 0}.get(a, a))

    And unsqueeze shifts axes at or after the insertion point::

        remap_sharded(placements, lambda a: a + 1 if a >= axis else a)
    """
    return tuple(
        Sharded(fn(p.axis)) if isinstance(p, Sharded) else p for p in placements
    )


def resolve_partials(
    placements: tuple[Placement, ...],
) -> tuple[Placement, ...]:
    """Replace all Partial placements with Replicated.

    Used by rules for non-linear ops that cannot operate on Partial data.
    The dispatch engine's redistribute step will execute the actual
    all_reduce to materialize the Replicated values.
    """
    return tuple(
        Replicated() if isinstance(p, Partial) else p for p in placements
    )


def resolve_partials_mapping(current: DeviceMapping) -> DeviceMapping:
    """DeviceMapping-level wrapper around ``resolve_partials``.

    Returns *current* unchanged when it has no Partial placements,
    otherwise builds a new PlacementMapping with Partials replaced
    by Replicated.
    """
    placements = current.to_placements()
    new_p = resolve_partials(placements)
    if new_p == placements:
        return current
    return PlacementMapping(current.mesh, new_p)


def replicated_placements(
    current: tuple[Placement, ...],
) -> tuple[Placement, ...]:
    """Replicate all placements."""
    return tuple(Replicated() for _ in current)


def replicated_mapping(
    current: DeviceMapping,
    mesh: DeviceMesh | None = None,
) -> DeviceMapping:
    """Create a new DeviceMapping with all placements replicated."""
    placements = current.to_placements()
    new_p = replicated_placements(placements)
    if new_p == placements:
        return current
    return PlacementMapping(mesh or current.mesh, new_p)
