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

"""Shared fixtures for pure-metadata placement rule tests.

These tests never create Tensors or graph ops — they only call rule
functions with (mappings, shapes, **kwargs) and check the output
placements.
"""

from __future__ import annotations

from max.driver import CPU
from max.experimental.sharding import (
    DeviceMesh,
    Partial,
    PlacementMapping,
    Replicated,
    Sharded,
)

# ── Convenience aliases ──────────────────────────────────────────────

R = Replicated()
P = Partial()


def S(d: int) -> Sharded:
    return Sharded(d)


# ── Standard test meshes ────────────────────────────────────────────

MESH_1D = DeviceMesh(
    devices=tuple(CPU() for _ in range(4)),
    mesh_shape=(4,),
    axis_names=("tp",),
)

MESH_2D = DeviceMesh(
    devices=tuple(CPU() for _ in range(4)),
    mesh_shape=(2, 2),
    axis_names=("dp", "tp"),
)

MESH_2 = DeviceMesh(
    devices=(CPU(), CPU()),
    mesh_shape=(2,),
    axis_names=("tp",),
)

# ── Mapping builders ────────────────────────────────────────────────


def M(
    mesh: DeviceMesh, *placements: Replicated | Sharded | Partial
) -> PlacementMapping:
    """Shorthand: M(MESH_1D, S(0)) -> PlacementMapping(MESH_1D, (S(0),))."""
    return PlacementMapping(mesh, tuple(placements))
