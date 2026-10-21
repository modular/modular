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

These tests never create Tensors or graph ops directly. They call
rule functions on :class:`TensorLayout` inputs and run the production
picker with every collective permitted to check what it selects.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from max.driver import CPU
from max.experimental.sharding import (
    ALL_TRANSITIONS,
    AxisAssignment,
    DeviceMapping,
    DeviceMesh,
    Partial,
    Replicated,
    Sharded,
    auto_reshard,
)
from max.experimental.sharding._auto_reshard import pick_reshard_action
from max.experimental.sharding.action import Action, tensor_layouts_in

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
) -> DeviceMapping:
    """Shorthand: M(MESH_1D, S(0)) -> DeviceMapping(MESH_1D, (S(0),))."""
    return DeviceMapping(mesh, tuple(placements))


# ── Picker for rule tests ────────────────────────────────────────────


def pick(
    rule: Callable[..., list[AxisAssignment]], *args: Any, **kwargs: Any
) -> Action:
    """Picks the cheapest :class:`Action` for ``rule(*args, **kwargs)``.

    Calls ``rule`` on the given :class:`TensorLayout` inputs and runs the
    production picker with every transition allowed, with no graph.
    """
    bound = inspect.signature(rule).bind(*args, **kwargs)
    bound.apply_defaults()
    layouts = tensor_layouts_in(tuple(bound.arguments.values()))
    with auto_reshard(ALL_TRANSITIONS, mode="silent"):
        return pick_reshard_action(
            rule(*args, **kwargs),
            layouts,
            op_name=rule.__name__,
            operand_names=(),
        )
