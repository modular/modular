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

"""Placement rules for unary, binary, and ternary elementwise ops."""

from __future__ import annotations

from typing import Any

from max.experimental.sharding.placements import Sharded
from max.experimental.sharding.types import TensorLayout

from ..action import ActionSet, AxisAssignment
from ..cost import P, R, build_action_set


def unary_rule(x: TensorLayout, *extra: Any) -> ActionSet:
    """Strategies for nonlinear unary ops (relu, exp, ...): Replicated or Sharded."""
    rows = [
        AxisAssignment((R,), R),
        *(AxisAssignment((Sharded(d),), Sharded(d)) for d in range(x.rank)),
    ]
    return build_action_set(rows, layouts=(x,), extras=extra)


def linear_unary_rule(x: TensorLayout, *extra: Any) -> ActionSet:
    """Strategies for linear unary ops (negate, cast, ...): adds Partial passthrough."""
    rows = [
        AxisAssignment((R,), R),
        *(AxisAssignment((Sharded(d),), Sharded(d)) for d in range(x.rank)),
        AxisAssignment((P,), P),
    ]
    return build_action_set(rows, layouts=(x,), extras=extra)


def binary_rule(lhs: TensorLayout, rhs: TensorLayout) -> ActionSet:
    """Strategies for elementwise binary ops (mul, div, ...): no Partial passthrough."""
    rows = [
        AxisAssignment((R, R), R),
        *(
            AxisAssignment((Sharded(d), Sharded(d)), Sharded(d))
            for d in range(lhs.rank)
        ),
        *(AxisAssignment((Sharded(d), R), Sharded(d)) for d in range(lhs.rank)),
        *(AxisAssignment((R, Sharded(d)), Sharded(d)) for d in range(lhs.rank)),
    ]
    return build_action_set(rows, layouts=(lhs, rhs))


def linear_binary_rule(lhs: TensorLayout, rhs: TensorLayout) -> ActionSet:
    """Strategies for linear binary ops (add, sub): Partial passthrough when both Partial."""
    rows = [
        AxisAssignment((R, R), R),
        *(
            AxisAssignment((Sharded(d), Sharded(d)), Sharded(d))
            for d in range(lhs.rank)
        ),
        *(AxisAssignment((Sharded(d), R), Sharded(d)) for d in range(lhs.rank)),
        *(AxisAssignment((R, Sharded(d)), Sharded(d)) for d in range(lhs.rank)),
        AxisAssignment((P, P), P),
    ]
    return build_action_set(rows, layouts=(lhs, rhs))


def ternary_rule(
    condition: TensorLayout, x: TensorLayout, y: TensorLayout
) -> ActionSet:
    """Strategies for elementwise ternary (``where``): any-broadcast Sharded."""
    n = condition.rank
    rows = [
        AxisAssignment((R, R, R), R),
        *(
            AxisAssignment((Sharded(d), Sharded(d), Sharded(d)), Sharded(d))
            for d in range(n)
        ),
        *(AxisAssignment((Sharded(d), R, R), Sharded(d)) for d in range(n)),
        *(AxisAssignment((R, Sharded(d), R), Sharded(d)) for d in range(n)),
        *(AxisAssignment((R, R, Sharded(d)), Sharded(d)) for d in range(n)),
    ]
    return build_action_set(rows, layouts=(condition, x, y))
