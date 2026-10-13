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

"""Placement rules for control-flow ops (``cond``, ``while_loop``)."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from max.experimental.sharding.placements import ShardingError
from max.experimental.sharding.types import TensorLayout
from max.graph.type import Type
from max.graph.value import TensorValue, Value

from ..action import AxisAssignment, replicated_rows


def cond_rule(
    pred: TensorLayout,
    out_types: Iterable[Type[Any]] | None,
    then_fn: Callable[..., Any],
    else_fn: Callable[..., Any],
) -> list[AxisAssignment]:
    """Auto-gathers the predicate; every device takes the same branch."""
    if out_types is not None and not isinstance(out_types, (list, tuple)):
        raise ShardingError(
            "Distributed cond requires a reusable sequence of output types."
        )
    return replicated_rows(
        pred, output_count=len(out_types) if out_types else 0
    )


def while_loop_rule(
    initial_values: (
        Iterable[TensorLayout | Value[Any]] | TensorLayout | Value[Any]
    ),
    predicate: Callable[..., TensorValue],
    body: Callable[..., Value[Any] | Iterable[Value[Any]]],
) -> list[AxisAssignment]:
    """Auto-gathers distributed initial values to Replicated.

    The loop body is not yet distribution-aware, so every carried tensor
    is gathered before the loop and each result takes that placement.
    """
    if isinstance(initial_values, TensorLayout):
        initial_values = (initial_values,)

    if not isinstance(initial_values, (list, tuple)):
        raise TypeError(
            "while_loop_rule: initial_values must be a TensorLayout, "
            f"list, or tuple; got {type(initial_values).__name__}."
        )

    tensor_layouts = tuple(
        v for v in initial_values if isinstance(v, TensorLayout)
    )
    if not tensor_layouts:
        raise ValueError(
            "while_loop_rule: no distributed TensorLayouts in initial_values."
        )

    return replicated_rows(*tensor_layouts, output_count=len(initial_values))
