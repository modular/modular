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

"""Placement rules for reduction ops (``reduce``, ``softmax``, ``sum``, ``mean``)."""

from __future__ import annotations

from typing import Any

from max.experimental.sharding import Sharded
from max.experimental.sharding.types import TensorLayout

from ..action import AxisAssignment
from ..cost import P, R


def _non_reduction_axis_rows(
    x: TensorLayout, axis: int
) -> list[AxisAssignment]:
    """``Sharded(d) -> Sharded(d)`` rows over every non-reduction axis."""
    norm = axis % x.rank
    return [
        AxisAssignment((Sharded(d),), (Sharded(d),))
        for d in range(x.rank)
        if d != norm
    ]


def reduce_rule(
    x: TensorLayout, axis: int = -1, *extra: Any, **kwargs: Any
) -> list[AxisAssignment]:
    """Non-linear reduction: shard any non-reduction axis.

    Shared by ``prod``, ``argmax``, ``argmin``, ``max``, ``min``;
    ``*extra`` absorbs op-specific trailing args.
    """
    return [AxisAssignment((R,), (R,)), *_non_reduction_axis_rows(x, axis)]


def softmax_rule(value: TensorLayout, axis: int = -1) -> list[AxisAssignment]:
    """Strategies for ``softmax`` / ``logsoftmax``: shard any non-softmax axis."""
    return [AxisAssignment((R,), (R,)), *_non_reduction_axis_rows(value, axis)]


def linear_reduce_rule(
    x: TensorLayout, axis: int = -1, *extra: Any
) -> list[AxisAssignment]:
    """Linear reduction: ``S(reduced_axis) -> Partial(SUM)``; ``P -> P``.

    Used by ``sum``. Not by ``cumsum``: a device's prefix sums along a
    sharded axis need the totals of the shards before it.
    """
    norm = axis % x.rank
    return [
        AxisAssignment((R,), (R,)),
        *_non_reduction_axis_rows(x, axis),
        AxisAssignment((Sharded(norm),), (P,)),
        AxisAssignment((P,), (P,)),
    ]
