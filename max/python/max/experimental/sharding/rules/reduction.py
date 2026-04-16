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

"""Placement rules for reduction ops.

- ``reduce_rule`` — nonlinear (softmax, argmax, etc.)
- ``linear_reduce_rule`` — linear (sum, cumsum)
"""

from __future__ import annotations

from max.experimental.sharding.mappings import PlacementMapping
from max.experimental.sharding.placements import Partial, Replicated, Sharded
from max.experimental.sharding.types import TensorLayout

from ._common import RuleSignature


def _reduce_impl(
    x: TensorLayout,
    axis: int = -1,
    linear: bool = False,
) -> RuleSignature:
    placements = x.mapping.to_placements()
    mesh = x.mapping.mesh
    norm_axis = axis % x.rank

    if not linear:
        suggested_p = tuple(
            Replicated() if isinstance(p, Partial) else p for p in placements
        )
    else:
        suggested_p = placements

    for p in suggested_p:
        if isinstance(p, Sharded) and p.axis == norm_axis:
            raise ValueError(
                f"reduce: cannot reduce along sharded axis {norm_axis}."
            )

    m = PlacementMapping(mesh, suggested_p)
    return (m, axis), (m,)


def reduce_rule(x: TensorLayout, axis: int = -1) -> RuleSignature:
    """Nonlinear reduction: resolves Partials."""
    return _reduce_impl(x, axis=axis, linear=False)


def linear_reduce_rule(x: TensorLayout, axis: int = -1) -> RuleSignature:
    """Linear reduction: Partials pass through."""
    return _reduce_impl(x, axis=axis, linear=True)


# Backward compatibility
reduce_single_axis = reduce_rule
