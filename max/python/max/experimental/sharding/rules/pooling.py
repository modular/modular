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

"""Placement rules for pooling ops (NHWC layout).

- ``pool_rule`` — nonlinear (max_pool2d)
- ``linear_pool_rule`` — linear (avg_pool2d)
"""

from __future__ import annotations

from max.experimental.sharding.mappings import PlacementMapping
from max.experimental.sharding.placements import Partial, Replicated, Sharded
from max.experimental.sharding.types import TensorLayout

from ._common import RuleSignature

_SPATIAL_AXES = {1, 2}


def _pool_impl(x: TensorLayout, *extra: object, linear: bool) -> RuleSignature:
    placements = x.mapping.to_placements()
    mesh = x.mapping.mesh
    if not linear:
        suggested_p = tuple(
            Replicated() if isinstance(p, Partial) else p for p in placements
        )
    else:
        suggested_p = placements
    for p in suggested_p:
        if isinstance(p, Sharded) and p.axis in _SPATIAL_AXES:
            raise ValueError(
                f"pool: cannot pool along spatially-sharded axis {p.axis}."
            )
    m = PlacementMapping(mesh, suggested_p)
    return (m, *extra), (m,)


def pool_rule(x: TensorLayout, *extra: object) -> RuleSignature:
    """Nonlinear pooling: resolves Partials."""
    return _pool_impl(x, *extra, linear=False)


def linear_pool_rule(x: TensorLayout, *extra: object) -> RuleSignature:
    """Linear pooling (avg_pool): Partials pass through."""
    return _pool_impl(x, *extra, linear=True)
