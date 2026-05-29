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

"""Placement rules for normalization."""

from __future__ import annotations

from max.experimental.sharding.mappings import PlacementMapping
from max.experimental.sharding.placements import (
    Partial,
    Replicated,
    Sharded,
)
from max.experimental.sharding.types import TensorLayout

from ._common import RuleSignature, is_replicated, replicated_mapping


def normalization_rule(
    x: TensorLayout,
    weight: TensorLayout,
    *args: object,
) -> RuleSignature:
    """Placement rule for normalization: rejects sharded norm dimensions."""
    # weight placement must be replicated.
    weight_placements = weight.mapping.to_placements()
    if any(not is_replicated(p) for p in weight_placements):
        raise ValueError(
            f"Normalization: weight must be replicated, but got {weight_placements}"
        )

    x_placements = x.mapping.to_placements()
    suggested_x_placements = tuple(
        Replicated() if isinstance(p, Partial) else p for p in x_placements
    )

    norm_start = x.rank - weight.rank
    for p in suggested_x_placements:
        if isinstance(p, Sharded) and p.axis >= norm_start:
            raise ValueError(
                f"layer_norm: cannot normalize along sharded axis "
                f"{p.axis}. Gather first or shard a different axis."
            )

    mesh = x.mapping.mesh
    suggested_x_mapping = PlacementMapping(mesh, suggested_x_placements)

    suggested_weight_mapping = replicated_mapping(weight.mapping, mesh)

    suggested_args: list[object] = []
    for arg in args:
        if isinstance(arg, TensorLayout):
            suggested_args.append(replicated_mapping(arg.mapping, mesh))
        else:
            suggested_args.append(arg)

    return (suggested_x_mapping, suggested_weight_mapping, *suggested_args), (
        suggested_x_mapping,
    )
