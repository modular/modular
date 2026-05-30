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

"""Distributed-tensor sharding: how a tensor is laid out across a device mesh.

Describes, for every op, what redistribution to perform before the op runs.
The pipeline is deliberately local: per-op rules over a placement vocabulary
(:class:`Replicated`, :class:`Sharded`, :class:`Partial`), scored by a single
cost model, with one pluggable :class:`Solver` making the choice at each
dispatch. There is no whole-graph trace.

A ``mode(...)`` block selects the solver for the ops inside it:

.. code-block:: python

    from max.experimental.functional import matmul, relu
    from max.experimental.sharding import GreedyReshard, mode

    with mode(GreedyReshard(on_reshard="warn")):
        y = relu(matmul(a, b))

Shipped solvers: :class:`GreedyReshard` (cheapest feasible action),
:class:`NoReshard` (first feasible, prefers passthrough), and
:class:`PartialsOnly` (only ``Partial -> Replicated`` resolutions).

This module avoids the overloaded word "rank". A *device* is one accelerator;
a *mesh axis* is one named dimension of the :class:`DeviceMesh` grid; a
*shard* is one device's piece of a tensor; a *tensor axis* is a dimension of
the tensor itself.
"""

from .action import (
    Action,
    ActionSet,
    AxisAssignment,
    PerShard,
)
from .cost import (
    FeasibilityContext,
    P,
    R,
    build_action_set,
    force_replicated_action_set,
    transition_cost,
)
from .mappings import (
    ConversionError,
    DeviceMapping,
    NamedMapping,
    PlacementMapping,
    SpecEntry,
    is_fully_replicated,
)
from .mesh import DeviceMesh, MeshContext, get_active_mesh

# Re-export so ``sharding.mode(...)`` resolves to the function, not the submodule.
from .mode import ShardingError, current_solver, isolated_solver, mode
from .per_shard_dim import (
    PerShardDim,
    cell_at,
    global_dim,
    global_shape,
    is_one,
    is_per_shard_dim,
    make_per_shard_dim,
    shape_at,
)
from .picker import (
    GreedyReshard,
    NoReshard,
    PartialsOnly,
    ReshardBehavior,
    Solver,
    cheapest_action,
    enumerate_feasible_actions,
)
from .placements import (
    Collective,
    Partial,
    Placement,
    ReduceOp,
    Replicated,
    Sharded,
    _shard_sizes_along_axis,
    local_shard_shape_from_global,
    shard_shape,
)
from .rules import *
from .types import (
    DistributedBufferType,
    DistributedTensorType,
    DistributedType,
    TensorLayout,
)

__all__ = [
    "Action",
    "ActionSet",
    "AxisAssignment",
    "Collective",
    "ConversionError",
    "DeviceMapping",
    "DeviceMesh",
    "DistributedBufferType",
    "DistributedTensorType",
    "DistributedType",
    "FeasibilityContext",
    "GreedyReshard",
    "MeshContext",
    "NamedMapping",
    "NoReshard",
    "P",
    "Partial",
    "PartialsOnly",
    "PerShard",
    "PerShardDim",
    "Placement",
    "PlacementMapping",
    "R",
    "ReduceOp",
    "Replicated",
    "ReshardBehavior",
    "Sharded",
    "ShardingError",
    "Solver",
    "SpecEntry",
    "TensorLayout",
    "_shard_sizes_along_axis",
    "build_action_set",
    "cell_at",
    "cheapest_action",
    "current_solver",
    "enumerate_feasible_actions",
    "force_replicated_action_set",
    "get_active_mesh",
    "global_dim",
    "global_shape",
    "is_fully_replicated",
    "is_one",
    "is_per_shard_dim",
    "isolated_solver",
    "local_shard_shape_from_global",
    "make_per_shard_dim",
    "mode",
    "shape_at",
    "shard_shape",
    "transition_cost",
]
