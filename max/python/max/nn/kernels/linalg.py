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

"""Linear-algebra kernels as ``Tensor`` ops, with their sharding rules."""

from max import functional as F
from max.nn._graph_ops import (
    row_invariant_matmul as _row_invariant_matmul,
)
from max.tensor.sharding.action import AxisAssignment
from max.tensor.sharding.cost import P, R
from max.tensor.sharding.placements import Sharded
from max.tensor.sharding.types import TensorLayout


def row_invariant_matmul_rule(
    x: TensorLayout, weight: TensorLayout
) -> list[AxisAssignment]:
    """Strategies for ``row_invariant_matmul`` (``x @ weight.T``)."""
    return [
        AxisAssignment((R, R), (R,)),
        # Column-parallel: weight's N axis sharded -> output's N axis.
        AxisAssignment((R, Sharded(0)), (Sharded(1),)),
        # Row-parallel: both K axes sharded -> partial sum.
        AxisAssignment((Sharded(1), Sharded(1)), (P,)),
        AxisAssignment((Sharded(0), R), (Sharded(0),)),
        AxisAssignment((P, R), (P,)),
        AxisAssignment((R, P), (P,)),
    ]


row_invariant_matmul = F.functional(
    _row_invariant_matmul, rule=row_invariant_matmul_rule
)
