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

"""Placement rules for buffer-mutation ops."""

from __future__ import annotations

from max.experimental.sharding.types import TensorLayout
from max.graph.ops.slice_tensor import SliceIndices

from ..action import AxisAssignment, pass_through_rows
from ..cost import R
from ..placements import Placement, Sharded, Unknown


def _placed_alike(actuals: tuple[Placement, ...]) -> bool:
    """Returns whether both operands share a placement, or either is Unknown."""
    return actuals[0] == actuals[1] or Unknown() in actuals


def buffer_store_rule(
    destination: TensorLayout, source: TensorLayout
) -> list[AxisAssignment]:
    """Strategies for ``buffer_store``: destination and source share placement."""
    return pass_through_rows(
        (destination, source),
        lambda actuals: () if _placed_alike(actuals) else None,
    )


def buffer_store_slice_rule(
    destination: TensorLayout,
    source: TensorLayout,
    indices: SliceIndices,
) -> list[AxisAssignment]:
    """Returns the rows for ``buffer_store_slice``: each device writes its own shard."""
    return pass_through_rows(
        (destination, source),
        lambda actuals: (
            ()
            if _placed_alike(actuals)
            or (isinstance(actuals[0], Sharded) and actuals[1] == R)
            else None
        ),
    )
