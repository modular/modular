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

"""Defines how a tensor is laid out across a device mesh and how ops reshard it.

Describes, for every op, what redistribution to perform before the op runs.
The pipeline is deliberately local: per-op rules over a placement vocabulary
(:class:`Replicated`, :class:`Sharded`, :class:`Partial`,
:class:`Unknown`), scored by a single cost model, with the cheapest allowed
row picked on each mesh axis. A dispatch whose every row needs a transition
the current policy disallows raises rather than picking one. There is no
whole-graph trace.

An op redistributes its inputs whenever the plan it picks needs it, so a
model can contain collectives it never wrote. Inside an
:func:`auto_reshard` block, ``mode="warn"`` reports each one and
``"raise"`` refuses it, both naming the collective and the ``transfer_to``
that would replace it:

.. code-block:: python

    from max.driver import CPU
    from max.experimental.functional import full, matmul, relu
    from max.experimental.sharding import (
        DeviceMesh,
        DeviceMapping,
        Sharded,
        auto_reshard,
    )

    # A simulated two-device mesh (both slots are the same CPU).
    mesh = DeviceMesh(
        devices=(CPU(), CPU()), mesh_shape=(2,), axis_names=("tp",)
    )

    # ``a`` is column-sharded, ``b`` is row-sharded: a @ b contracts the
    # sharded dimension, so the product is a partial sum on every device.
    a = full([4, 8], 1.0, device=DeviceMapping(mesh, (Sharded(1),)))
    b = full([8, 2], 1.0, device=DeviceMapping(mesh, (Sharded(0),)))

    # ``relu`` has no row for a partial sum, so its input moves to Replicated.
    # The default transition set makes that an allreduce: the cheaper
    # reduce-scatter into Sharded is not allowed outside an
    # ``auto_reshard(ALL_TRANSITIONS)`` block. This block reports the move.
    with auto_reshard(mode="warn"):
        y = relu(matmul(a, b))

.. invisible-code-block: python

    import numpy as np

    # full(4, 8) @ full(8, 2) = 8 * ones(4, 2), then relu is a no-op (positive).
    assert np.allclose(y.to_numpy(), np.full((4, 2), 8.0))

This module avoids the overloaded word "rank" in prose. A *device* is one
accelerator; a *mesh axis* is one named dimension of the :class:`DeviceMesh`
grid; a *shard* is one device's piece of a tensor; a *tensor axis* is a
dimension of the tensor itself. :attr:`TensorLayout.rank` is the number of
tensor axes.
"""

from ._auto_reshard import auto_reshard
from .action import (
    AxisAssignment,
    match_operand_placement,
    replicated_rows,
)
from .mappings import ConversionError, DeviceMapping, NamedMapping
from .mesh import DeviceMesh
from .placements import (
    ALL_TRANSITIONS,
    DEFAULT_TRANSITIONS,
    Partial,
    Placement,
    Replicated,
    Sharded,
    ShardingError,
    Transition,
    Unknown,
)
from .types import BufferLayout, TensorLayout
from .visualize import visualize

__all__ = [
    "ALL_TRANSITIONS",
    "DEFAULT_TRANSITIONS",
    "AxisAssignment",
    "BufferLayout",
    "ConversionError",
    "DeviceMapping",
    "DeviceMesh",
    "NamedMapping",
    "Partial",
    "Placement",
    "Replicated",
    "Sharded",
    "ShardingError",
    "TensorLayout",
    "Transition",
    "Unknown",
    "auto_reshard",
    "match_operand_placement",
    "replicated_rows",
    "visualize",
]
