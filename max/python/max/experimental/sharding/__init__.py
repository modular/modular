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
(:class:`Replicated`, :class:`Sharded`, :class:`Partial`), scored by a single
cost model, with the cheapest plan picked at each dispatch. There is no
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

    # ``relu`` needs the full sum, so its input is allreduced first. This
    # block reports that, instead of letting it pass unremarked.
    with auto_reshard(mode="warn"):
        y = relu(matmul(a, b))

.. invisible-code-block: python

    import numpy as np

    # full(4, 8) @ full(8, 2) = 8 * ones(4, 2), then relu is a no-op (positive).
    assert np.allclose(y.to_numpy(), np.full((4, 2), 8.0))

This module avoids the overloaded word "rank". A *device* is one accelerator;
a *mesh axis* is one named dimension of the :class:`DeviceMesh` grid; a
*shard* is one device's piece of a tensor; a *tensor axis* is a dimension of
the tensor itself.
"""

from ._auto_reshard import auto_reshard
from .action import ActionSet, AxisAssignment
from .cost import build_action_set, force_replicated_action_set
from .mappings import ConversionError, DeviceMapping, NamedMapping
from .mesh import DeviceMesh, mesh_context
from .placements import (
    ALL_TRANSITIONS,
    DEFAULT_TRANSITIONS,
    Partial,
    Placement,
    Replicated,
    Sharded,
    ShardingError,
    Transition,
)
from .types import BufferLayout, TensorLayout

__all__ = [
    "ALL_TRANSITIONS",
    "DEFAULT_TRANSITIONS",
    "ActionSet",
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
    "auto_reshard",
    "build_action_set",
    "force_replicated_action_set",
    "mesh_context",
]
