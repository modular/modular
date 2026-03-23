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

"""Distributed tensor abstractions for multi-device parallelism.

Provides :class:`DTensor`, :class:`DeviceMesh`, and placement types
(:class:`Replicated`, :class:`Sharded`, :class:`Partial`) for expressing how
tensors are distributed across devices.
"""

from .device_mesh import DeviceMesh
from .dtensor import DTensor
from .placement import Partial, Placement, ReduceOp, Replicated, Sharded

__all__ = [
    "DTensor",
    "DeviceMesh",
    "Partial",
    "Placement",
    "ReduceOp",
    "Replicated",
    "Sharded",
]
