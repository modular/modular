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

"""Placement types and sharding specifications for distributed tensors.

This package is the single source of truth for describing how tensor data is
distributed across a :class:`DeviceMesh`. It contains:

**Placement types** (mesh-axis-indexed primitives):

* :class:`Replicated`: every device holds a full copy.
* :class:`Sharded`: tensor is split along a dimension.
* :class:`Partial`: each device holds a partial result needing reduction.

**Sharding specifications** (high-level wrappers):

* :class:`PlacementMapping`, **mesh-axis-indexed** (PyTorch DTensor style).
  One :class:`Placement` per mesh axis. Suitable for eager dispatch.
* :class:`NamedMapping`, **tensor-dimension-indexed** (JAX PartitionSpec
  style). One entry per tensor dimension names the mesh axis that shards it.
  Suitable for compiler-driven sharding propagation.

Both spec types share the same :class:`DeviceMesh` and can be converted to
each other for the standard placement vocabulary. Conversions that would
lose information raise :class:`ConversionError`.
"""

from .mappings import (
    ConversionError,
    DeviceMapping,
    NamedMapping,
    PlacementMapping,
    SpecEntry,
)
from .mesh import DeviceMesh
from .placements import Partial, Placement, ReduceOp, Replicated, Sharded
from .shapes import (
    _shard_sizes_along_axis,
    global_shape_from_local,
    local_shard_shape_from_global,
    shard_shape,
)
from .types import (
    DistributedBufferType,
    DistributedTensorType,
    DistributedType,
    TensorLayout,
)

__all__ = [
    "ConversionError",
    "DeviceMapping",
    "DeviceMesh",
    "DistributedBufferType",
    "DistributedTensorType",
    "DistributedType",
    "NamedMapping",
    "Partial",
    "Placement",
    "PlacementMapping",
    "ReduceOp",
    "Replicated",
    "Sharded",
    "SpecEntry",
    "TensorLayout",
    "_shard_sizes_along_axis",
    "global_shape_from_local",
    "local_shard_shape_from_global",
    "shard_shape",
]
