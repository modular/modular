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
"""Placement helpers for Inkling's per-device kernels.

The kernels run on each device's shards and return tensors with no known
placement, so the caller claims the one it knows they have.
"""

from __future__ import annotations

from max.experimental import functional as F
from max.experimental.nn.common_layers.mesh_axis import TP
from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    NamedMapping,
    Partial,
)
from max.experimental.tensor import Tensor


def shard_axis(t: Tensor, axis: int) -> Tensor:
    """Shards ``t`` on ``axis`` across its mesh's tensor-parallel axis."""
    spec: list[str | None] = [None] * t.rank
    spec[axis] = TP
    return t.to(NamedMapping(t.mesh, tuple(spec)))


def as_replicated(t: Tensor, mesh: DeviceMesh) -> Tensor:
    """Claims every device holds the same ``t``."""
    if mesh.num_devices == 1:
        return t
    return t.rebind_mapping(DeviceMapping.replicated(mesh))


def as_sharded(t: Tensor, mesh: DeviceMesh, axis: int) -> Tensor:
    """Claims each device holds its slice of ``t`` along ``axis``."""
    if mesh.num_devices == 1:
        return t
    spec: list[str | None] = [None] * t.rank
    spec[axis] = TP
    return t.rebind_mapping(NamedMapping(mesh, tuple(spec)))


def as_partial(t: Tensor, mesh: DeviceMesh) -> Tensor:
    """Claims each device holds a partial sum of ``t``."""
    if mesh.num_devices == 1:
        return t
    return t.rebind_mapping(DeviceMapping(mesh, (Partial(),)))


def reduce_partial(t: Tensor) -> Tensor:
    """Sums a partial ``t`` across devices."""
    if t.mesh.num_devices == 1:
        return t
    return F.allreduce_sum(t)
