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

"""Shape utilities for distributed tensors."""

from __future__ import annotations

from collections.abc import Sequence

from max.graph import Dim, Shape

from .device_mesh import DeviceMesh
from .placement import Partial, Placement, Replicated, Sharded


def global_shape(
    local_shape: Sequence[Dim],
    mesh: DeviceMesh,
    placements: Sequence[Placement],
) -> Shape:
    """Derives the global shape from one local shard's shape and placements."""
    shape = list(local_shape)
    for axis_idx, placement in enumerate(placements):
        if isinstance(placement, Sharded):
            shape[placement.axis] = (
                shape[placement.axis] * mesh.mesh_shape[axis_idx]
            )
        elif isinstance(placement, (Replicated, Partial)):
            continue
        else:
            raise NotImplementedError(f"Unknown placement type: {placement}")
    return Shape(shape)
