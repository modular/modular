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
"""Shared test helpers for distributed tensor tests."""

from __future__ import annotations

from typing import Any

import numpy as np
from max.experimental.distributed_functional import transfer_to
from max.experimental.sharding import (
    DeviceMesh,
    Placement,
    PlacementMapping,
    Replicated,
)
from max.experimental.tensor import Tensor


def make_partial(
    data: np.ndarray[Any, Any],
    mesh: DeviceMesh,
    placements: tuple[Placement, ...],
) -> Tensor:
    """Creates a Partial tensor by replicating data across all devices.

    Distributes with Replicated placement (full copy on each device),
    then re-labels as the requested Partial placement.
    """
    replicated = PlacementMapping(mesh, tuple(Replicated() for _ in placements))
    t = transfer_to(Tensor(data), replicated)
    return Tensor._from_shards(
        tuple(s.driver_tensor for s in t.local_shards),
        mesh,
        placements,
        data.shape,
    )
