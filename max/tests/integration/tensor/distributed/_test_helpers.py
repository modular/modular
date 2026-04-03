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
"""Shared test helpers for distributed tensor tests.

Provides tensor creation, distribution, and materialization utilities.
All top-level imports depend only on sharding (PR1) and tensor (PR2);
distributed_functional imports are lazy so this module can be used by
tests landing with any PR.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from max.experimental import functional as F
from max.experimental.sharding import (
    DeviceMesh,
    Partial,
    Placement,
    Sharded,
)
from max.experimental.tensor import Tensor
from max.graph import Shape


def from_np(arr: np.ndarray[Any, Any]) -> Tensor:
    """Creates a Tensor from a numpy array."""
    return Tensor.from_dlpack(np.ascontiguousarray(arr))


def to_np(t: Tensor) -> np.ndarray[Any, Any]:
    """Converts a (possibly distributed) Tensor to numpy."""
    from max.experimental.distributed_functional.collectives import to_numpy

    return to_numpy(t)


def full_tensor(t: Tensor) -> Tensor:
    """Gathers a distributed Tensor into a single non-distributed Tensor."""
    from max.experimental.distributed_functional.collectives import materialize

    return materialize(t)


def make_partial(
    data: np.ndarray[Any, Any],
    mesh: DeviceMesh,
    placements: tuple[Placement, ...],
) -> Tensor:
    """Creates a Partial tensor by replicating data across all devices."""
    shards = tuple(from_np(data).driver_tensor for _ in range(mesh.num_devices))
    return Tensor._from_shards(shards, mesh, placements, data.shape)


def gpu_partial(
    data: np.ndarray[Any, Any],
    mesh: DeviceMesh,
    placements: tuple[Placement, ...],
) -> Tensor:
    """Creates a Partial tensor on GPU devices."""
    src = from_np(data)
    shards = tuple(
        F.transfer_to(src, mesh.devices[i]).driver_tensor
        for i in range(mesh.num_devices)
    )
    return Tensor._from_shards(shards, mesh, placements, data.shape)


def _even_chunks(total: int, n: int) -> list[int]:
    """Splits total into n chunks differing by at most 1."""
    base, remainder = divmod(total, n)
    return [base + (1 if i < remainder else 0) for i in range(n)]


def shard(
    t: Tensor,
    mesh: DeviceMesh,
    placements: list[Placement] | tuple[Placement, ...],
) -> Tensor:
    """Distributes a single-device Tensor across a mesh (CPU splitting)."""
    if len(placements) != mesh.ndim:
        raise ValueError(
            f"Need one placement per mesh axis ({mesh.ndim}), "
            f"got {len(placements)}."
        )
    ndim = len(t.shape)
    for p in placements:
        if isinstance(p, Partial):
            raise ValueError("Cannot distribute with Partial placement.")
        if isinstance(p, Sharded) and (p.axis < 0 or p.axis >= ndim):
            raise ValueError(
                f"Sharded(axis={p.axis}) is out of range for tensor "
                f"with {ndim} dimensions."
            )

    global_shape = Shape(t.shape)
    shards: list[Tensor] = [t] * mesh.num_devices

    stride = 1
    for mesh_axis in reversed(range(mesh.ndim)):
        p = placements[mesh_axis]
        if not isinstance(p, Sharded):
            stride *= mesh.mesh_shape[mesh_axis]
            continue
        n = mesh.mesh_shape[mesh_axis]
        chunk_sizes: int | list[int] = _even_chunks(int(t.shape[p.axis]), n)
        new_shards: list[Tensor] = []
        for i in range(0, len(shards), n * stride):
            for g in range(stride):
                new_shards.extend(shards[i + g].split(chunk_sizes, axis=p.axis))
        shards = new_shards
        stride *= n

    storages = tuple(
        F.transfer_to(shards[i], mesh.devices[i]).driver_tensor
        for i in range(mesh.num_devices)
    )
    return Tensor._from_shards(storages, mesh, tuple(placements), global_shape)
