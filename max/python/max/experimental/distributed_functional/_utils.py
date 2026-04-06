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

"""Tensor materialization and conversion utilities for distributed tensors.

These are convenience helpers, not collective ops.  They live here
rather than in ``collectives.py`` to keep that module focused on the
actual collective communication primitives.

TODO: ``materialize`` and ``to_numpy`` should eventually become methods
on the Tensor class itself.
"""

from __future__ import annotations

import numpy as np
from max.driver import CPU
from max.experimental import tensor
from max.experimental.sharding import Partial, Replicated, Sharded


def materialize(t: tensor.Tensor) -> tensor.Tensor:
    """Gather a distributed Tensor into a single non-distributed Tensor.

    Reduces all Partial axes via all-reduce-sum, gathers all Sharded
    axes via all-gather, and returns shard 0 of the fully-Replicated
    result.  No-op for non-distributed tensors.
    """
    if not t.is_distributed:
        return t

    # Lazy import to avoid circular dependency: _utils -> collectives -> _utils
    from .collectives import all_gather, all_reduce_sum

    result = t
    for ax, p in enumerate(result.placements):
        if isinstance(p, Replicated):
            pass
        elif isinstance(p, Partial):
            result = all_reduce_sum(result, mesh_axis=ax)
        elif isinstance(p, Sharded):
            result = all_gather(result, tensor_axis=p.axis, mesh_axis=ax)
    return result.local_shards[0]


def to_numpy(t: tensor.Tensor) -> np.ndarray:
    """Convert a (possibly distributed, possibly GPU) Tensor to numpy.

    Materializes distributed tensors and transfers to CPU if needed.
    """
    from max.experimental.sharding import DeviceMesh, PlacementMapping

    from .collectives import shard

    if t.is_distributed:
        t = materialize(t)
    if t.device != CPU():
        cpu_mapping = PlacementMapping(
            DeviceMesh.single(CPU()), (Replicated(),)
        )
        t = shard(t, cpu_mapping)
    return np.from_dlpack(t)
