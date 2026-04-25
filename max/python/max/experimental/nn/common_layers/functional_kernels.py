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

"""Functional wrappers for MAX kernel operations used in attention layers."""

import functools
from collections.abc import Callable
from typing import Any

from max.experimental import functional as F
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.sharding import (
    DeviceMesh,
    Placement,
    PlacementMapping,
    Replicated,
    Sharded,
)
from max.experimental.tensor import Tensor
from max.graph import TensorValue, ops
from max.nn.kernels import (
    flash_attention_ragged as _flash_attention_ragged,
)
from max.nn.kernels import (
    grouped_matmul_ragged as _grouped_matmul_ragged,
)
from max.nn.kernels import moe_create_indices as _moe_create_indices
from max.nn.kernels import rms_norm_key_cache as _rms_norm_key_cache
from max.nn.kernels import (
    rope_split_store_ragged as _rope_split_store_ragged,
)

grouped_matmul_ragged = F.functional(_grouped_matmul_ragged)
moe_create_indices = F.functional(_moe_create_indices)

inplace_custom = F.functional(ops.inplace_custom)


# ─── KVCache Operations ─────────────────────────────────────


def _wrap_kvcache_op(
    op: Callable[..., Any],
    output_sharded_axis: int | None = None,
) -> Callable[..., Any]:
    """Wraps a kernel op to dispatch per-device on distributed inputs.

    Args:
        op: The underlying kernel function.
        output_sharded_axis: If set, the output tensor's axis that is
            sharded across devices (e.g. 1 for column-parallel QKV).
            If ``None``, the output uses a Replicated placement.
    """

    @functools.wraps(op)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        results: list[Any] = []
        for a, kw in _loop_distributed_args(args, kwargs):
            results.append(op(*a, **kw))

        if results[0] is None:
            return None
        if isinstance(results[0], TensorValue):
            mesh = _find_mesh(*args, **kwargs)
            placements: tuple[Placement, ...]
            if output_sharded_axis is not None:
                placements = (Sharded(output_sharded_axis),)
            else:
                placements = (Replicated(),)
            return Tensor.from_shard_values(
                results, PlacementMapping(mesh, placements)
            )
        raise TypeError(f"Unexpected result type: {type(results[0])}")

    return wrapped


def _find_mesh(*args: Any, **kwargs: Any) -> DeviceMesh:
    """Returns the DeviceMesh from the first distributed Tensor arg."""
    for a in (*args, *kwargs.values()):
        if isinstance(a, Tensor) and a.is_distributed:
            return a.mesh
        if isinstance(a, PagedCacheValues) and a.n_devices > 1:
            # Get mesh from one of the PagedCacheValues fields.
            return a.kv_blocks.mesh

    # Backup plan: use the mesh of the first Tensor arg.
    for a in (*args, *kwargs.values()):
        if isinstance(a, Tensor):
            return a.mesh
    raise ValueError("No distributed tensors found in args or kwargs")


def _shard(a: Any, i: int) -> Any:
    if isinstance(a, Tensor):
        if a.is_distributed:
            return TensorValue(a.local_shards[i])
        else:
            return TensorValue(a)
    if isinstance(a, PagedCacheValues):
        return a.for_device(i)
    return a


def _loop_distributed_args(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
    """Unwrap distributed args into per-device (args, kwargs) tuples.

    Tensor args are split via ``local_shards``, PagedCacheValues via
    ``for_device``, and everything else is broadcast unchanged.
    """
    num_devices = 1
    for a in (*args, *kwargs.values()):
        if isinstance(a, Tensor) and a.is_distributed:
            num_devices = a.num_shards
            break
        if isinstance(a, PagedCacheValues) and a.n_devices > 1:
            num_devices = a.n_devices
            break

    return [
        (
            tuple(_shard(a, i) for a in args),
            {k: _shard(v, i) for k, v in kwargs.items()},
        )
        for i in range(num_devices)
    ]


flash_attention_ragged = _wrap_kvcache_op(
    _flash_attention_ragged, output_sharded_axis=1
)
rope_split_store_ragged = _wrap_kvcache_op(
    _rope_split_store_ragged, output_sharded_axis=1
)
# In-place op, returns None.
rms_norm_key_cache = _wrap_kvcache_op(_rms_norm_key_cache)


__all__ = [
    "flash_attention_ragged",
    "grouped_matmul_ragged",
    "moe_create_indices",
    "rms_norm_key_cache",
    "rope_split_store_ragged",
]
