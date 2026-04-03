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

"""Distributed shape-manipulation ops.

Covers permute, transpose, reshape, concat, split, unsqueeze, squeeze,
and gather — all with placement-aware dispatch.

All shape ops are linear in data values (they rearrange but don't change
values), so Partial placements always pass through unchanged.  Only
Sharded axes need remapping when the tensor shape changes.  These ops
use ``linear=None`` because they manage placement remapping themselves.
"""

from __future__ import annotations

from collections.abc import Sequence

from max.experimental import tensor
from max.experimental.sharding import (
    PlacementMapping,
    Sharded,
)
from max.graph import Shape, ShapeLike, TensorValue, ops
from max.graph.value import TensorValueLike

from ._context_provider import functional
from .collectives import _has_distributed, make_distributed, to_shard_tvs


@functional(linear=None)
def permute(x: TensorValueLike, dims: list[int]) -> TensorValueLike:
    """Distributed permute: remaps Sharded(axis) through inverse perm."""
    if not _has_distributed(x):
        return ops.permute(x, dims)
    assert isinstance(x, tensor.Tensor)
    inv = [0] * len(dims)
    for i, d in enumerate(dims):
        inv[d] = i
    # Sharded axes are remapped through the inverse permutation.
    # Replicated and Partial pass through unchanged (permute is linear).
    out_p = tuple(
        Sharded(inv[p.axis]) if isinstance(p, Sharded) else p
        for p in x.placements
    )
    results = [ops.permute(s, dims) for s in to_shard_tvs(x)]
    return make_distributed(results, PlacementMapping(x.mesh, out_p))


@functional(linear=None)
def transpose(x: TensorValueLike, axis_1: int, axis_2: int) -> TensorValueLike:
    """Distributed transpose: swaps sharded axis if affected."""
    if not _has_distributed(x):
        return ops.transpose(x, axis_1, axis_2)
    assert isinstance(x, tensor.Tensor)
    ndim = len(x.shape)
    # Normalize negative indices so they match Sharded(axis) which is always >= 0.
    a1 = axis_1 % ndim
    a2 = axis_2 % ndim
    swap = {a1: a2, a2: a1}
    out_p = tuple(
        Sharded(swap.get(p.axis, p.axis)) if isinstance(p, Sharded) else p
        for p in x.placements
    )
    results = [ops.transpose(s, axis_1, axis_2) for s in to_shard_tvs(x)]
    return make_distributed(results, PlacementMapping(x.mesh, out_p))


@functional(linear=None)
def reshape(x: TensorValueLike, shape: ShapeLike) -> TensorValueLike:
    """Distributed reshape: finds sharded dim in new shape."""
    if not _has_distributed(x):
        return ops.reshape(x, shape)
    assert isinstance(x, tensor.Tensor)
    new_global = tuple(int(d) for d in Shape(shape))
    out_p = list(x.placements)
    local_shape = list(new_global)
    used: set[int] = set()
    for i, p in enumerate(x.placements):
        if isinstance(p, Sharded):
            dim_size = int(x.shape[p.axis])
            candidates = [
                j
                for j, s in enumerate(new_global)
                if s == dim_size and j not in used
            ]
            if not candidates:
                raise ValueError(
                    f"reshape: sharded dim {p.axis} (size {dim_size}) "
                    f"doesn't appear in new shape {new_global}"
                )
            if len(candidates) > 1:
                raise ValueError(
                    f"reshape: sharded dim {p.axis} (size {dim_size}) "
                    f"is ambiguous in new shape {new_global} — "
                    f"{len(candidates)} dimensions have that size. "
                    f"Redistribute to Replicated first, then reshape."
                )
            new_axis = candidates[0]
            used.add(new_axis)
            out_p[i] = Sharded(new_axis)
            shard_count = x.mesh.mesh_shape[i]
            if local_shape[new_axis] % shard_count != 0:
                raise ValueError(
                    f"reshape: sharded dim {new_axis} (size "
                    f"{local_shape[new_axis]}) is not evenly divisible "
                    f"by {shard_count} devices on mesh axis {i}."
                )
            local_shape[new_axis] //= shard_count
    results = [ops.reshape(s, local_shape) for s in to_shard_tvs(x)]
    return make_distributed(results, PlacementMapping(x.mesh, tuple(out_p)))


# ─── concat ───────────────────────────────────────────────────────────────


@functional(linear=None)
def concat(
    tensors: Sequence[TensorValueLike], axis: int = 0
) -> TensorValueLike:
    """Distributed concat: concatenates along a tensor axis per shard.

    All inputs must have the same placements. Concatenating along the
    sharded axis increases each shard's local size; concatenating along
    a non-sharded axis is a plain per-shard concat.

    Rules per mesh axis:
        All Replicated → Replicated
        All Sharded(k), concat axis == k → Sharded(k)  (local size grows)
        All Sharded(k), concat axis != k → Sharded(k)
        All Partial → Partial
        Mixed → error
    """
    # Fast path: no distributed tensors.
    if not any(_has_distributed(t) for t in tensors):
        return ops.concat(list(tensors), axis)

    # Mesh mismatch is caught by _validate_distributed_args in @functional.
    # All distributed inputs must share the same placements.
    dist_tensors = [
        t for t in tensors if isinstance(t, tensor.Tensor) and t.is_distributed
    ]
    ref = dist_tensors[0]
    mesh = ref.mesh
    n = mesh.num_devices
    for t in dist_tensors[1:]:
        if t.placements != ref.placements:
            raise ValueError(
                f"concat: all inputs must have the same placements. "
                f"Got {ref.placements} and {t.placements}."
            )

    all_shards = [
        to_shard_tvs(t, n)
        if isinstance(t, tensor.Tensor)
        else [TensorValue(t)] * n
        for t in tensors
    ]
    results = [
        ops.concat([all_shards[j][i] for j in range(len(tensors))], axis)
        for i in range(n)
    ]
    return make_distributed(results, PlacementMapping(mesh, ref.placements))


# ─── split ────────────────────────────────────────────────────────────────


@functional(linear=None)
def split(
    x: TensorValueLike,
    split_sizes: list[int],
    axis: int = 0,
) -> list[TensorValueLike]:
    """Distributed split: splits along a tensor axis per shard.

    If splitting along the sharded axis, the split sizes are interpreted
    as global sizes and divided by the number of shards. If splitting
    along a non-sharded axis, sizes are used as-is per shard.

    Each output chunk inherits the input's placements.
    """
    if not _has_distributed(x):
        return list(ops.split(x, split_sizes, axis=axis))

    assert isinstance(x, tensor.Tensor)
    mesh = x.mesh
    n = mesh.num_devices
    ndim = len(x.shape)
    norm_axis = axis % ndim

    # Determine per-shard split sizes.
    local_sizes = list(split_sizes)
    for mesh_axis_idx, p in enumerate(x.placements):
        if isinstance(p, Sharded) and p.axis == norm_axis:
            # Splitting along the sharded axis: divide sizes by the
            # number of shards on THIS mesh axis (not total devices).
            shard_count = mesh.mesh_shape[mesh_axis_idx]
            for s in split_sizes:
                if s % shard_count != 0:
                    raise ValueError(
                        f"split: split size {s} along sharded axis "
                        f"{norm_axis} is not evenly divisible by "
                        f"{shard_count} devices on mesh axis "
                        f"{mesh_axis_idx}."
                    )
            local_sizes = [s // shard_count for s in split_sizes]
            break

    shards = to_shard_tvs(x)
    num_chunks = len(local_sizes)
    per_shard_splits = [ops.split(s, local_sizes, axis=axis) for s in shards]

    # Transpose: per_shard_splits[device][chunk] → result[chunk][device]
    results: list[TensorValueLike] = []
    for chunk_idx in range(num_chunks):
        chunk_shards = [per_shard_splits[dev][chunk_idx] for dev in range(n)]
        results.append(
            make_distributed(chunk_shards, PlacementMapping(mesh, x.placements))
        )
    return results


# ─── unsqueeze / squeeze ─────────────────────────────────────────────────


@functional(linear=None)
def unsqueeze(x: TensorValueLike, axis: int) -> TensorValueLike:
    """Distributed unsqueeze: adds a size-1 dimension.

    Sharded axes at or after the insertion point shift up by 1.
    """
    if not _has_distributed(x):
        return ops.unsqueeze(x, axis)
    assert isinstance(x, tensor.Tensor)
    ndim = len(x.shape)
    norm_axis = axis if axis >= 0 else axis + ndim + 1
    out_p = tuple(
        Sharded(p.axis + 1 if p.axis >= norm_axis else p.axis)
        if isinstance(p, Sharded)
        else p
        for p in x.placements
    )
    results = [ops.unsqueeze(s, axis) for s in to_shard_tvs(x)]
    return make_distributed(results, PlacementMapping(x.mesh, out_p))


@functional(linear=None)
def squeeze(x: TensorValueLike, axis: int) -> TensorValueLike:
    """Distributed squeeze: removes a size-1 dimension.

    Sharded axes after the removed dimension shift down by 1.
    Raises if squeezing the sharded dimension.
    """
    if not _has_distributed(x):
        return ops.squeeze(x, axis)
    assert isinstance(x, tensor.Tensor)
    ndim = len(x.shape)
    norm_axis = axis % ndim
    for p in x.placements:
        if isinstance(p, Sharded) and p.axis == norm_axis:
            raise ValueError(
                f"squeeze: cannot squeeze sharded axis {norm_axis}."
            )
    out_p = tuple(
        Sharded(p.axis - 1 if p.axis > norm_axis else p.axis)
        if isinstance(p, Sharded)
        else p
        for p in x.placements
    )
    results = [ops.squeeze(s, axis) for s in to_shard_tvs(x)]
    return make_distributed(results, PlacementMapping(x.mesh, out_p))


# ─── gather (index select) ───────────────────────────────────────────────


@functional(linear=None)
def gather(
    x: TensorValueLike, indices: TensorValueLike, axis: int = 0
) -> TensorValueLike:
    """Distributed gather: index-select along an axis.

    Rules:
        x Replicated, indices any → same as indices placement
        x Sharded(k), k != axis → Sharded(k), gather per shard
        x Sharded(k), k == axis → error (gathering across sharded dim
            requires vocab-parallel logic — use a custom dispatch)
    """
    if not _has_distributed(x) and not _has_distributed(indices):
        return ops.gather(x, indices, axis=axis)

    # Mixed distributed/non-distributed and mesh mismatch are caught
    # by _validate_distributed_args in the @functional wrapper.
    idx_is_tensor = isinstance(indices, tensor.Tensor)

    if isinstance(x, tensor.Tensor) and x.is_distributed:
        mesh = x.mesh
        n = mesh.num_devices
        ndim = len(x.shape)
        norm_axis = axis % ndim

        for p in x.placements:
            if isinstance(p, Sharded) and p.axis == norm_axis:
                raise ValueError(
                    f"gather: cannot gather along sharded axis {norm_axis}. "
                    "Use a custom dispatch for vocab-parallel embedding."
                )

        x_shards = to_shard_tvs(x, n)
        if idx_is_tensor:
            assert isinstance(indices, tensor.Tensor)
            idx_shards = to_shard_tvs(indices, n)
        else:
            idx_shards = [TensorValue(indices)] * n
        results = [
            ops.gather(xs, idxs, axis=axis)
            for xs, idxs in zip(x_shards, idx_shards, strict=False)
        ]
        return make_distributed(results, PlacementMapping(mesh, x.placements))

    # Only indices is distributed; x is non-distributed or a raw value.
    assert idx_is_tensor and isinstance(indices, tensor.Tensor)
    mesh = indices.mesh
    n = mesh.num_devices
    idx_shards = to_shard_tvs(indices, n)
    x_tv = TensorValue(x)
    results = [ops.gather(x_tv, idxs, axis=axis) for idxs in idx_shards]
    return make_distributed(results, PlacementMapping(mesh, indices.placements))
