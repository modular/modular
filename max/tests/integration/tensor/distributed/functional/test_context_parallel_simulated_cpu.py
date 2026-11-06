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
"""Decode context parallel helpers on simulated CPU meshes.

A one-axis ``tp`` mesh of 4 CPUs, and a ``(dp, tp)`` mesh of two replicas of
4 where each replica is its own DCP group.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.graph import Shape, ops
from max.nn.common_layers.context_parallel import (
    _pack_partials,
    check_dcp_mesh,
    combine_partial_attention,
    dcp_ranks,
    gather_topk_scores,
    split_selection,
)
from max.tensor import Tensor
from max.tensor.sharding import (
    DeviceMapping,
    DeviceMesh,
    Replicated,
    Sharded,
    Unknown,
)

MESH = DeviceMesh(
    devices=(CPU(), CPU(), CPU(), CPU()), mesh_shape=(4,), axis_names=("tp",)
)
DCP = 4
REPLICAS = 2
DP_MESH = DeviceMesh(
    devices=(CPU(),) * (REPLICAS * DCP),
    mesh_shape=(REPLICAS, DCP),
    axis_names=("dp", "tp"),
)


def _per_rank(
    fragments: list[np.ndarray[Any, Any]], mesh: DeviceMesh = MESH
) -> Tensor:
    """Places fragment ``r`` on device ``r``, claiming nothing about them."""
    shards = []
    for fragment, device in zip(fragments, mesh.devices, strict=True):
        t = Tensor(fragment)
        if t.device != device:
            t = Tensor(storage=t.driver_tensor.to(device))
        shards.append(t.driver_tensor)
    return Tensor._from_shards(
        tuple(shards), DeviceMapping(mesh, (Unknown(),) * mesh.ndim)
    )


def _cast(t: Tensor, dtype: DType) -> Tensor:
    """Casts each rank's fragment, keeping the placements it claims."""
    return F.call_on_mesh(
        lambda x: ops.cast(x, dtype), t.mesh, out_specs=t.mapping
    )(t)


def test_ranks_are_consecutive_devices() -> None:
    assert list(dcp_ranks(MESH, 4)) == [0, 1, 2, 3]
    assert list(dcp_ranks(MESH, 2)) == [0, 1, 0, 1]


def test_ranks_restart_in_each_replica() -> None:
    assert list(dcp_ranks(DP_MESH, DCP)) == [0, 1, 2, 3] * REPLICAS


def test_dcp_must_span_the_innermost_tp_axis() -> None:
    check_dcp_mesh(MESH, DCP)
    check_dcp_mesh(DP_MESH, DCP)
    with pytest.raises(NotImplementedError):
        check_dcp_mesh(DP_MESH, REPLICAS * DCP)
    with pytest.raises(NotImplementedError):
        check_dcp_mesh(MESH, 2)
    tp_outer = DeviceMesh(
        devices=DP_MESH.devices,
        mesh_shape=(DCP, REPLICAS),
        axis_names=("tp", "dp"),
    )
    with pytest.raises(NotImplementedError):
        check_dcp_mesh(tp_outer, DCP)
    no_tp = DeviceMesh(
        devices=DP_MESH.devices,
        mesh_shape=(REPLICAS, DCP),
        axis_names=("dp", "ep"),
    )
    with pytest.raises(NotImplementedError):
        check_dcp_mesh(no_tp, DCP)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_combine_returns_each_ranks_own_heads(dtype: Any) -> None:
    """On a one-axis mesh each rank keeps ``heads / dcp`` merged heads.

    Traces the merge without compiling or running it, in the layout
    ``o_proj`` reads. The merge is graph ops around one all-to-all; its
    numerics (LSE weights,
    max-shift, the ``-inf`` floor for a rank with no keys) are covered end
    to end by the DCP rows of the mach smoke accuracy tests, and the
    all-to-all's data movement by its Mojo kernel test.
    """
    tokens, heads, depth = 3, 8, 5
    outs = [np.zeros((tokens, heads, depth), np.float32) for _ in range(DCP)]
    lses = [np.zeros((tokens, heads), np.float32) for _ in range(DCP)]
    out, lse = _per_rank(outs), _per_rank(lses)
    expected = DType.bfloat16 if dtype == "bfloat16" else DType.float32
    with F.lazy():
        if expected != DType.float32:
            out = _cast(out, expected)
        merged = combine_partial_attention(out, lse, DCP)
    assert not merged.real
    assert merged.placements == (Sharded(1),)
    for shard in merged.local_shards:
        assert shard.shape == [tokens, heads // DCP, depth]
        assert shard.dtype == expected


def test_the_a2a_merge_is_one_collective_with_lse_lanes() -> None:
    """The a2a merge exchanges one packed tensor: outputs plus LSE lanes.

    Traces the packing without compiling or running it. A bfloat16 output
    carries each float32 LSE as three bfloat16 lanes, which sum back to it
    exactly; a float32 output carries it as one lane.
    """
    tokens, heads, depth = 2, 8, 5
    for dtype, lanes in [(DType.float32, 1), (DType.bfloat16, 3)]:
        out = _per_rank(
            [np.zeros((tokens, heads, depth), np.float32) for _ in range(DCP)]
        )
        lse = _per_rank(
            [np.zeros((tokens, heads), np.float32) for _ in range(DCP)]
        )
        with F.lazy():
            if dtype != DType.float32:
                out = _cast(out, dtype)
            packed = F.call_on_mesh(_pack_partials, MESH)(out, lse)
        assert not packed.real
        for shard in packed.local_shards:
            assert shard.shape == [tokens, heads, depth + lanes]
            assert shard.dtype == dtype


def test_topk_scores_gather_to_every_rank() -> None:
    """On a one-axis mesh every rank gets all ``dcp * k`` columns.

    Traces the gather without compiling or running it. The rank-major
    column order is the all-gather's own concatenation order.
    """
    k = 3
    frags = [np.zeros((2, k), np.float32) for _ in range(DCP)]
    scores = _per_rank(frags)
    with F.lazy():
        gathered = gather_topk_scores(scores, DCP)
    assert not gathered.real
    assert gathered.placements == (Replicated(),)
    for shard in gathered.local_shards:
        assert shard.shape == [2, DCP * k]
        assert shard.dtype == DType.float32


def test_combine_merges_within_each_replica() -> None:
    """On a ``(dp, tp)`` mesh each replica merges only its own ranks.

    Traces the merge without compiling or running it. The replicas hold
    different token counts, so a merge that crossed replicas would fail to
    trace or leave a rank with another replica's shape.
    """
    heads, depth = 8, 5
    tokens = [3, 2]  # Per replica.
    outs = [
        np.zeros((tokens[d], heads, depth), np.float32)
        for d in range(REPLICAS)
        for _ in range(DCP)
    ]
    lses = [
        np.zeros((tokens[d], heads), np.float32)
        for d in range(REPLICAS)
        for _ in range(DCP)
    ]
    out, lse = _per_rank(outs, DP_MESH), _per_rank(lses, DP_MESH)
    with F.lazy():
        merged = combine_partial_attention(out, lse, DCP)
    assert not merged.real
    assert merged.placements == (Unknown(), Sharded(1))
    for i, shard in enumerate(merged.local_shards):
        assert shard.shape == [tokens[i // DCP], heads // DCP, depth]
        assert shard.dtype == DType.float32


def test_topk_scores_gather_within_each_replica() -> None:
    """On a ``(dp, tp)`` mesh each replica gathers only its own ranks.

    Traces the gather without compiling or running it. The replicas hold
    different token counts, so a gather that crossed replicas would fail to
    trace or give a rank more than ``dcp * k`` columns.
    """
    k = 3
    tokens = [1, 2]  # Per replica.
    frags = [
        np.zeros((tokens[d], k), np.float32)
        for d in range(REPLICAS)
        for _ in range(DCP)
    ]
    scores = _per_rank(frags, DP_MESH)
    with F.lazy():
        gathered = gather_topk_scores(scores, DCP)
    assert not gathered.real
    assert gathered.placements == (Unknown(), Replicated())
    for i, shard in enumerate(gathered.local_shards):
        assert shard.shape == [tokens[i // DCP], DCP * k]
        assert shard.dtype == DType.float32


def _slice_bounds(t: Tensor) -> tuple[Shape, Shape, Shape]:
    """The ``(starts, stops, steps)`` of the slice that produced ``t``."""
    op: Any = t._graph_value._mlir_value.owner
    while not hasattr(op, "starts"):  # Look through a reshape.
        (operand,) = op.operands
        op = operand.value.owner
    return (
        Shape.from_mlir(op.starts),
        Shape.from_mlir(op.stops),
        Shape.from_mlir(op.steps),
    )


def test_a_packed_selection_splits_into_rows_and_their_count() -> None:
    """A selection row is ``k`` row slots, then the count attention reads.

    Traces the split without compiling or running it, and reads the slices
    each rank emits.
    """
    k = 4
    # Rank r holds r + 1 tokens, so each rank's shapes are its own.
    frags = [np.zeros((r + 1, k + 1), dtype=np.int32) for r in range(DCP)]
    selection = _per_rank(frags)
    with F.lazy():
        rows, counts = split_selection(selection, k)
    assert not rows.real and not counts.real
    assert rows.mapping == counts.mapping == selection.mapping
    for r, (row, count) in enumerate(
        zip(rows.local_shards, counts.local_shards, strict=True)
    ):
        t = r + 1
        assert row.shape == [t, k]
        assert count.shape == [t]
        assert row.dtype == count.dtype == DType.int32
        assert _slice_bounds(row) == ([0, 0], [t, k], [1, 1])
        assert _slice_bounds(count) == ([0, k], [t, k + 1], [1, 1])
