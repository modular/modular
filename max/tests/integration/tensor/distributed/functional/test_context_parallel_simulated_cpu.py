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

A one-axis ``tp`` mesh of 4 CPUs, a ``(dp, tp)`` mesh of two replicas of 4
where each replica is its own DCP group, and a one-axis ``tp`` mesh of 8
CPUs split into DCP groups smaller than TP.
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
    dcp_group_heads,
    dcp_ranks,
    dcp_view,
    gather_topk_scores,
    split_selection,
    to_dcp_mesh,
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
# TP8: DCP groups of 4 (or 2) inside it, each holding a copy of the cache.
TP8 = DeviceMesh(devices=(CPU(),) * 8, mesh_shape=(8,), axis_names=("tp",))


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


def test_ranks_restart_in_each_group_smaller_than_tp() -> None:
    assert list(dcp_ranks(TP8, 4)) == [0, 1, 2, 3] * 2
    assert list(dcp_ranks(dcp_view(TP8, 4), 4)) == [0, 1, 2, 3] * 2
    assert list(dcp_ranks(dcp_view(TP8, 2), 2)) == [0, 1] * 4


def test_dcp_must_tile_the_innermost_tp_axis() -> None:
    check_dcp_mesh(MESH, DCP)
    check_dcp_mesh(DP_MESH, DCP)
    # Any divisor of tp: groups of consecutive ranks.
    check_dcp_mesh(MESH, 2)
    check_dcp_mesh(TP8, 4)
    check_dcp_mesh(DP_MESH, 2)
    with pytest.raises(NotImplementedError):
        check_dcp_mesh(DP_MESH, REPLICAS * DCP)
    with pytest.raises(NotImplementedError):
        check_dcp_mesh(MESH, 3)
    with pytest.raises(NotImplementedError):
        check_dcp_mesh(TP8, 16)
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


def test_the_dcp_view_splits_tp_into_groups() -> None:
    """The view keeps the devices and splits ``tp`` into ``(tp/dcp, dcp)``."""
    view = dcp_view(TP8, 4)
    assert view.devices == TP8.devices
    assert view.mesh_shape == (2, 4)
    assert view.axis_names == ("tp", "dcp")
    assert dcp_view(view, 4) is view
    # A group spanning TP leaves one group on the view's tp axis.
    assert dcp_view(MESH, DCP).mesh_shape == (1, DCP)
    dp_view = dcp_view(DP_MESH, 2)
    assert dp_view.mesh_shape == (REPLICAS, 2, 2)
    assert dp_view.axis_names == ("dp", "tp", "dcp")
    with pytest.raises(NotImplementedError):
        dcp_view(view, 2)


def _claim(t: Tensor, mesh: DeviceMesh, *placements: Any) -> Tensor:
    return Tensor._from_shards(
        tuple(s.driver_tensor for s in t.local_shards),
        DeviceMapping(mesh, tuple(placements)),
    )


def test_tensors_move_to_the_view_and_back_without_data() -> None:
    """Placements carry over; group-only ones are Unknown on the model mesh.

    A tensor sharded over ``tp`` is sharded over both of the view's axes,
    the groups' pieces first, and folds back. One sharded across groups and
    replicated within each has no ``tp`` placement, so it comes back
    ``Unknown``; with one group spanning ``tp`` it comes back replicated.
    """
    view = dcp_view(TP8, 4)
    frags = [np.full((2, 3), r, np.float32) for r in range(8)]
    base = _per_rank(frags, TP8)
    sharded = _claim(base, TP8, Sharded(1))
    on_view = to_dcp_mesh(sharded, view)
    assert on_view.mesh == view
    assert on_view.placements == (Sharded(1), Sharded(1))
    back = to_dcp_mesh(on_view, TP8)
    assert back.mesh == TP8 and back.placements == (Sharded(1),)
    for r, shard in enumerate(back.local_shards):
        np.testing.assert_array_equal(shard.to_numpy(), frags[r])
    replicated = _claim(base, TP8, Replicated())
    assert to_dcp_mesh(replicated, view).placements == (
        Replicated(),
        Replicated(),
    )
    per_group = _claim(_per_rank(frags, view), view, Sharded(0), Replicated())
    assert to_dcp_mesh(per_group, TP8).placements == (Unknown(),)
    whole = dcp_view(MESH, DCP)
    one_group = _claim(
        _per_rank(frags[:4], whole), whole, Sharded(0), Replicated()
    )
    assert to_dcp_mesh(one_group, MESH).placements == (Replicated(),)
    with pytest.raises(ValueError):
        to_dcp_mesh(base, MESH)


def test_a_group_attends_its_tensor_parallel_heads() -> None:
    assert dcp_group_heads(TP8, 64, 8) == 64
    assert dcp_group_heads(TP8, 64, 4) == 32
    assert dcp_group_heads(TP8, 64, 2) == 16
    assert dcp_group_heads(dcp_view(TP8, 4), 64, 4) == 32
    assert dcp_group_heads(DP_MESH, 64, 2) == 32


def test_combine_below_tp_leaves_each_rank_its_tp_heads() -> None:
    """TP8 x DCP4: each group merges its 4 heads of 8 down to one per rank.

    Traces the merge without compiling or running it. The partials cover a
    group's heads, and the merged heads are each rank's TP heads, sharded
    over the whole ``tp`` axis for ``o_proj``.
    """
    tokens, heads, depth, dcp = 3, 8, 5, 4
    group_heads = dcp_group_heads(TP8, heads, dcp)
    outs = [np.zeros((tokens, group_heads, depth), np.float32)] * 8
    lses = [np.zeros((tokens, group_heads), np.float32)] * 8
    out, lse = _per_rank(outs, TP8), _per_rank(lses, TP8)
    with F.lazy():
        merged = combine_partial_attention(out, lse, dcp)
    assert not merged.real
    assert merged.mesh == TP8
    assert merged.placements == (Sharded(1),)
    assert int(merged.shape[1]) == heads
    for shard in merged.local_shards:
        assert shard.shape == [tokens, heads // 8, depth]


def test_topk_scores_gather_within_each_group_below_tp() -> None:
    """TP8 x DCP4: each group gathers only its own four ranks' scores.

    Traces the gather without compiling or running it. The groups hold
    different token counts, so a gather that crossed groups would fail to
    trace or give a rank more than ``dcp * k`` columns. The groups gathered
    different scores, so the result claims nothing along ``tp``.
    """
    k, dcp = 3, 4
    tokens = [1, 2]  # Per group.
    frags = [np.zeros((tokens[r // dcp], k), np.float32) for r in range(8)]
    scores = _per_rank(frags, TP8)
    with F.lazy():
        gathered = gather_topk_scores(scores, dcp)
    assert not gathered.real
    assert gathered.mesh == TP8
    assert gathered.placements == (Unknown(),)
    for r, shard in enumerate(gathered.local_shards):
        assert shard.shape == [tokens[r // dcp], dcp * k]


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_combine_returns_each_ranks_own_heads(dtype: Any) -> None:
    """On a one-axis mesh each rank keeps ``heads / dcp`` merged heads.

    Traces the merge without compiling or running it, in the layout
    ``o_proj`` reads: this checks shapes, dtypes and placements only.

    No test executes the merge's numerics (the LSE weights, the max shift,
    the ``-1e30`` floor for a rank with no keys), since tests here may not
    compile and run a MAX graph. The all-to-all's data movement has its
    Mojo kernel test, and the DCP rows of the mach smoke tests check gsm8k
    accuracy end to end, which catches a broken merge but does not pin its
    numerics. A fused merge kernel (MAX-1124) will carry a Mojo numerics
    test.
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
    exactly; a float32 output carries it as one lane. Each destination's
    chunk holds its heads' output features, then their lanes, head-minor.
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
            packed = F.call_on_mesh(
                lambda x, y: _pack_partials(x, y, DCP), MESH
            )(out, lse)
        assert not packed.real
        for shard in packed.local_shards:
            assert shard.shape == [
                tokens,
                DCP,
                heads // DCP * (depth + lanes),
            ]
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
