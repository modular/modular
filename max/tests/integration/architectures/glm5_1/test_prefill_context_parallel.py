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
"""Prefill context parallelism's row split on simulated CPU meshes.

The split, the all-gather back to batch order, the all-to-all to head
sharding and the step predicate are plain graph ops and collectives with no
kernel of their own, so they run here on a CPU mesh: TP4 with one DCP group
of four, TP8 with two groups of four, and two data-parallel replicas of
two. The cache packing kernels and the attention numerics are covered by
the kernel tests and the GPU serving runs.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from max.driver import CPU
from max.pipelines.architectures.deepseekV3_2_modulev3.layers.prefill_context_parallel import (
    PCP_MAX_BATCH_CONTEXT,
    PCP_MIN_STEP_TOKENS,
    PrefillContextParallel,
    local_row_count,
    prefill_predicate,
    row_ranges,
    rows_carrier,
    split_rows,
)
from max.pipelines.architectures.deepseekV3_2_modulev3.model_config import (
    resolve_prefill_context_parallel,
)
from max.tensor import default_device
from max.tensor import realization_context as rc
from max.tensor.sharding import (
    DeviceMapping,
    DeviceMesh,
    Replicated,
    Sharded,
)
from max.tensor.tensor import Tensor, realization_context

TP4 = DeviceMesh((CPU(),) * 4, (4,), ("tp",))
TP8 = DeviceMesh((CPU(),) * 8, (8,), ("tp",))

# Uneven prompts; a batch shorter than the group, which leaves ranks with no
# rows at all; one request; and short and long requests mixed.
BATCHES = {
    "uneven": [5, 9, 2],
    "fewer_rows_than_ranks": [3],
    "one_short_of_even": [7],
    "mixed": [1, 30, 2, 7, 1],
}

_PENDING: list[tuple[Tensor, Any]] = []


def _expect(actual: Tensor, expected: Any) -> None:
    """Checks ``actual`` once the test's eager graph realizes."""
    _PENDING.append((actual, expected))


@pytest.fixture(autouse=True)
def _realization() -> Iterator[None]:
    """Builds each test in one eager graph on the CPU, then runs its queued
    value checks."""
    _PENDING.clear()
    with (
        default_device(CPU()),
        rc.EagerRealizationContext() as ctx,
        realization_context(ctx),
    ):
        yield
    for actual, expected in _PENDING:
        np.testing.assert_array_equal(actual.to_numpy(), expected)
    _PENDING.clear()


def _offsets(lengths: list[int]) -> np.ndarray[Any, Any]:
    return np.concatenate([[0], np.cumsum(lengths)]).astype(np.uint32)


def _replicated(values: np.ndarray[Any, Any], mesh: DeviceMesh) -> Tensor:
    return Tensor(values).to(DeviceMapping(mesh, (Replicated(),) * mesh.ndim))


_NAMESPACES = itertools.count()


def _split(offsets: Tensor, rows: Tensor, dcp: int) -> PrefillContextParallel:
    """The split alone: the scratch and prefix fields stay unset, since
    nothing here reads them. Each split takes its own namespace: a test
    builds several batches into one eager graph, and a rank's row count is
    a named symbol that would otherwise bind two values."""
    row_offsets, query_starts, restore = split_rows(offsets, rows, dcp)
    row_start, row_stop = row_ranges(rows, dcp)
    namespace = f"pcp{next(_NAMESPACES)}"
    unset = cast(Any, None)
    return PrefillContextParallel(
        dcp=dcp,
        interleave=1,
        namespace=namespace,
        batch_row_offsets=offsets,
        row_offsets=row_offsets,
        query_starts=query_starts,
        row_start=row_start,
        row_stop=row_stop,
        rows=rows_carrier(row_start, row_stop, rows.mesh, dcp, namespace),
        restore=restore,
        prefix_offsets=unset,
        prefix_rows=unset,
        mla_scratch=unset,
        indexer_scratch=unset,
    )


def _rank_rows(total: int, rank: int, dcp: int) -> range:
    """Rank ``rank``'s flat rows: the ceiling-first split."""
    start = sum(int(local_row_count(total, r, dcp)) for r in range(rank))
    return range(start, start + int(local_row_count(total, rank, dcp)))


def _expected_layout(
    lengths: list[int], rank: int, dcp: int
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Rank ``rank``'s row offsets and query starts, from first principles."""
    offsets = _offsets(lengths)
    own = _rank_rows(int(offsets[-1]), rank, dcp)
    row_offsets = [0]
    starts = []
    for b in range(len(lengths)):
        begin, end = int(offsets[b]), int(offsets[b + 1])
        mine = [t for t in own if begin <= t < end]
        row_offsets.append(row_offsets[-1] + len(mine))
        # The request's rows before the rank's run, all of them when the
        # run starts past the request.
        starts.append(min(max(own.start, begin), end) - begin)
    return (
        np.array(row_offsets, dtype=np.uint32),
        np.array(starts, dtype=np.uint32),
    )


def test_local_row_count_is_the_ceiling_first_split() -> None:
    for total in range(20):
        for dcp in (1, 2, 4, 8):
            counts = [int(local_row_count(total, r, dcp)) for r in range(dcp)]
            assert sum(counts) == total
            assert counts == sorted(counts, reverse=True)
            assert counts[0] - counts[-1] <= 1


def test_rows_keep_their_request_and_position() -> None:
    """Each rank's rows, row offsets and query starts on a TP4 group, and
    the rows all-gather back in batch order on every rank."""
    for lengths in BATCHES.values():
        _check_rows(lengths)


def _check_rows(lengths: list[int]) -> None:
    offsets = _offsets(lengths)
    total = int(offsets[-1])
    dcp = 4
    rows = _replicated(np.arange(total, dtype=np.float32).reshape(-1, 1), TP4)
    split = _split(_replicated(offsets, TP4), rows, dcp)
    local = split.local_rows(rows)
    assert local.placements == (Sharded(0),)
    for rank in range(dcp):
        expected_rows = np.array(
            list(_rank_rows(total, rank, dcp)), dtype=np.float32
        ).reshape(-1, 1)
        _expect(local.local_shards[rank], expected_rows)
        row_offsets, starts = _expected_layout(lengths, rank, dcp)
        _expect(split.row_offsets.local_shards[rank], row_offsets)
        _expect(split.query_starts.local_shards[rank], starts)
    gathered = split.gather_rows(local, rows)
    assert gathered.placements == rows.placements
    for shard in gathered.local_shards:
        _expect(shard, np.arange(total, dtype=np.float32).reshape(-1, 1))


def test_head_sharding_restores_batch_order() -> None:
    """Each rank attends its rows with all of the group's heads; the
    all-to-all leaves rank ``j`` its run of heads over every row in batch
    order, padding and all."""
    for lengths in BATCHES.values():
        _check_heads(lengths)


def _check_heads(lengths: list[int]) -> None:
    total = int(_offsets(lengths)[-1])
    dcp, heads, depth = 4, 8, 2
    # Row t, head h, lane d holds 100 * t + 10 * h + d.
    whole = (
        100 * np.arange(total)[:, None, None]
        + 10 * np.arange(heads)[None, :, None]
        + np.arange(depth)[None, None, :]
    ).astype(np.float32)
    rows = _replicated(whole, TP4)
    split = _split(_replicated(_offsets(lengths), TP4), rows, dcp)
    merged = split.to_head_sharded(split.local_rows(rows), rows)
    assert merged.placements == (Sharded(1),)
    per_rank = heads // dcp
    for rank in range(dcp):
        _expect(
            merged.local_shards[rank],
            whole[:, rank * per_rank : (rank + 1) * per_rank],
        )


def test_groups_below_tp_split_and_restore_on_their_own() -> None:
    """TP8 x DCP4: each group of four splits the rows among its ranks; the
    all-to-all leaves every rank its own tensor-parallel heads."""
    lengths = BATCHES["mixed"]
    total = int(_offsets(lengths)[-1])
    dcp, heads, depth = 4, 4, 1
    # Each group attends its own heads: group g holds heads [4g, 4g + 4).
    groups = [
        (
            100 * np.arange(total)[:, None, None]
            + 10 * (g * heads + np.arange(heads))[None, :, None]
            + np.zeros((1, 1, depth))
        ).astype(np.float32)
        for g in range(2)
    ]
    rows = Tensor.from_shard_values(
        [Tensor(groups[i // dcp]) for i in range(8)],
        DeviceMapping(TP8, (Replicated(),)),
    )
    split = _split(_replicated(_offsets(lengths), TP8), rows, dcp)
    local = split.local_rows(rows)
    # Pinned: with groups below TP the rows are ``Sharded(0)`` on ``tp``,
    # though no group spans it. Only row-wise ops with ``tp``-replicated
    # weights and the split's own collectives read them, and the
    # ``__pcp4_dcp4_mtp`` smoke runs on this placement.
    assert local.placements == (Sharded(0),)
    for i in range(8):
        expected = groups[i // dcp][list(_rank_rows(total, i % dcp, dcp))]
        _expect(local.local_shards[i], expected)
    merged = split.to_head_sharded(local, rows)
    assert merged.placements == (Sharded(1),)
    for i in range(8):
        head = i % dcp
        _expect(merged.local_shards[i], groups[i // dcp][:, head : head + 1])


def _contexts(values: list[int]) -> Tensor:
    """Each replica's ``[1]`` host context bound, one shard each."""
    mesh = DeviceMesh((CPU(),) * len(values), (len(values),), ("dp",))
    return Tensor.from_shard_values(
        [Tensor(np.array([v], np.uint32)) for v in values],
        DeviceMapping(mesh, (Replicated(),)),
    )


def test_prefill_steps_take_the_split_and_decode_steps_do_not() -> None:
    """A step qualifies when its rows average more than eight a request and
    number at least ``PCP_MIN_STEP_TOKENS``."""

    def step(tokens: int, requests: int, context: int | None = None) -> Tensor:
        return prefill_predicate(
            Tensor(np.zeros(tokens, np.uint8)),
            Tensor(np.zeros(requests + 1, np.uint32)),
            _contexts([tokens if context is None else context]),
        )

    _expect(step(8192, 1), True)
    _expect(step(32, 4), False)  # A K = 7 verify of four requests.
    _expect(step(4, 4), False)  # Decode.
    least = PCP_MIN_STEP_TOKENS
    _expect(step(least, 1), True)
    _expect(step(least - 1, 1), False)  # A chunked prefill's short tail.
    _expect(step(least, least // 8), False)
    # The scratch grows with the context: past the cap the step keeps DCP.
    most = PCP_MAX_BATCH_CONTEXT
    _expect(step(8192, 1, most), True)
    _expect(step(8192, 1, most + 1), False)


def test_prefill_context_parallel_is_opt_in() -> None:
    # Unset means off, whatever the DCP layout: PCP is opt-in.
    assert not resolve_prefill_context_parallel(None)
    assert not resolve_prefill_context_parallel(False)
    # An explicit request passes through, for the config to refuse where PCP
    # cannot run (no DCP, a gathered query, interleave above 1, FP8 MLA or
    # more than one data-parallel replica).
    assert resolve_prefill_context_parallel(True)


def test_the_restore_index_skips_each_ranks_padding() -> None:
    """Row ``t`` lands past the padding of every short rank before it."""
    for total in (1, 3, 8, 9, 15):
        for dcp in (2, 8):
            mesh = DeviceMesh((CPU(),) * dcp, (dcp,), ("tp",))
            offsets = _replicated(np.array([0, total], np.uint32), mesh)
            rows = _replicated(np.zeros((total, 1), np.float32), mesh)
            _, _, restore = split_rows(offsets, rows, dcp)
            first = int(local_row_count(total, 0, dcp))
            expected = [
                r * first + (t - _rank_rows(total, r, dcp).start)
                for r in range(dcp)
                for t in _rank_rows(total, r, dcp)
            ]
            _expect(restore.local_shards[0], np.array(expected, np.int32))


def test_the_query_view_describes_each_ranks_own_rows() -> None:
    """The scratch's decode records take the rank's longest query and the
    DCP record's partition count, and turn the mixed-step split off."""
    lengths = [1, 30, 2, 7, 1]
    dcp = 8
    mesh = DeviceMesh((CPU(),) * dcp, (dcp,), ("tp",))
    offsets = _replicated(_offsets(lengths), mesh)
    rows = _replicated(np.zeros((sum(lengths), 1), np.float32), mesh)
    split = _split(offsets, rows, dcp)
    dcp_record = np.array([5, 30, 6, 1, 1, 4], np.int64)
    partitions = np.array([6, 4], np.int64)
    record = np.array([3, 3, 1, 1, 30, 9, 4], np.uint32)
    collection = SimpleNamespace(
        attention_dispatch_metadata=_replicated(dcp_record, mesh),
        mla_num_partitions=_replicated(partitions, mesh),
        mixed_step_split=_replicated(record, mesh),
    )
    fields = split._dispatch_fields(cast(Any, collection))
    for i in range(dcp):
        rank_offsets, _ = _expected_layout(lengths, i, dcp)
        longest = int(np.diff(rank_offsets.astype(np.int64)).max())
        _expect(
            fields["attention_dispatch_metadata"].local_shards[i],
            np.array([5, longest, 6, 0, 0, 0], np.int64),
        )
        _expect(
            fields["mla_num_partitions"].local_shards[i],
            np.array([6, 0], np.int64),
        )
        _expect(
            fields["mixed_step_split"].local_shards[i],
            np.zeros(7, np.uint32),
        )


def test_two_dcp_groups_share_one_rows_symbol_per_rank() -> None:
    """With two DCP groups (TP8, ``dcp == 4``) the plan declares one rows
    carrier per rank, not per device: a group's ranks hold the same counts on
    every group, and two declarations of one value would be unified to one
    symbol under a layer subgraph signature that names both (the
    ``__pcp4_dcp4_mtp`` export failed to compile on that). Every device
    slices with its rank's carrier."""
    lengths = [5, 7, 4]
    total = sum(lengths)
    dcp = 4
    rows = _replicated(np.arange(total, dtype=np.float32).reshape(-1, 1), TP8)
    split = _split(_replicated(_offsets(lengths), TP8), rows, dcp)
    assert len(split.rows) == dcp
    assert [str(r.shape[0]) for r in split.rows] == [
        f"{split.namespace}_rows_{rank}" for rank in range(dcp)
    ]
    for rank, carrier in enumerate(split.rows):
        _expect(carrier, np.zeros(len(_rank_rows(total, rank, dcp)), np.uint8))
    local = split.local_rows(rows)
    for i in range(8):
        expected = np.arange(total, dtype=np.float32).reshape(-1, 1)
        _expect(
            local.local_shards[i],
            expected[list(_rank_rows(total, i % dcp, dcp))],
        )
