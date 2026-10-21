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

"""What ``TierConnector`` asks its Rust tiers for, and what it reports.

The Rust side knows leaves only by index and hashes only as bytes, so this
shim's whole job is the translation: the flat lookup answer cut back into a
mask per leaf, and the per-leaf load lists handed straight through. A fake
Rust connector stands in for the extension module, so these need no GPU -- the
real thing is covered in
``internal/dkv/test_tiered_connector_gpu.py``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import SimpleNamespace

import pytest
from max.nn.kv_cache import KVCacheGroupId
from max.pipelines.kv_cache.connectors.tier_connector import (
    TierConnector,
    _validate_leaves,
)
from max.pipelines.kv_cache.kv_connector import KVLoadRefused

PAGE_SIZE = 128
FULL = KVCacheGroupId.full()
WINDOW = KVCacheGroupId(type="sliding_window", window_size=2 * PAGE_SIZE + 1)


def _h(n: int) -> bytes:
    return n.to_bytes(8, "big", signed=True)


class _FakeRust:
    """Records what it is asked and answers from a fixed residency set."""

    def __init__(
        self, resident: Sequence[tuple[int, bytes]], decline: bool = False
    ) -> None:
        self.resident = set(resident)
        self.decline = decline
        self.calls: list[str] = []
        self.lookups: list[tuple[list[int], list[bytes]]] = []
        self.loads: list[tuple[list[list[int]], list[list[bytes]]]] = []
        # How much of each offload the fake takes: blocks posted per leaf
        # (every block when None), and skips reported as already held.
        self.offload_posts: int | None = None
        self.offload_already_held = 0

    def reclaim(self) -> None:
        self.calls.append("reclaim")

    def lookup(
        self, leaf_idxs: Sequence[int], block_hashes: Sequence[bytes]
    ) -> list[list[bool]]:
        self.calls.append("lookup")
        self.lookups.append((list(leaf_idxs), list(block_hashes)))
        return [
            [(leaf_idx, h) in self.resident for h in block_hashes]
            for leaf_idx in leaf_idxs
        ]

    def load(
        self,
        block_ids: Sequence[Sequence[int]],
        hashes_per_leaf: Sequence[Sequence[bytes]],
        replica_idx: int,
    ) -> SimpleNamespace:
        self.calls.append("load")
        plan = (
            [list(ids) for ids in block_ids],
            [list(hashes) for hashes in hashes_per_leaf],
        )
        self.loads.append(plan)
        # Both refusals the real connector raises on.
        for leaf_idx, hashes in enumerate(hashes_per_leaf):
            if len(block_ids[leaf_idx]) < len(hashes):
                raise ValueError(
                    f"leaf {leaf_idx} was given {len(block_ids[leaf_idx])} "
                    f"destination blocks for {len(hashes)} hashes"
                )
            for block_hash in hashes:
                if (leaf_idx, block_hash) not in self.resident:
                    raise RuntimeError(
                        f"neither tier holds {block_hash!r} for leaf {leaf_idx}"
                    )
        return SimpleNamespace(
            g0_blocks_per_leaf=(
                [[] for _ in block_ids] if self.decline else plan[0]
            ),
            direction="load",
            is_complete=lambda: True,
            synchronize=lambda: None,
        )

    def offload(
        self,
        block_ids: Sequence[Sequence[int]],
        hashes_per_leaf: Sequence[Sequence[bytes]],
        replica_idx: int,
    ) -> SimpleNamespace:
        self.calls.append("offload")
        g0: list[list[int]] = []
        for ids, hashes in zip(block_ids, hashes_per_leaf, strict=True):
            posts = (
                len(hashes)
                if self.offload_posts is None
                else self.offload_posts
            )
            g0.append(list(ids[:posts]))
        return SimpleNamespace(
            g0_blocks_per_leaf=g0,
            already_held=self.offload_already_held,
            is_complete=lambda: True,
            synchronize=lambda: None,
        )


def _connector(
    leaves: Mapping[str, KVCacheGroupId], rust: _FakeRust
) -> TierConnector:
    """A connector wired to ``rust``, with no device buffers behind it."""
    connector = TierConnector.__new__(TierConnector)
    connector._leaves = leaves
    connector._page_size = PAGE_SIZE
    connector._native = rust
    connector._loads_refused = 0
    connector._offload_blocks_dropped = 0
    return connector


# ============================================================================
# lookup
# ============================================================================


def test_one_lookup_covers_every_leaf_and_hash() -> None:
    leaves = {"a": FULL, "b": WINDOW}
    hashes = [_h(1), _h(2), _h(3)]
    rust = _FakeRust([(idx, h) for idx in (0, 1) for h in hashes])

    resident = _connector(leaves, rust).lookup(hashes)

    # The lanes' bookkeeping is applied first, or a block whose offload just
    # landed would read as absent.
    assert rust.calls == ["reclaim", "lookup"]
    # Group-major: each hash crosses the FFI boundary once, not once per leaf.
    assert rust.lookups == [([0, 1], hashes)]
    assert resident == {"a": [True] * 3, "b": [True] * 3}


def test_the_flat_answer_is_cut_back_into_one_mask_per_leaf() -> None:
    # The Rust side knows leaves only by index, so a mis-cut here would report
    # one leaf's residency against another's.
    leaves = {"a": FULL, "b": WINDOW}
    hashes = [_h(1), _h(2)]
    rust = _FakeRust([(0, _h(1)), (1, _h(2))])

    assert _connector(leaves, rust).lookup(hashes) == {
        "a": [True, False],
        "b": [False, True],
    }


def test_an_empty_request_asks_the_tiers_for_nothing() -> None:
    rust = _FakeRust([])

    assert _connector({"a": FULL}, rust).lookup([]) == {"a": []}
    assert rust.calls == []


def test_a_mis_sized_lookup_answer_fails_loudly() -> None:
    rust = _FakeRust([])
    rust.lookup = lambda leaf_idxs, block_hashes: [[True]]  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="lookup answered 1 leaves"):
        _connector({"a": FULL, "b": FULL}, rust).lookup([_h(1), _h(2)])


# ============================================================================
# load
# ============================================================================


def test_each_leaf_loads_exactly_the_hashes_it_was_handed() -> None:
    leaves = {"full": FULL, "window": WINDOW}
    hashes = [_h(1), _h(2), _h(3), _h(4)]
    rust = _FakeRust([(0, h) for h in hashes] + [(1, _h(3)), (1, _h(4))])

    transfer = _connector(leaves, rust).load(
        {"full": [10, 11, 12, 13], "window": [20, 21]},
        {"full": hashes, "window": [_h(3), _h(4)]},
    )

    assert rust.loads == [
        ([[10, 11, 12, 13], [20, 21]], [hashes, [_h(3), _h(4)]])
    ]
    # A load does no bookkeeping of its own: `lookup` already drained the
    # lanes, and everything this moves was resolved by the caller.
    assert rust.calls == ["load"]
    assert transfer.is_complete()


def test_a_short_row_is_a_bug_not_a_miss() -> None:
    # The Rust side raises ValueError for a shape mismatch, and the shim lets
    # it fly: turning a sizing bug into a cache miss would hide it forever.
    hashes = [_h(1), _h(2)]
    rust = _FakeRust([(0, h) for h in hashes])

    with pytest.raises(ValueError, match="destination blocks"):
        _connector({"a": FULL}, rust).load({"a": [7]}, {"a": hashes})


def test_a_hash_no_tier_holds_is_refused() -> None:
    leaves = {"a": FULL}
    rust = _FakeRust([(0, _h(1))])
    connector = _connector(leaves, rust)

    with pytest.raises(KVLoadRefused, match="refused a load"):
        connector.load({"a": [7, 8]}, {"a": [_h(1), _h(2)]})
    assert connector._loads_refused == 1


def test_a_declined_load_is_refused_rather_than_reported_short() -> None:
    hashes = [_h(1), _h(2)]
    rust = _FakeRust([(0, h) for h in hashes], decline=True)
    connector = _connector({"a": FULL}, rust)

    with pytest.raises(KVLoadRefused, match="saturated"):
        connector.load({"a": [7, 8]}, {"a": hashes})
    assert connector._loads_refused == 1


def test_a_partially_posted_load_fails_loudly() -> None:
    # Raising KVLoadRefused makes the caller free the rows, so a load that
    # really did post would strand the blocks its lanes pinned.
    hashes = [_h(1), _h(2)]
    rust = _FakeRust([(0, h) for h in hashes])
    rust.load = lambda block_ids, hashes_per_leaf, replica_idx: (  # type: ignore[method-assign]
        SimpleNamespace(
            g0_blocks_per_leaf=[[7]],
            direction="load",
            is_complete=lambda: True,
            synchronize=lambda: None,
        )
    )

    with pytest.raises(AssertionError, match="posted a partial load"):
        _connector({"a": FULL}, rust).load({"a": [7, 8]}, {"a": hashes})


def test_load_rejects_keys_that_are_not_the_connectors_leaves() -> None:
    connector = _connector({"a": FULL, "b": FULL}, _FakeRust([]))

    with pytest.raises(ValueError, match="do not both match"):
        connector.load({"a": [0]}, {"a": [_h(1)], "b": [_h(1)]})


# ============================================================================
# offload
# ============================================================================


def test_offload_counts_blocks_the_pool_had_no_room_for_as_dropped() -> None:
    rust = _FakeRust([])
    rust.offload_posts = 1
    connector = _connector({"a": FULL, "b": FULL}, rust)

    run = [_h(1), _h(2), _h(3)]
    connector.offload({"a": [1, 2, 3], "b": [4, 5, 6]}, {"a": run, "b": run})

    assert connector._offload_blocks_dropped == 4


def test_offload_does_not_count_blocks_the_host_tier_already_holds() -> None:
    # Re-offloading a prefix the host tier holds is routine: every later turn
    # of a session commits the blocks its earlier turns already offloaded.
    rust = _FakeRust([])
    rust.offload_posts = 1
    rust.offload_already_held = 4
    connector = _connector({"a": FULL, "b": FULL}, rust)

    run = [_h(1), _h(2), _h(3)]
    connector.offload({"a": [1, 2, 3], "b": [4, 5, 6]}, {"a": run, "b": run})

    assert connector._offload_blocks_dropped == 0


def test_leaves_of_different_depth_drop_nothing_they_were_not_given() -> None:
    rust = _FakeRust(resident=[])
    connector = _connector({"a": FULL, "b": FULL}, rust)

    connector.offload(
        {"a": [1, 2, 3], "b": [4]}, {"a": [_h(1), _h(2), _h(3)], "b": [_h(3)]}
    )

    assert connector._offload_blocks_dropped == 0


# ============================================================================
# The leaf tree
# ============================================================================


def test_a_recurrent_leaf_is_served_and_an_unhashed_one_refused() -> None:
    # A recurrent leaf is keyed by hash like the attention ones, and the
    # manager's rules know its hit is the deepest checkpoint rather than a
    # run. A scratch leaf carries no hash, so refusing at construction beats
    # claiming it as a shape it is not.
    _validate_leaves(
        {"full": FULL, "window": WINDOW, "ssm": KVCacheGroupId.recurrent()}
    )
    with pytest.raises(ValueError, match="recurrent leaves only"):
        _validate_leaves({"full": FULL, "ring": KVCacheGroupId.scratch()})
