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

"""What ``RustTierConnector`` asks its Rust tiers for, and what it pads.

The Rust side answers presence and loads exactly what it is told; the prefix
rules that turn one into the other are ``prefix_hit``'s, run here. A fake Rust
connector stands in for the extension module, so these need no GPU -- the real
thing is covered in ``internal/dkv/test_rust_tiered_connector_gpu.py``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import SimpleNamespace

import pytest
from max.nn.kv_cache import KVCacheGroupId
from max.pipelines.kv_cache.connectors.rust_tier_connector import (
    RustTierConnector,
    _validate_leaves,
)

PAGE_SIZE = 128
FULL = KVCacheGroupId.full()
# Two pages of window: `blocks_in_window` drops the query token's own page.
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
        self.lookups: list[list[tuple[int, bytes]]] = []
        self.loads: list[tuple[list[list[int]], list[list[bytes]]]] = []

    def reclaim(self) -> None:
        self.calls.append("reclaim")

    def lookup(self, leaf_hashes: Sequence[tuple[int, bytes]]) -> list[bool]:
        self.calls.append("lookup")
        self.lookups.append(list(leaf_hashes))
        return [pair in self.resident for pair in leaf_hashes]

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
                raise RuntimeError(
                    f"leaf {leaf_idx} was given "
                    f"{len(block_ids[leaf_idx])} destination blocks for "
                    f"{len(hashes)} hashes"
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


def _connector(
    leaves: Mapping[str, KVCacheGroupId], rust: _FakeRust
) -> RustTierConnector:
    """A connector wired to ``rust``, with no device buffers behind it."""
    connector = RustTierConnector.__new__(RustTierConnector)
    connector._leaves = leaves
    connector._page_size = PAGE_SIZE
    connector._rust = rust
    return connector


def _everything(
    leaves: Mapping[str, KVCacheGroupId], hashes: Sequence[bytes]
) -> list[tuple[int, bytes]]:
    return [(idx, h) for idx in range(len(leaves)) for h in hashes]


def test_one_lookup_covers_every_leaf_and_hash() -> None:
    leaves = {"a": FULL, "b": WINDOW}
    hashes = [_h(1), _h(2), _h(3)]
    rust = _FakeRust(_everything(leaves, hashes))

    _connector(leaves, rust).load({"a": [0, 1, 2], "b": [3, 4, 5]}, hashes)

    assert rust.lookups == [_everything(leaves, hashes)]
    # And the lanes' bookkeeping is applied first, so the lookup and the load
    # read the same tiers.
    assert rust.calls == ["reclaim", "lookup", "load"]


def test_a_leaf_missing_the_hash_shortens_the_joint_hit() -> None:
    # Every leaf has to answer for the candidate, so a hash only one of them
    # holds is no hit -- `lookup` answers per leaf and makes no claim about a
    # leaf's siblings.
    rust = _FakeRust([(0, _h(1)), (1, _h(1)), (0, _h(2))])

    connector = _connector({"values": FULL, "scales": FULL}, rust)
    assert connector.count_cached_prefix([_h(1), _h(2)]) == (1, 0)


def test_a_mis_sized_lookup_answer_fails_loudly() -> None:
    rust = _FakeRust([])
    rust.lookup = lambda leaf_hashes: [True]  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="lookup answered 1 blocks"):
        _connector({"a": FULL, "b": FULL}, rust).count_cached_prefix(
            [_h(1), _h(2)]
        )


def test_a_full_leaf_loads_the_whole_hit() -> None:
    hashes = [_h(1), _h(2), _h(3)]
    rust = _FakeRust([(0, _h(1)), (0, _h(2))])

    transfer = _connector({"a": FULL}, rust).load({"a": [7, 8, 9]}, hashes)

    assert rust.loads == [([[7, 8]], [[_h(1), _h(2)]])]
    assert transfer.g0_blocks_per_leaf == {"a": [7, 8]}


def test_a_windowed_leaf_loads_its_window_and_nulls_the_rest() -> None:
    # The full leaf holds the whole run, the windowed leaf only its last two
    # blocks -- which is a complete window, so all four blocks are a hit. The
    # windowed leaf carries only the window; the rest of its row is null
    # blocks, which the Rust connector is never told about.
    leaves = {"full": FULL, "window": WINDOW}
    hashes = [_h(1), _h(2), _h(3), _h(4)]
    rust = _FakeRust([(0, h) for h in hashes] + [(1, _h(3)), (1, _h(4))])

    transfer = _connector(leaves, rust).load(
        {"full": [10, 11, 12, 13], "window": [20, 21]}, hashes
    )

    assert rust.loads == [
        ([[10, 11, 12, 13], [20, 21]], [hashes, [_h(3), _h(4)]])
    ]
    assert transfer.g0_blocks_per_leaf == {
        "full": [10, 11, 12, 13],
        "window": [0, 0, 20, 21],
    }


def test_a_window_the_full_leaf_cannot_reach_is_not_a_hit() -> None:
    leaves = {"full": FULL, "window": WINDOW}
    # A complete window at the tail, but the full leaf holds nothing in front
    # of it, so there is no prefix to hang it off.
    rust = _FakeRust([(1, _h(2)), (1, _h(3))])

    transfer = _connector(leaves, rust).load(
        {"full": [10, 11, 12], "window": [20, 21]}, [_h(1), _h(2), _h(3)]
    )

    # A miss costs no load at all, not an empty one.
    assert rust.loads == []
    assert transfer.g0_blocks_per_leaf == {"full": [], "window": []}


def test_an_empty_request_asks_the_tiers_for_nothing() -> None:
    rust = _FakeRust([])

    transfer = _connector({"a": FULL}, rust).load({"a": []}, [])

    assert rust.calls == ["reclaim"]
    assert transfer.g0_blocks_per_leaf == {"a": []}


def test_a_declined_load_is_a_miss_rather_than_a_partial_row() -> None:
    hashes = [_h(1), _h(2)]
    rust = _FakeRust([(0, h) for h in hashes], decline=True)

    transfer = _connector({"a": FULL}, rust).load({"a": [7, 8]}, hashes)

    assert transfer.g0_blocks_per_leaf == {"a": []}


def test_a_partially_posted_load_fails_loudly() -> None:
    # Dropping the handle here would strand the blocks the Rust lanes pinned,
    # so a short post must not be quietly reported as a miss.
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
        _connector({"a": FULL}, rust).load({"a": [7, 8]}, hashes)


def test_a_staging_row_shorter_than_its_share_reaches_the_rust_refusal() -> (
    None
):
    hashes = [_h(1), _h(2)]
    rust = _FakeRust([(0, h) for h in hashes])

    with pytest.raises(RuntimeError, match="destination blocks"):
        _connector({"a": FULL}, rust).load({"a": [7]}, hashes)


def test_load_rejects_block_ids_that_are_not_the_connectors_leaves() -> None:
    connector = _connector({"a": FULL, "b": FULL}, _FakeRust([]))

    with pytest.raises(ValueError, match="do not match the connector's leaves"):
        connector.load({"a": [0]}, [_h(1)])


def test_a_recurrent_leaf_is_refused() -> None:
    # A recurrent leaf's hit is the deepest published state, not a run, so the
    # rules here cannot decide it. Refusing at construction beats claiming a
    # prefix whose state pages are not the ones the row needs.
    _validate_leaves({"full": FULL, "window": WINDOW})
    with pytest.raises(ValueError, match="sliding-window leaves only"):
        _validate_leaves({"full": FULL, "ssm": KVCacheGroupId.recurrent()})
