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
"""Tests for the attention-shape prefix-hit rules.

These are the rules every tier decides a cache hit with -- the device pools
through the group coordinators, and an external store through a KV connector
-- so they are exercised here against residency stated directly rather than
through either tier's plumbing.

Blocks are named A..G in the comments and indexed 0..6 in the code.
"""

from __future__ import annotations

import pytest
from max.nn.kv_cache import KVCacheGroupId
from max.pipelines.kv_cache.prefix_hit import (
    blocks_held_of_hit,
    longest_joint_prefix_hit,
)

# page_size 1, so a window size of N+1 spans N blocks and the masks below read
# as one block per entry.
PAGE = 1
FULL = KVCacheGroupId.full()


def swa(blocks_in_window: int) -> KVCacheGroupId:
    """A windowed shape whose window is ``blocks_in_window`` blocks wide."""
    return KVCacheGroupId("sliding_window", blocks_in_window + 1)


# ---------------------------------------------------------------- one group


def test_full_attention_stops_at_the_first_hole() -> None:
    # A B C held, D missing: a full group reads its whole history, so its hit
    # ends where the run from the root ends.
    mask = [True, True, True, False, True]
    assert FULL.longest_hit(PAGE, mask) == 3


def test_sliding_window_takes_the_deepest_complete_window() -> None:
    # Window 2. Both [C D] and [F G] are complete, so the deepest bound wins:
    # stopping at G needs F and G, and it has them.
    mask = [False, False, True, True, False, True, True]
    assert swa(2).longest_hit(PAGE, mask) == 7


def test_sliding_window_falls_back_to_a_root_anchored_run() -> None:
    # Window 3, but only A B are held. No complete window exists anywhere;
    # the run that survives reaches index 0, and nothing sits below the root
    # to be missing, so those two blocks are still a hit.
    mask = [True, True, False, False]
    assert swa(3).longest_hit(PAGE, mask) == 2


def test_a_window_spanning_no_whole_block_is_rejected() -> None:
    # window_size == 1: the query token attends to no history, so every block
    # would be a hit that attention never reads back. The shape refuses it at
    # construction (SERVOPT-1627), which is why the rule never sees one.
    with pytest.raises(ValueError, match="greater than 1"):
        swa(0)


def test_a_zero_width_window_reaching_the_rule_is_a_bug() -> None:
    # Built around the ctor, so this can only come from a caller that made
    # the shape some other way.
    zero = KVCacheGroupId.full()
    object.__setattr__(zero, "type", "sliding_window")
    object.__setattr__(zero, "window_size", 1)
    with pytest.raises(ValueError, match="at least one block"):
        zero.longest_hit(PAGE, [False] * 5)


# -------------------------------------------------------------- joint rules


def test_two_full_groups_agree_on_the_shorter_run() -> None:
    # The easy case, and the one a minimum handles: values hold A..F, scales
    # hold A..E, so A..E is present in both.
    values = [True] * 6 + [False]
    scales = [True] * 5 + [False, False]
    assert longest_joint_prefix_hit([(FULL, values), (FULL, scales)], PAGE) == 5


def test_full_and_windowed_agree_below_both_their_answers() -> None:
    # The case no minimum can reach. Full holds A..F; the windowed group holds
    # [C D] and [F G] with window 2, and is missing E.
    #
    #   stop at F -> window needs [E F] -> E missing
    #   stop at E -> window needs [D E] -> E missing
    #   stop at D -> window needs [C D] -> held
    #
    # Note that E rules out BOTH of the first two candidates, so the answer is
    # not reachable by shortening either group's own answer once.
    full = [True] * 6 + [False]
    window = [False, False, True, True, False, True, True]
    assert longest_joint_prefix_hit([(FULL, full), (swa(2), window)], PAGE) == 4


def test_a_window_the_full_group_cannot_reach_is_no_hit() -> None:
    # The windowed group holds a COMPLETE window, [F G], but it ends past
    # where the full group reaches. Capping the candidate at F leaves the
    # windowed group holding only F: a run of 1 against a window of 2, and
    # nothing shallower has two in a row either.
    full = [True] * 6 + [False]
    window = [False, False, False, False, False, True, True]
    assert longest_joint_prefix_hit([(FULL, full), (swa(2), window)], PAGE) == 0


def test_narrowing_reopens_a_group_already_asked() -> None:
    # Why the joint rule ITERATES rather than asking each group once. The
    # windowed group answers 2 (B alone is a complete window of 1), which
    # narrows the candidate to 2; the full group then answers 1, because its
    # run from the root is just A. Re-asking the windowed group at 1 is what
    # catches that B is gone from that candidate and A is not held: a single
    # round-robin pass stops at 1 and claims a hit the KV cannot back.
    window = [False, True]
    full = [True, False]
    assert longest_joint_prefix_hit([(swa(1), window), (FULL, full)], PAGE) == 0


def test_a_lone_windowed_group_needs_nothing_to_bound_it() -> None:
    # With no full group there is nothing to cap the candidate, so the whole
    # request is in play and the deepest complete window wins.
    window = [False, False, True, True]
    assert longest_joint_prefix_hit([(swa(2), window)], PAGE) == 4


def test_no_groups_is_no_hit() -> None:
    assert longest_joint_prefix_hit([], PAGE) == 0


# ------------------------------------------------------------ what is held


def test_a_full_group_holds_the_whole_hit() -> None:
    assert blocks_held_of_hit(4, None) == 4


def test_a_windowed_group_holds_only_its_window() -> None:
    # The distinction that is easy to lose: the HIT is 4 blocks deep, but the
    # group holds 2 of them, because the slots below its window are never
    # read. Treating the hit depth as the block count asks the group for
    # blocks it does not have.
    assert blocks_held_of_hit(4, 2) == 2


def test_a_window_wider_than_the_hit_holds_the_whole_hit() -> None:
    assert blocks_held_of_hit(2, 8) == 2


def test_masks_of_different_widths_are_refused() -> None:
    # The masks index one chain, so a short one would quietly answer about a
    # different prefix than the rest.
    with pytest.raises(ValueError, match="same chain"):
        longest_joint_prefix_hit([(FULL, [True, True]), (swa(1), [True])], PAGE)
