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

from collections.abc import Callable, Sequence

from max.pipelines.kv_cache.prefix_hit import (
    blocks_held_of_hit,
    longest_full_attention_hit,
    longest_joint_prefix_hit,
    longest_sliding_window_hit,
)


def held(mask: Sequence[bool]) -> Callable[[int], bool]:
    """A residency predicate over a positional mask."""
    return lambda idx: mask[idx]


def full_rule(mask: Sequence[bool]) -> Callable[[int], int]:
    return lambda candidate: longest_full_attention_hit(candidate, held(mask))


def window_rule(
    mask: Sequence[bool], blocks_in_window: int
) -> Callable[[int], int]:
    return lambda candidate: longest_sliding_window_hit(
        candidate, blocks_in_window, held(mask)
    )


# ---------------------------------------------------------------- one group


def test_full_attention_stops_at_the_first_hole() -> None:
    # A B C held, D missing: a full group reads its whole history, so its hit
    # ends where the run from the root ends.
    mask = [True, True, True, False, True]
    assert longest_full_attention_hit(5, held(mask)) == 3


def test_sliding_window_takes_the_deepest_complete_window() -> None:
    # Window 2. Both [C D] and [F G] are complete, so the deepest bound wins:
    # stopping at G needs F and G, and it has them.
    mask = [False, False, True, True, False, True, True]
    assert longest_sliding_window_hit(7, 2, held(mask)) == 7


def test_sliding_window_falls_back_to_a_root_anchored_run() -> None:
    # Window 3, but only A B are held. No complete window exists anywhere;
    # the run that survives reaches index 0, and nothing sits below the root
    # to be missing, so those two blocks are still a hit.
    mask = [True, True, False, False]
    assert longest_sliding_window_hit(4, 3, held(mask)) == 2


def test_a_window_spanning_no_whole_block_is_a_total_hit() -> None:
    # window_size == 1: the query token attends to no history, so nothing has
    # to be resident. Folding this onto "full attention" would instead demand
    # every block from the root.
    assert longest_sliding_window_hit(5, 0, held([False] * 5)) == 5


# -------------------------------------------------------------- joint rules


def test_two_full_groups_agree_on_the_shorter_run() -> None:
    # The easy case, and the one a minimum handles: values hold A..F, scales
    # hold A..E, so A..E is present in both.
    values = [True] * 6 + [False]
    scales = [True] * 5 + [False, False]
    assert (
        longest_joint_prefix_hit(7, [full_rule(values), full_rule(scales)]) == 5
    )


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
    assert (
        longest_joint_prefix_hit(7, [full_rule(full), window_rule(window, 2)])
        == 4
    )


def test_a_window_the_full_group_cannot_reach_is_no_hit() -> None:
    # The windowed group holds a COMPLETE window, [F G], but it ends past
    # where the full group reaches. Capping the candidate at F leaves the
    # windowed group holding only F: a run of 1 against a window of 2, and
    # nothing shallower has two in a row either.
    full = [True] * 6 + [False]
    window = [False, False, False, False, False, True, True]
    assert (
        longest_joint_prefix_hit(7, [full_rule(full), window_rule(window, 2)])
        == 0
    )


def test_narrowing_reopens_a_group_already_asked() -> None:
    # Why the joint rule ITERATES rather than asking each group once. The
    # windowed group answers 2 (B alone is a complete window of 1), which
    # narrows the candidate to 2; the full group then answers 1, because its
    # run from the root is just A. Re-asking the windowed group at 1 is what
    # catches that B is gone from that candidate and A is not held: a single
    # round-robin pass stops at 1 and claims a hit the KV cannot back.
    window = [False, True]
    full = [True, False]
    assert (
        longest_joint_prefix_hit(2, [window_rule(window, 1), full_rule(full)])
        == 0
    )


def test_a_lone_windowed_group_needs_nothing_to_bound_it() -> None:
    # With no full group there is nothing to cap the candidate, so the whole
    # request is in play and the deepest complete window wins.
    window = [False, False, True, True]
    assert longest_joint_prefix_hit(4, [window_rule(window, 2)]) == 4


def test_no_groups_is_no_hit() -> None:
    assert longest_joint_prefix_hit(5, []) == 0


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
