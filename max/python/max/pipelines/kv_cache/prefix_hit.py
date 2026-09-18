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

"""Which prefix of a hash chain a cache tree can reuse.

The rules here are about attention shape alone. They take residency as a
predicate, so the same code answers for any tier: the device pools
(:mod:`.paged_kv_cache.kv_group_coordinator`), or a KV connector reporting
what an external store holds (:meth:`.connectors.dkv.DKVConnector.load`).
They live beside :mod:`.kv_connector` rather than under
:mod:`.paged_kv_cache` so that both sides can reach them. A connector
therefore needs no notion of full / sliding-window / SSM -- it reports
presence, and this decides what that presence is worth.

Two shapes, and they behave differently under narrowing:

* a **full-attention** group reads its whole history, so its hit is the run
  from the root and shortening an answer can never invalidate it;
* a **sliding-window** group reads only the last ``blocks_in_window`` blocks
  before wherever the sequence stops, so its hit is a SUFFIX run whose
  validity moves with the stopping point. Shortening can invalidate it.

That second property is why :func:`longest_joint_prefix_hit` iterates instead
of taking a minimum: narrowing for one group can invalidate another's window,
which narrows it again.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

__all__ = [
    "blocks_held_of_hit",
    "longest_full_attention_hit",
    "longest_joint_prefix_hit",
    "longest_sliding_window_hit",
]


def longest_full_attention_hit(
    num_hashes: int, is_cached: Callable[[int], bool]
) -> int:
    """The run of cached blocks from the root, which is what a full group reads.

    Args:
        num_hashes: How many blocks the request wants.
        is_cached: Whether the block at an index is held.

    Returns:
        The prefix length. Downward-closed: a group serving ``n`` also serves
        anything shorter.
    """
    for idx in range(num_hashes):
        if not is_cached(idx):
            return idx
    return num_hashes


def longest_sliding_window_hit(
    num_hashes: int, blocks_in_window: int, is_cached: Callable[[int], bool]
) -> int:
    """The deepest stopping point whose window this group holds complete.

    Walks backwards counting a consecutive run, because the window sits at the
    END of the candidate: the first place the run reaches ``blocks_in_window``
    is the deepest stopping point that works. A shallower one may not -- its
    window covers different blocks -- so this cannot be found by shortening a
    full-attention answer.

    A run that survives to index 0 is a hit of just that run, with no window
    check: nothing sits below the root to be missing.

    ``blocks_in_window`` is at least 1 for every group that gets here. Zero is
    ``window_size == 1``, a query token attending to no history, which would
    make every block a hit that attention never reads back;
    :class:`~max.nn.kv_cache.cache_params.KVCacheGroupId` rejects that window
    and so does Mach's ``KVCacheConfig::validate``, so a zero arriving here is
    a caller bug rather than a shape to serve (SERVOPT-1627).

    Raises:
        ValueError: If ``blocks_in_window`` is not positive.
    """
    if blocks_in_window < 1:
        raise ValueError(
            "A sliding-window group spans at least one block; got"
            f" blocks_in_window={blocks_in_window}. window_size must be"
            " greater than 1."
        )

    run = 0
    for idx in range(num_hashes - 1, -1, -1):
        if not is_cached(idx):
            # The run is broken. Reset the run counter.
            run = 0
            continue
        run += 1
        if run >= blocks_in_window:
            return idx + run
    return run


def longest_joint_prefix_hit(
    num_hashes: int, group_hits: Sequence[Callable[[int], int]]
) -> int:
    """The longest prefix EVERY group can serve at once.

    Each group answers under the run the others have already allowed, so the
    run is settled once every group has accepted it in turn. A group asked
    again about a prefix it just returned has to return that same length, or
    the loop would walk the run to nothing.

    Iterating is required rather than tidy. A minimum over the groups' answers
    is wrong twice over: a windowed group's answer is not a depth that can be
    compared with a full group's, and narrowing for one windowed group can
    invalidate another's window, which narrows it again.

    Args:
        num_hashes: How many blocks the request wants.
        group_hits: One callable per group, each taking a candidate length and
            returning how much of that candidate the group can serve.

    Returns:
        The agreed prefix length; ``0`` when the groups cannot agree on any.
    """
    if not group_hits:
        return 0

    candidate = num_hashes
    accepted = 0
    turn = 0
    while candidate and accepted < len(group_hits):
        num_hit_blocks = group_hits[turn](candidate)
        accepted = accepted + 1 if num_hit_blocks == candidate else 1
        candidate = num_hit_blocks
        turn = (turn + 1) % len(group_hits)
    return candidate


def blocks_held_of_hit(
    num_hit_blocks: int, blocks_in_window: int | None
) -> int:
    """How many blocks a group actually holds of a hit that deep.

    The companion to the rules above, and not the same number: a windowed
    group's hit can cover a whole prefix while the group holds only the window
    at the end of it, because the slots below the window are never read. A full
    group holds all of it.

    Every group's blocks END at ``num_hit_blocks``, so a group holding ``n`` of
    them covers ``hashes[num_hit_blocks - n : num_hit_blocks]`` and the
    ``num_hit_blocks - n`` slots below are the null block.

    Args:
        num_hit_blocks: The agreed prefix length.
        blocks_in_window: The group's window in whole blocks, or ``None`` for a
            group that attends over its whole history.
    """
    if blocks_in_window is None:
        return num_hit_blocks
    return min(num_hit_blocks, blocks_in_window)
