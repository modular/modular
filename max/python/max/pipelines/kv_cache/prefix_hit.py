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

Reconciling one leaf's shape against a residency mask is
:meth:`~max.nn.kv_cache.KVCacheGroupId.longest_hit`, which lives on the
shape itself. This module settles what a whole TREE of them serves at once,
for any tier: the device pools
(:mod:`.paged_kv_cache.kv_group_coordinator`), or a KV connector reporting
what an external store holds. It lives beside :mod:`.kv_connector` rather
than under :mod:`.paged_kv_cache` so both sides can reach it. A connector
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

from collections.abc import Sequence

from max.nn.kv_cache import KVCacheGroupId

__all__ = [
    "blocks_held_of_hit",
    "longest_joint_prefix_hit",
]


def longest_joint_prefix_hit(
    leaf_hits: Sequence[tuple[KVCacheGroupId, Sequence[bool]]],
    page_size: int,
) -> int:
    """The longest prefix EVERY leaf can serve at once.

    Each leaf answers under the run the others have already allowed, so the
    run is settled once every leaf has accepted it in turn. A leaf asked
    again about a prefix it just returned has to return that same length, or
    the loop would walk the run to nothing.

    Iterating is required rather than tidy. A minimum over the leaves'
    answers is wrong twice over: a windowed leaf's answer is not a depth that
    can be compared with a full leaf's, and narrowing for one windowed leaf
    can invalidate another's window, which narrows it again.

    Args:
        leaf_hits: One ``(shape, resident)`` pair per leaf. A sequence rather
            than a mapping keyed by shape, because leaves routinely share one
            -- an FP8 tree's values and scales are both full attention -- and
            each still answers from its own mask.
        page_size: Tokens per block, which turns a window size into a block
            count.

    Returns:
        The agreed prefix length; ``0`` when the leaves cannot agree on any.

    Raises:
        ValueError: If the masks are not all the same width. They index one
            chain of hashes, so a short one would quietly answer about a
            different prefix than the rest.
    """
    if not leaf_hits:
        return 0
    widths = {len(resident) for _, resident in leaf_hits}
    if len(widths) > 1:
        raise ValueError(
            f"every leaf's mask covers the same chain; got widths {sorted(widths)}"
        )

    candidate = widths.pop()
    accepted = 0
    turn = 0
    while candidate and accepted < len(leaf_hits):
        group_id, resident = leaf_hits[turn]
        # The slice is what narrowing means: `longest_hit` reads the
        # candidate off the mask it is given.
        num_hit_blocks = group_id.longest_hit(page_size, resident[:candidate])
        accepted = accepted + 1 if num_hit_blocks == candidate else 1
        candidate = num_hit_blocks
        turn = (turn + 1) % len(leaf_hits)
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
