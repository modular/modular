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
"""Host-side KV cache bookkeeping shared by MAX and Mach.

A compiled extension; this stub is its public surface. It holds the
prefix-hit rules below and the Jenga block pool (:class:`JengaBlockPool`).

Which prefix of a hash chain a cache tree can reuse depends on how far back
each leaf's attention reads. The rules take residency alone, so they answer
for any tier: the device pools, or a KV connector reporting what an external
store holds. A connector therefore needs no notion of full / sliding-window /
SSM: it reports presence, and these rules decide what that presence is worth.

Two shapes, and they behave differently under narrowing:

* a full-attention leaf reads its whole history, so its hit is the run from
  the root and shortening an answer can never invalidate it;
* a sliding-window leaf reads only the last ``blocks_in_window`` blocks
  before wherever the sequence stops, so its hit is a SUFFIX run whose
  validity moves with the stopping point. Shortening can invalidate it.

That second property is why :func:`longest_joint_prefix_hit` iterates instead
of taking a minimum: narrowing for one leaf can invalidate another's window,
which narrows it again.
"""

from collections.abc import Iterator, Mapping, Sequence
from typing import TypeAlias, TypeVar, final, overload

import numpy as np
import numpy.typing as npt

# Whether each block of a hash chain is held, positionally. Stub-only: the
# extension has no such attribute.
_Residency: TypeAlias = Sequence[bool] | npt.NDArray[np.bool_]
_T = TypeVar("_T")

@final
class LeafShape:
    """A leaf's attention shape, which selects the rule that reads its residency.

    Built only through the static constructors.
    """

    @staticmethod
    def full() -> LeafShape:
        """Returns the shape of a leaf that reads its whole history."""

    @staticmethod
    def sliding_window(blocks_in_window: int) -> LeafShape:
        """Returns the shape of a leaf that reads its last ``blocks_in_window`` blocks.

        Raises:
            ValueError: If ``blocks_in_window`` is zero.
        """

    @staticmethod
    def recurrent() -> LeafShape:
        """Returns the shape of a leaf that holds one state per boundary.

        Any resident boundary is a complete hit, so the deepest one wins.
        """

    @staticmethod
    def scratch() -> LeafShape:
        """Returns the shape of a leaf that is never published.

        Such a leaf never narrows a hit.
        """

    def longest_hit(self, resident: _Residency) -> int:
        """Returns how much of ``resident`` a leaf of this shape can reuse.

        The answer is counted from the start even for a shape that only reads
        the tail, so shapes can be compared with one another.
        ``len(resident)`` bounds it, so narrowing means passing a shorter view.
        """

    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

def longest_joint_prefix_hit(
    leaves: Sequence[tuple[LeafShape, _Residency]],
) -> int:
    """Returns the longest prefix every leaf can serve at once.

    Each leaf answers under the run the others have already allowed, so the
    run is settled once every leaf has accepted it in turn. A leaf asked
    again about a prefix it just returned has to return that same length, or
    the loop would walk the run to nothing.

    Iterating is required rather than tidy. A minimum over the leaves'
    answers is wrong twice over: a windowed leaf's answer is not a depth that
    can be compared with a full leaf's, and narrowing for one windowed leaf
    can invalidate another's window, which narrows it again.

    Args:
        leaves: One ``(shape, resident)`` pair per leaf. A sequence rather
            than a mapping, because leaves routinely share a shape (an FP8
            tree's values and scales are both full attention) and each
            still answers from its own mask.

    Returns:
        The agreed prefix length; ``0`` when there are no leaves or they
        cannot agree on any.

    Raises:
        ValueError: If the masks are not all the same width. They index one
            chain of hashes, so a short one would quietly answer about a
            different prefix than the rest.
    """

def blocks_held_of_hit(
    num_hit_blocks: int, blocks_in_window: int | None
) -> int:
    """Returns how many blocks a leaf holds of a hit ``num_hit_blocks`` deep.

    The companion to the prefix-hit rules, and not the same number: a
    windowed leaf's hit can cover a whole prefix while the leaf holds only
    the window at the end of it, because the slots below the window are never
    read. A full leaf (``blocks_in_window=None``) holds all of it.

    Every leaf's blocks END at ``num_hit_blocks``, so a leaf holding ``n`` of
    them covers ``hashes[num_hit_blocks - n : num_hit_blocks]`` and the
    ``num_hit_blocks - n`` slots below are the null block.
    """

class InsufficientBlocksError(Exception):
    """Exception raised when there are insufficient free blocks to satisfy an allocation.

    Fragmentation can raise it while other caches still hold little blocks, so
    it is a signal to preempt, not a bug.
    """

@final
class LittleKVCacheBlock:
    """One cache's page: block ``bid`` of cache ``cache_id``.

    A handle on the pool's state, not a copy of it: ``ref_cnt`` and
    ``block_hash`` read the pool as it is now. A pool hands out the same
    object for a block every time, so blocks compare with ``is``.
    """

    @property
    def bid(self) -> int:
        """The block's page index into its cache's view of the pool buffer.

        The little blocks of huge block ``h`` are
        ``[h * ratio, (h + 1) * ratio)``, so ``bid // ratio`` is the huge
        block whose bytes back this one.
        """

    @property
    def cache_id(self) -> str:
        """The cache (leaf) this block belongs to."""

    @property
    def is_null(self) -> bool:
        """Whether this is the null block, ``bid`` 0, that dummy and padding requests share."""

    @property
    def ref_cnt(self) -> int:
        """How many requests hold this block.

        Always 0 for the null block, whose references are never counted.

        Raises:
            RuntimeError: If the block's pool no longer exists, as does
                :attr:`block_hash`.
        """

    @property
    def block_hash(self) -> bytes | None:
        """The hash this block is committed under, once it is full."""

@final
class PrefixCache:
    """One cache's committed blocks, by hash: a live, read-only mapping.

    It is not a ``dict``: ``keys()``, ``values()`` and ``items()`` return
    lists, and it never compares equal to a ``dict``, so test emptiness with
    ``not``.
    """

    def __len__(self) -> int: ...
    def __contains__(self, block_hash: object) -> bool: ...
    def __getitem__(self, block_hash: bytes) -> LittleKVCacheBlock: ...
    def __iter__(self) -> Iterator[bytes]:
        """Iterates over a snapshot of the committed hashes."""

    @overload
    def get(self, block_hash: bytes) -> LittleKVCacheBlock | None: ...
    @overload
    def get(
        self, block_hash: bytes, default: _T
    ) -> LittleKVCacheBlock | _T: ...
    def keys(self) -> list[bytes]: ...
    def values(self) -> list[LittleKVCacheBlock]: ...
    def items(self) -> list[tuple[bytes, LittleKVCacheBlock]]: ...

@final
class JengaBlockPool:
    """A pool of huge blocks, each subdividable into one cache's little blocks.

    Every cache tiles the same bytes at its own page size, so a huge block is
    ``cache_ratios[cache_id]`` blocks of cache ``cache_id``. Huge block 0 is
    spent on the null block (``N``) that dummy and padding requests share,
    which leaves the rest of it (``.``) unusable, and starts real ids at 1 and
    at ``ratio``::

      huge block     |     0     |     1     |     2     |     3     |
      global  (x4)   | N| .| .| .| 4| 5| 6| 7| 8| 9|10|11|12|13|14|15|
      sliding (x2)   |  N  |  .  |  2  |  3  |  4  |  5  |  6  |  7  |

    A little block's ``bid`` is thus its index in its own cache's id space.
    ``num_huge_blocks`` counts huge block 0, so a pool needs at least two of
    them to hand anything out.

    Those views alias, so a huge block serves one cache at a time -- its
    ``little_block_type`` -- and at most one row of each column exists. Here
    the global cache holds huge block 1, the sliding cache holds 2, and 3 is
    free for either of them to claim::

      huge block     |     0     |     1     |     2     |     3     |
      global  (x4)   | N| .| .| .| 4| 5| 6| 7| - - - - - | - - - - - |
      sliding (x2)   |  N  |  .  | - - - - - |  4  |  5  | - - - - - |

    Bytes change hands only while nothing references them, so the split
    between caches follows live demand rather than a knob. A huge block is
    therefore always in one of two states::

      parked                                      claimed by cache c
      +-------------------------------+           +--------------------------+
      | ref_cnt == 0                  |   claim   | ref_cnt >= 1             |
      | in free_huge_blocks           |  ------>  | little_block_type == c   |
      | any cache may claim it        |  <------  | only c allocates from it |
      | commits still in prefix cache |   park    |                          |
      +-------------------------------+           +--------------------------+

    Parked means no request holds any of its little blocks, so the bytes are
    up for grabs. Claimed means one cache owns them: each of that cache's
    little blocks in the huge block is either referenced by a request or
    queued in that cache's free list as an eviction candidate, never both and
    never neither.

    :meth:`alloc_block` claims a parked block when its cache has no free
    little block left, and :meth:`touch` claims one back when a prefix hit
    takes its reference count up from 0. :meth:`free_block` parks the block
    again as soon as its last reference goes away, which is what lets another
    cache reuse the bytes.

    Parking keeps commits: a parked block's little blocks stay in their
    cache's prefix cache, so that cache can reclaim it and still hit them.
    Only another cache claiming the bytes evicts them.

    Implemented in ``kv-core`` (``kv_core::pool``), which the tiered
    connector's host pool also runs on. Each block operation crosses into
    Rust once; a block of another pool is refused with ``ValueError``.
    Breaking an invariant (a double free, committing the null block, touching
    a block whose bytes another cache now holds) raises
    ``pyo3_runtime.PanicException``, a ``BaseException`` rather than an
    ``Exception``.

    Args:
        num_huge_blocks: How many huge blocks the pool tiles, huge block 0
            (the null block) included.
        cache_ratios: How many little blocks of each cache fit in a huge
            block, by cache id.

    Raises:
        ValueError: If ``num_huge_blocks`` is below 2, or ``cache_ratios`` is
            empty or not all positive.
    """

    def __init__(
        self, num_huge_blocks: int, cache_ratios: Mapping[str, int]
    ) -> None: ...
    @property
    def num_huge_blocks(self) -> int:
        """How many huge blocks the pool tiles, the null block's included."""

    @property
    def num_allocatable_huge_blocks(self) -> int:
        """How many huge blocks allocation can hand out: all but the null block's."""

    @property
    def num_free_huge_blocks(self) -> int:
        """How many huge blocks are parked."""

    @property
    def cache_ratios(self) -> dict[str, int]:
        """How many little blocks of each cache fit in a huge block (a copy)."""

    @property
    def prefix_caches(self) -> dict[str, PrefixCache]:
        """Each cache's committed blocks, by hash."""

    @property
    def null_little_blocks(self) -> dict[str, LittleKVCacheBlock]:
        """Each cache's null block, which dummy and padding requests share.

        Its reference count is never counted, so no path can free it, evict
        it, or hand it out.
        """

    def alloc_block(self, cache_id: str) -> LittleKVCacheBlock:
        """Returns a fresh block of ``cache_id``, claiming huge blocks as needed.

        Prefers claiming a pristine huge block over handing out a free little
        block that is still committed, so a prefix is evicted only when
        nothing else is left. Handing out a committed block evicts it: its
        bytes are about to be overwritten.

        Raises:
            InsufficientBlocksError: If the pool has no bytes left to serve
                this cache.
        """

    def free_block(self, block: LittleKVCacheBlock) -> None:
        """Drops one reference, parking the huge block once it holds none.

        Freeing the null block does nothing. A block that was never committed
        goes to the front of its cache's free list and is reused first; a
        committed one goes to the back, so the least recently used commit is
        evicted first.
        """

    def touch(self, block: LittleKVCacheBlock) -> None:
        """Takes a reference on a block, reviving it if it was out of use.

        Reviving a block whose huge block is parked claims the huge block back
        for its cache. Touching the null block does nothing.
        """

    def uncommit_block(self, block: LittleKVCacheBlock) -> None:
        """Drops a block from its cache's prefix cache, if it is committed."""

    def commit_into_prefix_cache(
        self, block_hash: bytes, block: LittleKVCacheBlock
    ) -> None:
        """Makes a filled block reusable by anyone hashing the same tokens.

        Raises:
            TypeError: If ``block_hash`` is not ``bytes``.
            ValueError: If ``block_hash`` is over 32 bytes, as for
                :meth:`get_or_commit_into_prefix_cache`.
        """

    def get_or_commit_into_prefix_cache(
        self, block_hash: bytes, block: LittleKVCacheBlock
    ) -> LittleKVCacheBlock | None:
        """Commits a block, or returns the twin already holding its bytes.

        Returns:
            The committed block to use instead of ``block``, which has been
            freed, or ``None`` if ``block`` itself now serves the hash.
        """

    def block(self, cache_id: str, bid: int) -> LittleKVCacheBlock:
        """Returns the little block ``bid`` of ``cache_id``.

        Raises:
            IndexError: If ``cache_id`` has no such block. Of huge block 0
                only bid 0, the null block, exists.
        """

    def num_free_blocks(self, cache_id: str) -> int:
        """Returns how many more blocks of ``cache_id`` the pool can still serve.

        Counts this cache's free little blocks plus every parked huge block
        not already typed to it, at its ratio. Each cache's count assumes it
        alone may retype those huge blocks, so the counts of several caches do
        not add up; ask :meth:`can_satisfy_demand` for a joint answer.
        """

    def can_satisfy_demand(
        self, demand: Mapping[str, int], at_capacity: bool = False
    ) -> bool:
        """Returns whether the pool can allocate the demanded number of little blocks.

        Every cache is charged to the same huge blocks. ``at_capacity`` asks
        the same of a pool that has handed nothing out yet, making the answer
        a property of the pool's geometry rather than of what it currently
        holds. A count of 0 or less asks for nothing.
        """

    def reset_prefix_cache(self) -> dict[str, int]:
        """Drops every commit no request is holding, in every cache.

        A commit a request still references survives, because its block
        cannot be handed out while it is in use.

        Returns:
            How many blocks were purged from each cache's prefix cache.
        """
