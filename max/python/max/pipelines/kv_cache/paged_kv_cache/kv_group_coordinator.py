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
"""Cache group coordinators: one per set of leaves written in lockstep."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from max.nn.kv_cache import KVCacheGroupId
from max.pipelines.context import TextContext
from max.pipelines.modeling.types import RequestID

from ..prefix_hit import blocks_held_of_hit
from .block_utils import LittleKVCacheBlock
from .jenga_block_pool import JengaBlockPool

__all__ = [
    "FullKVGroupCoordinator",
    "KVGroupCoordinatorInterface",
    "SlidingWindowKVGroupCoordinator",
]


@dataclass(frozen=True)
class KVGroupCoordinatorInterface:
    """Finds and claims the prefix-cache hit one leaf can serve.

    How deep it can resume depends on how far back its attention reads: a
    full leaf resumes its run from the root, a windowed one the window ending
    at the prefix.

    One of these per LEAF. It used to be one per attention group, holding
    every leaf of that group and ANDing residency across them, which is the
    same reconcile the manager runs across coordinators -- so the group-level
    copy is gone.

    TODO: rename this and its subclasses now that "group" means "leaf".

    The lifecycle every group implements, in order:

    * :meth:`claim`: a request arrives.
    * :meth:`longest_cache_hit`, :meth:`claim_hit_blocks`: find a prefix hit
      and take it.
    * :meth:`blocks_to_allocate`, :meth:`grow`: size and draw the next
      forward's blocks; :meth:`grow_with_padding` for a padding dummy.
    * :meth:`forward_blocks`: name the blocks the next forward touches.
    * :meth:`resume`: fill the block the next forward runs in.
    * :meth:`checkpoint`: fill the successor of the block just run in.
    * :meth:`commit`: publish what the forward filled.
    * :meth:`advance`: release what the group no longer reads.
    * :meth:`shrink_to_fit`: trim the row to the committed prefix.
    * :meth:`release`: the request is done.
    """

    pools: Sequence[JengaBlockPool]
    """The block pools this group draws from, one per data-parallel
    replica."""
    leaf_id: str
    """The pool leaf whose blocks this manages."""
    group_id: KVCacheGroupId
    page_size: int
    """Tokens per page, which turns a window width into a block count."""

    rows: dict[RequestID, dict[str, list[LittleKVCacheBlock]]] = field(
        default_factory=dict, kw_only=True
    )
    """Each live request's blocks, per leaf."""

    def find_replica_with_hash(
        self,
        block_hash: bytes,
        replica_idx: int,
        allow_cross_replica: bool = False,
    ) -> int | None:
        """The replica to serve ``block_hash`` to ``replica_idx`` from.

        ``replica_idx`` is checked first, so a hash it already holds is never
        copied. ``allow_cross_replica`` then widens the search to the other
        replicas, whose pages are copied over before use; without it a local
        miss is simply a miss.

        Returns:
            The replica to read the block from, or None when no replica the
            search covered holds it.
        """
        cache = self.pools[replica_idx].prefix_caches[self.leaf_id]
        if block_hash in cache:
            return replica_idx
        if not allow_cross_replica:
            return None
        holders = [
            candidate
            for candidate in range(len(self.pools))
            if candidate != replica_idx
            and block_hash in self.pools[candidate].prefix_caches[self.leaf_id]
        ]
        if not holders:
            return None
        # Taking the first holder would make the lowest-indexed one serve every
        # copy of a popular prefix. Keying the choice on the hash and the
        # destination splits the reads across the holders, and across the ranks
        # reading the same block, while staying stable per block.
        rotation = int.from_bytes(block_hash, "little") + replica_idx
        return holders[rotation % len(holders)]

    def claimable_hashes(
        self, desired_hashes: Sequence[bytes]
    ) -> Sequence[bytes]:
        """Which of ``desired_hashes`` this group would claim as a hit."""
        raise NotImplementedError("Subclasses must implement this method.")

    def residency(
        self,
        desired_hashes: Sequence[bytes],
        replica_idx: int,
        allow_cross_replica: bool = False,
    ) -> list[bool]:
        """Which of ``desired_hashes`` this leaf can serve, positionally."""
        return [
            self.find_replica_with_hash(
                block_hash, replica_idx, allow_cross_replica
            )
            is not None
            for block_hash in desired_hashes
        ]

    def longest_cache_hit(
        self,
        desired_hashes: Sequence[bytes],
        replica_idx: int,
        allow_cross_replica: bool = False,
    ) -> int:
        """Returns how many of ``desired_hashes`` this group can reuse.

        Counted from the start, so the answer is a prefix length.

        Asked again about a prefix it just returned, a group has to return
        that same length. The manager cycles through the groups, shortening
        the run to each answer until they all agree, so a group that
        shrinks a prefix it already accepted would take the run to nothing.

        Args:
            desired_hashes: The blocks the request wants, starting at its
                committed index.
            replica_idx: Which replica's pool to read.
            allow_cross_replica: Whether hashes held only by another replica
                count as hits.
        """
        # The shape owns the rule, so every coordinator answers the same way
        # and a connector's masks drive the identical code.
        return self.group_id.longest_hit(
            self.page_size,
            self.residency(desired_hashes, replica_idx, allow_cross_replica),
        )

    def claim_hit_blocks(
        self,
        desired_hashes: Sequence[bytes],
        replica_idx: int,
    ) -> dict[str, list[LittleKVCacheBlock]]:
        """Claims the blocks for the given hashes."""
        raise NotImplementedError("Subclasses must implement this method.")

    def claim(self, req_id: RequestID) -> None:
        """Starts an empty row in every leaf of the group."""
        self.rows[req_id] = {self.leaf_id: []}

    def release(self, req_id: RequestID, replica_idx: int) -> None:
        """Frees every block the request holds in this group."""
        pool = self.pools[replica_idx]
        for blocks in self.rows.pop(req_id, {}).values():
            # Free in reverse so the tail is evicted before the shared head.
            for block in reversed(blocks):
                pool.free_block(block)

    def blocks_of(
        self, req_id: RequestID
    ) -> dict[str, list[LittleKVCacheBlock]]:
        """Returns the request's blocks, per leaf."""
        return self.rows[req_id]

    def extend(
        self,
        req_id: RequestID,
        hit_blocks: Mapping[str, Sequence[LittleKVCacheBlock]],
        loaded_blocks: Mapping[str, Sequence[LittleKVCacheBlock]],
        replica_idx: int,
    ) -> None:
        """Appends the blocks a prefix hit found to the end of each row.

        The device blocks come first, then the ones an external tier
        loaded, matching the order their hashes were hashed in.

        Args:
            req_id: The request to extend.
            hit_blocks: Blocks found in this device's cache, per leaf.
            loaded_blocks: Blocks loaded from an external tier, per leaf.
            replica_idx: Which pool the blocks came from.
        """
        row = self.rows[req_id]
        leaf_id = self.leaf_id
        # A leaf read by row has nothing an external tier can onload.
        row[leaf_id].extend(
            [*hit_blocks[leaf_id], *loaded_blocks.get(leaf_id, ())]
        )

    def grow_with_padding(
        self, req_id: RequestID, num_required_blocks: int, replica_idx: int
    ) -> None:
        """Points every row at the null block."""
        pool = self.pools[replica_idx]
        self.rows[req_id] = {
            self.leaf_id: (
                [pool.null_little_blocks[self.leaf_id]] * num_required_blocks
            )
        }

    def shrink_to_fit(
        self, req_id: RequestID, num_committed_blocks: int, replica_idx: int
    ) -> None:
        """Drops the blocks past the committed index."""
        pool = self.pools[replica_idx]
        for req_blocks in self.rows[req_id].values():
            assert len(req_blocks) >= num_committed_blocks
            for _ in range(len(req_blocks) - num_committed_blocks):
                pool.free_block(req_blocks.pop())

    def _num_blocks_to_allocate(
        self, row: Sequence[LittleKVCacheBlock], num_required_blocks: int
    ) -> int:
        """Returns how many blocks the row still needs."""
        return max(num_required_blocks - len(row), 0)

    def blocks_to_allocate(
        self, req_id: RequestID, num_required_blocks: int
    ) -> dict[str, int]:
        """Returns how many blocks each leaf must be given."""
        row = self.rows[req_id]
        return {
            self.leaf_id: self._num_blocks_to_allocate(
                row[self.leaf_id], num_required_blocks
            )
        }

    def grow(
        self, req_id: RequestID, num_required_blocks: int, replica_idx: int
    ) -> None:
        """Allocates the blocks every row still needs."""
        pool = self.pools[replica_idx]
        leaf_id = self.leaf_id
        req_blocks = self.rows[req_id][leaf_id]
        for _ in range(
            self._num_blocks_to_allocate(req_blocks, num_required_blocks)
        ):
            req_blocks.append(pool.alloc_block(leaf_id))

    def _is_committable(
        self, row: Sequence[LittleKVCacheBlock], block_idx: int
    ) -> bool:
        """Whether the block at ``block_idx`` may be published."""
        block = row[block_idx]
        return not block.is_null and block.block_hash is None

    def commit(
        self,
        req_id: RequestID,
        hashes: Sequence[bytes],
        last_block: int,
        replica_idx: int,
    ) -> None:
        """Commits every uncommitted block below ``last_block``.

        Scans from the start of the row, not just this forward's blocks: a
        group can hold a block whose hash the chain only reaches later.

        Args:
            req_id: The request whose blocks to commit.
            hashes: The request's block hashes, by block index.
            last_block: One past the last block index to commit.
            replica_idx: Which pool the blocks belong to.
        """
        pool = self.pools[replica_idx]
        req_blocks = self.rows[req_id][self.leaf_id]
        for block_idx in range(min(last_block, len(req_blocks))):
            if not self._is_committable(req_blocks, block_idx):
                continue
            twin = pool.get_or_commit_into_prefix_cache(
                hashes[block_idx], req_blocks[block_idx]
            )
            if twin is not None:
                req_blocks[block_idx] = twin

    def advance(
        self,
        req_id: RequestID,
        num_committed_blocks: int,
        replica_idx: int,
    ) -> None:
        """Frees the blocks the group no longer reads, nulling their slots."""
        raise NotImplementedError("Subclasses must implement this method.")

    def blocks_held_of_connector_hit(self, num_hit_blocks: int) -> int:
        """How many blocks of a hit that deep this group actually holds.

        Not the hit depth: a windowed group's hit can cover a whole prefix
        while the group holds only the window at the end of it, because the
        slots below the window are never read. The manager sizes each leaf's
        staging row from this and nulls the slots below it.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def forward_blocks(
        self, batch: Sequence[TextContext], num_blocks: Sequence[int]
    ) -> dict[str, list[list[int]]]:
        """Returns the blocks each leaf's forward touches, per request.

        Args:
            batch: The requests the next forward runs, in row order.
            num_blocks: How far into each request's row the forward reaches.
        """
        plans: dict[str, list[list[int]]] = {self.leaf_id: []}
        for batch_idx, ctx in enumerate(batch):
            required = num_blocks[batch_idx]
            row = self.rows[ctx.request_id]
            leaf_id = self.leaf_id
            blocks = row[leaf_id]
            assert len(blocks) >= required, (
                f"leaf {leaf_id!r} holds {len(blocks)} blocks, needs {required}"
            )
            plans[leaf_id].append([b.bid for b in blocks[:required]])
        return plans

    def resume(
        self, ctx: TextContext, replica_idx: int
    ) -> Mapping[str, tuple[int | None, int]]:
        """Returns the block each leaf's next forward is filled from.

        The pair is the block to read and the block to write, a ``None``
        source meaning zeros. :meth:`checkpoint` returns the same pair.

        Empty for a group whose blocks a forward only appends to, which is
        every group whose entry is a page of tokens rather than a state.

        Args:
            ctx: The request whose forward runs next.
            replica_idx: Which pool its blocks belong to.
        """
        return {}

    def checkpoint(
        self, ctx: TextContext, replica_idx: int
    ) -> Mapping[str, tuple[int | None, int]]:
        """Returns the block each leaf carries onto its successor.

        The same pair :meth:`resume` returns, read the same way, though a
        checkpoint always has a block to copy.

        Empty for a group whose blocks are not overwritten in place. The leaf
        folds each pair into the rows the copy runs over.

        Args:
            ctx: The request whose forward just ran.
            replica_idx: Which pool its blocks belong to.
        """
        return {}


@dataclass(frozen=True)
class FullKVGroupCoordinator(KVGroupCoordinatorInterface):
    """A group whose caches read their whole history."""

    def claimable_hashes(
        self, desired_hashes: Sequence[bytes]
    ) -> Sequence[bytes]:
        """Every hash: this group reads its whole history."""
        return desired_hashes

    def claim_hit_blocks(
        self,
        desired_hashes: Sequence[bytes],
        replica_idx: int,
    ) -> dict[str, list[LittleKVCacheBlock]]:
        """Adopts every block of the hit: the group reads its whole history."""
        pool = self.pools[replica_idx]
        rows: dict[str, list[LittleKVCacheBlock]] = {self.leaf_id: []}
        for block_hash in desired_hashes:
            leaf_id = self.leaf_id
            block = pool.prefix_caches[leaf_id][block_hash]
            pool.touch(block)
            rows[leaf_id].append(block)
        return rows

    def advance(
        self,
        req_id: RequestID,
        num_committed_blocks: int,
        replica_idx: int,
    ) -> None:
        """Keeps every block: this group reads its whole history."""
        return

    def blocks_held_of_connector_hit(self, num_hit_blocks: int) -> int:
        """All of it: this group reads its whole history."""
        return blocks_held_of_hit(num_hit_blocks, None)


@dataclass(frozen=True)
class SlidingWindowKVGroupCoordinator(KVGroupCoordinatorInterface):
    """A group that needs ``blocks_in_window`` consecutive blocks for a hit."""

    window_size: int
    """The sliding window's width in tokens."""

    @property
    def _blocks_in_window(self) -> int:
        return self.group_id.blocks_in_window(self.page_size)

    def claimable_hashes(
        self, desired_hashes: Sequence[bytes]
    ) -> Sequence[bytes]:
        """Only the window: this group has slid past everything below it."""
        low = max(0, len(desired_hashes) - self._blocks_in_window)
        return desired_hashes[low:]

    def claim_hit_blocks(
        self,
        desired_hashes: Sequence[bytes],
        replica_idx: int,
    ) -> dict[str, list[LittleKVCacheBlock]]:
        """Adopts the window ending at the hit and nulls every slot below it."""
        pool = self.pools[replica_idx]
        cache = pool.prefix_caches[self.leaf_id]
        low = max(0, len(desired_hashes) - self._blocks_in_window)
        if not all(block_hash in cache for block_hash in desired_hashes[low:]):
            low = len(desired_hashes)

        rows: dict[str, list[LittleKVCacheBlock]] = {
            self.leaf_id: [pool.null_little_blocks[self.leaf_id]] * low
        }
        for block_hash in desired_hashes[low:]:
            leaf_id = self.leaf_id
            block = pool.prefix_caches[leaf_id][block_hash]
            pool.touch(block)
            rows[leaf_id].append(block)
        return rows

    def advance(
        self,
        req_id: RequestID,
        num_committed_blocks: int,
        replica_idx: int,
    ) -> None:
        """Frees the blocks below the window, nulling their slots."""
        pool = self.pools[replica_idx]
        first_needed = max(0, num_committed_blocks - self._blocks_in_window)
        req_blocks = self.rows[req_id][self.leaf_id]
        null_block = pool.null_little_blocks[self.leaf_id]
        for idx in range(first_needed - 1, -1, -1):
            if req_blocks[idx].is_null:
                break
            pool.free_block(req_blocks[idx])
            req_blocks[idx] = null_block

    def blocks_held_of_connector_hit(self, num_hit_blocks: int) -> int:
        """Only the window ending at the hit; the rest has slid out of reach."""
        return blocks_held_of_hit(num_hit_blocks, self._blocks_in_window)

    def extend(
        self,
        req_id: RequestID,
        hit_blocks: Mapping[str, Sequence[LittleKVCacheBlock]],
        loaded_blocks: Mapping[str, Sequence[LittleKVCacheBlock]],
        replica_idx: int,
    ) -> None:
        """Drops the device blocks when the loaded run starts with a null.

        A null first loaded block means that block sits below the window,
        and so does everything before it, including every device block. The
        group will never read them again, so free them rather than hold
        pages nothing can use.
        """
        pool = self.pools[replica_idx]
        row = self.rows[req_id]
        leaf_id = self.leaf_id
        hit = list(hit_blocks[leaf_id])
        loaded = loaded_blocks.get(leaf_id, ())
        if loaded and loaded[0].is_null:
            for block in hit:
                pool.free_block(block)
            hit = [pool.null_little_blocks[leaf_id]] * len(hit)
        row[leaf_id].extend([*hit, *loaded])
