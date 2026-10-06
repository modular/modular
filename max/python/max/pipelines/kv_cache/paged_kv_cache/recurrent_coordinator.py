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
"""The cache group whose entry is a recurrent state rather than a span."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from max.pipelines.context import TextContext
from max.pipelines.modeling.types import RequestID
from max.profiler import traced

from .block_utils import InsufficientBlocksError, LittleKVCacheBlock
from .kv_group_coordinator import KVGroupCoordinatorInterface

__all__ = ["RecurrentKVGroupCoordinator"]


@dataclass(frozen=True)
class RecurrentKVGroupCoordinator(KVGroupCoordinatorInterface):
    """A group whose entry is one state, held in the last slot of a row.

    ``row[-1]`` is the block the recurrence runs in, and it carries no hash.
    Every block behind it is null, a published checkpoint, or one a
    connector hit is still loading. The recurrence reads and writes one
    block, so reaching a boundary publishes the block it ran in and copies
    it into a successor rather than snapshotting it.
    """

    page_size: int = 0
    """Tokens per page: the granularity a state can be committed at."""

    enable_prefix_caching: bool = False
    """Whether the state checkpoints at page boundaries.

    Without prefix caching nothing is published, so the state stays in the
    block it ran in.
    """

    successors: dict[RequestID, LittleKVCacheBlock] = field(
        default_factory=dict, kw_only=True
    )
    """The block each request's next checkpoint continues the state in.

    Held from admission to release while prefix caching is on, which is
    the second block ``blocks_to_reserve`` budgets. Drawing it in ``step``
    instead would be a draw no admission check priced.
    """

    loading: dict[RequestID, LittleKVCacheBlock] = field(
        default_factory=dict, kw_only=True
    )
    """The checkpoint a connector hit is still loading, per request.

    Hashless until its copy lands, so without this it reads as the live block.
    """

    def _loading(self, req_id: RequestID) -> LittleKVCacheBlock | None:
        """Returns the loading checkpoint, forgetting it once published."""
        block = self.loading.get(req_id)
        if block is None:
            return None
        if block.block_hash is not None:
            del self.loading[req_id]
            return None
        return block

    def _live(
        self, req_id: RequestID, row: Sequence[LittleKVCacheBlock]
    ) -> LittleKVCacheBlock | None:
        """Returns the block the recurrence runs in, if the row has one."""
        if not row or row[-1].block_hash is not None:
            return None
        if row[-1] is self._loading(req_id):
            return None
        return row[-1]

    def _resumed_from(
        self, req_id: RequestID, row: Sequence[LittleKVCacheBlock]
    ) -> LittleKVCacheBlock | None:
        """Returns the checkpoint a hit left for this request to read."""
        loading = self._loading(req_id)
        for block in reversed(row):
            if block.is_null:
                continue
            if block.block_hash is not None or block is loading:
                return block
        return None

    def _awaits_hash(self, req_id: RequestID) -> bool:
        """Whether this request's own last checkpoint still lacks its hash.

        A loading checkpoint is unhashed too, but it belongs to a hit.
        """
        row = self.rows[req_id][self.leaf_id]
        loading = self._loading(req_id)
        return any(
            not block.is_null
            and block.block_hash is None
            and block is not loading
            for block in row[:-1]
        )

    # ============================================================================
    # The hit
    # ============================================================================

    def claimable_hashes(
        self, desired_hashes: Sequence[bytes]
    ) -> Sequence[bytes]:
        """Returns the deepest hash: a state is one boundary, not a run."""
        return desired_hashes[-1:]

    def claim_hit_blocks(
        self, desired_hashes: Sequence[bytes], replica_idx: int
    ) -> dict[str, list[LittleKVCacheBlock]]:
        """Takes the block at the granted block count, nulling the slots behind it.

        Returns empty rows when no published state stands there.
        """
        pool = self.pools[replica_idx]
        matched = (
            bool(desired_hashes)
            and desired_hashes[-1] in pool.prefix_caches[self.leaf_id]
        )
        if not matched:
            return {self.leaf_id: []}
        rows: dict[str, list[LittleKVCacheBlock]] = {}
        leaf_id = self.leaf_id
        null_block = pool.null_little_blocks[leaf_id]
        row = [null_block] * (len(desired_hashes) - 1)
        block = pool.prefix_caches[leaf_id][desired_hashes[-1]]
        pool.touch(block)
        row.append(block)
        rows[leaf_id] = row
        return rows

    def blocks_held_of_connector_hit(self, num_hit_blocks: int) -> int:
        """Returns one block, the state at the hit's depth."""
        return min(num_hit_blocks, 1)

    def extend(
        self,
        req_id: RequestID,
        hit_blocks: Mapping[str, Sequence[LittleKVCacheBlock]],
        loaded_blocks: Mapping[str, Sequence[LittleKVCacheBlock]],
        replica_idx: int,
    ) -> None:
        """Appends the hit, recording a loaded checkpoint as loading."""
        super().extend(req_id, hit_blocks, loaded_blocks, replica_idx)
        for block in loaded_blocks.get(self.leaf_id, ()):
            if not block.is_null and block.block_hash is None:
                self.loading[req_id] = block

    # ============================================================================
    # Demand
    # ============================================================================

    def _needs_successor(self, req_id: RequestID) -> bool:
        """Whether admission must draw the request's successor.

        Not while a checkpoint is still unpublished: ``checkpoint`` runs one
        at a time, so the request holds at most its live block and one more.
        """
        if not self.enable_prefix_caching or req_id in self.successors:
            return False
        return not self._awaits_hash(req_id)

    def blocks_to_allocate(
        self, req_id: RequestID, num_required_blocks: int
    ) -> dict[str, int]:
        """Returns the live block the row lacks, and its missing successor."""
        row = self.rows[req_id][self.leaf_id]
        demand = 0 if self._live(req_id, row) is not None else 1
        if self._needs_successor(req_id):
            demand += 1
        return {self.leaf_id: demand}

    def grow(
        self, req_id: RequestID, num_required_blocks: int, replica_idx: int
    ) -> None:
        """Draws the live block and successor the request lacks.

        Also pads the row out.
        """
        pool = self.pools[replica_idx]
        leaf_id = self.leaf_id
        row = self.rows[req_id][leaf_id]
        draws_live = self._live(req_id, row) is None
        draws_successor = self._needs_successor(req_id)
        # Checked up front so a pool with room for one of the two blocks
        # draws neither.
        if not pool.can_satisfy_demand({leaf_id: draws_live + draws_successor}):
            raise InsufficientBlocksError(
                f"No blocks left for the recurrent state of {req_id}"
            )
        live = pool.alloc_block(leaf_id) if draws_live else row.pop()
        if draws_successor:
            self.successors[req_id] = pool.alloc_block(leaf_id)

        null_block = pool.null_little_blocks[leaf_id]
        while len(row) < max(num_required_blocks - 1, 0):
            row.append(null_block)
        row.append(live)

    def release(self, req_id: RequestID, replica_idx: int) -> None:
        """Frees every block the request holds, its successor too."""
        self.loading.pop(req_id, None)
        successor = self.successors.pop(req_id, None)
        if successor is not None:
            self.pools[replica_idx].free_block(successor)
        super().release(req_id, replica_idx)

    def shrink_to_fit(
        self, req_id: RequestID, num_committed_blocks: int, replica_idx: int
    ) -> None:
        """Refits the row to the committed blocks, keeping the live block last.

        Frees every other block in the row. A loading checkpoint's copy
        holds its own pin.
        """
        pool = self.pools[replica_idx]
        leaf_id = self.leaf_id
        row = self.rows[req_id][leaf_id]
        live = row.pop() if self._live(req_id, row) is not None else None
        for block in reversed(row):
            pool.free_block(block)
        row.clear()
        self.loading.pop(req_id, None)
        # Nothing live means nothing to refit around, so the row stays empty.
        if live is not None:
            null_block = pool.null_little_blocks[leaf_id]
            while len(row) < max(num_committed_blocks - 1, 0):
                row.append(null_block)
            row.append(live)

    # ============================================================================
    # The blocks a forward runs in
    # ============================================================================

    def live_blocks(self, req_id: RequestID) -> dict[str, int] | None:
        """Returns the block per leaf the recurrence runs in, if one is drawn."""
        row = self.rows.get(req_id)
        if row is None:
            return None
        live = self._live(req_id, row[self.leaf_id])
        if live is None:
            return None
        return {self.leaf_id: live.bid}

    def advance(
        self, req_id: RequestID, num_committed_blocks: int, replica_idx: int
    ) -> None:
        """Frees every published block behind the live one, nulling its slot."""
        pool = self.pools[replica_idx]
        leaf_id = self.leaf_id
        row = self.rows[req_id][leaf_id]
        null_block = pool.null_little_blocks[leaf_id]
        live = self._live(req_id, row)
        for idx, block in enumerate(row):
            if block is live or block.is_null:
                continue
            if block.block_hash is None:
                continue
            pool.free_block(block)
            row[idx] = null_block

    # ============================================================================
    # Checkpoints
    # ============================================================================

    @traced
    def resume(
        self, ctx: TextContext, replica_idx: int
    ) -> Mapping[str, tuple[int | None, int]]:
        """Returns the block this forward resumes from and the one it fills.

        Filled from a checkpoint when the row holds one: what a prefix hit
        claimed or loaded, or the predecessor a checkpoint published. Filled
        with zeros when the request has processed nothing and matched no hit,
        since a drawn block holds whatever its last request wrote.

        Empty once the request runs in a block it has written itself.
        """
        del replica_idx  # the blocks name themselves; the caller holds the pool
        row = self.rows.get(ctx.request_id)
        runs_in = self.live_blocks(ctx.request_id)
        if row is None or runs_in is None:
            return {}
        fills: dict[str, tuple[int | None, int]] = {}
        leaf_id = self.leaf_id
        published = self._resumed_from(ctx.request_id, row[leaf_id])
        # Mid-sequence with nothing published means there is no state to
        # resume from, so this leaf contributes no fill.
        if published is not None or ctx.tokens.processed_length == 0:
            src = None if published is None else published.bid
            fills[leaf_id] = (src, runs_in[leaf_id])
        return fills

    @traced
    def checkpoint(
        self, ctx: TextContext, replica_idx: int
    ) -> Mapping[str, tuple[int | None, int]]:
        """Publishes the block just run in and fills the one that succeeds it.

        The block the forward ran in already holds the state the boundary
        hash names, so it is published where that hash lands rather than
        snapshotted. The request continues in a freshly drawn block, copied
        from it. Runs before the commit, so the copy reads a block nothing
        has freed yet.

        Empty unless the forward ended exactly on a block boundary and the
        row holds no unpublished predecessor already.
        """
        if not self.enable_prefix_caching:
            return {}
        row = self.rows.get(ctx.request_id)
        if row is None:
            return {}
        if row[self.leaf_id] and row[self.leaf_id][-1].is_null:
            return {}  # padding a batch out; there is no state to keep
        ran_in = self.live_blocks(ctx.request_id)
        if ran_in is None:
            return {}

        num_committed_blocks, past_boundary = divmod(
            ctx.tokens.processed_length, self.page_size
        )
        if num_committed_blocks == 0 or past_boundary:
            return {}
        if self._awaits_hash(ctx.request_id):
            return {}  # one checkpoint at a time

        # Admission reserves the successor alongside the block being
        # published, so a missing one is an accounting bug, not pressure.
        pool = self.pools[replica_idx]
        successor = self.successors.pop(ctx.request_id, None)
        if successor is None:
            raise AssertionError(
                f"{ctx.request_id} reached a checkpoint with no successor"
                " reserved; was the group built with enable_prefix_caching?"
            )
        drawn = {self.leaf_id: successor}

        fills: dict[str, tuple[int | None, int]] = {}
        leaf_id = self.leaf_id
        r = row[leaf_id]
        while len(r) <= num_committed_blocks:
            r.insert(len(r) - 1, pool.null_little_blocks[leaf_id])
        # The block that ran moves to the slot its hash indexes; the
        # successor takes over as the one the recurrence runs in.
        r[num_committed_blocks - 1] = r[-1]
        r[-1] = drawn[leaf_id]
        fills[leaf_id] = (ran_in[leaf_id], drawn[leaf_id].bid)
        return fills

    def _is_committable(
        self,
        req_id: RequestID,
        row: Sequence[LittleKVCacheBlock],
        block_idx: int,
    ) -> bool:
        """The live block is still being written, so it cannot be published."""
        if row[block_idx] is self._live(req_id, row):
            return False
        return super()._is_committable(req_id, row, block_idx)

    def commit(
        self,
        req_id: RequestID,
        hashes: Sequence[bytes],
        last_block: int,
        replica_idx: int,
    ) -> None:
        """Publishes the checkpoints below ``last_block``.

        Forgets a loading checkpoint that a twin holding its hash replaced.
        """
        super().commit(req_id, hashes, last_block, replica_idx)
        loading = self.loading.get(req_id)
        if loading is not None and not any(
            block is loading for block in self.rows[req_id][self.leaf_id]
        ):
            del self.loading[req_id]

    @traced
    def forward_blocks(
        self, batch: Sequence[TextContext], num_blocks: Sequence[int]
    ) -> dict[str, list[list[int]]]:
        """Returns the block each request's recurrence runs in."""
        plans: dict[str, list[list[int]]] = {self.leaf_id: []}
        for ctx in batch:
            runs_in = self.live_blocks(ctx.request_id)
            if runs_in is None:
                raise ValueError(
                    f"{ctx.request_id} has no state blocks; alloc must run"
                    " before its inputs are built"
                )
            leaf_id = self.leaf_id
            plans[leaf_id].append([runs_in[leaf_id]])
        return plans
