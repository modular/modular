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
"""The cache group whose block is scratch for the life of one request."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from max.pipelines.context import TextContext
from max.pipelines.modeling.types import RequestID
from max.profiler import traced

from .block_utils import InsufficientBlocksError, LittleKVCacheBlock
from .kv_group_coordinator import KVGroupCoordinatorInterface

__all__ = ["ScratchKVGroupCoordinator"]


@dataclass(frozen=True)
class ScratchKVGroupCoordinator(KVGroupCoordinatorInterface):
    """A group holding one never-published block per request.

    The block is drawn on the request's first :meth:`grow` and freed at
    :meth:`release`, and nothing in between moves it: a page boundary neither
    publishes it nor rotates it, so its id is stable for the request's whole
    life. That is what a ring of recent kernel inputs needs, and it is why
    :meth:`commit`, :meth:`advance` and :meth:`shrink_to_fit` do nothing here.

    Carrying no hash keeps the group out of every content-addressed path. It
    answers no hit, contributes no claimable hash, and asks an external tier
    for nothing.
    """

    def claimable_hashes(
        self, desired_hashes: Sequence[bytes]
    ) -> Sequence[bytes]:
        """Returns nothing: this group's block is never published."""
        return ()

    def longest_hit(
        self, num_hashes: int, is_cached: Callable[[int], bool]
    ) -> int:
        """Returns the whole run, leaving the length to the groups that cache.

        The manager settles a prefix by having every group accept it in turn,
        so a group with no opinion has to accept whatever the others agree
        on. It is excluded from that cycle as well, which makes this
        unreachable rather than merely harmless.
        """
        return num_hashes

    def claim_hit_blocks(
        self, desired_hashes: Sequence[bytes], replica_idx: int
    ) -> dict[str, list[LittleKVCacheBlock]]:
        """Returns empty rows: a hit leaves this group nothing to take."""
        return {leaf_id: [] for leaf_id in self.leaf_ids}

    def _num_blocks_to_allocate(
        self, row: Sequence[LittleKVCacheBlock], num_required_blocks: int
    ) -> int:
        """Returns one until the row holds its block, then none."""
        return 0 if row else 1

    def grow(
        self, req_id: RequestID, num_required_blocks: int, replica_idx: int
    ) -> None:
        """Draws the request's one block, if it does not hold it yet."""
        pool = self.pools[replica_idx]
        drawn: dict[str, LittleKVCacheBlock] = {}
        try:
            for leaf_id in self.leaf_ids:
                if self._num_blocks_to_allocate(
                    self.rows[req_id][leaf_id], num_required_blocks
                ):
                    drawn[leaf_id] = pool.alloc_block(leaf_id)
        except InsufficientBlocksError:
            for block in drawn.values():
                pool.free_block(block)
            raise InsufficientBlocksError(
                f"No blocks left for the live scratch of {req_id}"
            ) from None
        for leaf_id, block in drawn.items():
            self.rows[req_id][leaf_id].append(block)

    def grow_with_padding(
        self, req_id: RequestID, num_required_blocks: int, replica_idx: int
    ) -> None:
        """Points the row at one null block, however long the dummy is."""
        pool = self.pools[replica_idx]
        self.rows[req_id] = {
            leaf_id: [pool.null_little_blocks[leaf_id]]
            for leaf_id in self.leaf_ids
        }

    def shrink_to_fit(
        self, req_id: RequestID, num_committed_blocks: int, replica_idx: int
    ) -> None:
        """Keeps the block: there is no committed prefix to trim it to."""
        return

    def commit(
        self,
        req_id: RequestID,
        hashes: Sequence[bytes],
        last_block: int,
        replica_idx: int,
    ) -> None:
        """Publishes nothing."""
        return

    def advance(
        self, req_id: RequestID, num_committed_blocks: int, replica_idx: int
    ) -> None:
        """Frees nothing: the request reads its block until it is released."""
        return

    def blocks_held_of_connector_hit(self, num_hit_blocks: int) -> int:
        """Returns zero: an external tier holds nothing for this group."""
        return 0

    @traced
    def forward_blocks(
        self, batch: Sequence[TextContext], num_blocks: Sequence[int]
    ) -> dict[str, list[list[int]]]:
        """Returns the one block each request's scratch lives in."""
        plans: dict[str, list[list[int]]] = {
            leaf_id: [] for leaf_id in self.leaf_ids
        }
        for ctx in batch:
            row = self.rows[ctx.request_id]
            for leaf_id in self.leaf_ids:
                blocks = row[leaf_id]
                if not blocks:
                    raise ValueError(
                        f"{ctx.request_id} has no {leaf_id!r} block; alloc"
                        " must run before its inputs are built"
                    )
                plans[leaf_id].append([blocks[0].bid])
        return plans
