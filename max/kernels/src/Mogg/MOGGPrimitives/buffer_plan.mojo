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

from std.collections import BitSet, InlineArray


def _compute_unshareable[
    N: Int,
](can_share: InlineArray[Int, N * N], out result: BitSet[N],):
    """Compute which allocations cannot share memory with any other allocation.

    An allocation i is unshareable if no other allocation j has a
    non-overlapping lifetime with it. Two allocations can share memory when
    can_share[i * N + j] == 1, meaning their lifetimes do not overlap.
    """
    result = {}
    result.set_all()

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if can_share[i * N + j]:
                result.clear(i)
                break


def _compute_shareable_rows[
    N: Int,
](can_share: InlineArray[Int, N * N], out result: InlineArray[BitSet[N], N],):
    """Compute per-allocation sharing bitsets from the sharing matrix.

    result[i].test(j) == True iff can_share[i * N + j] == 1, i.e. allocations
    i and j have non-overlapping lifetimes and may share a memory block.
    """
    result = InlineArray[BitSet[N], N](uninitialized=True)
    for i in range(N):
        var row: BitSet[N] = {}
        for j in range(N):
            if can_share[i * N + j]:
                row.set(j)
        result[i] = row^


def _maximum(alignments: InlineArray[Int, _]) -> Int:
    """Return the maximum value in an InlineArray of Ints."""
    var result = 0
    for align in alignments:
        result = max(result, align)
    return result


struct MemoryBlock[N: Int](Copyable, Movable):
    """Memory block used by buffer planning algorithm."""

    var offset: Int
    var size: Int
    # Set of allocation indices that can still be assigned to this block.
    # An allocation j is in this set iff j can share memory with every
    # allocation already assigned to this block.
    var shareable: BitSet[Self.N]

    def __init__(
        out self, offset: Int, size: Int, var shareable: BitSet[Self.N]
    ):
        self.offset = offset
        self.size = size
        self.shareable = shareable^


struct BufferPlanState[
    num_allocs: Int,
    //,
    alignments: InlineArray[Int, num_allocs],
    can_share: InlineArray[Int, num_allocs * num_allocs],
](Movable):
    # Computed at comptime: which allocations interfere with all others and
    # therefore can never share a memory block with any other allocation.
    comptime unshareable = _compute_unshareable[Self.num_allocs](Self.can_share)
    # Per-allocation sharing bitsets: shareable_rows[i].test(j) == True iff
    # allocations i and j can share a memory block.
    comptime shareable_rows = _compute_shareable_rows[Self.num_allocs](
        Self.can_share
    )
    # If every allocation is unshareable, skip the greedy block-reuse logic
    # entirely and use the fast linear-allocation path.
    comptime enable_sharing = len(Self.unshareable) < Self.num_allocs
    comptime max_alignment = _maximum(Self.alignments)

    var blocks: List[MemoryBlock[Self.num_allocs]]
    var allocated: Int

    # Computed allocation offsets for all allocations.
    var offsets: InlineArray[Int, Self.num_allocs]
    var pool_size: Int

    # The set of per-allocation shareable bitsets.
    var shareable_sets: InlineArray[BitSet[Self.num_allocs], Self.num_allocs]

    def __init__(
        out self,
    ):
        comptime if Self.enable_sharing:
            self.blocks = List[MemoryBlock[Self.num_allocs]](
                capacity=Self.num_allocs
            )
            self.shareable_sets = materialize[Self.shareable_rows]()
        else:
            self.blocks = {}
            self.shareable_sets = {uninitialized = True}

        self.allocated = 0
        self.pool_size = 0
        self.offsets = InlineArray[Int, Self.num_allocs](fill=0)

    @always_inline
    def take_results(
        deinit self,
    ) -> Tuple[Int, InlineArray[Int, Self.num_allocs]]:
        assert self.allocated == Self.num_allocs
        return self.pool_size, self.offsets

    def find_block(
        self,
        index: Int,
        alloc_size: Int,
        out result: Int,
    ):
        result = -1

        comptime if not Self.enable_sharing:
            return

        var best_size = Int.MAX

        for block_idx in range(len(self.blocks)):
            if (
                alloc_size > self.blocks[block_idx].size
                or self.blocks[block_idx].size >= best_size
            ):
                continue

            if self.blocks[block_idx].shareable.test(index):
                best_size = self.blocks[block_idx].size
                result = block_idx

    @always_inline
    def append_result(mut self, index: Int, value: Int):
        self.offsets[index] = value
        self.allocated += 1

    @always_inline
    def shareable_set(self, index: Int) -> BitSet[Self.num_allocs]:
        assert Self.enable_sharing, "unable to get shareable set"
        return self.shareable_sets[index].copy()

    def allocate_new_block(mut self, index: Int, alloc_size: Int):
        var new_offset = (
            (self.pool_size + self.max_alignment - 1)
            // self.max_alignment
            * self.max_alignment
        )

        comptime if Self.enable_sharing:
            self.blocks.append(
                MemoryBlock[Self.num_allocs](
                    offset=new_offset,
                    size=alloc_size,
                    shareable=self.shareable_set(index),
                )
            )

        self.append_result(index, new_offset)
        self.pool_size = new_offset + alloc_size

    def try_reuse_block(mut self, result_idx: Int, alloc_size: Int):
        var best_block_idx = self.find_block(result_idx, alloc_size)
        if best_block_idx >= 0:
            # Intersect the block's shareable set with the precomputed row for
            # result_idx. Since the diagonal is zero, result_idx is
            # automatically cleared (it is now a member, not a future
            # candidate).
            self.blocks[best_block_idx].shareable = self.blocks[
                best_block_idx
            ].shareable.intersection(self.shareable_set(result_idx))
            self.append_result(result_idx, self.blocks[best_block_idx].offset)
        else:
            self.allocate_new_block(result_idx, alloc_size)

    @always_inline
    def allocate_greedy[
        start: Int = 0
    ](mut self, sizes: InlineArray[Int, _],):
        comptime if not Self.enable_sharing:
            # No allocations can be shared; skip the greedy search entirely.
            for i, size in enumerate(sizes):
                self.allocate_new_block(i + start, size)
        else:
            comptime for i in range(sizes.size):
                var alloc_size = sizes[i]
                comptime result_idx = i + start

                comptime if Self.unshareable.test(result_idx):
                    # This allocation cannot share with any other; skip search.
                    self.allocate_new_block(result_idx, alloc_size)
                else:
                    self.try_reuse_block(result_idx, alloc_size)
