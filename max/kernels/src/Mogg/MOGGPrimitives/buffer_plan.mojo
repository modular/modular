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


def _compute_unsharable[
    N: Int,
](
    min_pre: InlineArray[Int, N],
    min_post: InlineArray[Int, N],
    max_pre: InlineArray[Int, N],
    max_post: InlineArray[Int, N],
    out result: BitSet[N],
):
    """Compute which allocations cannot share memory with any other allocation.

    An allocation i is unsharable if no other allocation j has a
    non-overlapping lifetime with it. Two allocations can share memory when
    one's lifetime does not contain the other in the dependency-tree traversal
    order: can_share(i, j) := (max_pre[i] < min_pre[j] && min_post[i] >
    max_post[j]) || (max_pre[j] < min_pre[i] && min_post[j] > max_post[i]).
    """
    result = {}
    result.set_all()

    for i in range(N):
        for j in range(N):
            if i == j:
                continue

            if (max_pre[i] < min_pre[j] and min_post[i] > max_post[j]) or (
                max_pre[j] < min_pre[i] and min_post[j] > max_post[i]
            ):
                result.clear(i)
                break


def _maximum(alignments: InlineArray[Int, _]) -> Int:
    """Return the maximum value in an InlineArray of Ints."""
    var result = 0
    for align in alignments:
        result = max(result, align)
    return result


@fieldwise_init
struct MemoryBlock(ImplicitlyCopyable, Movable):
    """Memory block used by buffer planning algorithm."""

    var offset: Int
    var size: Int
    var min_pre: Int
    var max_pre: Int
    var min_post: Int
    var max_post: Int

    def extend_lifetime(
        mut self, min_pre: Int, max_pre: Int, min_post: Int, max_post: Int
    ):
        self.min_pre = min(self.min_pre, min_pre)
        self.max_pre = max(self.max_pre, max_pre)
        self.min_post = min(self.min_post, min_post)
        self.max_post = max(self.max_post, max_post)


struct BufferPlanState[
    num_allocs: Int,
    //,
    alignments: InlineArray[Int, num_allocs],
    min_pre: InlineArray[Int, num_allocs],
    min_post: InlineArray[Int, num_allocs],
    max_pre: InlineArray[Int, num_allocs],
    max_post: InlineArray[Int, num_allocs],
](Movable):
    # Computed at comptime: which allocations interfere with all others and
    # therefore can never share a memory block with any other allocation.
    comptime unsharable = _compute_unsharable[Self.num_allocs](
        Self.min_pre, Self.min_post, Self.max_pre, Self.max_post
    )
    # If every allocation is unsharable, skip the greedy block-reuse logic
    # entirely and use the fast linear-allocation path.
    comptime enable_sharing = len(Self.unsharable) < Self.num_allocs
    comptime max_alignment = _maximum(Self.alignments)

    var blocks: List[MemoryBlock]
    var allocated: Int

    # Computed allocation offsets for all allocations.
    var offsets: InlineArray[Int, Self.num_allocs]
    var pool_size: Int

    def __init__(
        out self,
    ):
        comptime if Self.enable_sharing:
            self.blocks = List[MemoryBlock](capacity=Self.num_allocs)
        else:
            self.blocks = {}
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
            var block = self.blocks[block_idx]

            if (
                alloc_size <= block.size
                and block.size < best_size
                and self.max_pre[index] < block.min_pre
                and self.min_post[index] > block.max_post
            ):
                best_size = block.size
                result = block_idx

    @always_inline
    def append_result(mut self, index: Int, value: Int):
        self.offsets[index] = value
        self.allocated += 1

    def allocate_new_block(mut self, index: Int, alloc_size: Int):
        var new_offset = (
            (self.pool_size + self.max_alignment - 1)
            // self.max_alignment
            * self.max_alignment
        )

        comptime if Self.enable_sharing:
            self.blocks.append(
                MemoryBlock(
                    offset=new_offset,
                    size=alloc_size,
                    min_pre=self.min_pre[index],
                    max_pre=self.max_pre[index],
                    min_post=self.min_post[index],
                    max_post=self.max_post[index],
                )
            )

        self.append_result(index, new_offset)
        self.pool_size = new_offset + alloc_size

    def try_reuse_block(mut self, result_idx: Int, alloc_size: Int):
        var best_block_idx = self.find_block(result_idx, alloc_size)
        if best_block_idx >= 0:
            # We found a memory block to reuse
            # Update lifetime of memory block
            self.blocks[best_block_idx].extend_lifetime(
                self.min_pre[result_idx],
                self.max_pre[result_idx],
                self.min_post[result_idx],
                self.max_post[result_idx],
            )
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

                comptime if Self.unsharable.test(result_idx):
                    # This allocation cannot share with any other; skip search.
                    self.allocate_new_block(result_idx, alloc_size)
                else:
                    self.try_reuse_block(result_idx, alloc_size)
