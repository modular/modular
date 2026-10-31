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
"""Paged KV cache fixtures, sized once for the largest config of a sweep.

A `PageTable` owns the block mapping and cache lengths; any number of
`PagedBlocks` (K, V, an MLA latent, indexer keys and their scales) share it,
as production allocates every cache's pages together. A smaller config uses
the leading rows of the page table, so nothing is refilled between configs.
"""

from std.math import ceildiv
from std.sys import size_of

from max.gpu.host import DeviceBuffer, DeviceContext

from .driver import Footprint
from .generators import (
    PageLayout,
    fill_linear_grade,
    fill_lut_view,
    fill_page_table,
    fill_uniform,
    fill_uniform_host_rand,
    page_table_host,
)


struct PageTable[page_size: Int](Movable):
    """A page table over a pool of blocks, and per-sequence lengths.

    `lut` maps a pool of `max(max_batch * pages_per_seq, pool_pages)`
    entries, each a distinct block; read as `[max_batch, pages_per_seq]` it
    is one cache. For rotation, `write_copies` writes a config's copies as
    separate `[batch, pages]` tables in `copy_luts`, and `copy_lut` returns
    one. Copies differ on every page a launch reads and may share the rest,
    so a sparse kernel's copies cost only the pages it touches. Block 0 is
    never referenced, matching production's null block, so `num_blocks` is
    one more than the pool.
    """

    var max_batch: Int
    var pages_per_seq: Int
    var num_blocks: Int
    var layout: PageLayout
    var seed: Int
    var lut: DeviceBuffer[DType.uint32]
    var cache_lengths: DeviceBuffer[DType.uint32]
    var copy_luts: DeviceBuffer[DType.uint32]
    var _copies: Int
    var _copy_entries: Int

    def __init__(
        out self,
        ctx: DeviceContext,
        max_batch: Int,
        max_tokens_per_seq: Int,
        layout: PageLayout = PageLayout.IDENTITY,
        seed: Int = 0,
        pool_pages: Int = 0,
        copy_lut_entries: Int = 0,
    ) raises:
        self.max_batch = max_batch
        self.pages_per_seq = max(1, ceildiv(max_tokens_per_seq, Self.page_size))
        var pool = max(max_batch * self.pages_per_seq, pool_pages)
        self.num_blocks = pool + 1
        self.layout = layout
        self.seed = seed
        self.lut = ctx.enqueue_create_buffer[DType.uint32](pool)
        self.copy_luts = ctx.enqueue_create_buffer[DType.uint32](
            max(1, copy_lut_entries)
        )
        self._copies = 0
        self._copy_entries = 0
        self.cache_lengths = ctx.enqueue_create_buffer[DType.uint32](max_batch)
        fill_page_table(ctx, self.lut, layout, seed)

    @staticmethod
    def require(
        mut footprint: Footprint,
        batch: Int,
        tokens_per_seq: Int,
        copies: Int = 1,
        touched_pages: Int = -1,
    ):
        """Records one config's needs under the names `__init__` reads.

        `copies` is how many copies of the config's cache the benchmark
        rotates through, and `touched_pages` how many of its
        `batch * pages` pages a launch reads; -1 means all of them.
        """
        var pages = batch * max(1, ceildiv(tokens_per_seq, Self.page_size))
        var touched = pages if touched_pages < 0 else touched_pages
        footprint.need("batch", batch)
        footprint.need("tokens_per_seq", tokens_per_seq)
        footprint.need("pool_pages", pages + (max(copies, 1) - 1) * touched)
        footprint.need("copy_lut_entries", max(copies, 1) * pages)

    @staticmethod
    def from_footprint(
        ctx: DeviceContext,
        footprint: Footprint,
        layout: PageLayout = PageLayout.IDENTITY,
        seed: Int = 0,
    ) raises -> Self:
        return Self(
            ctx,
            footprint["batch"],
            footprint["tokens_per_seq"],
            layout,
            seed,
            footprint["pool_pages"],
            footprint["copy_lut_entries"],
        )

    @staticmethod
    def _pool_for(footprint: Footprint) -> Int:
        var pages = max(1, ceildiv(footprint["tokens_per_seq"], Self.page_size))
        return max(footprint["batch"] * pages, footprint["pool_pages"])

    @staticmethod
    def rows_for(footprint: Footprint) -> Int:
        """Returns the KV rows a table sized for `footprint` addresses."""
        return (Self._pool_for(footprint) + 1) * Self.page_size

    @staticmethod
    def bytes_for(footprint: Footprint) -> Int:
        return 4 * (
            Self._pool_for(footprint)
            + footprint["batch"]
            + footprint["copy_lut_entries"]
        )

    def rows(self) -> Int:
        """Returns the number of KV rows the blocks span, block 0 included."""
        return self.num_blocks * Self.page_size

    def bytes(self) -> Int:
        return 4 * (
            len(self.lut) + len(self.cache_lengths) + len(self.copy_luts)
        )

    def set_cache_len(self, ctx: DeviceContext, cache_len: Int) raises:
        """Sets every sequence's cache length to `cache_len`."""
        ctx.enqueue_memset(self.cache_lengths, UInt32(cache_len))

    def set_cache_len_graded(
        self, ctx: DeviceContext, batch: Int, lo: Int, hi: Int
    ) raises:
        """Grades the first `batch` cache lengths linearly from `lo` to `hi`.

        See `linear_grade_at`; a batch of one takes the midpoint.
        """
        fill_linear_grade(ctx, self.cache_lengths, lo, hi, batch)

    def write_view(
        self,
        ctx: DeviceContext,
        dst: DeviceBuffer[DType.uint32, ...],
        batch: Int,
        cols: Int,
        used_cols: Int,
    ) raises:
        """Writes a `[batch, cols]` table over this one's leading rows.

        Columns past `used_cols` point at the null block, as a page table
        sized from frozen metadata does past a sequence's real pages.
        """
        fill_lut_view(
            ctx, dst, self.lut, self.pages_per_seq, batch, cols, used_cols
        )

    def write_copies(
        mut self,
        ctx: DeviceContext,
        copies: Int,
        batch: Int,
        pages: Int,
        touched: List[Bool] = [],
    ) raises:
        """Writes `copies` `[batch, pages]` page tables into `copy_luts`.

        Copy 0 is the pool's first `batch * pages` blocks. Every later copy
        takes fresh blocks for the pages `touched` marks, `[batch, pages]`,
        and shares copy 0's elsewhere; an empty `touched` marks every page.
        Raises if the pool or `copy_luts` is too small.
        """
        var n = batch * pages
        if copies * n > len(self.copy_luts):
            raise Error(
                copies,
                " copies of ",
                n,
                " pages exceed copy_luts' ",
                len(self.copy_luts),
            )
        if len(touched) != 0 and len(touched) != n:
            raise Error("touched has ", len(touched), " pages, not ", n)
        var pool = self.host_lut()
        var host = ctx.enqueue_create_host_buffer[DType.uint32](copies * n)
        var next = n
        for copy in range(copies):
            for i in range(n):
                if copy == 0 or (len(touched) != 0 and not touched[i]):
                    host[copy * n + i] = pool[i]
                    continue
                if next >= len(pool):
                    raise Error(
                        "page pool of ",
                        len(pool),
                        " has no room for copy ",
                        copy,
                    )
                host[copy * n + i] = pool[next]
                next += 1
        ctx.enqueue_copy(
            DeviceBuffer[DType.uint32](
                ctx, self.copy_luts.unsafe_ptr(), copies * n, owning=False
            ),
            host,
        )
        self._copies = copies
        self._copy_entries = n

    def copy_lut(
        self, ctx: DeviceContext, copy: Int, batch: Int, pages: Int
    ) raises -> DeviceBuffer[DType.uint32]:
        """Returns copy `copy`'s `[batch, pages]` page table, a view.

        Raises unless `write_copies` last wrote at least `copy + 1` copies
        of that shape.
        """
        if copy >= self._copies or batch * pages != self._copy_entries:
            raise Error(
                "no copy ",
                copy,
                " of ",
                batch,
                "x",
                pages,
                " pages; write_copies wrote ",
                self._copies,
                " of ",
                self._copy_entries,
            )
        return DeviceBuffer[DType.uint32](
            ctx,
            self.copy_luts.unsafe_ptr().unsafe_offset(copy * batch * pages),
            batch * pages,
            owning=False,
        )

    def host_lut(self) -> List[UInt32]:
        """Returns the CPU reference of `lut`."""
        return page_table_host(len(self.lut), self.layout, self.seed)


struct PagedBlocks[dtype: DType](Movable):
    """Paged storage of `row_elems` elements per KV row, for a `PageTable`.

    `row_elems` is `num_heads * head_size` for K or V, the latent width for
    MLA, or the number of scales per row. Contents are uniform on `[lo, hi)`
    and read-only by convention, so configs that share them do not depend on
    each other.
    """

    var row_elems: Int
    var buf: DeviceBuffer[Self.dtype]

    def __init__[
        page_size: Int
    ](
        out self,
        ctx: DeviceContext,
        table: PageTable[page_size],
        row_elems: Int,
        seed: Int,
        lo: Float32 = 0.0,
        hi: Float32 = 1.0,
        on_host: Bool = False,
    ) raises:
        """Allocates and fills the blocks.

        `on_host` fills through `std.random.rand` on a mapped buffer, the
        status quo the device fill is measured against.
        """
        self.row_elems = row_elems
        self.buf = ctx.enqueue_create_buffer[Self.dtype](
            table.rows() * row_elems
        )
        if on_host:
            fill_uniform_host_rand(self.buf, Float64(lo), Float64(hi))
        else:
            fill_uniform(ctx, self.buf, seed, lo, hi)

    @staticmethod
    def bytes_for[page_size: Int](footprint: Footprint, row_elems: Int) -> Int:
        return (
            PageTable[page_size].rows_for(footprint)
            * row_elems
            * size_of[Self.dtype]()
        )

    def bytes(self) -> Int:
        return len(self.buf) * size_of[Self.dtype]()
