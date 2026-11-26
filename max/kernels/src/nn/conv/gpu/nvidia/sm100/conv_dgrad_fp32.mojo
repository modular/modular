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
"""The fp32 transposed convolution (cuDNN's backward-data) on SIMT FMAs.

Target family: Blackwell sm_100a (B200).

    dx[n, co, ho, wo] = sum_{k, r, s} dy[n, k, hi, wi] * w[k, co, r, s]
        with hi = (ho + ph - r*dh)/sh, wi = (wo + pw - s*dw)/sw, both exact

This is the gather form: every output element has one writer, the fp32 sum of
an element runs in one thread in a fixed order, and there are no atomics and no
zeroed output. A CTA owns a `TCO x TW` output tile of one batch image and one
output row. Two block roles share a ring of `STAGES` slots:

- the load block (threads 0-127) stages one chunk of `bk` input channels per
  slot: the input window the tile reads and the filter taps of that chunk;
- the compute block (threads 128-255) runs the FMAs of a slot into a register
  tile of `CHANNELS x COLUMNS` per thread, then writes the tile once through
  the epilogue.

A tap `(r, s)` reaches output column `u` of a tile only when the input index
`wo0 + u + pw - s*dw` is a multiple of `sw`, so a transposed convolution is
`sw` interleaved correlations. The CTA's window phase is uniform, so the tap
mask is a comptime function of the phase: no divergent tap loop and no
predicated FMA. The row axis works the same way, one uniform branch per filter
row.

The filter is read in the caller's own `[C_in, C_out, R, S]` order, so no
repack pass runs before the kernel. For a fixed input channel the `(co, r, s)`
taps of a tile are contiguous in global memory, so a warp's reads coalesce
even though the slot keeps the output channels innermost, which the FMA loop
needs for its 16-byte filter reads.

The compute block is a straight port of the `conv_dgrad_plan` Moka component,
in the same FMA order, so the two agree bit for bit.
"""

from std.math import align_up, ceildiv, fma
from std.utils.static_tuple import StaticTuple

from layout import Coord, TensorEngine, TensorLayout, TileTensor, row_major
from layout.tma_async import SharedMemBarrier
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_idx,
    thread_idx,
)
from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.memory import async_copy, external_memory, fence_mbarrier_init
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from max.gpu.sync import async_copy_arrive, barrier
from max.gpu.host.info import B200

from nn.conv.conv_utils import ConvSimdEpilogueFn

comptime _BLOCK_THREADS = 256
comptime _LOAD_THREADS = 128
"""The load block: threads `0 .. 127`. The compute block follows it."""
comptime _WORK_THREADS = _BLOCK_THREADS - _LOAD_THREADS
comptime _STAGES = 3
"""Ring depth. The 512 -> 1024 channel geometry takes 103.1, 102.5 and 102.7
us at 2, 3 and 4 stages, so the depth is not the limiter."""
comptime _THREAD_COLS = 8
"""Compute threads along the output columns."""
comptime _THREAD_ROWS = 16
"""Compute threads along the output channels."""
comptime _COLS = 8
"""Output columns a compute thread accumulates."""
comptime _CHANNELS = 4
"""Output channels a compute thread accumulates."""


struct _DgradTile[
    r: Int,
    s: Int,
    sh: Int,
    sw: Int,
    dh: Int,
    dw: Int,
    ph: Int,
    pw: Int,
    bk: Int,
    row_axis: Bool,
]:
    """Compile-time geometry of one dgrad instance, and its block bodies.

    Parameters:
        r: Filter rows.
        s: Filter columns.
        sh: Row stride of the forward convolution.
        sw: Column stride of the forward convolution (1 or 2).
        dh: Row dilation.
        dw: Column dilation.
        ph: Row padding.
        pw: Column padding.
        bk: Input channels one ring slot holds.
        row_axis: Whether the image has rows to walk. Without it `H_in = H_out
            = r = 1` and the grid's z axis is the batch alone.
    """

    comptime ROWS = Self.r
    """Filter rows, and the rows a slot's window holds per channel."""
    comptime RS = Self.ROWS * Self.s
    """Filter taps per `(k, co)` pair."""
    comptime TILE_COLS = _THREAD_COLS * _COLS
    """Output columns one CTA owns."""
    comptime TILE_CHANNELS = _THREAD_ROWS * _CHANNELS
    """Output channels one CTA owns."""
    comptime ACC = _CHANNELS * _COLS
    """Accumulators per compute thread."""
    comptime COL_SPAN = (Self.s - 1) * Self.dw
    """The largest column tap offset inside the window."""
    comptime COL_BIAS = align_up(Self.COL_SPAN + 1, Self.sw)
    """A multiple of `sw` at or above `COL_SPAN`: added to the window base's
    numerator so its division is floor-exact on a non-negative operand."""
    comptime ROW_SPAN = (Self.r - 1) * Self.dh
    comptime ROW_BIAS = align_up(Self.ROW_SPAN + 1, Self.sh)
    """The row axis' counterpart of `COL_BIAS`."""
    comptime WINDOW = (_COLS // Self.sw) * (_THREAD_COLS - 1) + ceildiv(
        _COLS - 1 + Self.COL_SPAN, Self.sw
    ) + 1
    """Input columns the widest tile reads."""
    comptime WINDOW_PAD = align_up(Self.WINDOW, 4)
    """A window row's slot width, a 16-byte multiple."""
    comptime WIN_ELEMS = Self.bk * Self.ROWS * Self.WINDOW_PAD
    comptime WGT_PLANE = Self.bk * Self.TILE_CHANNELS + 4
    """Words between the slot's tap planes: four more than a plane's `bk x co`
    words, which staggers the planes over the banks so a warp's copy of
    consecutive `(co, rs)` taps writes at most two lanes to one bank."""
    comptime WGT_ELEMS = Self.RS * Self.WGT_PLANE
    comptime SLOT_ELEMS = Self.WIN_ELEMS + Self.WGT_ELEMS
    comptime SMEM_BYTES = _STAGES * Self.SLOT_ELEMS * 4 + 2 * _STAGES * 8
    """The slots, then the full and empty barriers of every stage."""
    comptime ASYNC = Self.bk >= 8
    """Copy with `cp.async`. Below it a trip stages a few hundred elements and
    the load-store copy has the shorter tail."""
    comptime UNROLL = 4 if Self.bk >= 16 and Self.bk % 4 == 0 else 1
    """Input channels one trip of the FMA loop covers; unrolled so ptxas hoists
    the next channel's shared loads over this channel's FMAs."""
    comptime WIDE_WINDOW = Self.bk >= 16 and (_COLS // Self.sw) % 4 == 0
    """Read a thread's window as 16-byte shared loads."""

    @staticmethod
    @always_inline
    def row_hi(ho: Int, h_in: Int, row: Int) -> Int:
        """The input row filter row `row` reads for output row `ho`.

        Args:
            ho: Output row.
            h_in: Input rows.
            row: Filter row.

        Returns:
            The input row, or -1 when the row contributes nothing: its stride
            divisibility fails or it lies outside the image.
        """
        var numerator = ho + Self.ph - row * Self.dh + Self.ROW_BIAS
        var quotient, remainder = divmod(numerator, Self.sh)
        var hi = quotient - Self.ROW_BIAS // Self.sh
        if remainder != 0 or hi < 0 or hi >= h_in:
            return -1
        return hi

    @staticmethod
    @always_inline
    def window_element[
        row: Int
    ](
        idx: Int,
        first_col: Int,
        row_base: Int,
        plane: Int,
        w_in: Int,
        k0: Int,
        c_in: Int,
    ) -> Tuple[Bool, Int, Int]:
        """Locates window element `idx` of filter row `row` in one chunk.

        An element is zero where the channel is past the tensor, the column is
        outside `[0, w_in)` or the window, or `row_base` is negative (a filter
        row the output row does not have).

        Parameters:
            row: Filter row.

        Args:
            idx: Index into the row's `bk x WINDOW_PAD` cells.
            first_col: The window's first input column.
            row_base: Element offset of the input row, or negative.
            plane: Elements between input channels.
            w_in: Input columns.
            k0: The chunk's first input channel.
            c_in: Input channels.

        Returns:
            Whether the element is live, its offset in the input image (0 when
            dead), and its offset in the slot's window.
        """
        var k, col = divmod(idx, Self.WINDOW_PAD)
        var j = first_col + col
        var live = (
            row_base >= 0
            and k0 + k < c_in
            and col < Self.WINDOW
            and j >= 0
            and j < w_in
        )
        var src_off = 0
        if live:
            src_off = (k0 + k) * plane + row_base + j
        return (live, src_off, (k * Self.ROWS + row) * Self.WINDOW_PAD + col)

    @staticmethod
    @always_inline
    def load_window_row[
        row: Int
    ](
        dst: MutPointer[Float32, _, address_space=.SHARED],
        src: ImmPointer[Float32, _, address_space=.GLOBAL],
        t: Int,
        first_col: Int,
        row_base: Int,
        plane: Int,
        w_in: Int,
        k0: Int,
        c_in: Int,
    ):
        """Stages filter row `row`'s window of one chunk.

        Parameters:
            row: Filter row.

        Args:
            dst: The slot's window.
            src: The input at this CTA's batch image.
            t: Index of the calling load thread.
            first_col: The window's first input column.
            row_base: Element offset of the input row, or negative.
            plane: Elements between input channels.
            w_in: Input columns.
            k0: The chunk's first input channel.
            c_in: Input channels.
        """
        comptime total = Self.bk * Self.WINDOW_PAD
        comptime trips = ceildiv(total, _LOAD_THREADS)

        comptime for u in range(trips):
            var idx = t + _LOAD_THREADS * u
            if idx < total:
                var live, src_off, dst_off = Self.window_element[row](
                    idx, first_col, row_base, plane, w_in, k0, c_in
                )
                async_copy[size=4, fill=Float32(0)](
                    src.unsafe_offset(src_off),
                    dst.unsafe_offset(dst_off),
                    src_size=Int32(4 if live else 0),
                )

    @staticmethod
    @always_inline
    def load_filter(
        dst: MutPointer[Float32, _, address_space=.SHARED],
        src: ImmPointer[Float32, _, address_space=.GLOBAL],
        t: Int,
        k0: Int,
        c_in: Int,
        co0: Int,
        c_out: Int,
    ):
        """Stages the filter taps of one chunk from the caller's
        `[C_in, C_out, R, S]` order into the slot's `[rs][k][co]` order.

        For one input channel the tile's `(co, r, s)` taps are one contiguous
        run of the filter, so consecutive threads copy consecutive words (a
        warp's copy is one 128-byte line) and each thread's split of its word
        into `(co, rs)` is the same for every input channel and every chunk.
        The chunk must lie inside the tensor: `c_in` is a multiple of `bk`,
        `c_out` of the tile's channels.

        Args:
            dst: The slot's filter taps.
            src: The filter.
            t: Index of the calling load thread.
            k0: The chunk's first input channel.
            c_in: Input channels.
            co0: The tile's first output channel.
            c_out: Output channels.
        """
        comptime run = Self.TILE_CHANNELS * Self.RS
        comptime trips = ceildiv(run, _LOAD_THREADS)
        var first = (k0 * c_out + co0) * Self.RS
        comptime for u in range(trips):
            var j = t + _LOAD_THREADS * u
            if j < run:
                var co, rs = divmod(j, Self.RS)
                var slot = rs * Self.WGT_PLANE + co
                comptime for k in range(Self.bk):
                    async_copy[size=4](
                        src.unsafe_offset(first + k * c_out * Self.RS + j),
                        dst.unsafe_offset(slot + k * Self.TILE_CHANNELS),
                    )

    @staticmethod
    @always_inline
    def stage_small(
        win: MutPointer[Float32, _, address_space=.SHARED],
        wgt: MutPointer[Float32, _, address_space=.SHARED],
        dy: ImmPointer[Float32, _, address_space=.GLOBAL],
        filter: ImmPointer[Float32, _, address_space=.GLOBAL],
        t: Int,
        first_col: Int,
        ho: Int,
        h_in: Int,
        w_in: Int,
        k0: Int,
        c_in: Int,
        co0: Int,
        c_out: Int,
    ):
        """Stages one chunk for a geometry too small for `cp.async` to pay.

        Every global load of the chunk, window rows and filter taps, is issued
        before the first shared store, so the load block waits on memory once.

        Args:
            win: The slot's window.
            wgt: The slot's filter taps.
            dy: The input at this CTA's batch image.
            filter: The filter.
            t: Index of the calling load thread.
            first_col: The window's first input column.
            ho: The CTA's output row.
            h_in: Input rows.
            w_in: Input columns.
            k0: The chunk's first input channel.
            c_in: Input channels.
            co0: The tile's first output channel.
            c_out: Output channels.
        """
        comptime win_total = Self.bk * Self.WINDOW_PAD
        comptime win_trips = ceildiv(win_total, _LOAD_THREADS)
        comptime pairs = Self.bk * Self.TILE_CHANNELS
        comptime pair_trips = ceildiv(pairs, _LOAD_THREADS)
        var plane = h_in * w_in
        var win_values = StaticTuple[Float32, Self.ROWS * win_trips](
            fill=Float32(0)
        )
        var wgt_values = StaticTuple[Float32, pair_trips * Self.RS](
            fill=Float32(0)
        )

        comptime for row in range(Self.ROWS):
            var row_base = 0
            comptime if Self.row_axis:
                row_base = Self.row_hi(ho, h_in, row) * w_in
            comptime for u in range(win_trips):
                var idx = t + _LOAD_THREADS * u
                if idx < win_total:
                    var live, src_off, _ = Self.window_element[row](
                        idx, first_col, row_base, plane, w_in, k0, c_in
                    )
                    if live:
                        win_values[row * win_trips + u] = dy.unsafe_offset(
                            src_off
                        )[]
        comptime for u in range(pair_trips):
            var pair = t + _LOAD_THREADS * u
            if pair < pairs:
                var k, co = divmod(pair, Self.TILE_CHANNELS)
                if k0 + k < c_in and co0 + co < c_out:
                    var first = ((k0 + k) * c_out + co0 + co) * Self.RS
                    comptime for rs in range(Self.RS):
                        wgt_values[u * Self.RS + rs] = filter.unsafe_offset(
                            first + rs
                        )[]

        comptime for row in range(Self.ROWS):
            comptime for u in range(win_trips):
                var idx = t + _LOAD_THREADS * u
                if idx < win_total:
                    var _live, _src_off, dst_off = Self.window_element[row](
                        idx, first_col, 0, plane, w_in, k0, c_in
                    )
                    win.unsafe_offset(dst_off)[] = win_values[
                        row * win_trips + u
                    ]
        comptime for u in range(pair_trips):
            var pair = t + _LOAD_THREADS * u
            if pair < pairs:
                comptime for rs in range(Self.RS):
                    wgt.unsafe_offset(
                        rs * Self.WGT_PLANE + pair
                    )[] = wgt_values[u * Self.RS + rs]

    @staticmethod
    @always_inline
    def accumulate_row[
        row: Int, phase: Int
    ](
        win: MutPointer[Float32, _, address_space=.SHARED],
        wgt: MutPointer[Float32, _, address_space=.SHARED],
        tw: Int,
        tc: Int,
        mut acc: StaticTuple[Float32, Self.ACC],
    ):
        """One live filter row's taps for the thread's tile: every
        `(register, tap)` pair `sw` divides becomes an FMA, the rest is
        nothing at all.

        Parameters:
            row: Filter row.
            phase: The CTA's window phase, `0 .. sw - 1`.

        Args:
            win: The slot's window.
            wgt: The slot's filter taps.
            tw: The thread's column group.
            tc: The thread's channel group.
            acc: The thread's accumulator tile.
        """
        comptime per_thread = _COLS // Self.sw
        var dbase = per_thread * tw
        comptime tap_base = row * Self.s
        comptime vecs = ceildiv(
            ((Self.s - 1) * Self.dw + _COLS - 1 + phase) // Self.sw + 1, 4
        )

        for kb in range(0, Self.bk, Self.UNROLL):
            comptime for ku in range(Self.UNROLL):
                var k = kb + ku
                var xb = (k * Self.ROWS + row) * Self.WINDOW_PAD
                var wide = StaticTuple[Float32, vecs * 4](fill=Float32(0))
                comptime if Self.WIDE_WINDOW:
                    comptime for v in range(vecs):
                        var q = win.unsafe_load[width=4, alignment=16](
                            xb + dbase + 4 * v
                        )
                        comptime for e in range(4):
                            wide[4 * v + e] = q[e]
                comptime for s in range(Self.s):
                    var dv = StaticTuple[Float32, _COLS](fill=Float32(0))
                    comptime for i in range(_COLS):
                        comptime tap = (Self.s - 1 - s) * Self.dw + i + phase
                        comptime q = divmod(tap, Self.sw)
                        comptime if q[1] == 0:
                            comptime if Self.WIDE_WINDOW:
                                dv[i] = wide[q[0]]
                            else:
                                dv[i] = win.unsafe_offset(xb + dbase + q[0])[]
                    var vl = wgt.unsafe_load[
                        width=_CHANNELS, alignment=_CHANNELS * 4
                    ](
                        (tap_base + s) * Self.WGT_PLANE
                        + k * Self.TILE_CHANNELS
                        + tc * _CHANNELS
                    )
                    comptime for c in range(_CHANNELS):
                        comptime for i in range(_COLS):
                            comptime if (
                                divmod(
                                    (Self.s - 1 - s) * Self.dw + i + phase,
                                    Self.sw,
                                )[1]
                                == 0
                            ):
                                acc[c * _COLS + i] = fma(
                                    vl[c], dv[i], acc[c * _COLS + i]
                                )

    @staticmethod
    @always_inline
    def accumulate[
        phase: Int
    ](
        win: MutPointer[Float32, _, address_space=.SHARED],
        wgt: MutPointer[Float32, _, address_space=.SHARED],
        tw: Int,
        tc: Int,
        ho: Int,
        h_in: Int,
        mut acc: StaticTuple[Float32, Self.ACC],
    ):
        """The thread's whole tap set for one window phase: every live filter
        row, in order. A row is live or dead for the whole CTA, so the branch
        is uniform.

        Parameters:
            phase: The CTA's window phase.

        Args:
            win: The slot's window.
            wgt: The slot's filter taps.
            tw: The thread's column group.
            tc: The thread's channel group.
            ho: The CTA's output row.
            h_in: Input rows.
            acc: The thread's accumulator tile.
        """
        comptime if Self.row_axis:
            comptime for row in range(Self.ROWS):
                if Self.row_hi(ho, h_in, row) >= 0:
                    Self.accumulate_row[row, phase](win, wgt, tw, tc, acc)
        else:
            Self.accumulate_row[0, phase](win, wgt, tw, tc, acc)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_BLOCK_THREADS))
)
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))
@__name(t"conv_dgrad_fp32_r{r}s{s}_sw{sw}_bk{bk}")
def conv_dgrad_fp32_kernel[
    dy_layout: TensorLayout,
    filter_layout: TensorLayout,
    dy_origin: ImmOrigin,
    filter_origin: ImmOrigin,
    EpilogueFnType: ConvSimdEpilogueFn,
    DyEngine: TensorEngine,
    FilterEngine: TensorEngine,
    r: Int,
    s: Int,
    sh: Int,
    sw: Int,
    dh: Int,
    dw: Int,
    ph: Int,
    pw: Int,
    bk: Int,
    row_axis: Bool,
](
    dy: TileTensor[.float32, dy_layout, dy_origin, Engine=DyEngine],
    filter: TileTensor[
        .float32, filter_layout, filter_origin, Engine=FilterEngine
    ],
    epilogue_fn: EpilogueFnType,
):
    """Computes the data gradient of a convolution into `epilogue_fn`.

    Parameters:
        dy_layout: Memory layout of the input gradient.
        filter_layout: Memory layout of the filter.
        dy_origin: Origin of the input gradient.
        filter_origin: Origin of the filter.
        EpilogueFnType: Type of `epilogue_fn`.
        DyEngine: Storage engine of the input gradient.
        FilterEngine: Storage engine of the filter.
        r: Filter rows.
        s: Filter columns.
        sh: Row stride of the forward convolution.
        sw: Column stride of the forward convolution.
        dh: Row dilation.
        dw: Column dilation.
        ph: Row padding.
        pw: Column padding.
        bk: Input channels one ring slot holds.
        row_axis: Whether the image has rows to walk.

    Args:
        dy: The input gradient, `[N, C_in, H_in, W_in]` row-major.
        filter: The filter, `[C_in, C_out, R, S]` row-major.
        epilogue_fn: Stores each output element, once, with its
            `(n, co, ho, wo)` coordinate.
    """
    comptime T = _DgradTile[r, s, sh, sw, dh, dw, ph, pw, bk, row_axis]
    comptime assert dy.is_row_major, "the input gradient must be row-major"
    comptime assert filter.is_row_major, "the filter must be row-major"
    comptime assert sw == 1 or sw == 2, "the tap mask hoists only for sw 1, 2"
    comptime assert (
        _COLS % sw == 0
    ), "a thread's columns must tile by the column stride"

    var smem = external_memory[UInt8, address_space=.SHARED, alignment=1024]()
    var win_ring = smem.unsafe_bitcast[Float32]()
    var wgt_ring = win_ring.unsafe_offset(_STAGES * T.WIN_ELEMS)
    var bars = smem.unsafe_offset(_STAGES * T.SLOT_ELEMS * 4).unsafe_bitcast[
        SharedMemBarrier
    ]()
    # FULL: the load block's 128 threads arrive. EMPTY: the compute block's.
    var full = bars
    var empty = bars.unsafe_offset(_STAGES)

    var tid = thread_idx.x
    if tid == 0:
        comptime for i in range(_STAGES):
            full.unsafe_offset(i)[].init(Int32(_LOAD_THREADS))
            empty.unsafe_offset(i)[].init(Int32(_WORK_THREADS))
    fence_mbarrier_init()
    barrier()

    # The input and the filter may be the output of the grid launched just
    # before this one, so the wait precedes every global read.
    wait_on_dependent_grids()

    var c_in = Int(dy.dim[1]())
    var h_in = Int(dy.dim[2]())
    var w_in = Int(dy.dim[3]())
    var c_out = Int(filter.dim[1]())
    var h_out = sh * (h_in - 1) + dh * (r - 1) - 2 * ph + 1
    var w_out = sw * (w_in - 1) + dw * (s - 1) - 2 * pw + 1
    var k_tiles = ceildiv(c_in, bk)

    var co0 = block_idx.y * T.TILE_CHANNELS
    var wo0 = block_idx.x * T.TILE_COLS
    # The window's first input column j0 = floor((wo0 + pw - span) / sw) and
    # its phase (wo0 + pw - span) mod sw; the bias keeps the operand >= 0.
    var numerator = wo0 + pw - T.COL_SPAN + T.COL_BIAS
    var quotient, phase = divmod(numerator, sw)
    var j0 = quotient - T.COL_BIAS // sw
    var n = block_idx.z
    var ho = 0
    comptime if row_axis:
        var image, row = divmod(block_idx.z, h_out)
        n = image
        ho = row

    if tid < _LOAD_THREADS:
        var plane = h_in * w_in
        var dy_image = (
            dy.unsafe_ptr()
            .unsafe_address_space_cast[.GLOBAL]()
            .unsafe_offset(n * c_in * plane)
        )
        var filter_ptr = filter.unsafe_ptr().unsafe_address_space_cast[
            .GLOBAL
        ]()
        var stage = 0
        var parity = UInt32(1)
        for kc in range(k_tiles):
            empty.unsafe_offset(stage)[].wait(parity)
            var win = win_ring.unsafe_offset(stage * T.WIN_ELEMS)
            var wgt = wgt_ring.unsafe_offset(stage * T.WGT_ELEMS)
            comptime if T.ASYNC:
                comptime if row_axis:
                    comptime for row in range(r):
                        T.load_window_row[row](
                            win,
                            dy_image,
                            tid,
                            j0,
                            T.row_hi(ho, h_in, row) * w_in,
                            plane,
                            w_in,
                            kc * bk,
                            c_in,
                        )
                else:
                    T.load_window_row[0](
                        win, dy_image, tid, j0, 0, plane, w_in, kc * bk, c_in
                    )
                T.load_filter(wgt, filter_ptr, tid, kc * bk, c_in, co0, c_out)
            else:
                T.stage_small(
                    win,
                    wgt,
                    dy_image,
                    filter_ptr,
                    tid,
                    j0,
                    ho,
                    h_in,
                    w_in,
                    kc * bk,
                    c_in,
                    co0,
                    c_out,
                )
            var arrival = full.unsafe_offset(stage)
            comptime if T.ASYNC:
                async_copy_arrive(arrival[].unsafe_ptr())
            _ = arrival[].arrive()
            stage += 1
            if stage == _STAGES:
                stage = 0
                parity ^= 1
        return

    var rt = tid - _LOAD_THREADS
    var tc, tw = divmod(rt, _THREAD_COLS)
    var acc = StaticTuple[Float32, T.ACC](fill=Float32(0))
    var stage = 0
    var parity = UInt32(0)
    for _ in range(k_tiles):
        full.unsafe_offset(stage)[].wait(parity)
        var win = win_ring.unsafe_offset(stage * T.WIN_ELEMS)
        var wgt = wgt_ring.unsafe_offset(stage * T.WGT_ELEMS)
        comptime if sw == 1:
            T.accumulate[0](win, wgt, tw, tc, ho, h_in, acc)
        else:
            if phase == 0:
                T.accumulate[0](win, wgt, tw, tc, ho, h_in, acc)
            else:
                T.accumulate[1](win, wgt, tw, tc, ho, h_in, acc)
        _ = empty.unsafe_offset(stage)[].arrive()
        stage += 1
        if stage == _STAGES:
            stage = 0
            parity ^= 1

    var co_live = min(c_out - co0, T.TILE_CHANNELS)
    var col_live = min(w_out - wo0, T.TILE_COLS)
    var col = _COLS * tw
    comptime for c in range(_CHANNELS):
        var co = _CHANNELS * tc + c
        if co < co_live:
            var run = SIMD[.float32, _COLS]()
            comptime for i in range(_COLS):
                run[i] = acc[c * _COLS + i]
            # A thread's columns are consecutive, so a whole run is one call;
            # the ragged edge of the output goes element by element.
            if col + _COLS <= col_live:
                epilogue_fn[alignment=4]((n, co0 + co, ho, wo0 + col), run)
            else:
                comptime for i in range(_COLS):
                    if col + i < col_live:
                        epilogue_fn[alignment=4](
                            (n, co0 + co, ho, wo0 + col + i), run[i]
                        )
    launch_dependent_grids()


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(64))
)
@__name(t"conv_dgrad_fp32_tiny_s{s}_c{c_in}")
def conv_dgrad_fp32_tiny_kernel[
    dy_layout: TensorLayout,
    filter_layout: TensorLayout,
    dy_origin: ImmOrigin,
    filter_origin: ImmOrigin,
    EpilogueFnType: ConvSimdEpilogueFn,
    DyEngine: TensorEngine,
    FilterEngine: TensorEngine,
    s: Int,
    sw: Int,
    dw: Int,
    pw: Int,
    c_in: Int,
](
    dy: TileTensor[.float32, dy_layout, dy_origin, Engine=DyEngine],
    filter: TileTensor[
        .float32, filter_layout, filter_origin, Engine=FilterEngine
    ],
    epilogue_fn: EpilogueFnType,
):
    """Computes a one-image 1-D data gradient with a thread per output
    element, for outputs of at most 16 columns.

    At this size a tile's ring prologue is the whole cost, and a thread per
    element is one launch tick shorter. Thread `16 * co + wo` of the one CTA
    owns `dx[co, wo]`.

    Parameters:
        dy_layout: Memory layout of the input gradient.
        filter_layout: Memory layout of the filter.
        dy_origin: Origin of the input gradient.
        filter_origin: Origin of the filter.
        EpilogueFnType: Type of `epilogue_fn`.
        DyEngine: Storage engine of the input gradient.
        FilterEngine: Storage engine of the filter.
        s: Filter columns.
        sw: Column stride of the forward convolution.
        dw: Column dilation.
        pw: Column padding.
        c_in: Input channels.

    Args:
        dy: The input gradient, `[1, C_in, 1, W_in]` row-major.
        filter: The filter, `[C_in, C_out, 1, S]` row-major.
        epilogue_fn: Stores each output element with its `(n, co, ho, wo)`
            coordinate.
    """
    comptime assert dy.is_row_major, "the input gradient must be row-major"
    comptime assert filter.is_row_major, "the filter must be row-major"

    var c_out = Int(filter.dim[1]())
    var w_in = Int(dy.dim[3]())
    var w_out = sw * (w_in - 1) + dw * (s - 1) - 2 * pw + 1
    var co, wo = divmod(thread_idx.x, 16)
    if co >= c_out or wo >= w_out:
        return
    var dy_ptr = dy.unsafe_ptr()
    var filter_ptr = filter.unsafe_ptr()
    # Every tap's loads are issued unconditionally, clamped in range, and
    # masked afterwards, so they are independent and all in flight at once; a
    # branch around each load would chain them.
    var acc = Float32(0)
    comptime for tap in range(s):
        var t = wo + pw - tap * dw
        var quotient, remainder = divmod(t, sw)
        var live = t >= 0 and remainder == 0 and quotient < w_in
        var wi = quotient if live else 0
        comptime for c in range(c_in):
            var a = dy_ptr.unsafe_offset(c * w_in + wi)[]
            var b = filter_ptr.unsafe_offset((c * c_out + co) * s + tap)[]
            acc = fma(a if live else Float32(0), b, acc)
    epilogue_fn[alignment=4]((0, co, 0, wo), acc)


def _launch_dgrad[
    EpilogueFnType: ConvSimdEpilogueFn,
    //,
    r: Int,
    s: Int,
    sh: Int,
    sw: Int,
    dh: Int,
    dw: Int,
    ph: Int,
    pw: Int,
    bk: Int,
    row_axis: Bool,
](
    dy: TileTensor[address_space=.GENERIC, ...],
    filter: TileTensor[address_space=.GENERIC, ...],
    epilogue_fn: EpilogueFnType,
    ctx: DeviceContext,
) raises:
    comptime T = _DgradTile[r, s, sh, sw, dh, dw, ph, pw, bk, row_axis]
    comptime c_in = dy.static_shape[1]
    comptime c_out = filter.static_shape[1]
    comptime h_in = dy.static_shape[2]
    comptime w_in = dy.static_shape[3]
    comptime h_out = sh * (h_in - 1) + dh * (r - 1) - 2 * ph + 1
    comptime w_out = sw * (w_in - 1) + dw * (s - 1) - 2 * pw + 1
    comptime assert (
        c_in > 0 and c_out > 0 and h_in > 0 and w_in > 0
    ), "the launch is specialized on static extents"
    comptime assert row_axis or (
        h_in == 1 and h_out == 1
    ), "a geometry without a row axis is a single row"
    comptime assert not T.ASYNC or (
        c_in % bk == 0 and c_out % T.TILE_CHANNELS == 0
    ), "the channels must fill whole tiles"
    comptime kernel = conv_dgrad_fp32_kernel[
        dy.LayoutType,
        filter.LayoutType,
        ImmOrigin(dy.origin),
        ImmOrigin(filter.origin),
        EpilogueFnType,
        dy.Engine,
        filter.Engine,
        r,
        s,
        sh,
        sw,
        dh,
        dw,
        ph,
        pw,
        bk,
        row_axis,
    ]
    ctx.enqueue_function[kernel](
        dy.as_imm(),
        filter.as_imm(),
        host_args=(epilogue_fn,),
        grid_dim=(
            ceildiv(w_out, T.TILE_COLS),
            ceildiv(c_out, T.TILE_CHANNELS),
            h_out if row_axis else 1,
        ),
        block_dim=_BLOCK_THREADS,
        shared_mem_bytes=T.SMEM_BYTES,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(T.SMEM_BYTES)
        ),
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )


comptime _WIDE_1D = 0
"""1-D `512 -> 1024` channels, 550 columns, 7 taps, padding 3."""
comptime _TINY_1D = 1
"""1-D `2 -> 3` channels, 9 columns, 5 taps."""
comptime _SMALL_2D = 2
"""2-D `4 -> 6` channels, `8 x 9`, `3 x 5` taps, column stride 2, dilation 2,
padding 1."""
comptime _CASES = 3
comptime _GEOMETRIES = (
    (512, 1024, 1, 550, 1, 7, 1, 1, 1, 1, 0, 3),
    (2, 3, 1, 9, 1, 5, 1, 1, 1, 1, 0, 0),
    (4, 6, 8, 9, 3, 5, 1, 2, 2, 2, 1, 1),
)
"""The served geometries, one image each: `(C_in, C_out, H_in, W_in, R, S, sh,
sw, dh, dw, ph, pw)`."""


def _match_geometry(
    input_shape: Coord[Int, Int, Int, Int],
    filter_shape: Coord[Int, Int, Int, Int],
    output_shape: Coord[Int, Int, Int, Int],
    stride: Coord[Int, Int],
    dilation: Coord[Int, Int],
    padding: Coord[Int, Int],
) -> Int:
    """The served geometry a convolution is, or -1."""
    comptime for which in range(_CASES):
        comptime g = _GEOMETRIES[which]
        comptime h_out = g[6] * (g[2] - 1) + g[8] * (g[4] - 1) - 2 * g[10] + 1
        comptime w_out = g[7] * (g[3] - 1) + g[9] * (g[5] - 1) - 2 * g[11] + 1
        if (
            input_shape == Coord[Int, Int, Int, Int](1, g[0], g[2], g[3])
            and filter_shape
            == Coord[Int, Int, Int, Int](g[0], g[1], g[4], g[5])
            and output_shape == Coord[Int, Int, Int, Int](1, g[1], h_out, w_out)
            and stride == Coord[Int, Int](g[6], g[7])
            and dilation == Coord[Int, Int](g[8], g[9])
            and padding == Coord[Int, Int](g[10], g[11])
        ):
            return which
    return -1


def _launch_geometry[
    EpilogueFnType: ConvSimdEpilogueFn,
    //,
    which: Int,
](
    input: TileTensor[address_space=.GENERIC, ...],
    filter: TileTensor[address_space=.GENERIC, ...],
    epilogue_fn: EpilogueFnType,
    ctx: DeviceContext,
) raises:
    """Launches served geometry `which` on views of the operands whose extents
    are compile-time constants.

    The caller's tensors carry their extents at run time, which the kernel does
    not need: it knows them from the geometry, folds every bound, and takes
    8-byte arguments in place of 64-byte layouts. At the launch floor the
    smaller launch is a replay tick faster.
    """
    comptime g = _GEOMETRIES[which]
    var dy = TileTensor(
        input.unsafe_ptr().unsafe_bitcast[Float32](),
        row_major[1, g[0], g[2], g[3]](),
    )
    var w = TileTensor(
        filter.unsafe_ptr().unsafe_bitcast[Float32](),
        row_major[g[0], g[1], g[4], g[5]](),
    )
    comptime if which == _TINY_1D:
        comptime assert (
            g[1] <= 4
            and g[7] * (g[3] - 1) + g[9] * (g[5] - 1) - 2 * g[11] + 1 <= 16
        ), "the tiny kernel maps thread 16 * co + wo of one CTA"
        comptime kernel = conv_dgrad_fp32_tiny_kernel[
            dy.LayoutType,
            w.LayoutType,
            ImmOrigin(dy.origin),
            ImmOrigin(w.origin),
            EpilogueFnType,
            dy.Engine,
            w.Engine,
            g[5],
            g[7],
            g[9],
            g[11],
            g[0],
        ]
        # No programmatic launch: a plain launch is ordered after its
        # producer, and one block gains nothing from overlap.
        ctx.enqueue_function[kernel](
            dy.as_imm(),
            w.as_imm(),
            host_args=(epilogue_fn,),
            grid_dim=1,
            block_dim=64,
        )
    else:
        _launch_dgrad[
            g[4],
            g[5],
            g[6],
            g[7],
            g[8],
            g[9],
            g[10],
            g[11],
            16 if which == _WIDE_1D else 4,
            which == _SMALL_2D,
        ](dy, w, epilogue_fn, ctx)


def conv_dgrad_fp32[
    EpilogueFnType: ConvSimdEpilogueFn,
    //,
    output_type: DType = .float32,
](
    input: TileTensor[address_space=.GENERIC, ...],
    filter: TileTensor[address_space=.GENERIC, ...],
    output_shape: Coord[Int, Int, Int, Int],
    stride: Coord[Int, Int],
    dilation: Coord[Int, Int],
    padding: Coord[Int, Int],
    epilogue_fn: EpilogueFnType,
    ctx: DeviceContext,
) raises -> Bool:
    """Runs a transposed convolution on a dedicated SM100 kernel when the
    geometry is one it was measured against cuDNN's backward-data on (the
    first and third beat it; the `2 -> 3` channel case ties it at the launch
    floor), and hands every output element to `epilogue_fn`.

    The geometries, all fp32, one image, NCHW input and output and a
    `[C_in, C_out, R, S]` filter, each on a B200:

    - 1-D `512 -> 1024` channels, 550 columns, 7 taps, padding 3;
    - 1-D `2 -> 3` channels, 9 columns, 5 taps;
    - 2-D `4 -> 6` channels, `8 x 9`, `3 x 5` taps, column stride 2, dilation
      2, padding 1.

    Anything else, and any other dtype or device, returns False with nothing
    enqueued.

    Parameters:
        EpilogueFnType: Type of `epilogue_fn` (inferred).
        output_type: Element type of the output the epilogue writes; only
            fp32 is served.

    Args:
        input: The input gradient, `[N, C_in, H_in, W_in]`.
        filter: The filter, `[C_in, C_out, R, S]`.
        output_shape: Shape of the output, `[N, C_out, H_out, W_out]`.
        stride: Stride along the height and width.
        dilation: Dilation along the height and width.
        padding: Symmetric padding along the height and width.
        epilogue_fn: Stores each output element once with its `(n, co, ho,
            wo)` coordinate. It may be called with a run of consecutive
            columns, and must honor its `alignment` parameter.
        ctx: Device context for the launch.

    Returns:
        True if a kernel ran.

    Raises:
        If the launch fails.
    """
    comptime if not (
        ctx.default_device_info == B200
        and input.dtype == .float32
        and filter.dtype == .float32
        and output_type == .float32
        and input.flat_rank == 4
        and filter.flat_rank == 4
        and input.is_row_major
        and filter.is_row_major
    ):
        return False
    else:
        var which = _match_geometry(
            Coord(
                Int(input.dim[0]()),
                Int(input.dim[1]()),
                Int(input.dim[2]()),
                Int(input.dim[3]()),
            ),
            Coord(
                Int(filter.dim[0]()),
                Int(filter.dim[1]()),
                Int(filter.dim[2]()),
                Int(filter.dim[3]()),
            ),
            output_shape,
            stride,
            dilation,
            padding,
        )
        comptime for served in range(_CASES):
            if which == served:
                _launch_geometry[served](input, filter, epilogue_fn, ctx)
                return True
        return False


def conv_dgrad_fp32(
    input: TileTensor[address_space=.GENERIC, ...],
    filter: TileTensor[address_space=.GENERIC, ...],
    output: TileTensor[mut=True, address_space=.GENERIC, ...],
    stride: Coord[Int, Int],
    dilation: Coord[Int, Int],
    padding: Coord[Int, Int],
    ctx: DeviceContext,
) raises -> Bool:
    """Runs a transposed convolution into `output` on a dedicated SM100 kernel
    when `conv_dgrad_fp32` with an epilogue would.

    The kernel stores through a view of `output` whose extents are the served
    geometry's, known at compile time, so a store costs no stride arithmetic.

    Args:
        input: The input gradient, `[N, C_in, H_in, W_in]`.
        filter: The filter, `[C_in, C_out, R, S]`.
        output: The output, `[N, C_out, H_out, W_out]`.
        stride: Stride along the height and width.
        dilation: Dilation along the height and width.
        padding: Symmetric padding along the height and width.
        ctx: Device context for the launch.

    Returns:
        True if a kernel ran.

    Raises:
        If the launch fails.
    """
    comptime if not (
        ctx.default_device_info == B200
        and input.dtype == .float32
        and filter.dtype == .float32
        and output.dtype == .float32
        and input.flat_rank == 4
        and filter.flat_rank == 4
        and output.flat_rank == 4
        and input.is_row_major
        and filter.is_row_major
        and output.is_row_major
    ):
        return False
    else:
        var which = _match_geometry(
            Coord(
                Int(input.dim[0]()),
                Int(input.dim[1]()),
                Int(input.dim[2]()),
                Int(input.dim[3]()),
            ),
            Coord(
                Int(filter.dim[0]()),
                Int(filter.dim[1]()),
                Int(filter.dim[2]()),
                Int(filter.dim[3]()),
            ),
            Coord(
                Int(output.dim[0]()),
                Int(output.dim[1]()),
                Int(output.dim[2]()),
                Int(output.dim[3]()),
            ),
            stride,
            dilation,
            padding,
        )
        comptime for served in range(_CASES):
            if which == served:
                comptime g = _GEOMETRIES[served]
                var view = TileTensor(
                    output.unsafe_ptr().unsafe_bitcast[Float32](),
                    row_major[
                        1,
                        g[1],
                        g[6] * (g[2] - 1) + g[8] * (g[4] - 1) - 2 * g[10] + 1,
                        g[7] * (g[3] - 1) + g[9] * (g[5] - 1) - 2 * g[11] + 1,
                    ](),
                )

                @always_inline
                def store[
                    dtype: DType, width: SIMDLength, alignment: Int = 1
                ](coords: Coord, val: SIMD[dtype, width]) {var view}:
                    view.store[width=Int(width), alignment=alignment](
                        coords, val.cast[.float32]()
                    )

                _launch_geometry[served](input, filter, store, ctx)
                return True
        return False
