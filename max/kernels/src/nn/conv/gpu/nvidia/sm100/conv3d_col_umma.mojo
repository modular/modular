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
"""SM100 3x3x3 conv3d on column tiles, for the VAE residual blocks.

Target family: Blackwell sm_100a, 1-SM UMMA, fp16 and bf16 with fp32
accumulation, or fp32 operands rounded to tf32 by the tensor core. NDHWC
activation and output, stride 1, dilation 1, a 3x3x3 filter, no padding along
the depth (the causal convs arrive pre-padded) and one pixel of padding along
the height and the width.

    out[d, h, w, f] = sum_{q, r, s, c} in[d + q, h + r - 1, w + s - 1, c]
                                       * w[f, q, r, s, c]

The rank-5 TMA im2col loader stages a swizzled slot wrongly at a 32-byte span,
so this kernel does not use it. A tile is `WC`
consecutive columns of `HP` rows, 128 output rows in all, of one depth `d`.
One plain tiled TMA box fetches the activation window of one (q, s, channel
block) tap pair, with zero fill outside the image as the padding, and the MMA
reads it at 0, 1 and 2 rows' offset for the three filter rows `r`. Nine
windows per channel block serve 27 taps.

The activation is read through a permuted view `(D, W, H, C)`, so the box
`(BK channels, rows of one column, WC columns)` lands in shared memory
row-major as `row = column * rows + h`. The filter-row shift is a whole-row
shift only when the columns are contiguous in the slot: a single-column tile
(`H >= 128`) takes a window of `HP + 8` rows, and a tile of several short
columns takes exactly the tile's rows per column, so the last two output rows
of every column read the next column's first rows (values that are never
stored: the output box is clipped at `H`).

Schedule. Seven warps: four epilogue warps, a filter-load warp, an
activation-load warp and an MMA warp. The grid is persistent: a CTA walks
`units` passes of `MPASS` tiles, pass `cta + u * grid`. A pass runs M-inner, so
a filter box is read by every tile of the pass, and the last `TAIL` K steps run
tile-outer with their filter boxes held, so tile 0's accumulator completes (and
its epilogue starts) while the later tiles still accumulate. With window reuse
(`RU`, the fp32 schedule) a pass is `MPASS` adjacent columns and a block loads
`MPASS + 2` windows instead of `3 * MPASS`.

Warp roles (7 warps, 224 threads):

| warps | role                                                      |
|-------|-----------------------------------------------------------|
| 0-3   | epilogue: TMEM -> smem -> TMA store, or the fused epilogue |
| 4     | filter load: one TMA box per K step                       |
| 5     | activation load: one TMA box per window                   |
| 6     | MMA issue, TMEM allocation                                |

The filter is `[F, K]` with `K = ((q * 3 + r) * 3 + s) * C + c`, which
`pack_conv3d_filter` builds from either FCQRS or QRSCF.
"""

from std.collections import Array
from std.math import align_up, ceildiv
from std.sys import size_of
from std.utils.static_tuple import StaticTuple

from layout import TensorEngine, TensorLayout, TileTensor
from layout.tma_async import SharedMemBarrier
from linalg.arch.sm100 import MmaOpSM100_SS
from linalg.arch.sm100.mma import _create_mma_desc_k_major
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    grid_dim,
    thread_idx,
)
from max.gpu.compute.arch.mma_nvidia_sm100 import MMASmemDescriptor
from max.gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_fence_after,
    tcgen05_fence_before,
    tcgen05_release_allocation_lock,
)
from max.gpu.host import DeviceBuffer, DeviceContext, FuncAttribute
from max.gpu.host.nvidia.tma import (
    TensorMapSwizzle,
    TMADescriptor,
    create_tma_descriptor,
    prefetch_tma_descriptor,
)
from max.gpu.memory import (
    cp_async_bulk_tensor_global_shared_cta,
    cp_async_bulk_tensor_shared_cluster_global,
    external_memory,
    fence_async_view_proxy,
    fence_mbarrier_init,
)
from max.gpu.primitives.cluster import elect_one_sync
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from max.gpu.sync import (
    barrier,
    cp_async_bulk_commit_group,
    cp_async_bulk_wait_group,
    syncwarp,
)
from std.utils.coord import Idx
from structured_kernels.barriers import WarpGroupBarrier

from nn.conv.conv_utils import ConvSimdComputeFn, ConvSimdEpilogueFn

from .conv_epilogue import apply_conv_epilogue
from .conv3d_umma_common import (
    RingCursor,
    pack_conv3d_filter,
    stage_columns_swizzled,
)


struct Conv3dColUmma[
    act_type: DType,
    cin: Int,
    fout: Int,
    height: Int,
    width: Int,
    has_epilogue_fn: Bool = False,
    has_compute_fn: Bool = False,
]:
    """The 3x3x3 column-tile conv3d kernel and its launch geometry.

    Every schedule number derives from the channel counts, the element size
    and the image height, as the measured campaign configuration does.

    Parameters:
        act_type: Element type of the activation, filter and output
            (`float16`, `bfloat16`, or `float32` for tf32 operands).
        cin: Input channels.
        fout: Output channels.
        height: Input (and output) height; one pixel of padding each side.
        width: Input (and output) width.
        has_epilogue_fn: An elementwise epilogue stores each output value.
        has_compute_fn: An elementwise epilogue transforms each output value.
    """

    # ========== Problem family ==========
    comptime EB = size_of[Self.act_type]()
    comptime BM = 128
    comptime BN = 192 if Self.fout % 192 == 0 else 96
    """Output channels per tile. An UMMA of N = 192 reads 10 KB of shared
    memory per 98 clocks of math against 7 KB per 49 at N = 96, which takes
    the tensor core off the shared-memory ceiling; 96 is the width a
    96-channel output allows."""
    comptime NH = Self.BN // 96
    comptime SWIZZLE = TensorMapSwizzle.SWIZZLE_64B
    comptime BK = 64 // Self.EB
    """Channels per K step: one 64-byte swizzle row."""
    comptime NCB = Self.cin // Self.BK
    comptime NGROUPS = 9 * Self.NCB
    """(channel block, depth tap, column tap) groups; each is three UMMAs."""
    comptime NBLK = 3 * Self.NCB
    """Window-reuse blocks: one per (channel block, depth tap)."""
    comptime HP = 32 if Self.height + 2 <= 32 else 128
    """Rows per column in the slot: a column that fits 32 rows with its halo
    shares a tile with three others; any other is cut in 128s."""
    comptime WC = 128 // Self.HP
    comptime RU = 1 if (Self.EB == 4 and Self.WC == 1) else 0
    """Window reuse across adjacent output columns, on at fp32 where the
    shared-memory port is the limit."""
    comptime WIN_ROWS = Self.HP + 8 if Self.WC == 1 else 128
    comptime HT = ceildiv(Self.height, Self.HP)
    comptime WT = ceildiv(Self.width, Self.WC)
    comptime MPASS = (4 if Self.RU == 1 else 3) if Self.BN == 96 else 2
    """Tiles per pass: as many as the 512 TMEM columns leave room for."""
    comptime WGRP = Self.width // Self.MPASS
    comptime STAGES = 2 if Self.RU == 1 else 3
    comptime TAIL = 0 if Self.RU == 1 else (3 if Self.BN == 192 else 9)
    """K steps walked tile-outer with their filter boxes held."""
    comptime TG = Self.TAIL // 3
    comptime FST = 12 if Self.RU == 1 else (
        (10 if Self.EB == 4 else Self.TAIL + 3) if Self.BN == 192 else Self.TAIL
    )
    comptime SCST = 1 if Self.EB == 4 else 2
    comptime NACC = 512 // Self.BN
    comptime RSLOTS = Self.STAGES * Self.MPASS

    # ========== Rings and shared memory ==========
    comptime A_BYTES = Self.WIN_ROWS * Self.BK * Self.EB
    comptime B_BYTES = Self.BN * Self.BK * Self.EB
    comptime C_BYTES = 128 * 32 * Self.EB
    comptime A_ELEMS = Self.A_BYTES // Self.EB
    comptime B_ELEMS = Self.B_BYTES // Self.EB
    comptime C_ELEMS = Self.C_BYTES // Self.EB
    comptime A_OFF = 0
    comptime B_OFF = align_up(Self.A_OFF + Self.RSLOTS * Self.A_BYTES, 1024)
    comptime C_OFF = align_up(Self.B_OFF + Self.FST * Self.B_BYTES, 1024)
    comptime BAR_OFF = Self.C_OFF + 3 * Self.SCST * Self.C_BYTES

    comptime A_FULL = 0
    comptime A_EMPTY = Self.A_FULL + Self.RSLOTS
    comptime B_FULL = Self.A_EMPTY + Self.RSLOTS
    comptime B_EMPTY = Self.B_FULL + Self.FST
    comptime ACC_FULL = Self.B_EMPTY + Self.FST
    comptime ACC_EMPTY = Self.ACC_FULL + Self.NACC
    comptime TMEM_FREE = Self.ACC_EMPTY + Self.NACC
    comptime NBAR = Self.TMEM_FREE + 1
    comptime TMEM_ADDR_OFF = Self.BAR_OFF + Self.NBAR * 8
    comptime SMEM_BYTES = Self.TMEM_ADDR_OFF + 16

    # ========== Threads ==========
    comptime EPI_THREADS = 128
    comptime NUM_THREADS = Self.EPI_THREADS + 3 * WARP_SIZE
    comptime TMEM_COLS = 512

    comptime MmaEpiSync = WarpGroupBarrier[Self.EPI_THREADS + WARP_SIZE, 1]
    comptime EpiSync = WarpGroupBarrier[Self.EPI_THREADS, 2]

    comptime MmaOp = MmaOpSM100_SS[
        .float32,
        Self.act_type,
        Self.act_type,
        (Self.BM, Self.BN, Self.BK),
        (Self.BM, Self.BN, 32 // Self.EB),
        accum_type=.float32,
        cta_group=1,
        a_swizzle=Self.SWIZZLE,
        b_swizzle=Self.SWIZZLE,
        transpose_b=True,
    ]

    comptime fused = Self.has_epilogue_fn or Self.has_compute_fn

    @staticmethod
    @always_inline
    def _checks():
        comptime assert (
            Self.cin % Self.BK == 0
        ), "a channel block divides the channels"
        comptime assert Self.fout % Self.BN == 0 and (
            Self.BN == 96 or Self.BN == 192
        ), "the output tile is 96 or 192 columns"
        comptime assert Self.EB == 2 or Self.EB == 4, "16-bit or tf32 operands"
        comptime assert Self.WC * Self.HP == Self.BM, "a tile is 128 rows"
        comptime assert Self.WC == 1 or (
            Self.HT == 1
            and Self.HP >= Self.height + 2
            and Self.width % Self.WC == 0
        ), (
            "a multi-column tile needs a short image column: one tile per"
            " column group, window rows per column covering the halo, whole"
            " groups"
        )
        comptime assert (
            Self.TAIL % 3 == 0 and Self.TG <= Self.NGROUPS
        ), "the tail is whole depth and column taps"
        comptime assert Self.RU == 0 or (
            Self.width % Self.MPASS == 0 and Self.FST >= 9
        ), (
            "window reuse: whole passes of adjacent columns, a block's boxes"
            " resident"
        )
        comptime assert (
            Self.MPASS * Self.BN <= 512 and Self.NACC >= Self.MPASS
        ), "a pass fits the TMEM columns"
        comptime assert (
            Self.RU == 1 or Self.FST >= Self.TAIL
        ), "the held filter boxes fit the ring"
        comptime assert (
            Self.RU == 1 or Self.RSLOTS >= Self.TG * Self.MPASS
        ), "the held activation windows fit the ring"
        comptime assert (
            Self.SMEM_BYTES <= 232448
        ), "shared memory exceeds the 227 KiB of an SM100 CTA"

    # ========== Kernel ==========

    @staticmethod
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.NUM_THREADS)
        )
    )
    @__llvm_arg_metadata(act_desc, `nvvm.grid_constant`)
    @__llvm_arg_metadata(filt_desc, `nvvm.grid_constant`)
    @__llvm_arg_metadata(out_desc, `nvvm.grid_constant`)
    @__name(t"sm100_conv3d_col_umma_{Self.act_type}_{Self.cin}x{Self.fout}")
    def run[
        output_type: DType,
        output_layout: TensorLayout,
        output_origin: MutOrigin,
        OutputEngine: TensorEngine,
        EpilogueFnType: ConvSimdEpilogueFn,
        ComputeFnType: ConvSimdComputeFn,
    ](
        act_desc: TMADescriptor,
        filt_desc: TMADescriptor,
        out_desc: TMADescriptor,
        output: TileTensor[
            output_type,
            output_layout,
            output_origin,
            Engine=OutputEngine,
        ],
        units: Int32,
        out_depth: Int32,
        epilogue_fn: EpilogueFnType,
        compute_fn: ComputeFnType,
    ):
        """Kernel entry point.

        Args:
            act_desc: Descriptor for the NDHWC activation viewed `(D, W, H,
                C)`, box `(1, WC, rows per column, BK)`.
            filt_desc: Descriptor for the `[F, K]` filter, box `(BN, BK)`.
            out_desc: Descriptor for the NDHWC output viewed `(D*H*W, H, F)`,
                box `(WC, HP, 32)`; unused with a fused epilogue.
            output: Output, NDHWC; written by the kernel when
                `has_compute_fn`, unused by the TMA path otherwise.
            units: Passes each CTA walks.
            out_depth: Output depth.
            epilogue_fn: Storing epilogue, used when `has_epilogue_fn`.
            compute_fn: Transforming epilogue, used when `has_compute_fn`.
        """
        Self._checks()
        comptime assert output_type == Self.act_type
        var smem = external_memory[
            UInt8, address_space=.SHARED, alignment=1024
        ]()
        var sa = smem.unsafe_offset(Self.A_OFF).unsafe_bitcast[
            Scalar[Self.act_type]
        ]()
        var sb = smem.unsafe_offset(Self.B_OFF).unsafe_bitcast[
            Scalar[Self.act_type]
        ]()
        var sc = smem.unsafe_offset(Self.C_OFF).unsafe_bitcast[
            Scalar[Self.act_type]
        ]()
        var bars = smem.unsafe_offset(Self.BAR_OFF).unsafe_bitcast[
            SharedMemBarrier
        ]()
        var tmem_slot = smem.unsafe_offset(Self.TMEM_ADDR_OFF).unsafe_bitcast[
            UInt32
        ]()

        var tid = thread_idx.x
        var warp = tid // WARP_SIZE
        var cta = block_idx.x
        var ncta = grid_dim.x
        var cty = block_idx.y
        var n_units = Int(units)

        if tid == 0:
            var a_desc = Pointer(to=act_desc).unsafe_bitcast[NoneType]()
            var b_desc = Pointer(to=filt_desc).unsafe_bitcast[NoneType]()
            var c_desc = Pointer(to=out_desc).unsafe_bitcast[NoneType]()
            prefetch_tma_descriptor(a_desc)
            prefetch_tma_descriptor(b_desc)
            comptime if not Self.fused:
                prefetch_tma_descriptor(c_desc)
            comptime for i in range(Self.A_FULL, Self.ACC_EMPTY):
                bars.unsafe_offset(i)[].init(1)
            # Every epilogue thread arrives once when it is done with an
            # accumulator slot, and once more when it is done with TMEM.
            comptime for i in range(Self.ACC_EMPTY, Self.TMEM_FREE + 1):
                bars.unsafe_offset(i)[].init(Int32(Self.EPI_THREADS))
        fence_mbarrier_init()
        barrier()

        if warp < 4:
            Self._epilogue(
                out_desc,
                output,
                sc,
                bars,
                tmem_slot,
                tid,
                cta,
                ncta,
                cty,
                n_units,
                Int(out_depth),
                epilogue_fn,
                compute_fn,
            )
        elif warp == 4:
            Self._filter_load(filt_desc, sb, bars, n_units, cty)
        elif warp == 5:
            Self._act_load(act_desc, sa, bars, cta, ncta, n_units)
        else:
            Self._mma(sa, sb, bars, tmem_slot, n_units)

    # ----- tile coordinates -----

    @staticmethod
    @always_inline
    def _tile_origin(cu: Int, mt: Int) -> Tuple[Int, Int, Int]:
        """(depth, first row, first column) of tile `mt` of pass `cu`."""
        comptime if Self.RU == 1:
            var dd, rem = divmod(cu, Self.HT * Self.WGRP)
            var htile, wgi = divmod(rem, Self.WGRP)
            return (dd, htile * Self.HP, wgi * Self.MPASS + mt)
        else:
            var tile = cu * Self.MPASS + mt
            var cg, htile = divmod(tile, Self.HT)
            var dd, wtile = divmod(cg, Self.WT)
            return (dd, htile * Self.HP, wtile * Self.WC)

    # ----- filter load warp -----

    @staticmethod
    @always_inline
    def _filter_load(
        filt_desc: TMADescriptor,
        sb: MutPointer[Scalar[Self.act_type], _, address_space=.SHARED],
        bars: MutPointer[SharedMemBarrier, _, address_space=.SHARED],
        n_units: Int,
        cty: Int,
    ):
        """Loads the filter boxes of every pass in the order the MMA warp
        consumes them."""
        var desc = Pointer(to=filt_desc).unsafe_bitcast[NoneType]()
        # The filter is the packing kernel's output.
        wait_on_dependent_grids()
        var ring = RingCursor[Self.FST]()
        for _ in range(n_units):
            comptime if Self.RU == 1:
                # A block's nine boxes in the order its windows first need
                # them: filter column outermost.
                for cb in range(Self.NCB):
                    comptime for q in range(3):
                        comptime for fs in range(3):
                            comptime for fr in range(3):
                                Self._load_filter_box(
                                    desc,
                                    sb,
                                    bars,
                                    ring,
                                    (Self.cin * ((q * 3 + fr) * 3 + fs))
                                    + cb * Self.BK,
                                    cty,
                                )
            else:
                # K step = ((cb * 3 + q) * 3 + s) * 3 + r -> the filter box
                # of tap (q * 3 + r) * 3 + s, channel block cb.
                for cb in range(Self.NCB):
                    comptime for q in range(3):
                        comptime for sx in range(3):
                            comptime for r in range(3):
                                Self._load_filter_box(
                                    desc,
                                    sb,
                                    bars,
                                    ring,
                                    Self.cin * ((q * 3 + r) * 3 + sx)
                                    + cb * Self.BK,
                                    cty,
                                )

    @staticmethod
    @always_inline
    def _load_filter_box(
        desc: OpaquePointer[mut=False, _],
        sb: MutPointer[Scalar[Self.act_type], _, address_space=.SHARED],
        bars: MutPointer[SharedMemBarrier, _, address_space=.SHARED],
        mut ring: RingCursor[Self.FST],
        k0: Int,
        cty: Int,
    ):
        bars.unsafe_offset(Self.B_EMPTY + ring.slot())[].wait(ring.phase ^ 1)
        if elect_one_sync():
            var full = bars.unsafe_offset(Self.B_FULL + ring.slot())
            full[].expect_bytes(Int32(Self.B_BYTES))
            cp_async_bulk_tensor_shared_cluster_global[cta_group=1](
                sb.unsafe_offset(ring.slot() * Self.B_ELEMS),
                desc,
                full,
                (k0, cty * Self.BN),
            )
        ring.advance()

    # ----- activation load warp -----

    @staticmethod
    @always_inline
    def _act_load(
        act_desc: TMADescriptor,
        sa: MutPointer[Scalar[Self.act_type], _, address_space=.SHARED],
        bars: MutPointer[SharedMemBarrier, _, address_space=.SHARED],
        cta: Int,
        ncta: Int,
        n_units: Int,
    ):
        """Loads the activation windows of every pass in the order the MMA
        warp consumes them."""
        var desc = Pointer(to=act_desc).unsafe_bitcast[NoneType]()
        # The activation is the predecessor's output.
        wait_on_dependent_grids()
        var ring = RingCursor[Self.RSLOTS]()
        for u in range(n_units):
            var cu = cta + u * ncta
            comptime if Self.RU == 1:
                # Window `j` of a block is the image column `w0 + j - 1`,
                # rows of tile half `htile`, depth `dd + q`; the pass's tiles
                # read it at the filter column that lines them up.
                var dd, row0, w0 = Self._tile_origin(cu, 0)
                for cb in range(Self.NCB):
                    comptime for q in range(3):
                        for j in range(Self.MPASS + 2):
                            Self._load_window(
                                desc,
                                sa,
                                bars,
                                ring,
                                cb * Self.BK,
                                row0 - 1,
                                w0 + j - 1,
                                dd + q,
                            )
            else:
                var dds = Array[Int32, Self.MPASS](uninitialized=True)
                var rows = Array[Int32, Self.MPASS](uninitialized=True)
                var cols = Array[Int32, Self.MPASS](uninitialized=True)
                comptime for t in range(Self.MPASS):
                    var dd, row0, col0 = Self._tile_origin(cu, t)
                    dds[t] = Int32(dd)
                    rows[t] = Int32(row0)
                    cols[t] = Int32(col0)
                # Group = (cb * 3 + q) * 3 + s.
                for cb in range(Self.NCB):
                    comptime for q in range(3):
                        comptime for sx in range(3):
                            comptime for t in range(Self.MPASS):
                                Self._load_window(
                                    desc,
                                    sa,
                                    bars,
                                    ring,
                                    cb * Self.BK,
                                    Int(rows[t]) - 1,
                                    Int(cols[t]) + sx - 1,
                                    Int(dds[t]) + q,
                                )

    @staticmethod
    @always_inline
    def _load_window(
        desc: OpaquePointer[mut=False, _],
        sa: MutPointer[Scalar[Self.act_type], _, address_space=.SHARED],
        bars: MutPointer[SharedMemBarrier, _, address_space=.SHARED],
        mut ring: RingCursor[Self.RSLOTS],
        channel: Int,
        row: Int,
        col: Int,
        depth: Int,
    ):
        """One window into ring slot `ring.slot()`; the box reads zeros
        outside the image, which is the padding."""
        bars.unsafe_offset(Self.A_EMPTY + ring.slot())[].wait(ring.phase ^ 1)
        if elect_one_sync():
            var full = bars.unsafe_offset(Self.A_FULL + ring.slot())
            full[].expect_bytes(Int32(Self.A_BYTES))
            cp_async_bulk_tensor_shared_cluster_global[cta_group=1](
                sa.unsafe_offset(ring.slot() * Self.A_ELEMS),
                desc,
                full,
                (channel, row, col, depth),
            )
        ring.advance()

    # ----- MMA warp -----

    @staticmethod
    @always_inline
    def _issue3(
        mma_op: Self.MmaOp,
        a_desc: MMASmemDescriptor,
        b_desc: MMASmemDescriptor,
        a_slot: Int,
        b_slot0: Int,
        b_slot1: Int,
        b_slot2: Int,
        acc: UInt32,
        first: Bool,
    ):
        """The three filter rows against one window: `A` read `r` rows on
        (`r * BK * EB` bytes) against the box of row `r`."""
        if elect_one_sync():
            comptime row_bytes = Self.BK * Self.EB
            var a = a_desc + a_slot * Self.A_BYTES
            mma_op.mma_from_desc(a, b_desc + b_slot0 * Self.B_BYTES, acc, first)
            mma_op.mma_from_desc(
                a + row_bytes, b_desc + b_slot1 * Self.B_BYTES, acc, False
            )
            mma_op.mma_from_desc(
                a + 2 * row_bytes, b_desc + b_slot2 * Self.B_BYTES, acc, False
            )

    @staticmethod
    @always_inline
    def _release(
        mma_op: Self.MmaOp,
        bars: MutPointer[SharedMemBarrier, _, address_space=.SHARED],
        barrier_index: Int,
    ):
        """Frees a ring slot once the UMMAs issued so far have read it."""
        if elect_one_sync():
            mma_op.commit(bars.unsafe_offset(barrier_index))

    @staticmethod
    @always_inline
    def _mma(
        sa: MutPointer[Scalar[Self.act_type], _, address_space=.SHARED],
        sb: MutPointer[Scalar[Self.act_type], _, address_space=.SHARED],
        bars: MutPointer[SharedMemBarrier, _, address_space=.SHARED],
        tmem_slot: MutPointer[UInt32, _, address_space=.SHARED],
        n_units: Int,
    ):
        """Allocates TMEM, issues every UMMA of every pass, and frees TMEM."""
        tcgen05_alloc[1](tmem_slot, UInt32(Self.TMEM_COLS))
        syncwarp()
        var tmem = tmem_slot[]
        Self.MmaEpiSync.arrive()

        var mma_op = Self.MmaOp()
        var a_desc = _create_mma_desc_k_major[Self.act_type, Self.SWIZZLE](sa)
        var b_desc = _create_mma_desc_k_major[Self.act_type, Self.SWIZZLE](sb)

        var ra = RingCursor[Self.RSLOTS]()
        var rb = RingCursor[Self.FST]()
        var acc = RingCursor[Self.NACC]()

        for _ in range(n_units):
            # This pass's accumulators.
            var acc_stage = Array[Int, Self.MPASS](uninitialized=True)
            comptime for mt in range(Self.MPASS):
                bars.unsafe_offset(Self.ACC_EMPTY + acc.slot())[].wait(
                    acc.phase ^ 1
                )
                acc_stage[mt] = acc.slot()
                acc.advance()
            tcgen05_fence_after()

            comptime if Self.RU == 1:
                Self._mma_window_reuse(
                    mma_op,
                    a_desc,
                    b_desc,
                    bars,
                    tmem,
                    ra,
                    rb,
                    acc_stage,
                )
            else:
                Self._mma_tail(
                    mma_op,
                    a_desc,
                    b_desc,
                    bars,
                    tmem,
                    ra,
                    rb,
                    acc_stage,
                )

        tcgen05_release_allocation_lock[1]()
        bars.unsafe_offset(Self.TMEM_FREE)[].wait(0)
        tcgen05_fence_after()
        tcgen05_dealloc[1](tmem, UInt32(Self.TMEM_COLS))

    @staticmethod
    @always_inline
    def _mma_tail(
        mma_op: Self.MmaOp,
        a_desc: MMASmemDescriptor,
        b_desc: MMASmemDescriptor,
        bars: MutPointer[SharedMemBarrier, _, address_space=.SHARED],
        tmem: UInt32,
        mut ra: RingCursor[Self.RSLOTS],
        mut rb: RingCursor[Self.FST],
        acc_stage: Array[Int, Self.MPASS],
    ):
        """One pass of the M-inner schedule: the first groups with a filter
        box read by every tile, then the last `TAIL` K steps tile by tile
        with their boxes held."""
        # The first groups: M-inner.
        for g in range(Self.NGROUPS - Self.TG):
            var a_slots = Array[Int, Self.MPASS](uninitialized=True)
            comptime for mt in range(Self.MPASS):
                bars.unsafe_offset(Self.A_FULL + ra.slot())[].wait(ra.phase)
                a_slots[mt] = ra.slot()
                ra.advance()
            var b_slots = Array[Int, 3](uninitialized=True)
            comptime for rr in range(3):
                bars.unsafe_offset(Self.B_FULL + rb.slot())[].wait(rb.phase)
                b_slots[rr] = rb.slot()
                rb.advance()
            comptime for mt in range(Self.MPASS):
                Self._issue3(
                    mma_op,
                    a_desc,
                    b_desc,
                    a_slots[mt],
                    b_slots[0],
                    b_slots[1],
                    b_slots[2],
                    tmem + UInt32(acc_stage[mt] * Self.BN),
                    g == 0,
                )
            comptime for rr in range(3):
                Self._release(mma_op, bars, Self.B_EMPTY + b_slots[rr])
            comptime for mt in range(Self.MPASS):
                Self._release(mma_op, bars, Self.A_EMPTY + a_slots[mt])

        comptime if Self.TG > 0:
            # The last groups: tile by tile. Tile 0 waits for each group's
            # windows and boxes; later tiles find them resident.
            var a_base = ra
            var b_base = rb
            comptime for mt in range(Self.MPASS):
                comptime for gi in range(Self.TG):
                    comptime if mt == 0:
                        comptime for _ in range(Self.MPASS):
                            bars.unsafe_offset(Self.A_FULL + ra.slot())[].wait(
                                ra.phase
                            )
                            ra.advance()
                        comptime for rr in range(3):
                            bars.unsafe_offset(Self.B_FULL + rb.slot())[].wait(
                                rb.phase
                            )
                            rb.advance()
                    var a_slot = a_base.slot_after(Int32(gi * Self.MPASS + mt))
                    var b0 = b_base.slot_after(Int32(gi * 3))
                    var b1 = b_base.slot_after(Int32(gi * 3 + 1))
                    var b2 = b_base.slot_after(Int32(gi * 3 + 2))
                    Self._issue3(
                        mma_op,
                        a_desc,
                        b_desc,
                        a_slot,
                        b0,
                        b1,
                        b2,
                        tmem + UInt32(acc_stage[mt] * Self.BN),
                        False,
                    )
                    comptime if mt == Self.MPASS - 1:
                        Self._release(mma_op, bars, Self.B_EMPTY + b0)
                        Self._release(mma_op, bars, Self.B_EMPTY + b1)
                        Self._release(mma_op, bars, Self.B_EMPTY + b2)
                    Self._release(mma_op, bars, Self.A_EMPTY + a_slot)
                if elect_one_sync():
                    mma_op.commit(
                        bars.unsafe_offset(Self.ACC_FULL + acc_stage[mt])
                    )
        else:
            comptime for mt in range(Self.MPASS):
                if elect_one_sync():
                    mma_op.commit(
                        bars.unsafe_offset(Self.ACC_FULL + acc_stage[mt])
                    )

    @staticmethod
    @always_inline
    def _mma_window_reuse(
        mma_op: Self.MmaOp,
        a_desc: MMASmemDescriptor,
        b_desc: MMASmemDescriptor,
        bars: MutPointer[SharedMemBarrier, _, address_space=.SHARED],
        tmem: UInt32,
        mut ra: RingCursor[Self.RSLOTS],
        mut rb: RingCursor[Self.FST],
        acc_stage: Array[Int, Self.MPASS],
    ):
        """One pass of the window-reuse schedule: window `j` of a block feeds
        tile `t` at filter column `s = j - t`, so the boxes of column `s` are
        waited for at window `s` and last used (by the last tile) at window
        `s + MPASS - 1`. A tile's accumulation order is still (block, s, r);
        tile `t` completes after the last block's window `t + 2`."""
        for blk in range(Self.NBLK):
            var b_slots = Array[Int, 9](uninitialized=True)
            comptime for j in range(Self.MPASS + 2):
                bars.unsafe_offset(Self.A_FULL + ra.slot())[].wait(ra.phase)
                var a_slot = ra.slot()
                ra.advance()
                comptime if j < 3:
                    comptime for rr in range(3):
                        bars.unsafe_offset(Self.B_FULL + rb.slot())[].wait(
                            rb.phase
                        )
                        b_slots[j * 3 + rr] = rb.slot()
                        rb.advance()
                comptime for t in range(Self.MPASS):
                    comptime fs = j - t
                    comptime if fs >= 0 and fs < 3:
                        Self._issue3(
                            mma_op,
                            a_desc,
                            b_desc,
                            a_slot,
                            b_slots[fs * 3],
                            b_slots[fs * 3 + 1],
                            b_slots[fs * 3 + 2],
                            tmem + UInt32(acc_stage[t] * Self.BN),
                            blk == 0 and fs == 0,
                        )
                Self._release(mma_op, bars, Self.A_EMPTY + a_slot)
                comptime done_s = j - (Self.MPASS - 1)
                comptime if done_s >= 0 and done_s < 3:
                    comptime for rr in range(3):
                        Self._release(
                            mma_op,
                            bars,
                            Self.B_EMPTY + b_slots[done_s * 3 + rr],
                        )
                comptime dtile = j - 2
                comptime if dtile >= 0 and dtile < Self.MPASS:
                    if blk == Self.NBLK - 1:
                        if elect_one_sync():
                            mma_op.commit(
                                bars.unsafe_offset(
                                    Self.ACC_FULL + acc_stage[dtile]
                                )
                            )

    # ----- epilogue warps -----

    @staticmethod
    @always_inline
    def _epilogue[
        output_type: DType,
        output_layout: TensorLayout,
        output_origin: MutOrigin,
        OutputEngine: TensorEngine,
        EpilogueFnType: ConvSimdEpilogueFn,
        ComputeFnType: ConvSimdComputeFn,
    ](
        out_desc: TMADescriptor,
        output: TileTensor[
            output_type,
            output_layout,
            output_origin,
            Engine=OutputEngine,
        ],
        sc: MutPointer[Scalar[Self.act_type], _, address_space=.SHARED],
        bars: MutPointer[SharedMemBarrier, _, address_space=.SHARED],
        tmem_slot: MutPointer[UInt32, _, address_space=.SHARED],
        tid: Int,
        cta: Int,
        ncta: Int,
        cty: Int,
        n_units: Int,
        out_depth: Int,
        epilogue_fn: EpilogueFnType,
        compute_fn: ComputeFnType,
    ):
        """Moves each accumulator tile TMEM -> registers -> smem -> TMA store,
        or applies the fused epilogue to the registers.

        The staging boxes are a ring of `SCST` slots, each three 32-column
        boxes: a slot is rewritten only after the store that read it has
        finished reading (`cp.async.bulk.wait_group.read`).
        """
        # A closure or the residual can read what the predecessor grid
        # writes. These warps skip the load warps' wait, and the barriers
        # after that wait do not keep their read-only loads below it (see
        # `wait_on_dependent_grids`).
        # No PDL level gates the wait: it returns at once in a launch
        # without PDL.
        comptime if Self.fused:
            wait_on_dependent_grids()
        Self.MmaEpiSync.wait()
        var tmem = tmem_slot[]
        var desc = Pointer(to=out_desc).unsafe_bitcast[NoneType]()
        var acc = RingCursor[Self.NACC]()
        var slot = 0
        var out_ptr = output.unsafe_ptr().unsafe_origin_cast[
            MutUntrackedOrigin
        ]()

        for u in range(n_units):
            var cu = cta + u * ncta
            for mt in range(Self.MPASS):
                var dd, row0, col0 = Self._tile_origin(cu, mt)
                bars.unsafe_offset(Self.ACC_FULL + acc.slot())[].wait(acc.phase)
                tcgen05_fence_after()
                var taddr = tmem + UInt32(acc.slot() * Self.BN)

                comptime if Self.fused:
                    # Lane `tid` is output row `tid` of the tile: column
                    # `tid // HP` of the tile, image row `row0 + tid % HP`.
                    var c_local, h_local = divmod(tid, Self.HP)
                    var h = row0 + h_local
                    var w = col0 + c_local
                    var live = (
                        dd < out_depth and h < Self.height and w < Self.width
                    )
                    var voxel = (dd * Self.height + h) * Self.width + w
                    comptime for half in range(Self.NH):
                        apply_conv_epilogue[
                            cols=96,
                            has_epilogue_fn=Self.has_epilogue_fn,
                            has_compute_fn=Self.has_compute_fn,
                            has_residual=False,
                            ndhwc=True,
                            round_first=True,
                        ](
                            taddr + UInt32(96 * half),
                            voxel,
                            cty * Self.BN + 96 * half,
                            Self.fout,
                            out_ptr,
                            out_ptr,
                            0.0,
                            epilogue_fn,
                            compute_fn,
                            live,
                            Self.height * Self.width,
                            Self.width,
                            out_depth,
                        )
                else:
                    var x = dd * (Self.height * Self.width) + col0
                    comptime for half in range(Self.NH):
                        if tid == 0:
                            cp_async_bulk_wait_group[Int32(Self.SCST - 1)]()
                        Self.EpiSync.sync()
                        var base = sc.unsafe_offset(slot * Self.C_ELEMS * 3)
                        stage_columns_swizzled[Self.act_type, 3](
                            taddr + UInt32(96 * half),
                            base,
                            Self.C_ELEMS,
                            tid,
                            True,
                        )
                        fence_async_view_proxy()
                        Self.EpiSync.sync()
                        if tid == 0:
                            comptime for pc in range(3):
                                cp_async_bulk_tensor_global_shared_cta(
                                    base.unsafe_offset(pc * Self.C_ELEMS),
                                    desc,
                                    (
                                        cty * Self.BN + 96 * half + pc * 32,
                                        row0,
                                        x,
                                    ),
                                )
                            cp_async_bulk_commit_group()
                        slot += 1
                        if slot == Self.SCST:
                            slot = 0

                tcgen05_fence_before()
                _ = bars.unsafe_offset(Self.ACC_EMPTY + acc.slot())[].arrive()
                acc.advance()

        tcgen05_fence_before()
        _ = bars.unsafe_offset(Self.TMEM_FREE)[].arrive()
        comptime if not Self.fused:
            if tid == 0:
                cp_async_bulk_wait_group[0, read=False]()

        # Release the dependent launch only once every store of the CTA is done.
        Self.EpiSync.sync()
        launch_dependent_grids()


def conv3d_col_umma_supported[
    act_type: DType, cin: Int, fout: Int, height: Int, width: Int
]() -> Bool:
    """Whether `launch_conv3d_col_umma` serves a channel and plane pair.

    The pairs are the three residual-block shapes of the WAN VAE the kernel
    was measured on, at each element type.

    Parameters:
        act_type: Element type of the activation, filter and output.
        cin: Input channels.
        fout: Output channels.
        height: Input height.
        width: Input width.

    Returns:
        True for a measured shape.
    """
    comptime if act_type != .float16 and act_type != .bfloat16 and (
        act_type != .float32
    ):
        return False
    return (
        (cin == 96 and fout == 96 and height == 240 and width == 416)
        or (cin == 192 and fout == 192 and height == 120 and width == 208)
        or (cin == 384 and fout == 384 and height == 30 and width == 52)
    )


def launch_conv3d_col_umma_packed[
    EpilogueFnType: ConvSimdEpilogueFn,
    ComputeFnType: ConvSimdComputeFn,
    packed_type: DType,
    //,
    act_type: DType,
    cin: Int,
    fout: Int,
    height: Int,
    width: Int,
    has_epilogue_fn: Bool,
    has_compute_fn: Bool,
](
    input: TileTensor[mut=True, act_type, address_space=.GENERIC, ...],
    packed: DeviceBuffer[packed_type],
    output: TileTensor[mut=True, address_space=.GENERIC, ...],
    ctx: DeviceContext,
    epilogue_fn: EpilogueFnType,
    compute_fn: ComputeFnType,
) raises:
    """Launches the 3x3x3 column-tile kernel on a packed filter.

    The caller has checked `conv3d_col_umma_supported`, a batch of one, dense
    row-major tensors and 16-byte aligned base pointers, and keeps `packed`
    alive until the launch is enqueued.

    Parameters:
        EpilogueFnType: Type of `epilogue_fn` (inferred).
        ComputeFnType: Type of `compute_fn` (inferred).
        act_type: Element type of the activation, filter and output.
        cin: Input channels.
        fout: Output channels.
        height: Input height.
        width: Input width.
        has_epilogue_fn: `epilogue_fn` stores each output value.
        has_compute_fn: `compute_fn` transforms each output value.

    Args:
        input: Input activation, NDHWC, batch 1.
        packed: The filter packed by `pack_conv3d_filter`.
        output: Output, NDHWC, batch 1.
        ctx: Device context for the launches.
        epilogue_fn: Storing epilogue, used when `has_epilogue_fn`.
        compute_fn: Transforming epilogue, used when `has_compute_fn`.

    Raises:
        If descriptor creation or a launch fails.
    """
    comptime K = Conv3dColUmma[
        act_type, cin, fout, height, width, has_epilogue_fn, has_compute_fn
    ]
    var depth = Int(input.dim[1]())
    var out_depth = depth - 2
    var passes = out_depth * K.HT * K.WGRP if K.RU == 1 else ceildiv(
        out_depth * K.WT * K.HT, K.MPASS
    )
    # A persistent grid of one wave, each CTA walking `units` passes; several
    # waves of one-pass CTAs would pay a launch, a TMEM allocation and a
    # pipeline fill per wave.
    var grid_y = fout // K.BN
    var units = ceildiv(passes, ctx.default_device_info.sm_count // grid_y)
    var grid_x = ceildiv(passes, units)

    comptime assert (
        output.dtype == act_type and packed_type == act_type
    ), "activation, filter and output share one element type"

    comptime out_swizzle = TensorMapSwizzle.SWIZZLE_64B if K.EB == 2 else (
        TensorMapSwizzle.SWIZZLE_128B
    )
    var act_desc = create_tma_descriptor[act_type, swizzle_mode=K.SWIZZLE](
        DeviceBuffer(
            ctx,
            input.unsafe_ptr(),
            1,
            owning=False,
        ),
        (depth, width, height, cin),
        (height * width * cin, cin, width * cin, 1),
        (1, K.WC, K.WIN_ROWS // K.WC, K.BK),
    )
    var filt_desc = create_tma_descriptor[packed_type, swizzle_mode=K.SWIZZLE](
        DeviceBuffer(ctx, packed.unsafe_ptr(), 1, owning=False),
        (fout, 27 * cin),
        (27 * cin, 1),
        (K.BN, K.BK),
    )
    var out_desc = create_tma_descriptor[
        output.dtype, swizzle_mode=out_swizzle
    ](
        DeviceBuffer(
            ctx,
            output.unsafe_ptr(),
            1,
            owning=False,
        ),
        (out_depth * height * width, height, fout),
        (fout, width * fout, 1),
        (K.WC, K.HP, 32),
    )

    comptime kernel = K.run[
        output.dtype,
        output.LayoutType,
        output.origin,
        output.Engine,
        EpilogueFnType,
        ComputeFnType,
    ]
    ctx.enqueue_function[kernel](
        act_desc,
        filt_desc,
        out_desc,
        output,
        Int32(units),
        Int32(out_depth),
        host_args=(epilogue_fn, compute_fn),
        grid_dim=(grid_x, grid_y, 1),
        block_dim=K.NUM_THREADS,
        shared_mem_bytes=K.SMEM_BYTES,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(K.SMEM_BYTES)
        ),
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )


def launch_conv3d_col_umma[
    EpilogueFnType: ConvSimdEpilogueFn,
    ComputeFnType: ConvSimdComputeFn,
    //,
    act_type: DType,
    cin: Int,
    fout: Int,
    height: Int,
    width: Int,
    filter_is_fcrs: Bool,
    has_epilogue_fn: Bool,
    has_compute_fn: Bool,
](
    input: TileTensor[mut=True, act_type, address_space=.GENERIC, ...],
    filter: TileTensor[address_space=.GENERIC, ...],
    output: TileTensor[mut=True, address_space=.GENERIC, ...],
    ctx: DeviceContext,
    epilogue_fn: EpilogueFnType,
    compute_fn: ComputeFnType,
) raises:
    """Packs the filter and launches the 3x3x3 column-tile kernel.

    The caller has checked `conv3d_col_umma_supported`, a batch of one, dense
    row-major tensors and 16-byte aligned base pointers.

    Parameters:
        EpilogueFnType: Type of `epilogue_fn` (inferred).
        ComputeFnType: Type of `compute_fn` (inferred).
        act_type: Element type of the activation, filter and output.
        cin: Input channels.
        fout: Output channels.
        height: Input height.
        width: Input width.
        filter_is_fcrs: The filter is FCQRS rather than QRSCF.
        has_epilogue_fn: `epilogue_fn` stores each output value.
        has_compute_fn: `compute_fn` transforms each output value.

    Args:
        input: Input activation, NDHWC, batch 1.
        filter: Filter, FCQRS or QRSCF.
        output: Output, NDHWC, batch 1.
        ctx: Device context for the launches.
        epilogue_fn: Storing epilogue, used when `has_epilogue_fn`.
        compute_fn: Transforming epilogue, used when `has_compute_fn`.

    Raises:
        If descriptor creation or a launch fails.
    """
    comptime assert (
        filter.dtype == act_type
    ), "activation, filter and output share one element type"
    var packed = pack_conv3d_filter[filter_is_fcrs, 27, cin](filter, fout, ctx)
    launch_conv3d_col_umma_packed[
        act_type, cin, fout, height, width, has_epilogue_fn, has_compute_fn
    ](input, packed, output, ctx, epilogue_fn, compute_fn)
    # The descriptor holds the packed filter's address, which Mojo cannot see.
    _ = packed^
