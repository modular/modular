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
"""SM100 Conv2D fprop for non-overlapping patches (stride == filter, no
padding), fp32 operands on the tf32 tensor core.

Target family: Blackwell sm_100a, 1-SM UMMA, tf32 with fp32 accumulation
(the contract of the cuDNN fp32 engine this replaces). The vision patch
embeds are this shape: 896x896 pixels, 3 channels, a 14x14 filter at stride
14, 1152 output channels.

A patch row (`FW` pixels of `C` channels) is `px` contiguous elements and the
patch rows of one output row are `px` apart, so the NHWC activation reshapes
for free to `[N * OH, FH, OW / 2, 2 * px]`. A pixel pair is a legal TMA stride
(a 16-byte multiple) where one 168-byte pixel is not, so a tile is the
pixels of ONE parity of `BM / (OW / 2)` consecutive output rows and one tiled
TMA box `[orows, 1, OW / 2, BK]` fills a K tile of the whole 128-row slot. An
odd pixel starts `px` elements into the pair, which is not 16-byte aligned, so
its boxes start `lead = px % (16 / eb)` elements early and the odd tiles use a
filter shifted by as many K positions: the packed filter holds the even
matrix and the odd one, `[2 * F, K]`. Columns of a patch row past `px` read the
neighbouring pixel or TMA's zero fill and meet a zero weight; the epilogue
warps zero them in shared memory while the mainloop runs, so a
non-finite neighbour cannot reach the product.

Warp roles (7 warps, 224 threads), one output tile per CTA:

| warps | role                                                      |
|-------|-----------------------------------------------------------|
| 0-3   | clear neighbour columns of A slots; epilogue (TMEM -> out)  |
| 4     | filter load: one TMA box per K step                       |
| 5     | activation load: one tiled TMA box per K step             |
| 6     | MMA issue, TMEM allocation                                |

The grid is `(2 * N * OH / orows, F / BN)`: `blockIdx.x` is the parity and the
group of output rows, `blockIdx.y` the output-channel tile.
"""

from std.math import align_up, ceildiv, clamp
from std.sys import size_of
from std.utils.static_tuple import StaticTuple

from layout import (
    Coord,
    TensorEngine,
    TensorLayout,
    TileTensor,
    coord,
    row_major,
)
from layout.tma_async import SharedMemBarrier, create_tensor_tile
from linalg.arch.sm100 import MmaOpSM100_SS
from linalg.arch.sm100.mma import _create_mma_desc_k_major
from linalg.utils import (
    ElementwiseComputeFn,
    ElementwiseEpilogueFn,
    no_compute_fn,
    no_epilogue_fn,
)
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    global_idx,
    thread_idx,
)
from max.gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_fence_after,
    tcgen05_fence_before,
    tcgen05_ld,
    tcgen05_load_wait,
    tcgen05_release_allocation_lock,
)
from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.host.info import B200
from max.gpu.host.nvidia.tma import (
    TensorMapSwizzle,
    TMADescriptor,
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
from structured_kernels.barriers import WarpGroupBarrier

from .conv_epilogue import apply_conv_epilogue


struct Conv2dPatchFprop[
    ow: Int,
    cin: Int,
    fh: Int,
    fw: Int,
    has_epilogue_fn: Bool = False,
    has_compute_fn: Bool = False,
    has_residual: Bool = False,
]:
    """The patch-embed conv2d fprop kernel and its launch helpers.

    Parameters:
        ow: Output width, even; `OW / 2` pixel pairs fill one tile row group.
        cin: Input channels.
        fh: Patch rows (the filter height).
        fw: Patch columns (the filter width).
        has_epilogue_fn: An elementwise epilogue stores each output value.
        has_compute_fn: An elementwise epilogue transforms each output value.
        has_residual: Add `beta * source` to each output value first.

    With any of the three flags the epilogue warps apply
    `conv_epilogue.apply_conv_epilogue` in registers instead of staging the
    tile for a TMA store.
    """

    # ========== Problem family ==========
    comptime act_type = DType.float32
    comptime px = Self.fw * Self.cin
    """Elements in one patch row."""
    comptime EB = size_of[Self.act_type]()
    comptime BM = 128
    comptime BN = 128
    comptime SWIZZLE = TensorMapSwizzle.SWIZZLE_128B
    comptime BK = Self.SWIZZLE.bytes() // Self.EB
    """Reduction elements per K tile: one swizzle row."""
    comptime MMA_K = 32 // Self.EB
    comptime PAIRS = Self.ow // 2
    comptime OROWS = Self.BM // Self.PAIRS
    """Output rows per tile."""
    comptime LEAD = Self.px % (16 // Self.EB)
    """Elements an odd pixel's boxes start early, to be 16-byte aligned."""
    comptime PER_ROW = ceildiv(Self.px, Self.BK)
    """K tiles per patch row."""
    comptime KROW = align_up(Self.px, Self.BK)
    comptime K = Self.fh * Self.KROW
    """Reduction extent of the packed filter, padded per patch row."""
    comptime KT = Self.fh * Self.PER_ROW

    # ========== Rings and shared memory ==========
    comptime STAGES = 4
    comptime CHUNK = 32
    """Output columns per staged TMA store box."""
    comptime NCHUNK = Self.BN // Self.CHUNK
    comptime A_ELEMS = Self.BM * Self.BK
    comptime B_ELEMS = Self.BN * Self.BK
    comptime C_ELEMS = Self.BM * Self.CHUNK
    comptime A_BYTES = Self.A_ELEMS * Self.EB
    comptime B_BYTES = Self.B_ELEMS * Self.EB
    comptime C_BYTES = Self.C_ELEMS * Self.EB
    comptime A_OFF = 0
    comptime B_OFF = Self.A_OFF + Self.STAGES * Self.A_BYTES
    comptime C_OFF = Self.B_OFF + Self.STAGES * Self.B_BYTES
    comptime BAR_OFF = Self.C_OFF + Self.NCHUNK * Self.C_BYTES

    comptime A_FULL = 0
    comptime A_EMPTY = Self.A_FULL + Self.STAGES
    comptime B_FULL = Self.A_EMPTY + Self.STAGES
    comptime B_EMPTY = Self.B_FULL + Self.STAGES
    comptime ACC_FULL = Self.B_EMPTY + Self.STAGES
    comptime A_CLEAN = Self.ACC_FULL + 1
    """Activation slots whose neighbour columns the epilogue warps cleared."""
    comptime TMEM_FREE = Self.A_CLEAN + Self.STAGES
    comptime NBAR = Self.TMEM_FREE + 1
    comptime TMEM_ADDR_OFF = Self.BAR_OFF + Self.NBAR * 8
    comptime SMEM_BYTES = Self.TMEM_ADDR_OFF + 16

    # ========== Threads ==========
    comptime EPI_THREADS = 128
    comptime NUM_THREADS = Self.EPI_THREADS + 3 * WARP_SIZE
    comptime TMEM_COLS = Self.BN

    comptime MmaEpiSync = WarpGroupBarrier[Self.EPI_THREADS + WARP_SIZE, 1]
    comptime EpiSync = WarpGroupBarrier[Self.EPI_THREADS, 2]

    comptime MmaOp = MmaOpSM100_SS[
        Self.act_type,
        Self.act_type,
        Self.act_type,
        (Self.BM, Self.BN, Self.BK),
        (Self.BM, Self.BN, Self.MMA_K),
        accum_type=.float32,
        cta_group=1,
        a_swizzle=Self.SWIZZLE,
        b_swizzle=Self.SWIZZLE,
        transpose_b=True,
    ]

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
    @__name(t"sm100_conv2d_patch_fprop_{Self.act_type}")
    def run[
        EpilogueFnType: ElementwiseEpilogueFn,
        ComputeFnType: ElementwiseComputeFn,
    ](
        act_desc: TMADescriptor,
        filt_desc: TMADescriptor,
        out_desc: TMADescriptor,
        out_h: Int32,
        out_channels: Int32,
        out_ptr: MutPointer[Scalar[Self.act_type], MutUntrackedOrigin],
        src_ptr: MutPointer[Scalar[Self.act_type], MutUntrackedOrigin],
        beta: Float32,
        epilogue_fn: EpilogueFnType,
        compute_fn: ComputeFnType,
    ):
        """Kernel entry point.

        Args:
            act_desc: Descriptor for the activation viewed
                `[N * OH, FH, OW / 2, 2 * px]`, box `[orows, 1, OW / 2, BK]`.
            filt_desc: Descriptor for the packed filter `[2 * F, K]`, box
                `[BN, BK]`.
            out_desc: Descriptor for the output viewed `[N * OH, OW / 2,
                2 * F]`, box `[orows, OW / 2, 32]`.
            out_h: Output height of one image.
            out_channels: Output channels `F`.
            out_ptr: Output; written by the kernel unless `has_epilogue_fn`,
                and only on the register epilogue path.
            src_ptr: Residual source, NHWC like the output.
            beta: Residual scale factor.
            epilogue_fn: Storing epilogue, used when `has_epilogue_fn`.
            compute_fn: Transforming epilogue, used when `has_compute_fn`.
        """
        Self._run_impl(
            act_desc,
            filt_desc,
            out_desc,
            Int(out_h),
            Int(out_channels),
            out_ptr,
            src_ptr,
            beta,
            epilogue_fn,
            compute_fn,
        )

    @staticmethod
    @inline(.always)
    def _run_impl[
        EpilogueFnType: ElementwiseEpilogueFn,
        ComputeFnType: ElementwiseComputeFn,
    ](
        act_desc: TMADescriptor,
        filt_desc: TMADescriptor,
        out_desc: TMADescriptor,
        out_h: Int,
        out_channels: Int,
        out_ptr: MutPointer[Scalar[Self.act_type], MutUntrackedOrigin],
        src_ptr: MutPointer[Scalar[Self.act_type], MutUntrackedOrigin],
        beta: Float32,
        epilogue_fn: EpilogueFnType,
        compute_fn: ComputeFnType,
    ):
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
        var tile = block_idx.x
        var rows_group, par = divmod(tile, 2)
        var n0 = block_idx.y * Self.BN
        # The odd tiles read the second filter matrix.
        var filt_row = par * out_channels + n0

        if tid == 0:
            var a = Pointer(to=act_desc).unsafe_bitcast[NoneType]()
            var f = Pointer(to=filt_desc).unsafe_bitcast[NoneType]()
            var o = Pointer(to=out_desc).unsafe_bitcast[NoneType]()
            prefetch_tma_descriptor(a)
            prefetch_tma_descriptor(f)
            prefetch_tma_descriptor(o)
            comptime for i in range(Self.A_FULL, Self.A_CLEAN):
                bars.unsafe_offset(i)[].init(1)
            # Every epilogue thread clears its own row of a slot, and later
            # frees TMEM.
            comptime for i in range(Self.A_CLEAN, Self.TMEM_FREE + 1):
                bars.unsafe_offset(i)[].init(Int32(Self.EPI_THREADS))
        fence_mbarrier_init()
        barrier()

        if warp < 4:
            Self._epilogue(
                out_desc,
                sa,
                sc,
                bars,
                tmem_slot,
                tid,
                par,
                rows_group,
                n0,
                out_h,
                out_channels,
                out_ptr,
                src_ptr,
                beta,
                epilogue_fn,
                compute_fn,
            )
        elif warp == 4:
            Self._filter_load(filt_desc, sb, bars, filt_row)
        elif warp == 5:
            Self._act_load(act_desc, sa, bars, par, rows_group)
        else:
            Self._mma(sa, sb, bars, tmem_slot)

    # ----- load warps -----

    @staticmethod
    @inline(.always)
    def _filter_load(
        filt_desc: TMADescriptor,
        sb: MutPointer[Scalar[Self.act_type], _, address_space=.SHARED],
        bars: MutPointer[SharedMemBarrier, _, address_space=.SHARED],
        filt_row: Int,
    ):
        """Loads one `[BN, BK]` filter box per K step."""
        var desc = Pointer(to=filt_desc).unsafe_bitcast[NoneType]()
        # The predecessor may be the pack that writes this filter.
        wait_on_dependent_grids()
        var stage = 0
        var phase = UInt32(0)
        for kt in range(Self.KT):
            bars.unsafe_offset(Self.B_EMPTY + stage)[].wait(phase ^ 1)
            if elect_one_sync():
                var full = bars.unsafe_offset(Self.B_FULL + stage)
                full[].expect_bytes(Int32(Self.B_BYTES))
                cp_async_bulk_tensor_shared_cluster_global[cta_group=1](
                    sb.unsafe_offset(stage * Self.B_ELEMS),
                    desc,
                    full,
                    (kt * Self.BK, filt_row),
                )
            stage += 1
            if stage == Self.STAGES:
                stage = 0
                phase ^= 1

    @staticmethod
    @inline(.always)
    def _act_load(
        act_desc: TMADescriptor,
        sa: MutPointer[Scalar[Self.act_type], _, address_space=.SHARED],
        bars: MutPointer[SharedMemBarrier, _, address_space=.SHARED],
        par: Int,
        rows_group: Int,
    ):
        """Loads one tiled box of the tile's pixels per K step: patch row
        `r`, elements `jt * BK ..` of the pixel pair, shifted to a 16-byte
        boundary for the odd pixels."""
        var desc = Pointer(to=act_desc).unsafe_bitcast[NoneType]()
        wait_on_dependent_grids()
        var stage = 0
        var phase = UInt32(0)
        var first_row = rows_group * Self.OROWS
        var shift = par * (Self.px - Self.LEAD)
        for kt in range(Self.KT):
            var r, jt = divmod(kt, Self.PER_ROW)
            bars.unsafe_offset(Self.A_EMPTY + stage)[].wait(phase ^ 1)
            if elect_one_sync():
                var full = bars.unsafe_offset(Self.A_FULL + stage)
                full[].expect_bytes(Int32(Self.A_BYTES))
                cp_async_bulk_tensor_shared_cluster_global[cta_group=1](
                    sa.unsafe_offset(stage * Self.A_ELEMS),
                    desc,
                    full,
                    (jt * Self.BK + shift, 0, r, first_row),
                )
            stage += 1
            if stage == Self.STAGES:
                stage = 0
                phase ^= 1

    # ----- epilogue warps, mainloop part -----

    @staticmethod
    @inline(.always)
    def _clear_neighbour_columns[
        jt: Int
    ](
        slot: MutPointer[Scalar[Self.act_type], _, address_space=.SHARED],
        row: Int,
        par: Int,
    ):
        """Zeroes the columns of K tile `jt` in one row of a slot that hold
        another pixel's elements.

        A box covers a whole pixel pair, so the even tile's second box and the
        odd tile's first and second boxes carry elements of the neighbouring
        pixel. The filter is zero there, but `0 * inf` is NaN, and one
        non-finite pixel would poison its neighbour's whole output (cuDNN
        keeps it to the pixel).

        Args:
            slot: The activation slot, `[BM, BK]` in the 128-byte swizzle.
            row: The row of the slot to clear.
            par: Pixel parity of the tile.
        """
        comptime CH = 16 // Self.EB
        comptime NCH = Self.BK // CH
        var first_valid = max(0, par * Self.LEAD - jt * Self.BK)
        var end_valid = clamp(
            Self.px + par * Self.LEAD - jt * Self.BK, 0, Self.BK
        )
        var base = slot.unsafe_offset(row * Self.BK)
        var key = row & 7
        comptime for c in range(NCH):
            comptime lo = c * CH
            var dst = base.unsafe_offset((c ^ key) * CH)
            if lo + CH <= first_valid or lo >= end_valid:
                dst.unsafe_store[alignment=16](SIMD[Self.act_type, CH](0))
            elif lo < first_valid or lo + CH > end_valid:
                comptime for i in range(CH):
                    if lo + i < first_valid or lo + i >= end_valid:
                        dst.unsafe_offset(i).unsafe_store(
                            Scalar[Self.act_type](0)
                        )

    # ----- MMA warp -----

    @staticmethod
    @inline(.always)
    def _mma(
        sa: MutPointer[Scalar[Self.act_type], _, address_space=.SHARED],
        sb: MutPointer[Scalar[Self.act_type], _, address_space=.SHARED],
        bars: MutPointer[SharedMemBarrier, _, address_space=.SHARED],
        tmem_slot: MutPointer[UInt32, _, address_space=.SHARED],
    ):
        """Allocates TMEM, issues every UMMA of the tile, and frees TMEM."""
        tcgen05_alloc[1](tmem_slot, UInt32(Self.TMEM_COLS))
        syncwarp()
        var tmem = tmem_slot[]
        Self.MmaEpiSync.arrive()

        var mma_op = Self.MmaOp()
        var a_desc = _create_mma_desc_k_major[Self.act_type, Self.SWIZZLE](sa)
        var b_desc = _create_mma_desc_k_major[Self.act_type, Self.SWIZZLE](sb)

        var stage = 0
        var phase = UInt32(0)
        for r in range(Self.fh):
            comptime for jt in range(Self.PER_ROW):
                bars.unsafe_offset(Self.B_FULL + stage)[].wait(phase)
                bars.unsafe_offset(Self.A_CLEAN + stage)[].wait(phase)
                if elect_one_sync():
                    mma_op.mma_from_desc(
                        a_desc + stage * Self.A_BYTES,
                        b_desc + stage * Self.B_BYTES,
                        tmem,
                        init_c=(r == 0 and jt == 0),
                    )
                    mma_op.commit(bars.unsafe_offset(Self.A_EMPTY + stage))
                    mma_op.commit(bars.unsafe_offset(Self.B_EMPTY + stage))
                stage += 1
                if stage == Self.STAGES:
                    stage = 0
                    phase ^= 1
        if elect_one_sync():
            mma_op.commit(bars.unsafe_offset(Self.ACC_FULL))

        tcgen05_release_allocation_lock[1]()
        bars.unsafe_offset(Self.TMEM_FREE)[].wait(0)
        tcgen05_fence_after()
        tcgen05_dealloc[1](tmem, UInt32(Self.TMEM_COLS))

    # ----- epilogue warps -----

    @staticmethod
    @inline(.always)
    def _epilogue[
        EpilogueFnType: ElementwiseEpilogueFn,
        ComputeFnType: ElementwiseComputeFn,
    ](
        out_desc: TMADescriptor,
        sc_a: MutPointer[Scalar[Self.act_type], _, address_space=.SHARED],
        sc: MutPointer[Scalar[Self.act_type], _, address_space=.SHARED],
        bars: MutPointer[SharedMemBarrier, _, address_space=.SHARED],
        tmem_slot: MutPointer[UInt32, _, address_space=.SHARED],
        tid: Int,
        par: Int,
        rows_group: Int,
        n0: Int,
        out_h: Int,
        out_channels: Int,
        out_ptr: MutPointer[Scalar[Self.act_type], MutUntrackedOrigin],
        src_ptr: MutPointer[Scalar[Self.act_type], MutUntrackedOrigin],
        beta: Float32,
        epilogue_fn: EpilogueFnType,
        compute_fn: ComputeFnType,
    ):
        """Clears the neighbour columns of the activation slots, then moves the
        accumulator tile TMEM -> registers -> output.

        Without an epilogue the tile is staged as `NCHUNK` swizzled boxes of
        32 columns, whose rows a lane writes in 16-byte chunks without bank
        conflicts, and stored by TMA. With one, each thread owns the output
        pixel of its TMEM lane and applies `apply_conv_epilogue` to its
        columns.
        """
        # While the mainloop runs the epilogue warps would idle: each thread
        # clears its row of every activation slot as the slot lands, so the
        # MMA warp only waits for the cleared slot.
        var clean_stage = 0
        var clean_phase = UInt32(0)
        for _ in range(Self.fh):
            comptime for jt in range(Self.PER_ROW):
                bars.unsafe_offset(Self.A_FULL + clean_stage)[].wait(
                    clean_phase
                )
                Self._clear_neighbour_columns[jt](
                    sc_a.unsafe_offset(clean_stage * Self.A_ELEMS), tid, par
                )
                fence_async_view_proxy()
                _ = bars.unsafe_offset(Self.A_CLEAN + clean_stage)[].arrive()
                clean_stage += 1
                if clean_stage == Self.STAGES:
                    clean_stage = 0
                    clean_phase ^= 1

        Self.MmaEpiSync.wait()
        var tmem = tmem_slot[]
        bars.unsafe_offset(Self.ACC_FULL)[].wait(0)
        tcgen05_fence_after()

        comptime if Self.has_epilogue_fn or Self.has_compute_fn or Self.has_residual:
            # Lane `tid` is row `tid` of the tile: output row
            # `first_row + tid / pairs`, column `2 * (tid % pairs) + par`.
            var row_in_group, pair = divmod(tid, Self.PAIRS)
            var out_row = rows_group * Self.OROWS + row_in_group
            var out_col = 2 * pair + par
            var pixel = out_row * Self.ow + out_col
            comptime for piece in range(Self.NCHUNK):
                apply_conv_epilogue[
                    cols=Self.CHUNK,
                    has_epilogue_fn=Self.has_epilogue_fn,
                    has_compute_fn=Self.has_compute_fn,
                    has_residual=Self.has_residual,
                    nhwc=True,
                ](
                    tmem + UInt32(piece * Self.CHUNK),
                    pixel,
                    n0 + piece * Self.CHUNK,
                    out_channels,
                    out_ptr,
                    src_ptr,
                    beta,
                    epilogue_fn,
                    compute_fn,
                    True,
                    out_h * Self.ow,
                    Self.ow,
                )
            tcgen05_fence_before()
            _ = bars.unsafe_offset(Self.TMEM_FREE)[].arrive()
            # Only a CTA's first release counts, so the dependents are
            # released here, after this path's stores, and not by the loaders.
            launch_dependent_grids()
            return

        comptime CH = 16 // Self.EB
        """Output elements per 16-byte chunk."""
        comptime NCH = Self.CHUNK // CH
        """Chunks per staged row."""
        var key = tid & 7
        comptime for piece in range(Self.NCHUNK):
            var regs = tcgen05_ld[
                datapaths=32,
                bits=32,
                repeat=Self.CHUNK,
                dtype=.float32,
                pack=False,
            ](tmem + UInt32(piece * Self.CHUNK))
            tcgen05_load_wait()
            var dst = sc.unsafe_offset(piece * Self.C_ELEMS + tid * Self.CHUNK)
            comptime for c in range(NCH):
                var v = SIMD[Self.act_type, CH]()
                comptime for e in range(CH):
                    v[e] = regs[c * CH + e].cast[Self.act_type]()
                dst.unsafe_offset((c ^ key) * CH).unsafe_store[alignment=16](v)
        tcgen05_fence_before()
        _ = bars.unsafe_offset(Self.TMEM_FREE)[].arrive()
        fence_async_view_proxy()
        Self.EpiSync.sync()
        if tid == 0:
            var desc = Pointer(to=out_desc).unsafe_bitcast[NoneType]()
            comptime for piece in range(Self.NCHUNK):
                cp_async_bulk_tensor_global_shared_cta(
                    sc.unsafe_offset(piece * Self.C_ELEMS),
                    desc,
                    (
                        par * out_channels + n0 + piece * Self.CHUNK,
                        0,
                        rows_group * Self.OROWS,
                    ),
                )
            cp_async_bulk_commit_group()
            cp_async_bulk_wait_group[0, read=False]()
        Self.EpiSync.sync()
        launch_dependent_grids()


comptime _PACK_VEC = 4
"""Packed elements one thread writes with a single 16-byte store."""


@__name(t"pack_patch_filter_{dtype}")
def _pack_patch_filter[
    dtype: DType,
    SrcLayoutType: TensorLayout,
    src_origin: ImmOrigin,
    SrcEngine: TensorEngine,
    cin: Int,
    fh: Int,
    fw: Int,
    krow: Int,
    lead: Int,
    pdl_level: PDLLevel,
](
    src: TileTensor[dtype, SrcLayoutType, src_origin, Engine=SrcEngine],
    dst_ptr: MutPointer[Scalar[dtype], MutUntrackedOrigin],
    F: Int32,
):
    """Packs an FCRS filter `[F, C, R, S]` into the two `[F, R * krow]`
    matrices the kernel reads, the even pixels' rows `0 .. F` and the odd
    pixels' rows `F .. 2F`: `k = r * krow + (s * C + c) + par * lead`, zero
    elsewhere. A thread writes `_PACK_VEC` consecutive packed elements.

    Parameters:
        dtype: Element type.
        SrcLayoutType: Layout of the FCRS filter.
        src_origin: Origin of the filter.
        SrcEngine: Engine of the filter.
        cin: Input channels.
        fh: Filter height.
        fw: Filter width.
        krow: Packed elements per patch row.
        lead: Positions the odd matrix is shifted by.
        pdl_level: Programmatic dependent launch level.

    Args:
        src: The FCRS filter.
        dst_ptr: The packed matrices.
        F: Output channels.
    """
    comptime k_total = fh * krow
    comptime assert krow % _PACK_VEC == 0, "a vector stays inside a patch row"
    # The filter may be the output of the kernel launched just before.
    comptime if pdl_level > PDLLevel.OFF:
        wait_on_dependent_grids()
    # Releasing the dependents here lets the conv kernel's CTAs become
    # resident while the pack runs; its loaders wait for the pack's stores.
    comptime if pdl_level == PDLLevel.OVERLAP_AT_BEGINNING:
        launch_dependent_grids()
    var first = global_idx.x * _PACK_VEC
    var matrix = Int(F) * k_total
    if first < 2 * matrix:
        var par, rem = divmod(first, matrix)
        var f, kk = divmod(rem, k_total)
        var r, jj = divmod(kk, krow)
        var v = SIMD[dtype, _PACK_VEC](0)
        comptime for e in range(_PACK_VEC):
            var j = jj + e - par * lead
            if 0 <= j and j < fw * cin:
                var s, c = divmod(j, cin)
                v[e] = src.load((f, c, r, s))[0]
        dst_ptr.unsafe_offset(first).unsafe_store[
            alignment=_PACK_VEC * size_of[dtype]()
        ](v)
    comptime if pdl_level > PDLLevel.OFF and (
        pdl_level != PDLLevel.OVERLAP_AT_BEGINNING
    ):
        launch_dependent_grids()


def conv2d_patch_supported(
    input_shape: Coord[Int, Int, Int, Int],
    filter_shape: Coord[Int, Int, Int, Int],
    stride: Coord[Int, Int],
    dilation: Coord[Int, Int],
    padding: Coord[Int, Int],
    num_groups: Int,
) -> Bool:
    """Whether `conv2d_patch_dispatch` serves a convolution.

    The envelope is the one shape the kernel was timed ahead of cuDNN on: a
    14x14 filter at stride 14 over a 896x896 image of 3 channels, 1152 output
    channels, one image.

    Args:
        input_shape: Activation `[N, H, W, C]`.
        filter_shape: Filter `[F, C, R, S]`.
        stride: Stride along the height and width.
        dilation: Dilation along the height and width.
        padding: Symmetric padding along the height and width.
        num_groups: Number of convolution groups.

    Returns:
        True when the convolution is that shape.
    """
    return (
        num_groups == 1
        and input_shape[0] == 1
        and input_shape[1] == 896
        and input_shape[2] == 896
        and input_shape[3] == 3
        and filter_shape[0] == 1152
        and filter_shape[1] == 3
        and filter_shape[2] == 14
        and filter_shape[3] == 14
        and stride[0] == 14
        and stride[1] == 14
        and dilation[0] == 1
        and dilation[1] == 1
        and padding[0] == 0
        and padding[1] == 0
    )


def conv2d_patch_fprop[
    EpilogueFnType: ElementwiseEpilogueFn,
    ComputeFnType: ElementwiseComputeFn,
    //,
    ow: Int,
    cin: Int,
    fh: Int,
    fw: Int,
    has_epilogue_fn: Bool,
    has_compute_fn: Bool,
](
    output: TileTensor[mut=True, address_space=.GENERIC, ...],
    activation: TileTensor[mut=True, address_space=.GENERIC, ...],
    packed_filter: TileTensor[address_space=.GENERIC, ...],
    ctx: DeviceContext,
    epilogue_fn: EpilogueFnType,
    compute_fn: ComputeFnType,
) raises:
    """Launches the patch-embed kernel.

    Parameters:
        EpilogueFnType: Type of `epilogue_fn` (inferred).
        ComputeFnType: Type of `compute_fn` (inferred).
        ow: Output width.
        cin: Input channels.
        fh: Filter height, equal to the stride.
        fw: Filter width, equal to the stride.
        has_epilogue_fn: `epilogue_fn` stores each output value; the kernel
            writes nothing to `output`.
        has_compute_fn: `compute_fn` transforms each output value.

    Args:
        output: Output `[N, OH, OW, F]`, NHWC.
        activation: Input `[N, OH * fh, OW * fw, cin]`, NHWC.
        packed_filter: The even and odd matrices `_pack_patch_filter` writes,
            `[2 * F, K]`.
        ctx: Device context for the launch.
        epilogue_fn: Storing epilogue.
        compute_fn: Transforming epilogue.

    Raises:
        If descriptor creation or the launch fails.
    """
    comptime assert (
        output.dtype == .float32
        and activation.dtype == .float32
        and packed_filter.dtype == .float32
    ), "the kernel runs fp32 operands on the tf32 tensor core"
    comptime K = Conv2dPatchFprop[
        ow, cin, fh, fw, has_epilogue_fn, has_compute_fn
    ]
    var batch_rows = Int(output.dim[0]()) * Int(output.dim[1]())
    var fout = Int(output.dim[3]())

    var act_view = activation.reshape(
        row_major(batch_rows, fh, K.PAIRS, 2 * K.px)
    )
    var act_desc = create_tensor_tile[
        coord[K.OROWS, 1, K.PAIRS, K.BK],
        swizzle_mode=K.SWIZZLE,
        __tile_shape=coord[K.OROWS, 1, K.PAIRS, K.BK],
        __desc_shape=coord[K.OROWS, 1, K.PAIRS, K.BK],
    ](ctx, act_view).descriptor
    var filt_desc = create_tensor_tile[
        coord[K.BN, K.BK],
        swizzle_mode=K.SWIZZLE,
        __tile_shape=coord[K.BN, K.BK],
        __desc_shape=coord[K.BN, K.BK],
    ](ctx, packed_filter).descriptor
    var out_view = output.reshape(row_major(batch_rows, K.PAIRS, 2 * fout))
    var out_desc = create_tensor_tile[
        coord[K.OROWS, K.PAIRS, K.CHUNK],
        swizzle_mode=K.SWIZZLE,
        __tile_shape=coord[K.OROWS, K.PAIRS, K.CHUNK],
        __desc_shape=coord[K.OROWS, K.PAIRS, K.CHUNK],
    ](ctx, out_view).descriptor

    var out_ptr = rebind[MutPointer[Scalar[.float32], MutUntrackedOrigin]](
        output.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    )
    comptime kernel = K.run[EpilogueFnType, ComputeFnType]
    ctx.enqueue_function[kernel](
        act_desc,
        filt_desc,
        out_desc,
        Int32(output.dim[1]()),
        Int32(fout),
        out_ptr,
        out_ptr,
        Float32(0),
        host_args=(epilogue_fn, compute_fn),
        grid_dim=(2 * (batch_rows // K.OROWS), fout // K.BN, 1),
        block_dim=K.NUM_THREADS,
        shared_mem_bytes=K.SMEM_BYTES,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(K.SMEM_BYTES)
        ),
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )


def conv2d_patch_dispatch[
    EpilogueFnType: ElementwiseEpilogueFn,
    ComputeFnType: ElementwiseComputeFn,
    //,
    has_epilogue_fn: Bool,
    has_compute_fn: Bool,
](
    input: TileTensor[mut=True, address_space=.GENERIC, ...],
    filter: TileTensor[address_space=.GENERIC, ...],
    output: TileTensor[mut=True, address_space=.GENERIC, ...],
    stride: Coord[Int, Int],
    dilation: Coord[Int, Int],
    padding: Coord[Int, Int],
    num_groups: Int,
    ctx: DeviceContext,
    epilogue_fn: EpilogueFnType,
    compute_fn: ComputeFnType,
) raises -> Bool:
    """Runs an fp32 FCRS patch-embed convolution on the tf32 kernel if it
    serves the problem.

    Packs the filter into the kernel's two matrices first, with PDL so the
    pack overlaps the kernel's launch.

    Parameters:
        EpilogueFnType: Type of `epilogue_fn` (inferred).
        ComputeFnType: Type of `compute_fn` (inferred).
        has_epilogue_fn: `epilogue_fn` stores each output value.
        has_compute_fn: `compute_fn` transforms each output value.

    Args:
        input: Input activation, NHWC.
        filter: Filter, FCRS.
        output: Output, NHWC.
        stride: Stride along the height and width.
        dilation: Dilation along the height and width.
        padding: Symmetric padding along the height and width.
        num_groups: Number of convolution groups.
        ctx: Device context for the launch.
        epilogue_fn: Storing epilogue.
        compute_fn: Transforming epilogue.

    Returns:
        True if the kernel ran; False, with nothing enqueued, if the device
        is not a B200 (the only GPU the kernel was timed on) or the problem or
        the operand layouts are outside what it serves.

    Raises:
        If descriptor creation or a launch fails.
    """
    comptime assert input.flat_rank == 4, "input must be rank 4 (NHWC)"
    comptime assert filter.flat_rank == 4, "filter must be rank 4 (FCRS)"
    comptime assert output.flat_rank == 4, "output must be rank 4 (NHWC)"
    comptime if (
        not (ctx.default_device_info == B200)
        or input.dtype != .float32
        or filter.dtype != .float32
        or output.dtype != .float32
        or not (
            input.is_row_major and filter.is_row_major and output.is_row_major
        )
    ):
        return False
    else:
        if not conv2d_patch_supported(
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
            stride,
            dilation,
            padding,
            num_groups,
        ):
            return False

        comptime OW = 64
        comptime CIN = 3
        comptime FH = 14
        comptime FW = 14
        comptime K = Conv2dPatchFprop[OW, CIN, FH, FW]
        var fout = Int(filter.dim[0]())
        var packed_elems = 2 * fout * K.K
        var packed = ctx.enqueue_create_buffer[.float32](packed_elems)
        comptime pack_block = 256
        comptime pack = _pack_patch_filter[
            .float32,
            filter.LayoutType,
            ImmOrigin(filter.origin),
            filter.Engine,
            CIN,
            FH,
            FW,
            K.KROW,
            K.LEAD,
            PDLLevel.OVERLAP_AT_BEGINNING,
        ]
        ctx.enqueue_function[pack](
            filter.as_imm(),
            packed.unsafe_ptr(),
            Int32(fout),
            grid_dim=ceildiv(packed_elems // _PACK_VEC, pack_block),
            block_dim=pack_block,
            attributes=pdl_launch_attributes(PDLLevel.OVERLAP_AT_BEGINNING),
        )
        var packed_tt = TileTensor(packed, row_major(2 * fout, K.K))
        conv2d_patch_fprop[OW, CIN, FH, FW, has_epilogue_fn, has_compute_fn](
            output, input, packed_tt, ctx, epilogue_fn, compute_fn
        )
        return True
