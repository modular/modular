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
"""SM100 Conv2D fprop on column tiles: 3x3, stride 1, padding 1.

Target family: Blackwell sm_100a, 1-SM UMMA, fp16 and bf16 (fp32 accumulate)
or fp32 on the tf32 tensor core. The activation is NHWC, the filter KRSC
`[F, R, S, C]` (K contiguous), the output NHWC.

A tile is `BM = 128` consecutive rows of ONE image column. Every window a tile
needs is then a column of the image, so the 3x3 filter does not need an
im2col box per tap: one plain 4-D TMA box of `BM + 8` rows by `BK` channels at
image column `w + s - 1` serves the three filter rows, whose A operands are
the same swizzled tile read `r` rows later (the UMMA descriptor's start moves
by `r` rows of `BK * eb` bytes). A row or column outside the image reads as
zero, which is the padding, so the activation is never materialized and no
im2col descriptor is needed. That is three times fewer activation loads than
the im2col schedule of `conv2d_f96_fprop`.

One CTA walks `units` passes of `MPASS = 3` adjacent tiles. K is walked
channel block outermost, then the window column `s`, then the filter row `r`.
Every filter box is loaded once for the three tiles of a pass (M-inner order),
except the last `tail` boxes, which stay resident while each tile takes them in
turn, so tile 0's accumulator completes while tiles 1 and 2 are still in the
mainloop and its epilogue overlaps them. Each tile owns a 96-column TMEM
accumulator; a ring of five (480 of the 512 columns) lets a persistent CTA's
second pass start while the first pass drains.

A plain convolution leaves TMEM as three 32-column boxes written to shared
memory in the swizzled layout the TMA store reads, so a lane's 16-byte chunks
do not bank-conflict, and is stored by TMA. With an elementwise epilogue each
thread instead owns the output pixel of its TMEM lane and applies
`conv_epilogue.apply_conv_epilogue` in registers.

Warp roles (7 warps, 224 threads):

| warps | role                                                      |
|-------|-----------------------------------------------------------|
| 0-3   | epilogue: TMEM -> smem -> TMA store, or registers         |
| 4     | filter load: one TMA box per K step                       |
| 5     | activation load: one window per (channel block, s, tile)  |
| 6     | MMA issue, TMEM allocation                                |

The grid is `(ceildiv(OW * ceildiv(OH, BM), 3 * units), F / 96)`.
"""

from std.collections import Array, OptionalReg
from std.math import align_up, ceildiv
from std.sys import size_of
from std.utils.static_tuple import StaticTuple

from layout import (
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
from .conv_config import Conv2dProblemShape
from .dispatch import _transpose_fcrs_to_krsc
from .dedicated_fprop import (
    assert_dedicated_operands,
    create_box_tma,
    dedicated_problem_shape,
)


struct Conv2dColumnFprop[
    act_type: DType,
    cin: Int,
    fout: Int,
    row_tiles: Int,
    units: Int,
    tail: Int,
    stages: Int,
    cstages: Int,
    has_epilogue_fn: Bool = False,
    has_compute_fn: Bool = False,
    has_residual: Bool = False,
    nhwc_coords: Bool = True,
    round_first: Bool = False,
]:
    """The column-tile conv2d fprop kernel and its launch helpers.

    Parameters:
        act_type: Element type of the activation, filter and output (fp16,
            bf16, or fp32 for the tf32 tensor core).
        cin: Input channels, 96 or 192.
        fout: Output channels, a multiple of 96.
        row_tiles: Tiles per image column, `ceildiv(OH, 128)`. A compile-time
            count, so splitting a tile index into (column, half) is a shift
            rather than a division on the loader's critical path.
        units: Passes each CTA walks, as a compile-time count so the ring
            positions fold into constants: CTA `c` of a grid of `G` takes
            passes `c`, `c + G`, ..., `c + (units - 1) G`.
        tail: Filter boxes held through the tail while each tile finishes, a
            multiple of the filter height.
        stages: Activation ring depth in passes (slots are `stages * 3`).
        cstages: Output staging slots.
        has_epilogue_fn: An elementwise epilogue stores each output value.
        has_compute_fn: An elementwise epilogue transforms each output value.
        has_residual: Add `beta * source` to each output value first.
        nhwc_coords: The closures take NHWC coordinates, not `[M, N]` ones.
        round_first: Run the epilogue arithmetic in the output dtype.
    """

    # ========== Problem family ==========
    comptime EB = size_of[Self.act_type]()
    comptime R = 3
    comptime S = 3
    comptime PAD = 1
    comptime BM = 128
    comptime BN = 96
    comptime WINDOW = Self.BM + 8
    """Rows of one activation window: the tile, the row above, the row below,
    padded to a whole 8-row swizzle atom."""
    comptime MPASS = 3
    """Tiles per pass; one filter box serves all of them."""
    comptime SWIZZLE = TensorMapSwizzle.SWIZZLE_128B if (
        Self.cin * Self.EB
    ) % 128 == 0 else TensorMapSwizzle.SWIZZLE_64B
    """The widest swizzle span dividing a channel row."""
    comptime BK = Self.SWIZZLE.bytes() // Self.EB
    """Channels per K tile: one swizzle row."""
    comptime MMA_K = 32 // Self.EB
    comptime CBLOCKS = Self.cin // Self.BK
    comptime GROUPS = Self.S * Self.CBLOCKS
    """(channel block, window column) pairs: the activation windows per tile."""
    comptime KT = Self.GROUPS * Self.R
    """Filter boxes of one pass."""
    comptime HELD_GROUPS = Self.tail // Self.R
    comptime ROW_BYTES = Self.BK * Self.EB
    """Bytes between window rows: the shift of one filter row."""
    comptime OUT_SWIZZLE = TensorMapSwizzle.SWIZZLE_64B if Self.EB == 2 else TensorMapSwizzle.SWIZZLE_128B
    """Swizzle of the staged output boxes: one 32-column row spans it."""
    comptime CHUNK = 32
    """Output columns per staged TMA store box."""
    comptime NCHUNK = Self.BN // Self.CHUNK

    # ========== Rings and shared memory ==========
    comptime ASTAGES = Self.stages * Self.MPASS
    comptime BSTAGES = Self.tail
    comptime NACC = min(512 // Self.BN, Self.MPASS * Self.units)
    """Accumulator slots in tensor memory, 96 columns each."""
    comptime A_ELEMS = Self.WINDOW * Self.BK
    comptime B_ELEMS = Self.BN * Self.BK
    comptime C_ELEMS = Self.BM * Self.CHUNK
    comptime A_BYTES = Self.A_ELEMS * Self.EB
    comptime B_BYTES = Self.B_ELEMS * Self.EB
    comptime C_BYTES = Self.C_ELEMS * Self.EB
    comptime A_OFF = 0
    comptime B_OFF = align_up(Self.A_OFF + Self.ASTAGES * Self.A_BYTES, 1024)
    comptime C_OFF = align_up(Self.B_OFF + Self.BSTAGES * Self.B_BYTES, 1024)
    comptime BAR_OFF = Self.C_OFF + Self.NCHUNK * Self.cstages * Self.C_BYTES

    comptime A_FULL = 0
    comptime A_EMPTY = Self.A_FULL + Self.ASTAGES
    comptime B_FULL = Self.A_EMPTY + Self.ASTAGES
    comptime B_EMPTY = Self.B_FULL + Self.BSTAGES
    comptime ACC_FULL = Self.B_EMPTY + Self.BSTAGES
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

    @staticmethod
    @always_inline
    def _grid_ctas(out_w: Int) -> Int:
        """CTAs along the grid's x axis."""
        return ceildiv(out_w * Self.row_tiles, Self.MPASS * Self.units)

    @staticmethod
    @always_inline
    def _tile_pos(cta: Int, ctas: Int, u: Int, mt: Int) -> Tuple[Int, Int]:
        """The (image column, row half) of tile `mt` of CTA `cta`'s pass `u`."""
        return divmod((cta + u * ctas) * Self.MPASS + mt, Self.row_tiles)

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
    @__name(t"sm100_conv2d_column_fprop_{Self.act_type}_{Self.cin}_{Self.fout}")
    def run[
        EpilogueFnType: ElementwiseEpilogueFn,
        ComputeFnType: ElementwiseComputeFn,
    ](
        act_desc: TMADescriptor,
        filt_desc: TMADescriptor,
        out_desc: TMADescriptor,
        out_h: Int32,
        out_w: Int32,
        out_ptr: MutPointer[Scalar[Self.act_type], MutUntrackedOrigin],
        src_ptr: MutPointer[Scalar[Self.act_type], MutUntrackedOrigin],
        beta: Float32,
        epilogue_fn: EpilogueFnType,
        compute_fn: ComputeFnType,
    ):
        """Kernel entry point.

        Args:
            act_desc: Descriptor for the NHWC activation `[N, H, W, C]`, box
                `[1, BM + 8, 1, BK]`.
            filt_desc: Descriptor for the KRSC filter viewed `[F, K]`, box
                `[96, BK]`.
            out_desc: Descriptor for the NHWC output `[N, OH, OW, F]`, box
                `[1, BM, 1, 32]`.
            out_h: Output height.
            out_w: Output width.
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
            Int(out_w),
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
        out_w: Int,
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
        var cta = block_idx.x
        var n0 = block_idx.y * Self.BN
        var ctas = Self._grid_ctas(out_w)

        if tid == 0:
            prefetch_tma_descriptor(
                Pointer(to=act_desc).unsafe_bitcast[NoneType]()
            )
            prefetch_tma_descriptor(
                Pointer(to=filt_desc).unsafe_bitcast[NoneType]()
            )
            prefetch_tma_descriptor(
                Pointer(to=out_desc).unsafe_bitcast[NoneType]()
            )
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
                sc,
                bars,
                tmem_slot,
                tid,
                cta,
                ctas,
                n0,
                out_h,
                out_w,
                out_ptr,
                src_ptr,
                beta,
                epilogue_fn,
                compute_fn,
            )
        elif warp == 4:
            Self._filter_load(filt_desc, sb, bars, n0)
        elif warp == 5:
            Self._act_load(act_desc, sa, bars, cta, ctas)
        else:
            Self._mma(sa, sb, bars, tmem_slot)

    # ----- filter load warp -----

    @staticmethod
    @inline(.always)
    def _filter_load(
        filt_desc: TMADescriptor,
        sb: MutPointer[Scalar[Self.act_type], _, address_space=.SHARED],
        bars: MutPointer[SharedMemBarrier, _, address_space=.SHARED],
        n0: Int,
    ):
        """Loads the filter boxes of every pass in the order the MMA warp
        consumes them: channel block outermost, window column, filter row."""
        var desc = Pointer(to=filt_desc).unsafe_bitcast[NoneType]()
        # The predecessor may be the pack that writes this filter.
        wait_on_dependent_grids()
        var stage = 0
        var phase = UInt32(0)
        comptime for _ in range(Self.units):
            for kt in range(Self.KT):
                var group, r = divmod(kt, Self.R)
                var cb, s = divmod(group, Self.S)
                var box = (r * Self.S + s) * Self.CBLOCKS + cb
                bars.unsafe_offset(Self.B_EMPTY + stage)[].wait(phase ^ 1)
                if elect_one_sync():
                    var full = bars.unsafe_offset(Self.B_FULL + stage)
                    full[].expect_bytes(Int32(Self.B_BYTES))
                    cp_async_bulk_tensor_shared_cluster_global[cta_group=1](
                        sb.unsafe_offset(stage * Self.B_ELEMS),
                        desc,
                        full,
                        (box * Self.BK, n0),
                    )
                stage += 1
                if stage == Self.BSTAGES:
                    stage = 0
                    phase ^= 1

    # ----- activation load warp -----

    @staticmethod
    @inline(.always)
    def _act_load(
        act_desc: TMADescriptor,
        sa: MutPointer[Scalar[Self.act_type], _, address_space=.SHARED],
        bars: MutPointer[SharedMemBarrier, _, address_space=.SHARED],
        cta: Int,
        ctas: Int,
    ):
        """Loads one window per (channel block, window column, tile) in the
        order the MMA warp consumes them."""
        var desc = Pointer(to=act_desc).unsafe_bitcast[NoneType]()
        # The activation is the predecessor's output.
        wait_on_dependent_grids()
        var stage = 0
        var phase = UInt32(0)
        comptime for u in range(Self.units):
            for group in range(Self.GROUPS):
                var cb, s = divmod(group, Self.S)
                comptime for t in range(Self.MPASS):
                    var col, half = Self._tile_pos(cta, ctas, u, t)
                    bars.unsafe_offset(Self.A_EMPTY + stage)[].wait(phase ^ 1)
                    if elect_one_sync():
                        var full = bars.unsafe_offset(Self.A_FULL + stage)
                        full[].expect_bytes(Int32(Self.A_BYTES))
                        cp_async_bulk_tensor_shared_cluster_global[cta_group=1](
                            sa.unsafe_offset(stage * Self.A_ELEMS),
                            desc,
                            full,
                            (
                                cb * Self.BK,
                                col + s - Self.PAD,
                                half * Self.BM - Self.PAD,
                                0,
                            ),
                        )
                    stage += 1
                    if stage == Self.ASTAGES:
                        stage = 0
                        phase ^= 1

    # ----- MMA warp -----

    @staticmethod
    @always_inline
    def _ring_slot(
        stage0: Int, phase0: UInt32, offset: Int, depth: Int
    ) -> Tuple[Int, UInt32]:
        """The slot `offset` entries after ring position `(stage0, phase0)`,
        and the phase its barrier is waited with (`offset <= depth`)."""
        var pos = stage0 + offset
        var wrapped = pos >= depth
        return (
            pos - depth if wrapped else pos,
            phase0 ^ UInt32(1) if wrapped else phase0,
        )

    @staticmethod
    @inline(.always)
    def _mma(
        sa: MutPointer[Scalar[Self.act_type], _, address_space=.SHARED],
        sb: MutPointer[Scalar[Self.act_type], _, address_space=.SHARED],
        bars: MutPointer[SharedMemBarrier, _, address_space=.SHARED],
        tmem_slot: MutPointer[UInt32, _, address_space=.SHARED],
    ):
        """Allocates TMEM, issues every UMMA of every pass, and frees TMEM."""
        tcgen05_alloc[1](tmem_slot, UInt32(Self.TMEM_COLS))
        syncwarp()
        var tmem = tmem_slot[]
        Self.MmaEpiSync.arrive()

        var mma_op = Self.MmaOp()
        var a_desc = _create_mma_desc_k_major[Self.act_type, Self.SWIZZLE](sa)
        var b_desc = _create_mma_desc_k_major[Self.act_type, Self.SWIZZLE](sb)

        var a_stage = 0
        var a_phase = UInt32(0)
        var b_stage = 0
        var b_phase = UInt32(0)

        comptime for u in range(Self.units):
            # This pass's three accumulators: tile `t` lives in slot
            # `(u * 3 + t) % NACC` once the epilogue has drained the tile that
            # last used it.
            var acc = Array[UInt32, Self.MPASS](uninitialized=True)
            comptime for mt in range(Self.MPASS):
                comptime lap = (u * Self.MPASS + mt) // Self.NACC
                comptime sl = (u * Self.MPASS + mt) % Self.NACC
                acc[mt] = tmem + UInt32(sl * Self.BN)
                bars.unsafe_offset(Self.ACC_EMPTY + sl)[].wait(
                    UInt32(lap & 1) ^ 1
                )
            tcgen05_fence_after()

            # Main groups: the three tiles' windows are consecutive slots, each
            # filter box serves all three, and the window of filter row `r`
            # is the tile read `r` rows later.
            for group in range(Self.GROUPS - Self.HELD_GROUPS):
                comptime for mt in range(Self.MPASS):
                    bars.unsafe_offset(Self.A_FULL + a_stage + mt)[].wait(
                        a_phase
                    )
                comptime for r in range(Self.R):
                    bars.unsafe_offset(Self.B_FULL + b_stage)[].wait(b_phase)
                    if elect_one_sync():
                        comptime for mt in range(Self.MPASS):
                            mma_op.mma_from_desc(
                                a_desc
                                + (a_stage + mt) * Self.A_BYTES
                                + r * Self.ROW_BYTES,
                                b_desc + b_stage * Self.B_BYTES,
                                acc[mt],
                                init_c=(group == 0 and r == 0),
                            )
                        mma_op.commit(
                            bars.unsafe_offset(Self.B_EMPTY + b_stage)
                        )
                    b_stage += 1
                    if b_stage == Self.BSTAGES:
                        b_stage = 0
                        b_phase ^= 1
                if elect_one_sync():
                    comptime for mt in range(Self.MPASS):
                        mma_op.commit(
                            bars.unsafe_offset(Self.A_EMPTY + a_stage + mt)
                        )
                a_stage += Self.MPASS
                if a_stage == Self.ASTAGES:
                    a_stage = 0
                    a_phase ^= 1

            # Tail: the last groups' filter boxes stay resident while each
            # tile takes them in turn, so a tile's accumulator completes early.
            var a0 = a_stage
            var a0_phase = a_phase
            var b0 = b_stage
            var b0_phase = b_phase
            comptime for mt in range(Self.MPASS):
                comptime for gi in range(Self.HELD_GROUPS):
                    # Tile 0 takes each held group as it lands; later tiles
                    # find them all resident.
                    comptime if mt == 0:
                        comptime for t in range(Self.MPASS):
                            var wa = Self._ring_slot(
                                a0,
                                a0_phase,
                                gi * Self.MPASS + t,
                                Self.ASTAGES,
                            )
                            bars.unsafe_offset(Self.A_FULL + wa[0])[].wait(
                                wa[1]
                            )
                        comptime for r in range(Self.R):
                            var wb = Self._ring_slot(
                                b0, b0_phase, gi * Self.R + r, Self.BSTAGES
                            )
                            bars.unsafe_offset(Self.B_FULL + wb[0])[].wait(
                                wb[1]
                            )
                    var held_a = Self._ring_slot(
                        a0, a0_phase, gi * Self.MPASS + mt, Self.ASTAGES
                    )
                    if elect_one_sync():
                        comptime for r in range(Self.R):
                            var held_b = Self._ring_slot(
                                b0, b0_phase, gi * Self.R + r, Self.BSTAGES
                            )
                            mma_op.mma_from_desc(
                                a_desc
                                + held_a[0] * Self.A_BYTES
                                + r * Self.ROW_BYTES,
                                b_desc + held_b[0] * Self.B_BYTES,
                                acc[mt],
                                init_c=False,
                            )
                            comptime if mt == Self.MPASS - 1:
                                mma_op.commit(
                                    bars.unsafe_offset(Self.B_EMPTY + held_b[0])
                                )
                        mma_op.commit(
                            bars.unsafe_offset(Self.A_EMPTY + held_a[0])
                        )
                if elect_one_sync():
                    comptime sl = (u * Self.MPASS + mt) % Self.NACC
                    mma_op.commit(bars.unsafe_offset(Self.ACC_FULL + sl))
            var a_next = Self._ring_slot(
                a0, a0_phase, Self.HELD_GROUPS * Self.MPASS, Self.ASTAGES
            )
            a_stage = a_next[0]
            a_phase = a_next[1]
            var b_next = Self._ring_slot(b0, b0_phase, Self.tail, Self.BSTAGES)
            b_stage = b_next[0]
            b_phase = b_next[1]

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
        sc: MutPointer[Scalar[Self.act_type], _, address_space=.SHARED],
        bars: MutPointer[SharedMemBarrier, _, address_space=.SHARED],
        tmem_slot: MutPointer[UInt32, _, address_space=.SHARED],
        tid: Int,
        cta: Int,
        ctas: Int,
        n0: Int,
        out_h: Int,
        out_w: Int,
        out_ptr: MutPointer[Scalar[Self.act_type], MutUntrackedOrigin],
        src_ptr: MutPointer[Scalar[Self.act_type], MutUntrackedOrigin],
        beta: Float32,
        epilogue_fn: EpilogueFnType,
        compute_fn: ComputeFnType,
    ):
        """Moves each accumulator tile TMEM -> smem -> TMA store.

        With `has_epilogue_fn`, `has_compute_fn` or `has_residual` each thread
        instead owns the output pixel of its TMEM lane and applies the
        epilogue to its 96 channels in registers (`apply_conv_epilogue`).
        """
        Self.MmaEpiSync.wait()
        var tmem = tmem_slot[]

        comptime if Self.has_epilogue_fn or Self.has_compute_fn or Self.has_residual:
            comptime for u in range(Self.units):
                comptime for mt in range(Self.MPASS):
                    comptime lap = (u * Self.MPASS + mt) // Self.NACC
                    comptime slot = (u * Self.MPASS + mt) % Self.NACC
                    bars.unsafe_offset(Self.ACC_FULL + slot)[].wait(
                        UInt32(lap & 1)
                    )
                    tcgen05_fence_after()
                    var col, half = Self._tile_pos(cta, ctas, u, mt)
                    var y = half * Self.BM + tid
                    var active = col < out_w and y < out_h
                    # One call for all 96 columns: the TMEM loads are
                    # warp-collective, so lanes without a pixel (a ragged
                    # last tile) must not leave the call before the others
                    # have issued them.
                    apply_conv_epilogue[
                        cols=Self.BN,
                        has_epilogue_fn=Self.has_epilogue_fn,
                        has_compute_fn=Self.has_compute_fn,
                        has_residual=Self.has_residual,
                        nhwc=Self.nhwc_coords,
                        round_first=Self.round_first,
                    ](
                        tmem + UInt32(slot * Self.BN),
                        y * out_w + col,
                        n0,
                        Self.fout,
                        out_ptr,
                        src_ptr,
                        beta,
                        epilogue_fn,
                        compute_fn,
                        active,
                        out_h * out_w,
                        out_w,
                    )
                    tcgen05_fence_before()
                    _ = bars.unsafe_offset(Self.ACC_EMPTY + slot)[].arrive()
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
        # A 64-byte row (16-bit) swizzles its chunk by `(row / 2) % 4`, a
        # 128-byte row (fp32) by `row % 8`.
        var key = (tid >> 1) & 3 if Self.EB == 2 else tid & 7
        var desc = Pointer(to=out_desc).unsafe_bitcast[NoneType]()

        comptime for u in range(Self.units):
            comptime for mt in range(Self.MPASS):
                comptime j = u * Self.MPASS + mt
                comptime lap = j // Self.NACC
                comptime slot = j % Self.NACC
                comptime cslot = j % Self.cstages
                bars.unsafe_offset(Self.ACC_FULL + slot)[].wait(UInt32(lap & 1))
                tcgen05_fence_after()
                var col, half = Self._tile_pos(cta, ctas, u, mt)

                # The staging slot's previous store must have read it out.
                comptime if j >= Self.cstages:
                    if tid == 0:
                        cp_async_bulk_wait_group[Int32(Self.cstages - 1)]()
                    Self.EpiSync.sync()

                comptime for piece in range(Self.NCHUNK):
                    var regs = tcgen05_ld[
                        datapaths=32,
                        bits=32,
                        repeat=Self.CHUNK,
                        dtype=.float32,
                        pack=False,
                    ](tmem + UInt32(slot * Self.BN + piece * Self.CHUNK))
                    tcgen05_load_wait()
                    var dst = sc.unsafe_offset(
                        (cslot * Self.NCHUNK + piece) * Self.C_ELEMS
                        + tid * Self.CHUNK
                    )
                    comptime for c in range(NCH):
                        var v = SIMD[Self.act_type, CH]()
                        comptime for e in range(CH):
                            v[e] = regs[c * CH + e].cast[Self.act_type]()
                        dst.unsafe_offset((c ^ key) * CH).unsafe_store[
                            alignment=16
                        ](v)
                # Every column of the slot is in registers: hand it back to
                # the MMA warp for a later tile.
                tcgen05_fence_before()
                _ = bars.unsafe_offset(Self.ACC_EMPTY + slot)[].arrive()
                fence_async_view_proxy()
                Self.EpiSync.sync()
                if tid == 0:
                    comptime for piece in range(Self.NCHUNK):
                        cp_async_bulk_tensor_global_shared_cta(
                            sc.unsafe_offset(
                                (cslot * Self.NCHUNK + piece) * Self.C_ELEMS
                            ),
                            desc,
                            (n0 + piece * Self.CHUNK, col, half * Self.BM, 0),
                        )
                    cp_async_bulk_commit_group()

        tcgen05_fence_before()
        _ = bars.unsafe_offset(Self.TMEM_FREE)[].arrive()
        if tid == 0:
            cp_async_bulk_wait_group[0, read=False]()
        Self.EpiSync.sync()
        launch_dependent_grids()


comptime _ROW_TILES = 2
"""Tiles per image column of every served plan: 240 output rows."""


comptime _PLAN_NONE = 0
comptime _PLAN_16B_96_96 = 1
comptime _PLAN_F32_96_96 = 2
comptime _PLAN_F32_192_96 = 3


def _column_plan[act_type: DType](problem: Conv2dProblemShape) -> Int:
    """The launch configuration serving `problem`, or `_PLAN_NONE`.

    Each plan is a (dtype, channels, image size) the kernel was timed ahead
    of the route it replaces on, filter pack included: the 240x416 images of
    the 96 -> 96 channel conv2d in all three dtypes and of the fp32 192 -> 96
    one. The 192 -> 192 channel conv2d ties with its route once the pack is
    counted, and the 16-bit 192 -> 96 one loses or ties with a store epilogue,
    the form the graph calls, so neither is served.
    """
    if (
        problem.batch != 1
        or problem.filter_h != 3
        or problem.filter_w != 3
        or problem.pad_h != 1
        or problem.pad_w != 1
        or problem.stride_h != 1
        or problem.stride_w != 1
        or problem.dilation_h != 1
        or problem.dilation_w != 1
        or problem.groups != 1
    ):
        return _PLAN_NONE
    var big = problem.in_height == 240 and problem.in_width == 416
    var cin = problem.in_channels
    var fout = problem.out_channels
    comptime match act_type:
        case .float32:
            if big and cin == 96 and fout == 96:
                return _PLAN_F32_96_96
            if big and cin == 192 and fout == 96:
                return _PLAN_F32_192_96
        case .float16 | .bfloat16:
            if big and cin == 96 and fout == 96:
                return _PLAN_16B_96_96
        case _:
            pass
    return _PLAN_NONE


def conv2d_column_supported[
    act_type: DType
](problem: Conv2dProblemShape) -> Bool:
    """Whether `conv2d_column_fprop` can run `problem`.

    Parameters:
        act_type: Element type of the activation, filter and output.

    Args:
        problem: The convolution geometry.

    Returns:
        True when the problem is one of the shapes the kernel serves.
    """
    return _column_plan[act_type](problem) != _PLAN_NONE


def _launch_column[
    EpilogueFnType: ElementwiseEpilogueFn,
    ComputeFnType: ElementwiseComputeFn,
    //,
    act_type: DType,
    cin: Int,
    fout: Int,
    row_tiles: Int,
    units: Int,
    tail: Int,
    stages: Int,
    cstages: Int,
    has_epilogue_fn: Bool,
    has_compute_fn: Bool,
    has_residual: Bool,
    nhwc_coords: Bool,
    round_first: Bool,
](
    output: TileTensor[mut=True, ...],
    activation: TileTensor[mut=True, act_type, address_space=.GENERIC, ...],
    filter: TileTensor[...],
    ctx: DeviceContext,
    source_ptr: OptionalReg[MutPointer[Scalar[act_type], _]],
    beta: Float32,
    epilogue_fn: EpilogueFnType,
    compute_fn: ComputeFnType,
) raises:
    comptime K = Conv2dColumnFprop[
        act_type,
        cin,
        fout,
        row_tiles,
        units,
        tail,
        stages,
        cstages,
        has_epilogue_fn,
        has_compute_fn,
        has_residual,
        nhwc_coords,
        round_first,
    ]
    var out_h = Int(output.dim[1]())
    var out_w = Int(output.dim[2]())

    var act_desc = create_tensor_tile[
        coord[1, K.WINDOW, 1, K.BK],
        swizzle_mode=K.SWIZZLE,
        __tile_shape=coord[1, K.WINDOW, 1, K.BK],
        __desc_shape=coord[1, K.WINDOW, 1, K.BK],
    ](ctx, activation).descriptor
    var filter_matrix = filter.reshape(row_major(fout, K.R * K.S * cin))
    var filt_desc = create_box_tma[K.BN, K.BK, K.SWIZZLE](
        ctx, filter_matrix
    ).descriptor
    var out_desc = create_tensor_tile[
        coord[1, K.BM, 1, K.CHUNK],
        swizzle_mode=K.OUT_SWIZZLE,
        __tile_shape=coord[1, K.BM, 1, K.CHUNK],
        __desc_shape=coord[1, K.BM, 1, K.CHUNK],
    ](ctx, output).descriptor

    # `output.dtype == act_type` is asserted by the caller.
    var out_ptr = rebind[MutPointer[Scalar[act_type], MutUntrackedOrigin]](
        output.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    )
    var src_ptr = out_ptr
    comptime if has_residual:
        # SAFETY: set when has_residual == True
        src_ptr = source_ptr.unsafe_value().unsafe_origin_cast[
            MutUntrackedOrigin
        ]()

    var ctas = K._grid_ctas(out_w)

    comptime kernel = K.run[EpilogueFnType, ComputeFnType]
    ctx.enqueue_function[kernel](
        act_desc,
        filt_desc,
        out_desc,
        Int32(out_h),
        Int32(out_w),
        out_ptr,
        src_ptr,
        beta,
        host_args=(epilogue_fn, compute_fn),
        grid_dim=(ctas, fout // K.BN, 1),
        block_dim=K.NUM_THREADS,
        shared_mem_bytes=K.SMEM_BYTES,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(K.SMEM_BYTES)
        ),
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )


def conv2d_column_fprop[
    EpilogueFnType: ElementwiseEpilogueFn = type_of(no_epilogue_fn),
    ComputeFnType: ElementwiseComputeFn = type_of(no_compute_fn),
    //,
    act_type: DType,
    has_epilogue_fn: Bool = False,
    has_compute_fn: Bool = False,
    has_residual: Bool = False,
    nhwc_coords: Bool = True,
    round_first: Bool = False,
](
    output: TileTensor[mut=True, ...],
    activation: TileTensor[mut=True, act_type, address_space=.GENERIC, ...],
    filter: TileTensor[...],
    problem: Conv2dProblemShape,
    ctx: DeviceContext,
    source_ptr: OptionalReg[MutPointer[Scalar[act_type], MutAnyOrigin]] = None,
    beta: Float32 = 0.0,
    epilogue_fn: EpilogueFnType = no_epilogue_fn,
    compute_fn: ComputeFnType = no_compute_fn,
) raises:
    """Launches the column-tile conv2d fprop kernel.

    Accepts only problems `conv2d_column_supported` accepts; the caller checks.

    Parameters:
        EpilogueFnType: Type of `epilogue_fn` (inferred).
        ComputeFnType: Type of `compute_fn` (inferred).
        act_type: Element type of the activation, filter and output (fp16,
            bf16 or fp32 with a tf32 tensor core).
        has_epilogue_fn: `epilogue_fn` stores each output value; the kernel
            writes nothing to `output`.
        has_compute_fn: `compute_fn` transforms each output value; excludes
            `has_epilogue_fn` and `has_residual`.
        has_residual: Store `Conv(A, B) + beta * source` (through
            `epilogue_fn` when `has_epilogue_fn`).
        nhwc_coords: The closures take NHWC coordinates `(n, h, w, c)`
            instead of `[M, N]` ones `(pixel, c)`.
        round_first: Run the epilogue arithmetic in the output dtype.

    Args:
        output: Output tensor `[N, H_out, W_out, F]`, NHWC.
        activation: Input activation `[N, H, W, C]`, NHWC.
        filter: Filter `[F, 3, 3, C]`, KRSC (channel innermost).
        problem: The convolution geometry.
        ctx: Device context for the launch.
        source_ptr: Residual source, NHWC like `output`; required when
            `has_residual`.
        beta: Residual scale factor.
        epilogue_fn: Storing elementwise epilogue.
        compute_fn: Transforming elementwise epilogue.

    Raises:
        If `problem` is outside the supported family, or if descriptor
        creation or the launch fails.
    """
    comptime assert (
        output.dtype == act_type and filter.dtype == act_type
    ), "activation, filter and output must share one element type"
    assert_dedicated_operands[has_epilogue_fn, has_compute_fn, has_residual](
        output, activation, filter
    )
    var plan = _column_plan[act_type](problem)

    @always_inline
    @__allow_legacy_any_origin_fields
    def launch[
        cin: Int,
        fout: Int,
        units: Int,
        tail: Int,
        stages: Int,
        cstages: Int,
    ]() raises {imm}:
        _launch_column[
            act_type,
            cin,
            fout,
            _ROW_TILES,
            units,
            tail,
            stages,
            cstages,
            has_epilogue_fn,
            has_compute_fn,
            has_residual,
            nhwc_coords,
            round_first,
        ](
            output,
            activation,
            filter,
            ctx,
            source_ptr,
            beta,
            epilogue_fn,
            compute_fn,
        )

    comptime if act_type == .float32:
        if plan == _PLAN_F32_96_96:
            launch[96, 96, 2, 6, 2, 1]()
        elif plan == _PLAN_F32_192_96:
            launch[192, 96, 2, 6, 2, 1]()
        else:
            raise Error("conv2d_column_fprop: unsupported problem shape")
    else:
        if plan == _PLAN_16B_96_96:
            launch[96, 96, 2, 9, 3, 2]()
        else:
            raise Error("conv2d_column_fprop: unsupported problem shape")


def conv2d_column_dispatch[
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    EpilogueFnType: ElementwiseEpilogueFn = type_of(no_epilogue_fn),
    ComputeFnType: ElementwiseComputeFn = type_of(no_compute_fn),
    //,
    filter_is_fcrs: Bool,
    has_residual: Bool = False,
    has_epilogue_fn: Bool = False,
    has_compute_fn: Bool = False,
    nhwc_coords: Bool = True,
    round_first: Bool = False,
](
    input: TileTensor[mut=True, input_type, address_space=.GENERIC, ...],
    filter: TileTensor[filter_type, ...],
    output: TileTensor[mut=True, output_type, ...],
    pad_h: Int,
    pad_w: Int,
    ctx: DeviceContext,
    source_ptr: OptionalReg[
        MutPointer[Scalar[output_type], MutAnyOrigin]
    ] = None,
    beta: Float32 = 0.0,
    epilogue_fn: EpilogueFnType = no_epilogue_fn,
    compute_fn: ComputeFnType = no_compute_fn,
) raises -> Bool:
    """Runs a stride 1, dilation 1 convolution on the column-tile kernel if it
    serves the problem.

    The kernel reads a KRSC filter, so the caller's FCRS filter is packed into
    a scratch buffer first (PDL overlaps the pack with the kernel's launch).

    Parameters:
        input_type: Element type of the activation.
        filter_type: Element type of the filter.
        output_type: Element type of the output.
        EpilogueFnType: Type of `epilogue_fn` (inferred).
        ComputeFnType: Type of `compute_fn` (inferred).
        filter_is_fcrs: Whether `filter` is FCRS `[F, C, R, S]`; an RSCF
            filter is not served.
        has_residual: Fuse `D = Conv(A, B) + beta * source`.
        has_epilogue_fn: `epilogue_fn` stores each output value.
        has_compute_fn: `compute_fn` transforms each output value; excludes
            `has_epilogue_fn` and `has_residual`.
        nhwc_coords: The closures take NHWC coordinates, as `conv_gpu`'s
            callers write them, instead of `[M, N]` ones.
        round_first: Run the epilogue arithmetic in the output dtype.

    Args:
        input: Input activation, NHWC.
        filter: Filter weights.
        output: Output, NHWC.
        pad_h: Symmetric padding along the height.
        pad_w: Symmetric padding along the width.
        ctx: Device context for the launch.
        source_ptr: Residual source, NHWC like `output`.
        beta: Residual scale factor.
        epilogue_fn: Storing elementwise epilogue.
        compute_fn: Transforming elementwise epilogue.

    Returns:
        True if the kernel ran; False, with nothing enqueued, if the device
        is not a B200 (the only GPU the kernel was timed on) or the problem or
        the operand layouts are outside what it serves.

    Raises:
        If descriptor creation or a launch fails.
    """
    comptime assert input.flat_rank == 4, "input must be rank 4 (NHWC)"
    comptime assert filter.flat_rank == 4, "filter must be rank 4"
    comptime assert output.flat_rank == 4, "output must be rank 4 (NHWC)"
    comptime assert not (
        has_epilogue_fn and has_compute_fn
    ), "a store epilogue excludes the compute epilogue"
    comptime assert not (
        has_residual and has_compute_fn
    ), "the residual kernel has no compute epilogue"
    comptime if (
        not (ctx.default_device_info == B200)
        or not filter_is_fcrs
        or input_type != output_type
        or input_type != filter_type
        or not (
            input.is_row_major and filter.is_row_major and output.is_row_major
        )
    ):
        return False
    else:
        var problem = dedicated_problem_shape[filter_is_fcrs](
            input, filter, output, pad_h, pad_w
        )
        if not conv2d_column_supported[input_type](problem):
            return False

        var filter_size = filter.num_elements()
        var krsc = ctx.enqueue_create_buffer[filter_type](filter_size)
        comptime transpose_block = 256
        comptime pack = _transpose_fcrs_to_krsc[
            filter_type,
            filter.LayoutType,
            ImmOrigin(filter.origin),
            filter.Engine,
        ]
        ctx.enqueue_function[pack](
            filter.as_imm(),
            krsc.unsafe_ptr(),
            Int32(problem.out_channels),
            Int32(problem.in_channels),
            Int32(problem.filter_h),
            Int32(problem.filter_w),
            grid_dim=ceildiv(filter_size, transpose_block),
            block_dim=transpose_block,
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )
        var out_tt = output.reshape(
            row_major(
                problem.batch,
                problem.out_height(),
                problem.out_width(),
                problem.out_channels,
            )
        )
        var act_tt = input.reshape(
            row_major(
                problem.batch,
                problem.in_height,
                problem.in_width,
                problem.in_channels,
            )
        )
        var filt_tt = TileTensor(
            krsc,
            row_major(
                problem.out_channels,
                problem.filter_h,
                problem.filter_w,
                problem.in_channels,
            ),
        )
        conv2d_column_fprop[
            input_type,
            has_epilogue_fn=has_epilogue_fn,
            has_compute_fn=has_compute_fn,
            has_residual=has_residual,
            nhwc_coords=nhwc_coords,
            round_first=round_first,
        ](
            out_tt,
            act_tt,
            filt_tt,
            problem,
            ctx,
            rebind[OptionalReg[MutPointer[Scalar[input_type], MutAnyOrigin]]](
                source_ptr
            ),
            beta,
            epilogue_fn,
            compute_fn,
        )
        return True
