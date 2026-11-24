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
"""SM100 Conv2D fprop for 96 output channels, 3x3, pad 1.

Target family: Blackwell sm_100a, 1-SM UMMA, fp16 and bf16 with fp32
accumulation. This is a shape-gated route beside the general structured
kernel in `conv2d_fprop_kernel.mojo`; `dispatch.mojo` selects it only when
`conv2d_f96_supported` accepts the problem.

Why a separate kernel. The general kernel pads N to 128 and walks K in
the order the TMA im2col transaction emits, one output tile per CTA, so at
`F = 96` every CTA re-reads the whole 324 KiB filter and the tensor pipe idles
behind the load warp. This kernel is shaped for `F = 96` with `C_in` of 192
or 96:

- N = 96 exactly (one UMMA N, no padded columns).
- One CTA computes three consecutive 128-pixel output tiles (`MPASS = 3`)
  and loads each filter box once for all three, cutting filter traffic 3x.
  Each tile owns a 96-column TMEM accumulator, 288 of the 512 columns.
- One im2col transaction loads a whole 128-pixel tile. The hardware walks the
  box through output rows and images, so `OW` need not divide the tile, and the
  (n, h, w) start of each of the CTA's three tiles is computed once, not per
  transaction.
- K tiles are one swizzle row of channels (64 at C = 192, 32 at C = 96), 27
  of them either way. K runs 18 steps with one filter box live for the three
  tiles, then a nine step tail that holds the last nine filter boxes while
  each tile finishes,
  so tile 0's accumulator is complete (and its epilogue starts) while tiles 1
  and 2 are still in the mainloop.
- A problem of two waves of work units runs as persistent CTAs that take two
  units each: the second unit's loads run during the first unit's tail, its
  accumulators take the next slots of a five-slot TMEM ring while the first
  unit's epilogue drains, and the CTA is launched and set up once. The unit
  loop is unrolled at compile time so the ring positions fold into constants.
- The activation ring has six stages (twelve at C = 96, the same bytes) and
  the output staging ring one: the payload is the same 225280 bytes at C = 192
  with (5, 2) and (6, 1) stages, and the sixth activation stage buys more than
  the second store slot.
- The accumulator leaves TMEM as two pieces (64 + 32 columns), is staged
  into shared memory with `stmatrix`, and is written with two TMA stores.

Warp roles (7 warps, 224 threads):

| warps | role                                                   |
|-------|--------------------------------------------------------|
| 0-3   | epilogue: TMEM -> smem -> TMA store, slot release        |
| 4     | filter load: one TMA box per K step                     |
| 5     | activation load: im2col TMA, cached coordinates         |
| 6     | MMA issue, TMEM allocation                              |

The filter is KRSC `[F, R, S, C]` (K contiguous, channel innermost), which
is the layout `dispatch.mojo` already prepares for the general kernel.
"""

from std.collections import Array, OptionalReg
from std.math import ceildiv
from std.sys import size_of
from std.utils.static_tuple import StaticTuple

from layout import TileTensor, coord, row_major
from layout.tma_async import SharedMemBarrier, TMATensorTileIm2col
from linalg.arch.sm100 import MmaOpSM100_SS
from linalg.arch.sm100.mma import _create_mma_desc_k_major
from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, WARP_SIZE, block_idx
from max.gpu import grid_dim, thread_idx
from max.gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_fence_after,
    tcgen05_fence_before,
    tcgen05_release_allocation_lock,
)
from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.memory import (
    cp_async_bulk_tensor_global_shared_cta,
    cp_async_bulk_tensor_shared_cluster_global,
    cp_async_bulk_tensor_shared_cluster_global_im2col,
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
from structured_kernels.tile_types import TmaOpType, static_row_major

from linalg.utils import (
    ElementwiseComputeFn,
    ElementwiseEpilogueFn,
    no_compute_fn,
    no_epilogue_fn,
)

from .conv_config import Conv2dProblemShape
from .conv_epilogue import apply_conv_epilogue
from .dedicated_fprop import (
    assert_dedicated_operands,
    create_act_im2col_tma,
    create_box_tma,
    stage_accumulator,
)


def conv2d_f96_supported[
    act_type: DType
](problem: Conv2dProblemShape, ctx: DeviceContext) -> Bool:
    """Whether `conv2d_f96_fprop` can run `problem`.

    Mirrors the kernel's compile-time and launch-time requirements so the
    dispatcher can fall through to the general path for every other shape.

    Parameters:
        act_type: Element type of the activation, filter and output.

    Args:
        problem: The convolution geometry.
        ctx: The device the kernel would run on.

    Returns:
        True when the problem is exactly the family this kernel serves.
    """
    comptime if act_type != .float16 and act_type != .bfloat16:
        return False
    else:
        var out_w = problem.out_width()
        var m_total = problem.batch * problem.out_height() * out_w
        return (
            m_total > 0
            and (problem.in_channels == 192 or problem.in_channels == 96)
            and problem.out_channels == Conv2dFpropF96[act_type, 96].BN
            and problem.filter_h == Conv2dFpropF96[act_type, 96].R
            and problem.filter_w == Conv2dFpropF96[act_type, 96].S
            and problem.pad_h == 1
            and problem.pad_w == 1
            and problem.stride_h == 1
            and problem.stride_w == 1
            and problem.dilation_h == 1
            and problem.dilation_w == 1
            and problem.groups == 1
            and m_total
            % (
                Conv2dFpropF96[act_type, 96].BM
                * Conv2dFpropF96[act_type, 96].MPASS
            )
            == 0
        )


struct Conv2dFpropF96[
    act_type: DType,
    cin: Int,
    units: Int = 1,
    has_epilogue_fn: Bool = False,
    has_compute_fn: Bool = False,
    has_residual: Bool = False,
    nhwc_coords: Bool = False,
    round_first: Bool = False,
]:
    """The 96 output channel conv2d fprop kernel and its launch helpers.

    Parameters:
        act_type: Element type of the activation, filter and output
            (`DType.float16` or `DType.bfloat16`).
        cin: Input channels, 96 or 192.
        units: Work units each CTA walks, as a compile-time count so the ring
            positions fold into constants: CTA `c` of a grid of `G` takes
            units `c`, `c + G`, ..., `c + (units - 1) G`.
        has_epilogue_fn: An elementwise epilogue stores each output value.
        has_compute_fn: An elementwise epilogue transforms each output value.
        has_residual: Add `beta * source` to each output value first.
        nhwc_coords: The closures take NHWC coordinates, not `[M, N]` ones.
        round_first: Run the epilogue arithmetic in the output dtype.

    With any of the three flags the epilogue warps apply
    `conv_epilogue.apply_conv_epilogue` in registers instead of staging the
    tile for a TMA store.
    """

    # ========== Problem family ==========
    comptime CIN = Self.cin
    comptime R = 3
    comptime S = 3
    comptime BM = 128
    comptime BN = 96
    comptime EB = size_of[Self.act_type]()
    comptime SWIZZLE = TensorMapSwizzle.SWIZZLE_128B if (
        Self.CIN * Self.EB
    ) % 128 == 0 else TensorMapSwizzle.SWIZZLE_64B
    """The widest swizzle span dividing a channel row: 128 bytes at C = 192,
    64 at C = 96."""
    comptime BK = Self.SWIZZLE.bytes() // Self.EB
    """Channels per K tile: one swizzle row, so a TMA im2col transaction
    never straddles a filter tap."""
    comptime MMA_K = 16
    comptime MPASS = 3
    """Output tiles one CTA computes; one filter box serves all of them."""
    comptime PIX = Self.BM
    """Output pixels one im2col transaction loads: a whole tile. The hardware
    walks the box through output rows and images, so `OW` need not divide it.
    Narrower boxes (32 pixels) cost the load warp four transactions per K step
    and run 10% to 40% slower."""
    comptime BOXES = Self.BM // Self.PIX
    """Im2col transactions per output tile."""
    comptime NBOX = Self.BOXES * Self.MPASS
    """Im2col transactions per CTA, whose (n, h, w) starts are cached."""
    comptime KT = Self.CIN * Self.R * Self.S // Self.BK
    """K tiles of one output tile (27 at both channel counts)."""
    comptime CBLOCKS = Self.CIN // Self.BK
    """Channel blocks per filter tap (3)."""
    comptime TAPS = Self.R * Self.S
    comptime HELD = Self.TAPS
    """Filter boxes held through the tail while each tile finishes."""
    comptime MAIN_STEPS = Self.KT - Self.HELD
    """K steps walked with the three tiles interleaved (18)."""

    # ========== Rings and shared memory ==========
    comptime ASTAGES = 6 * 64 // Self.BK
    """Activation ring slots: the same bytes at both channel counts. Depths
    from 10 to 18 slots measure the same at C = 96."""
    comptime BSTAGES = Self.TAPS
    comptime CSTAGES = 1
    comptime A_ELEMS = Self.BM * Self.BK
    comptime B_ELEMS = Self.BN * Self.BK
    comptime C_ELEMS = Self.BM * 64
    comptime A_BYTES = Self.A_ELEMS * Self.EB
    comptime B_BYTES = Self.B_ELEMS * Self.EB
    comptime C_BYTES = Self.C_ELEMS * Self.EB
    comptime A_OFF = 0
    comptime B_OFF = Self.A_OFF + Self.ASTAGES * Self.A_BYTES
    comptime C_OFF = Self.B_OFF + Self.BSTAGES * Self.B_BYTES
    comptime BAR_OFF = Self.C_OFF + Self.CSTAGES * Self.C_BYTES

    comptime A_FULL = 0
    comptime A_EMPTY = Self.A_FULL + Self.ASTAGES
    comptime B_FULL = Self.A_EMPTY + Self.ASTAGES
    comptime B_EMPTY = Self.B_FULL + Self.BSTAGES
    comptime ACC_SLOTS = 5
    """Accumulator tiles in tensor memory: 5 x 96 of the 512 columns. A CTA
    walks its output tiles in order, tile `t` in slot `t % ACC_SLOTS`, so a
    tile's epilogue overlaps the next tiles' mainloop."""
    comptime ACC_FULL = Self.B_EMPTY + Self.BSTAGES
    comptime ACC_EMPTY = Self.ACC_FULL + Self.ACC_SLOTS
    comptime TMEM_FREE = Self.ACC_EMPTY + Self.ACC_SLOTS
    comptime NBAR = Self.TMEM_FREE + 1
    comptime TMEM_ADDR_OFF = Self.BAR_OFF + Self.NBAR * 8
    comptime SMEM_BYTES = Self.TMEM_ADDR_OFF + 16

    # ========== Threads ==========
    comptime EPI_THREADS = 128
    comptime NUM_THREADS = Self.EPI_THREADS + 3 * WARP_SIZE
    comptime TMEM_COLS = 512

    comptime MmaEpiSync = WarpGroupBarrier[Self.EPI_THREADS + WARP_SIZE, 1]
    comptime EpiSync = WarpGroupBarrier[Self.EPI_THREADS, 2]

    # ========== TMA descriptor types ==========
    comptime ActTma = TMATensorTileIm2col[
        Self.act_type,
        coord[Self.BM, Self.BK],
        coord[Self.PIX, Self.BK],
    ]
    comptime FiltTma = TmaOpType[
        Self.act_type,
        static_row_major[Self.BN, Self.BK],
        static_row_major[Self.BN, Self.BK],
    ]
    comptime OutWideTma = TmaOpType[
        Self.act_type,
        static_row_major[Self.BM, 64],
        static_row_major[Self.BM, 64],
    ]
    comptime OutNarrowTma = TmaOpType[
        Self.act_type,
        static_row_major[Self.BM, Self.BN - 64],
        static_row_major[Self.BM, Self.BN - 64],
    ]

    comptime OutWideSwizzle = TensorMapSwizzle.SWIZZLE_128B
    """Swizzle of the output staging tile and TMA box for columns 0-63."""
    comptime OutNarrowSwizzle = TensorMapSwizzle.SWIZZLE_64B
    """Swizzle of the output staging tile and TMA box for the 32 columns
    after them."""

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
    @__llvm_arg_metadata(act_tma, `nvvm.grid_constant`)
    @__llvm_arg_metadata(filt_tma, `nvvm.grid_constant`)
    @__llvm_arg_metadata(out_wide_tma, `nvvm.grid_constant`)
    @__llvm_arg_metadata(out_narrow_tma, `nvvm.grid_constant`)
    @__name(t"sm100_conv2d_f96_fprop_{Self.act_type}")
    def run[
        EpilogueFnType: ElementwiseEpilogueFn,
        ComputeFnType: ElementwiseComputeFn,
    ](
        act_tma: Self.ActTma,
        filt_tma: Self.FiltTma,
        out_wide_tma: Self.OutWideTma,
        out_narrow_tma: Self.OutNarrowTma,
        out_w: Int32,
        out_h: Int32,
        out_ptr: MutPointer[Scalar[Self.act_type], MutUntrackedOrigin],
        src_ptr: MutPointer[Scalar[Self.act_type], MutUntrackedOrigin],
        beta: Float32,
        epilogue_fn: EpilogueFnType,
        compute_fn: ComputeFnType,
    ):
        """Kernel entry point.

        Args:
            act_tma: Im2col descriptor for the NHWC activation, box
                `[PIX, BK]`.
            filt_tma: Descriptor for the KRSC filter viewed `[F, K]`, box
                `[BN, BK]`.
            out_wide_tma: Descriptor for the NHWC output viewed `[M, F]`,
                box `[BM, 64]`, 128-byte swizzle (output columns 0-63).
            out_narrow_tma: The same view with box `[BM, 32]`, 64-byte
                swizzle (output columns 64-95).
            out_w: Output width.
            out_h: Output height.
            out_ptr: Output, `[M, F]`; written by the kernel unless
                `has_epilogue_fn`, and only on the register epilogue path.
            src_ptr: Residual source `[M, F]`, read when `has_residual`.
            beta: Residual scale factor.
            epilogue_fn: Storing epilogue, used when `has_epilogue_fn`.
            compute_fn: Transforming epilogue, used when `has_compute_fn`.
        """
        Self._run_impl(
            act_tma,
            filt_tma,
            out_wide_tma,
            out_narrow_tma,
            Int(out_w),
            Int(out_h),
            out_ptr,
            src_ptr,
            beta,
            epilogue_fn,
            compute_fn,
        )
        launch_dependent_grids()

    @staticmethod
    @inline(.always)
    def _run_impl[
        EpilogueFnType: ElementwiseEpilogueFn,
        ComputeFnType: ElementwiseComputeFn,
    ](
        act_tma: Self.ActTma,
        filt_tma: Self.FiltTma,
        out_wide_tma: Self.OutWideTma,
        out_narrow_tma: Self.OutNarrowTma,
        out_w: Int,
        out_h: Int,
        out_ptr: MutPointer[Scalar[Self.act_type], _],
        src_ptr: MutPointer[Scalar[Self.act_type], _],
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
        var stride = grid_dim.x

        if tid == 0:
            act_tma.prefetch_descriptor()
            filt_tma.prefetch_descriptor()
            out_wide_tma.prefetch_descriptor()
            out_narrow_tma.prefetch_descriptor()
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
                out_wide_tma,
                out_narrow_tma,
                sc,
                bars,
                tmem_slot,
                tid,
                cta,
                stride,
                out_ptr,
                src_ptr,
                beta,
                epilogue_fn,
                compute_fn,
                out_h * out_w,
                out_w,
            )
        elif warp == 4:
            Self._filter_load(filt_tma, sb, bars)
        elif warp == 5:
            Self._act_load(act_tma, sa, bars, cta, stride, out_w, out_h)
        else:
            Self._mma(sa, sb, bars, tmem_slot)

    # ----- filter load warp -----

    @staticmethod
    @inline(.always)
    def _filter_load(
        filt_tma: Self.FiltTma,
        sb: MutPointer[Scalar[Self.act_type], _, address_space=.SHARED],
        bars: MutPointer[SharedMemBarrier, _, address_space=.SHARED],
    ):
        """Loads the 27 filter boxes of every work unit in the order the MMA
        warp consumes them: channel block outer, filter tap inner."""
        var desc = Pointer(to=filt_tma.descriptor).unsafe_bitcast[NoneType]()
        # The predecessor may be the filter transpose that writes this
        # filter: the barrier setup, TMEM allocation and coordinate caching
        # of the other warps run ahead of this wait.
        wait_on_dependent_grids()
        var stage = 0
        var phase = UInt32(0)
        comptime for _ in range(Self.units):
            for step in range(Self.KT):
                var cb, tap = divmod(step, Self.TAPS)
                bars.unsafe_offset(Self.B_EMPTY + stage)[].wait(phase ^ 1)
                if elect_one_sync():
                    var full = bars.unsafe_offset(Self.B_FULL + stage)
                    full[].expect_bytes(Int32(Self.B_BYTES))
                    cp_async_bulk_tensor_shared_cluster_global[cta_group=1](
                        sb.unsafe_offset(stage * Self.B_ELEMS),
                        desc,
                        full,
                        ((tap * Self.CBLOCKS + cb) * Self.BK, 0),
                    )
                stage += 1
                if stage == Self.BSTAGES:
                    stage = 0
                    phase ^= 1

    # ----- activation load warp -----

    @staticmethod
    @inline(.always)
    def _act_load(
        act_tma: Self.ActTma,
        sa: MutPointer[Scalar[Self.act_type], _, address_space=.SHARED],
        bars: MutPointer[SharedMemBarrier, _, address_space=.SHARED],
        cta: Int,
        stride: Int,
        out_w: Int,
        out_h: Int,
    ):
        """Loads the activation tiles of every work unit in the order the MMA
        warp consumes them: 18 K steps with the three tiles interleaved, then
        three tiles of nine tail steps each."""
        var desc = Pointer(to=act_tma.descriptor).unsafe_bitcast[NoneType]()

        # The activation is the predecessor's output.
        wait_on_dependent_grids()
        var stage = 0
        var phase = UInt32(0)

        comptime for local in range(Self.units):
            # (n, h, w) of the first pixel of each of this unit's three
            # boxes, with the padding offset already applied.
            var box_n = Array[Int32, Self.NBOX](uninitialized=True)
            var box_h = Array[Int32, Self.NBOX](uninitialized=True)
            var box_w = Array[Int32, Self.NBOX](uninitialized=True)
            var unit = cta + local * stride
            comptime for q in range(Self.NBOX):
                var row, col = divmod(Self.PIX * (Self.NBOX * unit + q), out_w)
                var image, row_in_image = divmod(row, out_h)
                box_n[q] = Int32(image)
                box_h[q] = Int32(row_in_image - 1)
                box_w[q] = Int32(col - 1)

            for step in range(Self.MAIN_STEPS):
                var cb, tap = divmod(step, Self.TAPS)
                var r, s = divmod(tap, Self.S)
                comptime for mt in range(Self.MPASS):
                    Self._load_act_tile[mt](
                        desc,
                        sa,
                        bars,
                        stage,
                        phase,
                        box_n,
                        box_h,
                        box_w,
                        cb * Self.BK,
                        r,
                        s,
                    )
                    stage += 1
                    if stage == Self.ASTAGES:
                        stage = 0
                        phase ^= 1

            comptime for mt in range(Self.MPASS):
                comptime for tap in range(Self.TAPS):
                    comptime r, s = divmod(tap, Self.S)
                    Self._load_act_tile[mt](
                        desc,
                        sa,
                        bars,
                        stage,
                        phase,
                        box_n,
                        box_h,
                        box_w,
                        (Self.CBLOCKS - 1) * Self.BK,
                        r,
                        s,
                    )
                    stage += 1
                    if stage == Self.ASTAGES:
                        stage = 0
                        phase ^= 1

    @staticmethod
    @inline(.always)
    def _load_act_tile[
        mt: Int
    ](
        desc: OpaquePointer[mut=False, _],
        sa: MutPointer[Scalar[Self.act_type], _, address_space=.SHARED],
        bars: MutPointer[SharedMemBarrier, _, address_space=.SHARED],
        stage: Int,
        phase: UInt32,
        box_n: Array[Int32, Self.NBOX],
        box_h: Array[Int32, Self.NBOX],
        box_w: Array[Int32, Self.NBOX],
        c: Int,
        r: Int,
        s: Int,
    ):
        """One output tile's activation transactions into ring slot `stage`."""
        bars.unsafe_offset(Self.A_EMPTY + stage)[].wait(phase ^ 1)
        if elect_one_sync():
            var full = bars.unsafe_offset(Self.A_FULL + stage)
            full[].expect_bytes(Int32(Self.A_BYTES))
            var dst = sa.unsafe_offset(stage * Self.A_ELEMS)
            comptime for b in range(Self.BOXES):
                comptime q = Self.BOXES * mt + b
                cp_async_bulk_tensor_shared_cluster_global_im2col[cta_group=1](
                    dst.unsafe_offset(b * Self.PIX * Self.BK),
                    desc,
                    full,
                    (c, Int(box_w[q]), Int(box_h[q]), Int(box_n[q])),
                    (s, r),
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
        """Allocates TMEM, issues every UMMA of every work unit, and frees
        TMEM."""
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

        comptime for local in range(Self.units):
            # This unit's three accumulators: tile `t` of the CTA lives in
            # slot `t % ACC_SLOTS` once the epilogue has drained the tile
            # that last used it.
            var acc = Array[UInt32, Self.MPASS](uninitialized=True)
            var slot = Array[Int, Self.MPASS](uninitialized=True)
            comptime for mt in range(Self.MPASS):
                var lap, sl = divmod(local * Self.MPASS + mt, Self.ACC_SLOTS)
                slot[mt] = sl
                acc[mt] = tmem + UInt32(sl * Self.BN)
                bars.unsafe_offset(Self.ACC_EMPTY + sl)[].wait(
                    UInt32(lap & 1) ^ 1
                )
            tcgen05_fence_after()

            # Main steps: one filter box, three activation tiles.
            for step in range(Self.MAIN_STEPS):
                bars.unsafe_offset(Self.B_FULL + b_stage)[].wait(b_phase)
                comptime for mt in range(Self.MPASS):
                    bars.unsafe_offset(Self.A_FULL + a_stage)[].wait(a_phase)
                    if elect_one_sync():
                        mma_op.mma_from_desc(
                            a_desc + a_stage * Self.A_BYTES,
                            b_desc + b_stage * Self.B_BYTES,
                            acc[mt],
                            init_c=(step == 0),
                        )
                        mma_op.commit(
                            bars.unsafe_offset(Self.A_EMPTY + a_stage)
                        )
                    a_stage += 1
                    if a_stage == Self.ASTAGES:
                        a_stage = 0
                        a_phase ^= 1
                if elect_one_sync():
                    mma_op.commit(bars.unsafe_offset(Self.B_EMPTY + b_stage))
                b_stage += 1
                if b_stage == Self.BSTAGES:
                    b_stage = 0
                    b_phase ^= 1

            # Tail: the last nine filter boxes stay resident while each tile
            # takes them in turn, so a tile's accumulator completes early.
            # The main steps are a whole number of laps, so box `tap` is in
            # slot `tap` on the ring's next lap; the last tile releases each
            # box for the next unit's loads.
            comptime assert Self.MAIN_STEPS % Self.BSTAGES == 0
            comptime for mt in range(Self.MPASS):
                comptime for tap in range(Self.HELD):
                    # Tile 0 takes each held box as it lands; later tiles
                    # find them all resident.
                    comptime if mt == 0:
                        bars.unsafe_offset(Self.B_FULL + tap)[].wait(b_phase)
                    bars.unsafe_offset(Self.A_FULL + a_stage)[].wait(a_phase)
                    if elect_one_sync():
                        mma_op.mma_from_desc(
                            a_desc + a_stage * Self.A_BYTES,
                            b_desc + tap * Self.B_BYTES,
                            acc[mt],
                            init_c=False,
                        )
                        mma_op.commit(
                            bars.unsafe_offset(Self.A_EMPTY + a_stage)
                        )
                        comptime if mt == Self.MPASS - 1:
                            mma_op.commit(
                                bars.unsafe_offset(Self.B_EMPTY + tap)
                            )
                    a_stage += 1
                    if a_stage == Self.ASTAGES:
                        a_stage = 0
                        a_phase ^= 1
                if elect_one_sync():
                    mma_op.commit(bars.unsafe_offset(Self.ACC_FULL + slot[mt]))
            b_phase ^= 1

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
        out_wide_tma: Self.OutWideTma,
        out_narrow_tma: Self.OutNarrowTma,
        sc: MutPointer[Scalar[Self.act_type], _, address_space=.SHARED],
        bars: MutPointer[SharedMemBarrier, _, address_space=.SHARED],
        tmem_slot: MutPointer[UInt32, _, address_space=.SHARED],
        tid: Int,
        cta: Int,
        stride: Int,
        out_ptr: MutPointer[Scalar[Self.act_type], _],
        src_ptr: MutPointer[Scalar[Self.act_type], _],
        beta: Float32,
        epilogue_fn: EpilogueFnType,
        compute_fn: ComputeFnType,
        out_hw: Int,
        out_w: Int,
    ):
        """Moves each accumulator tile TMEM -> registers -> smem -> TMA store.

        The one staging slot is reused for the 64-column and the 32-column
        piece of every tile, so each store must finish reading it before the
        next piece is staged.

        With `has_epilogue_fn`, `has_compute_fn` or `has_residual` each thread
        instead owns one output pixel of the tile and applies the epilogue to
        its 96 channels in registers (`apply_conv_epilogue`).
        """
        # A closure or the residual can read what the predecessor grid
        # writes. These warps skip the load warps' wait, and the barriers
        # after that wait do not keep their read-only loads below it (see
        # `wait_on_dependent_grids`).
        # No PDL level gates the wait: it returns at once in a launch
        # without PDL.
        comptime if Self.has_epilogue_fn or Self.has_compute_fn or Self.has_residual:
            wait_on_dependent_grids()
        Self.MmaEpiSync.wait()
        var tmem = tmem_slot[]
        var wide_desc = Pointer(to=out_wide_tma.descriptor).unsafe_bitcast[
            NoneType
        ]()
        var narrow_desc = Pointer(to=out_narrow_tma.descriptor).unsafe_bitcast[
            NoneType
        ]()
        var warp = tid // WARP_SIZE

        comptime if Self.has_epilogue_fn or Self.has_compute_fn or Self.has_residual:
            comptime for local in range(Self.units):
                var unit = cta + local * stride
                comptime for mt in range(Self.MPASS):
                    var lap, slot = divmod(
                        local * Self.MPASS + mt, Self.ACC_SLOTS
                    )
                    bars.unsafe_offset(Self.ACC_FULL + slot)[].wait(
                        UInt32(lap & 1)
                    )
                    tcgen05_fence_after()
                    var pixel = (unit * Self.MPASS + mt) * Self.BM + tid
                    apply_conv_epilogue[
                        cols=Self.BN,
                        has_epilogue_fn=Self.has_epilogue_fn,
                        has_compute_fn=Self.has_compute_fn,
                        has_residual=Self.has_residual,
                        nhwc=Self.nhwc_coords,
                        round_first=Self.round_first,
                    ](
                        tmem + UInt32(slot * Self.BN),
                        pixel,
                        0,
                        Self.BN,
                        out_ptr,
                        src_ptr,
                        beta,
                        epilogue_fn,
                        compute_fn,
                        True,
                        out_hw,
                        out_w,
                    )
                    tcgen05_fence_before()
                    _ = bars.unsafe_offset(Self.ACC_EMPTY + slot)[].arrive()
            tcgen05_fence_before()
            _ = bars.unsafe_offset(Self.TMEM_FREE)[].arrive()
            return

        comptime for local in range(Self.units):
            var unit = cta + local * stride
            comptime for mt in range(Self.MPASS):
                var lap, slot = divmod(local * Self.MPASS + mt, Self.ACC_SLOTS)
                bars.unsafe_offset(Self.ACC_FULL + slot)[].wait(UInt32(lap & 1))
                tcgen05_fence_after()
                var row = (unit * Self.MPASS + mt) * Self.BM
                var acc = tmem + UInt32(slot * Self.BN)

                stage_accumulator[Self.act_type, 64, 32, Self.OutWideSwizzle](
                    acc, sc, warp
                )
                fence_async_view_proxy()
                Self.EpiSync.sync()
                if tid == 0:
                    cp_async_bulk_tensor_global_shared_cta(
                        sc, wide_desc, (0, row)
                    )
                    cp_async_bulk_commit_group()
                    cp_async_bulk_wait_group[0]()
                Self.EpiSync.sync()

                stage_accumulator[
                    Self.act_type, Self.BN - 64, 32, Self.OutNarrowSwizzle
                ](acc + UInt32(64), sc, warp)
                # Every column of the slot is in registers: hand it back to
                # the MMA warp for a later tile.
                tcgen05_fence_before()
                _ = bars.unsafe_offset(Self.ACC_EMPTY + slot)[].arrive()
                fence_async_view_proxy()
                Self.EpiSync.sync()
                if tid == 0:
                    cp_async_bulk_tensor_global_shared_cta(
                        sc, narrow_desc, (64, row)
                    )
                    cp_async_bulk_commit_group()
                    cp_async_bulk_wait_group[0]()
                Self.EpiSync.sync()

        tcgen05_fence_before()
        _ = bars.unsafe_offset(Self.TMEM_FREE)[].arrive()
        if tid == 0:
            cp_async_bulk_wait_group[0, read=False]()


def _launch_f96[
    EpilogueFnType: ElementwiseEpilogueFn,
    ComputeFnType: ElementwiseComputeFn,
    //,
    act_type: DType,
    cin: Int,
    units: Int,
    has_epilogue_fn: Bool,
    has_compute_fn: Bool,
    has_residual: Bool,
    nhwc_coords: Bool,
    round_first: Bool,
](
    output: TileTensor[mut=True, ...],
    activation: TileTensor[mut=True, act_type, address_space=.GENERIC, ...],
    filter: TileTensor[...],
    problem: Conv2dProblemShape,
    ctx: DeviceContext,
    source_ptr: OptionalReg[MutPointer[Scalar[act_type], _]],
    beta: Float32,
    epilogue_fn: EpilogueFnType,
    compute_fn: ComputeFnType,
) raises:
    comptime K = Conv2dFpropF96[
        act_type,
        cin,
        units,
        has_epilogue_fn,
        has_compute_fn,
        has_residual,
        nhwc_coords,
        round_first,
    ]
    var out_h = problem.out_height()
    var out_w = problem.out_width()
    var m_total = problem.gemm_m()

    var act_tma = create_act_im2col_tma[act_type, K.BM, K.BK, K.PIX, K.SWIZZLE](
        ctx, activation, problem
    )

    var filter_tensor = filter.reshape(
        row_major(problem.gemm_n(), problem.gemm_k())
    )
    var filt_tma = create_box_tma[K.BN, K.BK, K.SWIZZLE](ctx, filter_tensor)

    var out_tensor = output.reshape(row_major(m_total, problem.gemm_n()))
    var out_wide_tma = create_box_tma[K.BM, 64, K.OutWideSwizzle](
        ctx, out_tensor
    )
    var out_narrow_tma = create_box_tma[K.BM, K.BN - 64, K.OutNarrowSwizzle](
        ctx, out_tensor
    )

    var ctas = m_total // (K.BM * K.MPASS * units)

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

    comptime kernel = K.run[EpilogueFnType, ComputeFnType]
    ctx.enqueue_function[kernel](
        act_tma,
        filt_tma,
        out_wide_tma,
        out_narrow_tma,
        Int32(out_w),
        Int32(out_h),
        out_ptr,
        src_ptr,
        beta,
        host_args=(epilogue_fn, compute_fn),
        grid_dim=(ctas, 1, 1),
        block_dim=K.NUM_THREADS,
        shared_mem_bytes=K.SMEM_BYTES,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(K.SMEM_BYTES)
        ),
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )


def conv2d_f96_fprop[
    EpilogueFnType: ElementwiseEpilogueFn = type_of(no_epilogue_fn),
    ComputeFnType: ElementwiseComputeFn = type_of(no_compute_fn),
    //,
    act_type: DType,
    has_epilogue_fn: Bool = False,
    has_compute_fn: Bool = False,
    has_residual: Bool = False,
    nhwc_coords: Bool = False,
    round_first: Bool = False,
](
    output: TileTensor[mut=True, ...],  # NHWC
    activation: TileTensor[mut=True, act_type, address_space=.GENERIC, ...],
    filter: TileTensor[...],  # KRSC (out_ch, R, S, in_ch)
    problem: Conv2dProblemShape,
    ctx: DeviceContext,
    source_ptr: OptionalReg[MutPointer[Scalar[act_type], MutAnyOrigin]] = None,
    beta: Float32 = 0.0,
    epilogue_fn: EpilogueFnType = no_epilogue_fn,
    compute_fn: ComputeFnType = no_compute_fn,
) raises:
    """Launches the 96 output channel conv2d fprop kernel.

    Accepts only problems `conv2d_f96_supported` accepts; the caller checks.

    Parameters:
        EpilogueFnType: Type of `epilogue_fn` (inferred).
        ComputeFnType: Type of `compute_fn` (inferred).
        act_type: Element type of the activation, filter and output.
        has_epilogue_fn: `epilogue_fn` stores each output value; the kernel
            writes nothing to `output`.
        has_compute_fn: `compute_fn` transforms each output value, with
            coordinates in the `[M, N]` GEMM view of the output; excludes
            `has_epilogue_fn` and `has_residual`.
        has_residual: Store `Conv(A, B) + beta * source` (through
            `epilogue_fn` when `has_epilogue_fn`).
        nhwc_coords: The closures take NHWC coordinates `(n, h, w, c)`
            instead of `[M, N]` ones `(pixel, c)`.
        round_first: Run the epilogue arithmetic in the output dtype (what
            cuDNN followed by an elementwise kernel does) rather than the
            general kernel's float32 for fp16.

    Args:
        output: Output tensor `[N, H_out, W_out, 96]`, NHWC.
        activation: Input activation `[N, H, W, C]`, NHWC, `C` 96 or 192.
        filter: Filter `[96, 3, 3, C]`, KRSC (channel innermost).
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
    if not conv2d_f96_supported[act_type](problem, ctx):
        raise Error("conv2d_f96_fprop: unsupported problem shape")

    # A problem of two waves of work units runs as persistent CTAs that take
    # two units each, so the second unit's loads and the first unit's epilogue
    # overlap the neighbouring mainloop and the CTA is launched once.
    var m_total = problem.gemm_m()
    var total_units = m_total // (
        Conv2dFpropF96[act_type, 96, 1].BM
        * Conv2dFpropF96[act_type, 96, 1].MPASS
    )
    var persistent = (
        ceildiv(total_units, ctx.default_device_info.sm_count) == 2
        and total_units % 2 == 0
    )

    @always_inline
    @__allow_legacy_any_origin_fields
    def launch[cin: Int, units: Int]() raises {imm}:
        _launch_f96[
            act_type,
            cin,
            units,
            has_epilogue_fn,
            has_compute_fn,
            has_residual,
            nhwc_coords,
            round_first,
        ](
            output,
            activation,
            filter,
            problem,
            ctx,
            source_ptr,
            beta,
            epilogue_fn,
            compute_fn,
        )

    if problem.in_channels == 192:
        if persistent:
            launch[192, 2]()
        else:
            launch[192, 1]()
    elif persistent:
        launch[96, 2]()
    else:
        launch[96, 1]()
