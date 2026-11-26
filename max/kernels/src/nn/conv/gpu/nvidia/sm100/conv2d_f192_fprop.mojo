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
"""SM100 Conv2D fprop for the 192 -> 192 channel, 3x3, pad 1 shape family.

Target family: Blackwell sm_100a, 1-SM UMMA, fp16 and bf16 with fp32
accumulation. This is a shape-gated route beside the general structured
kernel in `conv2d_fprop_kernel.mojo`; `dispatch.mojo` selects it only when
`conv2d_f192_supported` accepts the problem.

Why a separate kernel. At `F = 192` the general kernel runs one 128-pixel
tile per CTA over two 128-wide filter tiles, so each CTA re-reads the 663 KB
filter and the tensor pipe idles behind the load warp. This kernel splits the
output channels into two halves of 96 (grid `y`) and gives each CTA the
`conv2d_f96_fprop` schedule for its half:

- N = 96 exactly per CTA (one UMMA N, no padded columns); CTA `(x, y)`
  computes output channels `96 * y .. 96 * y + 95`.
- One CTA computes three consecutive 128-pixel output tiles (`MPASS = 3`)
  and loads each filter box once for all three. Each tile owns a 96-column
  TMEM accumulator, 288 of the 512 columns.
- One im2col transaction loads a whole 128-pixel tile. The hardware walks the
  box through output rows and images, so `OW` need not divide the tile, and
  the (n, h, w) start of each of the CTA's three tiles is computed once, not
  per transaction. Narrower boxes (16 to 32 pixels) measure 11 to 45 % slower
  because the load warp issues more transactions per K step.
- K runs 18 steps with one filter box live for the three tiles, then a nine
  step tail that holds the last nine filter boxes while each tile finishes,
  so tile 0's accumulator is complete (and its epilogue starts) while tiles 1
  and 2 are still in the mainloop.
- The activation ring has six stages and the output staging ring one.
- The accumulator leaves TMEM as two pieces (64 + 32 columns), is staged
  into shared memory with `stmatrix`, and is written with two TMA stores.

Warp roles (7 warps, 224 threads):

| warps | role                                                   |
|-------|--------------------------------------------------------|
| 0-3   | epilogue: TMEM -> smem -> TMA store, TMEM release       |
| 4     | filter load: one TMA box per K step                     |
| 5     | activation load: im2col TMA, cached coordinates         |
| 6     | MMA issue, TMEM allocation                              |

The filter comes in either of the layouts a caller holds (`conv2d_f192_dispatch`):

- RSCF `[R, S, C, F]`, which is what the graph compiler hands over, is read in
  place. The output channel is contiguous, so B is MN-major: a K step is
  three `[64, 32]` TMA boxes with a 64-byte swizzle and the UMMA reads them as
  a 96-wide MN-major operand. The shared transpose to KRSC in `dispatch.mojo`
  (one element per thread through divisions and strided gathers) took 4.2 us on
  this 663 KB filter, a fifth of the convolution; reading RSCF directly
  removes that launch.
- FCRS `[F, C, R, S]` is packed to KRSC `[F, R, S, C]` (K contiguous) by a
  bandwidth-bound kernel of its own, and B is K-major with a 128-byte swizzle.
"""

from std.collections import Array, OptionalReg
from std.sys import size_of
from std.utils.static_tuple import StaticTuple

from layout import TensorEngine, TensorLayout, TileTensor, coord, row_major
from layout.tma_async import SharedMemBarrier, TMATensorTileIm2col
from linalg.arch.sm100 import MmaOpSM100_SS
from linalg.arch.sm100.mma import _create_mma_desc_k_major
from max.gpu.compute.arch.mma_nvidia_sm100 import (
    MMASmemDescriptor,
    UMMAInsDescriptor,
    UMMAKind,
    mma,
)
from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, WARP_SIZE, block_idx
from max.gpu import thread_idx
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
    PDL,
    PDLLevel,
    pdl_launch_attributes,
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
    dedicated_problem_shape,
    stage_accumulator,
)


def conv2d_f192_supported[
    act_type: DType
](problem: Conv2dProblemShape, ctx: DeviceContext) -> Bool:
    """Whether `conv2d_f192_fprop` can run `problem`.

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
        comptime K = Conv2dFpropF192[act_type]
        var out_w = problem.out_width()
        var m_total = problem.batch * problem.out_height() * out_w
        return (
            problem.in_channels == K.CIN
            and problem.out_channels == K.FOUT
            and problem.filter_h == K.R
            and problem.filter_w == K.S
            and problem.pad_h == 1
            and problem.pad_w == 1
            and problem.stride_h == 1
            and problem.stride_w == 1
            and problem.dilation_h == 1
            and problem.dilation_w == 1
            and problem.groups == 1
            and m_total % (K.BM * K.MPASS) == 0
        )


struct Conv2dFpropF192[
    act_type: DType,
    rscf_filter: Bool = False,
    has_epilogue_fn: Bool = False,
    has_compute_fn: Bool = False,
    has_residual: Bool = False,
    nhwc_coords: Bool = False,
    round_first: Bool = False,
]:
    """The 192 -> 192 channel conv2d fprop kernel and its launch helpers.

    Parameters:
        act_type: Element type of the activation, filter and output
            (`DType.float16` or `DType.bfloat16`).
        rscf_filter: Read the filter in its RSCF `[R, S, C, F]` layout, where
            the output channel is contiguous, instead of KRSC. The B operand
            is then MN-major in shared memory: each K step lands as three
            `[64, 32]` boxes with a 64-byte swizzle.
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
    comptime CIN = 192
    comptime FOUT = 192
    comptime R = 3
    comptime S = 3
    comptime BM = 128
    comptime BN = 96
    comptime EB = size_of[Self.act_type]()
    comptime SWIZZLE = TensorMapSwizzle.SWIZZLE_128B
    comptime BK = Self.SWIZZLE.bytes() // Self.EB
    """Channels per K tile: one 128-byte swizzle row, so a TMA im2col
    transaction never straddles a filter tap."""
    comptime NB_SPAN = 32
    """Output channels per RSCF filter box: one 64-byte swizzle row."""
    comptime NHALVES = Self.FOUT // Self.BN
    """Output-channel halves; grid `y` selects one."""
    comptime MMA_K = 16
    comptime MPASS = 3
    """Output tiles one CTA computes; one filter box serves all of them."""
    comptime PIX = 128
    """Output pixels one im2col transaction loads."""
    comptime BOXES = Self.BM // Self.PIX
    """Im2col transactions per output tile."""
    comptime NBOX = Self.BOXES * Self.MPASS
    """Im2col transactions per CTA, whose (n, h, w) starts are cached."""
    comptime KT = Self.CIN * Self.R * Self.S // Self.BK
    """K tiles of one output tile (27)."""
    comptime CBLOCKS = Self.CIN // Self.BK
    """Channel blocks per filter tap (3)."""
    comptime TAPS = Self.R * Self.S
    comptime HELD = Self.TAPS
    """Filter boxes held through the tail while each tile finishes."""
    comptime MAIN_STEPS = Self.KT - Self.HELD
    """K steps walked with the three tiles interleaved (18)."""

    # ========== Rings and shared memory ==========
    comptime ASTAGES = 6
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
    comptime ACC_FULL = Self.B_EMPTY + Self.BSTAGES
    comptime TMEM_FREE = Self.ACC_FULL + Self.MPASS
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
    comptime FILT_BOX_ROWS = Self.BK if Self.rscf_filter else Self.BN
    comptime FILT_BOX_COLS = Self.NB_SPAN if Self.rscf_filter else Self.BK
    comptime FiltTma = TmaOpType[
        Self.act_type,
        static_row_major[Self.FILT_BOX_ROWS, Self.FILT_BOX_COLS],
        static_row_major[Self.FILT_BOX_ROWS, Self.FILT_BOX_COLS],
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
    @__name(t"sm100_conv2d_f192_fprop_{Self.act_type}")
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
                box `[BM, 64]`, 128-byte swizzle (the first 64 columns of
                the CTA's half).
            out_narrow_tma: The same view with box `[BM, 32]`, 64-byte
                swizzle (the last 32 columns of the CTA's half).
            out_w: Output width.
            out_h: Output height.
            out_ptr: Output, `[M, F]`; written by the kernel unless
                `has_epilogue_fn`, and only on the register epilogue path.
            src_ptr: Residual source `[M, F]`, read when `has_residual`.
            beta: Residual scale factor.
            epilogue_fn: Storing epilogue, used when `has_epilogue_fn`.
            compute_fn: Transforming epilogue, used when `has_compute_fn`.

        Grid `x` selects a group of `MPASS` output tiles and grid `y` the
        half of the output channels.
        """
        with PDL():
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

    @staticmethod
    @always_inline
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

        if tid == 0:
            act_tma.prefetch_descriptor()
            filt_tma.prefetch_descriptor()
            out_wide_tma.prefetch_descriptor()
            out_narrow_tma.prefetch_descriptor()
            comptime for i in range(Self.A_FULL, Self.TMEM_FREE):
                bars.unsafe_offset(i)[].init(1)
            # Every epilogue thread arrives once when it is done with TMEM.
            bars.unsafe_offset(Self.TMEM_FREE)[].init(Int32(Self.EPI_THREADS))
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
                block_idx.x,
                block_idx.y * Self.BN,
                out_ptr,
                src_ptr,
                beta,
                epilogue_fn,
                compute_fn,
                out_h * out_w,
                out_w,
            )
        elif warp == 4:
            Self._filter_load(filt_tma, sb, bars, block_idx.y * Self.BN)
        elif warp == 5:
            Self._act_load(act_tma, sa, bars, block_idx.x, out_w, out_h)
        else:
            Self._mma(sa, sb, bars, tmem_slot)

    # ----- filter load warp -----

    @staticmethod
    @always_inline
    def _filter_load(
        filt_tma: Self.FiltTma,
        sb: MutPointer[Scalar[Self.act_type], _, address_space=.SHARED],
        bars: MutPointer[SharedMemBarrier, _, address_space=.SHARED],
        col0: Int,
    ):
        """Loads the 27 filter boxes of output channels `col0 .. col0 + BN`
        in the order the MMA warp consumes them: channel block outer, filter
        tap inner."""
        var desc = Pointer(to=filt_tma.descriptor).unsafe_bitcast[NoneType]()
        var stage = 0
        var phase = UInt32(0)
        for step in range(Self.KT):
            var cb, tap = divmod(step, Self.TAPS)
            bars.unsafe_offset(Self.B_EMPTY + stage)[].wait(phase ^ 1)
            if elect_one_sync():
                var full = bars.unsafe_offset(Self.B_FULL + stage)
                full[].expect_bytes(Int32(Self.B_BYTES))
                comptime if Self.rscf_filter:
                    comptime for j in range(Self.BN // Self.NB_SPAN):
                        cp_async_bulk_tensor_shared_cluster_global[cta_group=1](
                            sb.unsafe_offset(
                                stage * Self.B_ELEMS
                                + j * Self.BK * Self.NB_SPAN
                            ),
                            desc,
                            full,
                            (
                                col0 + j * Self.NB_SPAN,
                                tap * Self.CIN + cb * Self.BK,
                            ),
                        )
                else:
                    cp_async_bulk_tensor_shared_cluster_global[cta_group=1](
                        sb.unsafe_offset(stage * Self.B_ELEMS),
                        desc,
                        full,
                        ((tap * Self.CBLOCKS + cb) * Self.BK, col0),
                    )
            stage += 1
            if stage == Self.BSTAGES:
                stage = 0
                phase ^= 1

    # ----- activation load warp -----

    @staticmethod
    @always_inline
    def _act_load(
        act_tma: Self.ActTma,
        sa: MutPointer[Scalar[Self.act_type], _, address_space=.SHARED],
        bars: MutPointer[SharedMemBarrier, _, address_space=.SHARED],
        cta: Int,
        out_w: Int,
        out_h: Int,
    ):
        """Loads the activation tiles in the order the MMA warp consumes them:
        18 K steps with the three tiles interleaved, then three tiles of nine
        tail steps each."""
        var desc = Pointer(to=act_tma.descriptor).unsafe_bitcast[NoneType]()

        # (n, h, w) of the first pixel of each of this CTA's boxes,
        # with the padding offset already applied.
        var box_n = Array[Int32, Self.NBOX](uninitialized=True)
        var box_h = Array[Int32, Self.NBOX](uninitialized=True)
        var box_w = Array[Int32, Self.NBOX](uninitialized=True)
        comptime for q in range(Self.NBOX):
            var pixel = Self.PIX * (Self.NBOX * cta + q)
            var row, col = divmod(pixel, out_w)
            var image, y = divmod(row, out_h)
            box_n[q] = Int32(image)
            box_h[q] = Int32(y - 1)
            box_w[q] = Int32(col - 1)

        var stage = 0
        var phase = UInt32(0)

        for step in range(Self.MAIN_STEPS):
            var cb, tap = divmod(step, Self.TAPS)
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
                    tap // Self.S,
                    tap % Self.S,
                )
                stage += 1
                if stage == Self.ASTAGES:
                    stage = 0
                    phase ^= 1

        comptime for mt in range(Self.MPASS):
            comptime for tap in range(Self.TAPS):
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
                    tap // Self.S,
                    tap % Self.S,
                )
                stage += 1
                if stage == Self.ASTAGES:
                    stage = 0
                    phase ^= 1

    @staticmethod
    @always_inline
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
    @always_inline
    def _issue(
        mma_op: Self.MmaOp,
        a_desc: MMASmemDescriptor,
        b_desc: MMASmemDescriptor,
        c_tmem: UInt32,
        init_c: Bool,
    ):
        """Issues the `BK / MMA_K` UMMAs of one K step."""
        comptime if Self.rscf_filter:
            comptime idesc = UMMAInsDescriptor[UMMAKind.KIND_F16].create[
                .float32,
                Self.act_type,
                Self.act_type,
                (UInt32(Self.BM), UInt32(Self.BN)),
                transpose_b=False,
            ]()
            comptime for k in range(Self.BK // Self.MMA_K):
                var c_scale: UInt32 = 0 if (init_c and k == 0) else 1
                mma[1](
                    a_desc + k * Self.MMA_K * Self.EB,
                    b_desc + k * Self.MMA_K * Self.NB_SPAN * Self.EB,
                    c_tmem,
                    idesc,
                    c_scale=c_scale,
                )
        else:
            mma_op.mma_from_desc(a_desc, b_desc, c_tmem, init_c)

    @staticmethod
    @always_inline
    def _mma(
        sa: MutPointer[Scalar[Self.act_type], _, address_space=.SHARED],
        sb: MutPointer[Scalar[Self.act_type], _, address_space=.SHARED],
        bars: MutPointer[SharedMemBarrier, _, address_space=.SHARED],
        tmem_slot: MutPointer[UInt32, _, address_space=.SHARED],
    ):
        """Allocates TMEM, issues every UMMA, and frees TMEM."""
        tcgen05_alloc[1](tmem_slot, UInt32(Self.TMEM_COLS))
        syncwarp()
        var tmem = tmem_slot[]
        Self.MmaEpiSync.arrive()

        var mma_op = Self.MmaOp()
        var a_desc = _create_mma_desc_k_major[Self.act_type, Self.SWIZZLE](sa)
        var b_desc: MMASmemDescriptor
        comptime if Self.rscf_filter:
            # MN-major B: K rows of 64 bytes, eight to a swizzle atom
            # (SBO 512), the three 32-column boxes 4096 bytes apart (LBO).
            b_desc = MMASmemDescriptor.create[
                8 * Self.NB_SPAN * Self.EB,
                Self.BK * Self.NB_SPAN * Self.EB,
                TensorMapSwizzle.SWIZZLE_64B,
            ](sb)
        else:
            b_desc = _create_mma_desc_k_major[Self.act_type, Self.SWIZZLE](sb)

        var a_stage = 0
        var a_phase = UInt32(0)

        # Main steps: one filter box, three activation tiles.
        var b_stage = 0
        var b_phase = UInt32(0)
        for step in range(Self.MAIN_STEPS):
            bars.unsafe_offset(Self.B_FULL + b_stage)[].wait(b_phase)
            comptime for mt in range(Self.MPASS):
                bars.unsafe_offset(Self.A_FULL + a_stage)[].wait(a_phase)
                if elect_one_sync():
                    Self._issue(
                        mma_op,
                        a_desc + a_stage * Self.A_BYTES,
                        b_desc + b_stage * Self.B_BYTES,
                        tmem + UInt32(mt * Self.BN),
                        step == 0,
                    )
                    mma_op.commit(bars.unsafe_offset(Self.A_EMPTY + a_stage))
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
        # takes them in turn, so a tile's accumulator completes early. The
        # main steps are a whole number of laps, so box `tap` is in slot
        # `tap` on the ring's next lap.
        comptime assert Self.MAIN_STEPS % Self.BSTAGES == 0
        comptime for tap in range(Self.HELD):
            bars.unsafe_offset(Self.B_FULL + tap)[].wait(b_phase)
        comptime for mt in range(Self.MPASS):
            comptime for tap in range(Self.HELD):
                bars.unsafe_offset(Self.A_FULL + a_stage)[].wait(a_phase)
                if elect_one_sync():
                    Self._issue(
                        mma_op,
                        a_desc + a_stage * Self.A_BYTES,
                        b_desc + tap * Self.B_BYTES,
                        tmem + UInt32(mt * Self.BN),
                        False,
                    )
                    mma_op.commit(bars.unsafe_offset(Self.A_EMPTY + a_stage))
                a_stage += 1
                if a_stage == Self.ASTAGES:
                    a_stage = 0
                    a_phase ^= 1
            if elect_one_sync():
                mma_op.commit(bars.unsafe_offset(Self.ACC_FULL + mt))

        tcgen05_release_allocation_lock[1]()
        bars.unsafe_offset(Self.TMEM_FREE)[].wait(0)
        tcgen05_fence_after()
        tcgen05_dealloc[1](tmem, UInt32(Self.TMEM_COLS))

    # ----- epilogue warps -----

    @staticmethod
    @always_inline
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
        col0: Int,
        out_ptr: MutPointer[Scalar[Self.act_type], _],
        src_ptr: MutPointer[Scalar[Self.act_type], _],
        beta: Float32,
        epilogue_fn: EpilogueFnType,
        compute_fn: ComputeFnType,
        out_hw: Int,
        out_w: Int,
    ):
        """Moves each accumulator tile TMEM -> registers -> smem -> TMA store
        into output columns `col0 .. col0 + BN`.

        The one staging slot is reused for the 64-column and the 32-column
        piece of every tile, so each store must finish reading it before the
        next piece is staged.

        With `has_epilogue_fn`, `has_compute_fn` or `has_residual` each thread
        instead owns one output pixel of the tile and applies the epilogue to
        its 96 channels in registers (`apply_conv_epilogue`).
        """
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
            comptime for mt in range(Self.MPASS):
                bars.unsafe_offset(Self.ACC_FULL + mt)[].wait(0)
                tcgen05_fence_after()
                var pixel = (cta * Self.MPASS + mt) * Self.BM + tid
                var acc = tmem + UInt32(mt * Self.BN)
                apply_conv_epilogue[
                    cols=Self.BN,
                    has_epilogue_fn=Self.has_epilogue_fn,
                    has_compute_fn=Self.has_compute_fn,
                    has_residual=Self.has_residual,
                    nhwc=Self.nhwc_coords,
                    round_first=Self.round_first,
                ](
                    acc,
                    pixel,
                    col0,
                    Self.FOUT,
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
            _ = bars.unsafe_offset(Self.TMEM_FREE)[].arrive()
            return

        comptime for mt in range(Self.MPASS):
            bars.unsafe_offset(Self.ACC_FULL + mt)[].wait(0)
            tcgen05_fence_after()
            var row = (cta * Self.MPASS + mt) * Self.BM
            var acc = tmem + UInt32(mt * Self.BN)

            stage_accumulator[Self.act_type, 64, 32, Self.OutWideSwizzle](
                acc, sc, warp
            )
            fence_async_view_proxy()
            Self.EpiSync.sync()
            if tid == 0:
                cp_async_bulk_tensor_global_shared_cta(
                    sc, wide_desc, (col0, row)
                )
                cp_async_bulk_commit_group()
                cp_async_bulk_wait_group[0]()
            Self.EpiSync.sync()

            stage_accumulator[
                Self.act_type, Self.BN - 64, 32, Self.OutNarrowSwizzle
            ](acc + UInt32(64), sc, warp)
            fence_async_view_proxy()
            Self.EpiSync.sync()
            if tid == 0:
                cp_async_bulk_tensor_global_shared_cta(
                    sc, narrow_desc, (col0 + 64, row)
                )
                cp_async_bulk_commit_group()
                cp_async_bulk_wait_group[0]()
            Self.EpiSync.sync()

        tcgen05_fence_before()
        _ = bars.unsafe_offset(Self.TMEM_FREE)[].arrive()
        if tid == 0:
            cp_async_bulk_wait_group[0, read=False]()


def conv2d_f192_fprop[
    EpilogueFnType: ElementwiseEpilogueFn = type_of(no_epilogue_fn),
    ComputeFnType: ElementwiseComputeFn = type_of(no_compute_fn),
    //,
    act_type: DType,
    rscf_filter: Bool = False,
    has_epilogue_fn: Bool = False,
    has_compute_fn: Bool = False,
    has_residual: Bool = False,
    nhwc_coords: Bool = False,
    round_first: Bool = False,
](
    output: TileTensor[mut=True, ...],  # NHWC
    activation: TileTensor[mut=True, act_type, address_space=.GENERIC, ...],
    filter: TileTensor[...],  # KRSC (out_ch, R, S, in_ch), or RSCF
    problem: Conv2dProblemShape,
    ctx: DeviceContext,
    source_ptr: OptionalReg[MutPointer[Scalar[act_type], MutAnyOrigin]] = None,
    beta: Float32 = 0.0,
    epilogue_fn: EpilogueFnType = no_epilogue_fn,
    compute_fn: ComputeFnType = no_compute_fn,
) raises:
    """Launches the 192 -> 192 channel conv2d fprop kernel.

    Accepts only problems `conv2d_f192_supported` accepts; the caller checks.

    Parameters:
        EpilogueFnType: Type of `epilogue_fn` (inferred).
        ComputeFnType: Type of `compute_fn` (inferred).
        act_type: Element type of the activation, filter and output.
        rscf_filter: Whether `filter` is RSCF rather than KRSC.
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
        output: Output tensor `[N, H_out, W_out, 192]`, NHWC.
        activation: Input activation `[N, H, W, 192]`, NHWC.
        filter: Filter `[192, 3, 3, 192]`, KRSC (channel innermost), or
            `[3, 3, 192, 192]` RSCF (output channel innermost).
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
    if not conv2d_f192_supported[act_type](problem, ctx):
        raise Error("conv2d_f192_fprop: unsupported problem shape")

    comptime K = Conv2dFpropF192[
        act_type,
        rscf_filter,
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

    # RSCF is the filter viewed [K, F]; KRSC is [F, K].
    var filter_tensor = filter.reshape(
        row_major(
            problem.gemm_k() if rscf_filter else problem.gemm_n(),
            problem.gemm_n() if rscf_filter else problem.gemm_k(),
        )
    )
    var filt_tma = create_box_tma[
        K.FILT_BOX_ROWS,
        K.FILT_BOX_COLS,
        TensorMapSwizzle.SWIZZLE_64B if rscf_filter else K.SWIZZLE,
    ](ctx, filter_tensor)

    var out_tensor = output.reshape(row_major(m_total, problem.gemm_n()))
    var out_wide_tma = create_box_tma[K.BM, 64, K.OutWideSwizzle](
        ctx, out_tensor
    )
    var out_narrow_tma = create_box_tma[K.BM, K.BN - 64, K.OutNarrowSwizzle](
        ctx, out_tensor
    )

    # `output.dtype == act_type` is asserted above.
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
        grid_dim=(m_total // (K.BM * K.MPASS), K.NHALVES, 1),
        block_dim=K.NUM_THREADS,
        shared_mem_bytes=K.SMEM_BYTES,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(K.SMEM_BYTES)
        ),
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )


# ----- filter pack: FCRS or RSCF -> KRSC -----

comptime _PACK_V = 8
"""Elements one thread moves with a single 16-byte access."""


@__name(t"pack_fcrs_to_krsc_f192_{dtype}")
def _pack_fcrs[
    dtype: DType,
    SrcLayoutType: TensorLayout,
    src_origin: ImmOrigin,
    SrcEngine: TensorEngine,
    DstLayoutType: TensorLayout,
    dst_origin: MutOrigin,
    DstEngine: TensorEngine,
](
    src: TileTensor[dtype, SrcLayoutType, src_origin, Engine=SrcEngine],
    dst: TileTensor[dtype, DstLayoutType, dst_origin, Engine=DstEngine],
):
    """One block per output channel `f`: thread `t` gathers eight channels of
    one tap from the row `src[f]` (contiguous, so the block's reads stay in
    L1) and stores them as one 16-byte vector."""
    comptime C = Conv2dFpropF192[dtype].CIN
    comptime TAPS = Conv2dFpropF192[dtype].TAPS
    comptime K = C * TAPS
    var f = block_idx.x
    var j = thread_idx.x * _PACK_V
    var tap, c = divmod(j, C)
    with PDL():
        var from_offset = f * K + c * TAPS + tap
        var v = SIMD[dtype, _PACK_V]()
        comptime for i in range(_PACK_V):
            v[i] = src.raw_load(from_offset + i * TAPS)
        dst.raw_store(f * K + j, v)


def conv2d_f192_dispatch[
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
    nhwc_coords: Bool = False,
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
    """Runs the convolution on the 192 -> 192 kernel if it serves the problem.

    The filter is in the caller's layout, FCRS `[F, C, R, S]` or RSCF
    `[R, S, C, F]`. An RSCF filter is read in place as the MN-major B operand;
    an FCRS one is packed to KRSC into a scratch buffer first.

    Parameters:
        input_type: Element type of the activation.
        filter_type: Element type of the filter.
        output_type: Element type of the output.
        EpilogueFnType: Type of `epilogue_fn` (inferred).
        ComputeFnType: Type of `compute_fn` (inferred).
        filter_is_fcrs: Whether `filter` is FCRS rather than RSCF.
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
        epilogue_fn: Storing elementwise epilogue, coordinates in the
            `[M, N]` GEMM view of the output.
        compute_fn: Transforming elementwise epilogue, same coordinates.

    Returns:
        True if the kernel ran; False, with nothing enqueued, if the problem
        or the operand layouts are outside what it serves.

    Raises:
        If descriptor creation or a launch fails.
    """
    comptime assert input.flat_rank == 4, "input must be rank 4 (NHWC)"
    comptime assert filter.flat_rank == 4, "filter must be rank 4"
    comptime assert output.flat_rank == 4, "output must be rank 4 (NHWC)"
    comptime if (
        input_type != output_type
        or input_type != filter_type
        or not (
            input.is_row_major and filter.is_row_major and output.is_row_major
        )
    ):
        return False
    elif input_type != .float16 and input_type != .bfloat16:
        return False
    else:
        comptime K = Conv2dFpropF192[input_type]
        var problem = dedicated_problem_shape[filter_is_fcrs](
            input, filter, output, pad_h, pad_w
        )
        if not conv2d_f192_supported[input_type](problem, ctx):
            return False

        var out_tt = output.reshape(
            row_major(
                problem.batch,
                problem.in_height,
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
        var source = rebind[
            OptionalReg[MutPointer[Scalar[input_type], MutAnyOrigin]]
        ](source_ptr)
        comptime if filter_is_fcrs:
            var krsc_buf = ctx.enqueue_create_buffer[input_type](
                K.FOUT * K.TAPS * K.CIN
            )
            var krsc = TileTensor(
                krsc_buf, row_major[K.FOUT, K.R, K.S, K.CIN]()
            )
            comptime pack = _pack_fcrs[
                filter_type,
                filter.LayoutType,
                ImmOrigin(filter.origin),
                filter.Engine,
                krsc.LayoutType,
                krsc.origin,
                krsc.Engine,
            ]
            ctx.enqueue_function[pack](
                filter.as_imm(),
                krsc,
                grid_dim=K.FOUT,
                block_dim=K.TAPS * K.CIN // _PACK_V,
                attributes=pdl_launch_attributes(PDLLevel.ON),
            )
            conv2d_f192_fprop[
                input_type,
                has_epilogue_fn=has_epilogue_fn,
                has_compute_fn=has_compute_fn,
                has_residual=has_residual,
                nhwc_coords=nhwc_coords,
                round_first=round_first,
            ](
                out_tt,
                act_tt,
                krsc,
                problem,
                ctx,
                source,
                beta,
                epilogue_fn,
                compute_fn,
            )
        else:
            conv2d_f192_fprop[
                input_type,
                rscf_filter=True,
                has_epilogue_fn=has_epilogue_fn,
                has_compute_fn=has_compute_fn,
                has_residual=has_residual,
                nhwc_coords=nhwc_coords,
                round_first=round_first,
            ](
                out_tt,
                act_tt,
                filter.reshape(row_major[K.R, K.S, K.CIN, K.FOUT]()),
                problem,
                ctx,
                source,
                beta,
                epilogue_fn,
                compute_fn,
            )
        return True
