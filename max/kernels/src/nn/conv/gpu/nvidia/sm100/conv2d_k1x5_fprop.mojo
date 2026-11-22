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
"""SM100 Conv2D fprop for a 1x5 filter over 512 input channels (Conv1D).

Target family: Blackwell sm_100a, 1-SM UMMA, fp16 and bf16 with fp32
accumulation. This is a shape-gated route beside the general structured
kernel in `conv2d_fprop_kernel.mojo`; `dispatch.mojo` selects it only when
`conv2d_k1x5_supported` accepts the problem.

Why a separate kernel. A 1-D convolution run as a rank-4 one (1x550, 512 -> 1024
channels, 1x5 filter) is a GEMM with M = 546 output pixels, N = 1024 and
K = 2560. The general kernel tiles M by 128, so it launches 5 x 8 = 40 CTAs on
a 148-SM part and its 128 KB of shared memory pins one CTA to an SM. This
kernel tiles M by 64 and N by 64: 9 x 16 = 144 CTAs, one per SM. Each CTA owns
one output tile and walks the 40 K steps (5 taps x 8 channel blocks of 64)
through a three-slot ring of four K tiles each, with one 64-pixel im2col box
per K tile. The load warp's issue rate and the per-slot handshake bound the
loop: 13.2, 11.1, 10.2 and 8.7 us with 8, 16, 32 and 64-pixel boxes (one tile
per slot), then 7.1 us with four tiles per slot; ring depth alone does not
move it.

Warp roles (6 warps, 192 threads):

| warps | role                                                   |
|-------|--------------------------------------------------------|
| 0-3   | epilogue: TMEM -> smem -> TMA store, TMEM release       |
| 4     | load: one filter box and one im2col box per K step      |
| 5     | MMA issue, TMEM allocation                              |

The M = 64 accumulator occupies the first 16 lanes of each warp's TMEM
quadrant: warp `w` holds rows `16 w .. 16 w + 15`.

The filter comes in either layout a caller holds (`conv2d_k1x5_dispatch`):

- RSCF `[R, S, C, F]`, which is what the graph compiler hands over, is read in
  place. The output channel is contiguous, so B is MN-major: a K step is one
  `[64, 64]` TMA box with a 128-byte swizzle and the UMMA reads it as a 64-wide
  MN-major operand. The shared transpose in `dispatch.mojo` divides 64-bit
  indices per element and takes 22 us on this 5 MB filter, longer than the
  convolution itself; a pack kernel of our own cost 2.8 us per call.
- FCRS `[F, C, R, S]` is packed to KRSC `[F, R, S, C]` (K contiguous) by a
  bandwidth-bound kernel of its own, and B is K-major with a 128-byte swizzle.
"""

from std.collections import Array, OptionalReg
from std.math import ceildiv
from std.sys import size_of
from std.utils.static_tuple import StaticTuple

from layout import (
    TensorEngine,
    TensorLayout,
    TileTensor,
    coord,
    row_major,
    stack_allocation,
)
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
    dedicated_problem_shape,
    stage_accumulator,
)


def conv2d_k1x5_supported[
    act_type: DType
](problem: Conv2dProblemShape, ctx: DeviceContext) -> Bool:
    """Whether `conv2d_k1x5_fprop` can run `problem`.

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
        comptime K = Conv2dFpropK1x5[act_type]
        return (
            problem.in_channels == K.CIN
            and problem.out_channels == K.FOUT
            and problem.filter_h == 1
            and problem.filter_w == K.S
            and problem.pad_h == 0
            and problem.pad_w == 0
            and problem.stride_h == 1
            and problem.stride_w == 1
            and problem.dilation_h == 1
            and problem.dilation_w == 1
            and problem.groups == 1
            and problem.gemm_m() <= K.MAX_M
        )


struct Conv2dFpropK1x5[
    act_type: DType,
    rscf_filter: Bool = False,
    has_epilogue_fn: Bool = False,
    has_compute_fn: Bool = False,
    has_residual: Bool = False,
    nhwc_coords: Bool = False,
    round_first: Bool = False,
]:
    """The 1x5 filter, 512 -> 1024 channel conv2d fprop kernel.

    Parameters:
        act_type: Element type of the activation, filter and output
            (`DType.float16` or `DType.bfloat16`).
        rscf_filter: Read the filter in its RSCF `[R, S, C, F]` layout, where
            the output channel is contiguous, instead of KRSC. The B operand
            is then MN-major in shared memory: each K step lands as one
            `[64, 64]` box with a 128-byte swizzle.
        has_epilogue_fn: An elementwise epilogue stores each output value.
        has_compute_fn: An elementwise epilogue transforms each output value.
        has_residual: Add `beta * source` to each output value first.
        nhwc_coords: The closures take NHWC coordinates, not `[M, N]` ones.
        round_first: Run the epilogue arithmetic in the output dtype.

    With any of the three flags the epilogue warps apply
    `conv_epilogue.apply_conv_epilogue` in registers instead of staging the
    tile for a TMA store. The M = 64 accumulator leaves lanes 16 to 31 of
    every warp empty, so those lanes take part in the load and nothing else.
    """

    # ========== Problem family ==========
    comptime CIN = 512
    comptime FOUT = 1024
    comptime R = 1
    comptime S = 5
    comptime MAX_M = 1024
    """Largest GEMM M the one-tile-per-CTA grid is shaped for: at 64 x 64
    tiles it stays within one resident wave of two CTAs per SM."""
    comptime BM = 64
    comptime BN = 64
    comptime EB = size_of[Self.act_type]()
    comptime SWIZZLE = TensorMapSwizzle.SWIZZLE_128B
    comptime BK = Self.SWIZZLE.bytes() // Self.EB
    """Channels per K tile: one swizzle row, so an im2col transaction never
    straddles a filter tap."""
    comptime MMA_K = 16
    comptime PIX = 64
    """Output pixels one im2col transaction loads."""
    comptime BOXES = Self.BM // Self.PIX
    comptime CBLOCKS = Self.CIN // Self.BK
    comptime KT = Self.S * Self.CBLOCKS
    """K tiles of one output tile (40)."""

    # ========== Ring and shared memory ==========
    comptime KG = 4
    """K tiles per ring slot. Each slot costs a fixed full/empty handshake of
    about 100 ns however little it carries, so fewer, larger slots win: the
    kernel measures 8.7, 7.4 and 7.1 us at 1, 2 and 4 tiles per slot."""
    comptime KS = Self.KT // Self.KG
    """Ring slots one output tile walks."""
    comptime STAGES = 3
    comptime A_ELEMS = Self.BM * Self.BK
    comptime B_ELEMS = Self.BN * Self.BK
    comptime C_ELEMS = Self.BM * Self.BN
    comptime A_BYTES = Self.A_ELEMS * Self.EB
    comptime B_BYTES = Self.B_ELEMS * Self.EB
    comptime C_BYTES = Self.C_ELEMS * Self.EB
    comptime A_STAGE_ELEMS = Self.KG * Self.A_ELEMS
    comptime B_STAGE_ELEMS = Self.KG * Self.B_ELEMS
    comptime A_STAGE_BYTES = Self.KG * Self.A_BYTES
    comptime B_STAGE_BYTES = Self.KG * Self.B_BYTES
    comptime A_OFF = 0
    comptime B_OFF = Self.A_OFF + Self.STAGES * Self.A_STAGE_BYTES
    comptime C_OFF = Self.B_OFF + Self.STAGES * Self.B_STAGE_BYTES
    comptime BAR_OFF = Self.C_OFF + Self.C_BYTES

    comptime FULL = 0
    comptime EMPTY = Self.FULL + Self.STAGES
    comptime ACC_FULL = Self.EMPTY + Self.STAGES
    comptime TMEM_FREE = Self.ACC_FULL + 1
    comptime NBAR = Self.TMEM_FREE + 1
    comptime TMEM_ADDR_OFF = Self.BAR_OFF + Self.NBAR * 8
    comptime SMEM_BYTES = Self.TMEM_ADDR_OFF + 16

    # ========== Threads ==========
    comptime EPI_THREADS = 128
    comptime NUM_THREADS = Self.EPI_THREADS + 2 * WARP_SIZE
    comptime TMEM_COLS = 64

    comptime MmaEpiSync = WarpGroupBarrier[Self.EPI_THREADS + WARP_SIZE, 1]
    comptime EpiSync = WarpGroupBarrier[Self.EPI_THREADS, 2]

    # ========== TMA descriptor types ==========
    comptime ActTma = TMATensorTileIm2col[
        Self.act_type,
        coord[Self.BM, Self.BK],
        coord[Self.PIX, Self.BK],
    ]
    comptime FILT_BOX_ROWS = Self.BK if Self.rscf_filter else Self.BN
    comptime FILT_BOX_COLS = Self.BN if Self.rscf_filter else Self.BK
    comptime FiltTma = TmaOpType[
        Self.act_type,
        static_row_major[Self.FILT_BOX_ROWS, Self.FILT_BOX_COLS],
        static_row_major[Self.FILT_BOX_ROWS, Self.FILT_BOX_COLS],
    ]
    comptime OutTma = TmaOpType[
        Self.act_type,
        static_row_major[Self.BM, Self.BN],
        static_row_major[Self.BM, Self.BN],
    ]

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
    @__llvm_arg_metadata(out_tma, `nvvm.grid_constant`)
    @__name(t"sm100_conv2d_k1x5_fprop_{Self.act_type}")
    def run[
        EpilogueFnType: ElementwiseEpilogueFn,
        ComputeFnType: ElementwiseComputeFn,
    ](
        act_tma: Self.ActTma,
        filt_tma: Self.FiltTma,
        out_tma: Self.OutTma,
        out_h: Int32,
        out_w: Int32,
        m_total: Int32,
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
            out_tma: Descriptor for the NHWC output viewed `[M, F]`, box
                `[BM, BN]`, 128-byte swizzle.
            out_h: Output height.
            out_w: Output width.
            m_total: Output pixels in the whole launch.
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
            out_tma,
            Int(out_h),
            Int(out_w),
            Int(m_total),
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
        out_tma: Self.OutTma,
        out_h: Int,
        out_w: Int,
        m_total: Int,
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
            out_tma.prefetch_descriptor()
            comptime for i in range(Self.FULL, Self.TMEM_FREE):
                bars.unsafe_offset(i)[].init(1)
            # Every epilogue thread arrives once when it is done with TMEM.
            bars.unsafe_offset(Self.TMEM_FREE)[].init(Int32(Self.EPI_THREADS))
        fence_mbarrier_init()
        barrier()

        var m0 = block_idx.x * Self.BM
        var n0 = block_idx.y * Self.BN
        if warp < 4:
            Self._epilogue(
                out_tma,
                sc,
                bars,
                tmem_slot,
                tid,
                m0,
                n0,
                m_total,
                out_ptr,
                src_ptr,
                beta,
                epilogue_fn,
                compute_fn,
                out_h * out_w,
                out_w,
            )
        elif warp == 4:
            Self._load(act_tma, filt_tma, sa, sb, bars, m0, n0, out_h, out_w)
        else:
            Self._mma(sa, sb, bars, tmem_slot)

    # ----- load warp -----

    @staticmethod
    @always_inline
    def _load(
        act_tma: Self.ActTma,
        filt_tma: Self.FiltTma,
        sa: MutPointer[Scalar[Self.act_type], _, address_space=.SHARED],
        sb: MutPointer[Scalar[Self.act_type], _, address_space=.SHARED],
        bars: MutPointer[SharedMemBarrier, _, address_space=.SHARED],
        m0: Int,
        n0: Int,
        out_h: Int,
        out_w: Int,
    ):
        """Loads, per K step, one filter box and the tile's im2col boxes into
        ring slot `stage`, tap outer and channel block inner."""
        var act_desc = Pointer(to=act_tma.descriptor).unsafe_bitcast[NoneType]()
        var filt_desc = Pointer(to=filt_tma.descriptor).unsafe_bitcast[
            NoneType
        ]()

        # (n, h, w) of the first pixel of each of the tile's boxes. A box that
        # starts near the end of an output row continues into the next row
        # in the hardware's own walk; a box past the last pixel lies outside
        # the image and loads zeros into rows the output store drops.
        var box_n = Array[Int32, Self.BOXES](uninitialized=True)
        var box_h = Array[Int32, Self.BOXES](uninitialized=True)
        var box_w = Array[Int32, Self.BOXES](uninitialized=True)
        comptime for b in range(Self.BOXES):
            var p = m0 + b * Self.PIX
            var row, col = divmod(p, out_w)
            var image, y = divmod(row, out_h)
            box_n[b] = Int32(image)
            box_h[b] = Int32(y)
            box_w[b] = Int32(col)

        # The activation does not depend on the filter pack that precedes
        # this grid, so the first `STAGES` activation slots are in flight
        # before the wait; the filter boxes follow it.
        comptime for ks in range(Self.STAGES):
            bars.unsafe_offset(Self.EMPTY + ks)[].wait(UInt32(1))
            if elect_one_sync():
                Self._load_act(act_desc, sa, bars, ks, ks, box_n, box_h, box_w)
        wait_on_dependent_grids()

        var stage = 0
        var phase = UInt32(0)
        for ks in range(Self.KS):
            if ks >= Self.STAGES:
                bars.unsafe_offset(Self.EMPTY + stage)[].wait(phase ^ 1)
                if elect_one_sync():
                    Self._load_act(
                        act_desc, sa, bars, stage, ks, box_n, box_h, box_w
                    )
            if elect_one_sync():
                comptime for g in range(Self.KG):
                    var kt = ks * Self.KG + g
                    cp_async_bulk_tensor_shared_cluster_global[cta_group=1](
                        sb.unsafe_offset(
                            stage * Self.B_STAGE_ELEMS + g * Self.B_ELEMS
                        ),
                        filt_desc,
                        bars.unsafe_offset(Self.FULL + stage),
                        (n0, kt * Self.BK) if Self.rscf_filter else (
                            kt * Self.BK,
                            n0,
                        ),
                    )
            stage += 1
            if stage == Self.STAGES:
                stage = 0
                phase ^= 1

    @staticmethod
    @always_inline
    def _load_act(
        act_desc: OpaquePointer[mut=False, _],
        sa: MutPointer[Scalar[Self.act_type], _, address_space=.SHARED],
        bars: MutPointer[SharedMemBarrier, _, address_space=.SHARED],
        stage: Int,
        ks: Int,
        box_n: Array[Int32, Self.BOXES],
        box_h: Array[Int32, Self.BOXES],
        box_w: Array[Int32, Self.BOXES],
    ):
        """Arms ring slot `stage`'s full barrier for ring step `ks` and issues
        its im2col boxes; the filter boxes complete on the same barrier."""
        var full = bars.unsafe_offset(Self.FULL + stage)
        full[].expect_bytes(Int32(Self.A_STAGE_BYTES + Self.B_STAGE_BYTES))
        comptime for g in range(Self.KG):
            var kt = ks * Self.KG + g
            var dst = sa.unsafe_offset(
                stage * Self.A_STAGE_ELEMS + g * Self.A_ELEMS
            )
            var tap, cb = divmod(kt, Self.CBLOCKS)
            comptime for b in range(Self.BOXES):
                cp_async_bulk_tensor_shared_cluster_global_im2col[cta_group=1](
                    dst.unsafe_offset(b * Self.PIX * Self.BK),
                    act_desc,
                    full,
                    (
                        cb * Self.BK,
                        Int(box_w[b]),
                        Int(box_h[b]),
                        Int(box_n[b]),
                    ),
                    (tap, 0),
                )

    # ----- MMA warp -----

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
            # MN-major B: K rows of one 128-byte swizzle row, eight to an
            # atom (SBO 1024); the N extent is one atom wide, so the leading
            # offset is unused.
            b_desc = MMASmemDescriptor.create[
                8 * Self.BN * Self.EB,
                Self.BK * Self.BN * Self.EB,
                Self.SWIZZLE,
            ](sb)
        else:
            b_desc = _create_mma_desc_k_major[Self.act_type, Self.SWIZZLE](sb)

        var stage = 0
        var phase = UInt32(0)
        for ks in range(Self.KS):
            bars.unsafe_offset(Self.FULL + stage)[].wait(phase)
            if elect_one_sync():
                comptime for g in range(Self.KG):
                    var a_slot = (
                        a_desc + stage * Self.A_STAGE_BYTES + g * Self.A_BYTES
                    )
                    var b_slot = (
                        b_desc + stage * Self.B_STAGE_BYTES + g * Self.B_BYTES
                    )
                    comptime if Self.rscf_filter:
                        comptime idesc = UMMAInsDescriptor[
                            UMMAKind.KIND_F16
                        ].create[
                            .float32,
                            Self.act_type,
                            Self.act_type,
                            (UInt32(Self.BM), UInt32(Self.BN)),
                            transpose_b=False,
                        ]()
                        comptime for k in range(Self.BK // Self.MMA_K):
                            var c_scale: UInt32 = 0 if (
                                ks == 0 and g == 0 and k == 0
                            ) else 1
                            mma[1](
                                a_slot + k * Self.MMA_K * Self.EB,
                                b_slot + k * Self.MMA_K * Self.BN * Self.EB,
                                tmem,
                                idesc,
                                c_scale=c_scale,
                            )
                    else:
                        mma_op.mma_from_desc(
                            a_slot, b_slot, tmem, init_c=(ks == 0 and g == 0)
                        )
                mma_op.commit(bars.unsafe_offset(Self.EMPTY + stage))
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
    @always_inline
    def _epilogue[
        EpilogueFnType: ElementwiseEpilogueFn,
        ComputeFnType: ElementwiseComputeFn,
    ](
        out_tma: Self.OutTma,
        sc: MutPointer[Scalar[Self.act_type], _, address_space=.SHARED],
        bars: MutPointer[SharedMemBarrier, _, address_space=.SHARED],
        tmem_slot: MutPointer[UInt32, _, address_space=.SHARED],
        tid: Int,
        m0: Int,
        n0: Int,
        m_total: Int,
        out_ptr: MutPointer[Scalar[Self.act_type], _],
        src_ptr: MutPointer[Scalar[Self.act_type], _],
        beta: Float32,
        epilogue_fn: EpilogueFnType,
        compute_fn: ComputeFnType,
        out_hw: Int,
        out_w: Int,
    ):
        """Moves the accumulator TMEM -> registers -> smem -> TMA store.

        With `has_epilogue_fn`, `has_compute_fn` or `has_residual` each live
        thread instead owns one output pixel of the tile and applies the
        epilogue to its 64 channels in registers (`apply_conv_epilogue`).
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
        var out_desc = Pointer(to=out_tma.descriptor).unsafe_bitcast[NoneType]()
        var warp = tid // WARP_SIZE

        bars.unsafe_offset(Self.ACC_FULL)[].wait(0)
        tcgen05_fence_after()

        comptime if Self.has_epilogue_fn or Self.has_compute_fn or Self.has_residual:
            # Warp `w` holds rows 16 w .. 16 w + 15 in its lanes 0 to 15; the
            # last tile may hold fewer than 64 pixels.
            var lane = tid % WARP_SIZE
            var pixel = m0 + warp * 16 + lane
            apply_conv_epilogue[
                cols=Self.BN,
                has_epilogue_fn=Self.has_epilogue_fn,
                has_compute_fn=Self.has_compute_fn,
                has_residual=Self.has_residual,
                nhwc=Self.nhwc_coords,
                round_first=Self.round_first,
            ](
                tmem,
                pixel,
                n0,
                Self.FOUT,
                out_ptr,
                src_ptr,
                beta,
                epilogue_fn,
                compute_fn,
                lane < 16 and pixel < m_total,
                out_hw,
                out_w,
            )
            tcgen05_fence_before()
            _ = bars.unsafe_offset(Self.TMEM_FREE)[].arrive()
            return

        stage_accumulator[Self.act_type, Self.BN, 16, Self.SWIZZLE](
            tmem, sc, warp
        )
        fence_async_view_proxy()
        Self.EpiSync.sync()
        if tid == 0:
            cp_async_bulk_tensor_global_shared_cta(sc, out_desc, (n0, m0))
            cp_async_bulk_commit_group()

        tcgen05_fence_before()
        _ = bars.unsafe_offset(Self.TMEM_FREE)[].arrive()
        if tid == 0:
            cp_async_bulk_wait_group[0, read=False]()


def conv2d_k1x5_fprop[
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
    filter: TileTensor[...],  # KRSC (out_ch, 1, 5, in_ch)
    problem: Conv2dProblemShape,
    ctx: DeviceContext,
    source_ptr: OptionalReg[MutPointer[Scalar[act_type], MutAnyOrigin]] = None,
    beta: Float32 = 0.0,
    epilogue_fn: EpilogueFnType = no_epilogue_fn,
    compute_fn: ComputeFnType = no_compute_fn,
) raises:
    """Launches the 1x5 filter, 512 -> 1024 channel conv2d fprop kernel.

    Accepts only problems `conv2d_k1x5_supported` accepts; the caller checks.

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
        output: Output tensor `[N, H_out, W_out, 1024]`, NHWC.
        activation: Input activation `[N, H, W, 512]`, NHWC.
        filter: Filter `[1024, 1, 5, 512]`, KRSC (channel innermost), or
            `[1, 5, 512, 1024]` RSCF (output channel innermost).
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
    if not conv2d_k1x5_supported[act_type](problem, ctx):
        raise Error("conv2d_k1x5_fprop: unsupported problem shape")

    comptime K = Conv2dFpropK1x5[
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
    var filt_tma = create_box_tma[K.FILT_BOX_ROWS, K.FILT_BOX_COLS, K.SWIZZLE](
        ctx, filter_tensor
    )

    var out_tensor = output.reshape(row_major(m_total, problem.gemm_n()))
    var out_tma = create_box_tma[K.BM, K.BN, K.SWIZZLE](ctx, out_tensor)

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
        out_tma,
        Int32(out_h),
        Int32(out_w),
        Int32(m_total),
        out_ptr,
        src_ptr,
        beta,
        host_args=(epilogue_fn, compute_fn),
        grid_dim=(ceildiv(m_total, K.BM), problem.gemm_n() // K.BN, 1),
        block_dim=K.NUM_THREADS,
        shared_mem_bytes=K.SMEM_BYTES,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(K.SMEM_BYTES)
        ),
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )


# ----- filter pack: FCRS or RSCF -> KRSC -----

comptime _PACK_C = Conv2dFpropK1x5[.bfloat16].CIN
comptime _PACK_S = Conv2dFpropK1x5[.bfloat16].S
comptime _PACK_K = _PACK_C * _PACK_S
comptime _PACK_V = 8
"""Elements one thread moves with a single 16-byte store."""
comptime _PACK_ROWS = 4
"""Output channels one `_pack_fcrs` block packs."""


@__name(t"pack_fcrs_to_krsc_k1x5_{dtype}")
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
    """`_PACK_ROWS` output channels per block. The rows of `src` are
    contiguous, so a block reads them coalesced into shared memory, then each
    thread gathers eight channels of one tap and writes them as one 16-byte
    store. Grid `F / _PACK_ROWS`."""
    comptime N = _PACK_ROWS * _PACK_K
    comptime THREADS = 256
    var tile = stack_allocation[dtype, address_space=.SHARED](row_major[N]())
    var base = block_idx.x * N
    var t = thread_idx.x
    with PDL[overlap_at_beginning=True]():
        comptime for u in range(N // THREADS):
            tile.raw_store(
                t + u * THREADS, src.raw_load(base + t + u * THREADS)
            )
        barrier()
        comptime for u in range(N // _PACK_V // THREADS):
            var v_idx = t + u * THREADS
            var row, v_col = divmod(v_idx, _PACK_K // _PACK_V)
            var j = v_col * _PACK_V
            var from_offset = (
                row * _PACK_K + (j % _PACK_C) * _PACK_S + j // _PACK_C
            )
            var v = SIMD[dtype, _PACK_V]()
            comptime for i in range(_PACK_V):
                v[i] = tile.raw_load(from_offset + i * _PACK_S)
            dst.raw_store[alignment=16](base + v_idx * _PACK_V, v)


def conv2d_k1x5_dispatch[
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
    """Runs the convolution on the 1x5 kernel if it serves the problem.

    The filter is in the caller's layout, FCRS `[F, C, 1, 5]` or RSCF
    `[1, 5, C, F]`. An RSCF filter is read in place as the MN-major B operand;
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
        (input_type != .float16 and input_type != .bfloat16)
        or input_type != output_type
        or input_type != filter_type
        or not (
            input.is_row_major and filter.is_row_major and output.is_row_major
        )
    ):
        return False
    else:
        comptime K = Conv2dFpropK1x5[input_type]
        var problem = dedicated_problem_shape[filter_is_fcrs](
            input, filter, output, pad_h, pad_w
        )
        if not conv2d_k1x5_supported[input_type](problem, ctx):
            return False
        # The TMA descriptors need 16-byte aligned bases, and the register
        # epilogue moves 16-byte vectors of the output and the residual.
        comptime if not filter_is_fcrs:
            if Int(filter.unsafe_ptr()) % 16 != 0:
                return False
        comptime if has_epilogue_fn or has_compute_fn or has_residual:
            if Int(output.unsafe_ptr()) % 16 != 0:
                return False
        comptime if has_residual:
            if Int(source_ptr.unsafe_value()) % 16 != 0:
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
                problem.out_channels * K.S * K.CIN
            )
            var krsc = TileTensor(
                krsc_buf, row_major(problem.out_channels, 1, K.S, K.CIN)
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
                grid_dim=problem.out_channels // _PACK_ROWS,
                block_dim=256,
                attributes=pdl_launch_attributes(PDLLevel.ON),
            )
            conv2d_k1x5_fprop[
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
            conv2d_k1x5_fprop[
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
                filter,
                problem,
                ctx,
                source,
                beta,
                epilogue_fn,
                compute_fn,
            )
        return True
