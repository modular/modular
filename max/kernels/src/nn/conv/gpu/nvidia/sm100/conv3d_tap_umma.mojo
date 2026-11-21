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
"""SM100 3-D convolutions whose reduction is a short list of taps over plain
tiled TMA boxes: the VAE's time conv and its thin-channel input conv.

Target family: Blackwell sm_100a, 1-SM UMMA, fp16 and bf16 with fp32
accumulation, or fp32 operands rounded to tf32 by the tensor core. NDHWC
activation and output, stride 1, dilation 1.

Two kernels share this file's rings, warp roles and epilogue:

`Conv3dDepthUmma`, the depth conv (`Q x 1 x 1`, no padding). A depth tap moves
the whole `H x W` plane, so with the plane flattened to `PLANE = H * W` pixels
the activation is one matrix `X [D * PLANE, C]` and tap `q` of output row
`m = d * PLANE + p` is row `m + q * PLANE` of it. The convolution is `Q`
row-shifted 1x1 GEMMs accumulated into one accumulator. Every operand box is
one tiled TMA box of 128 consecutive rows of `X` (a row past the end reads as
zero), and the output descriptor drops rows past the plane and depths past
the last. A pass is `MPASS` consecutive output depths of one pixel tile: the
planes stream through the activation ring once per channel block, and plane
`j` serves tap `q` of output depth `j - q`, so each plane is loaded once for
up to `Q` taps.

`Conv3dWindowUmma`, the 3x3x3 conv with padding 1 and 16 input channels. A
tile is a `HT x WT` block of pixels of one depth slice. The padded
`HP x WP` window of the input that the block needs is one tiled box per depth
tap (zero fill is the padding), and each of the nine in-plane taps reads it
`qh * WP + qw` rows on. One tap is one K step of 16 channels, so the filter is
27 resident boxes and `K = 27 * 16` has no tail. The accumulator row of pixel
`(h, w)` is `h * WP + w`; the `WP - WT` columns per row that belong to no
output compute throwaway values and write nothing.

Warp roles (7 warps, 224 threads):

| warps | role                                                       |
|-------|------------------------------------------------------------|
| 0-3   | epilogue: TMEM -> smem -> TMA store, or the fused epilogue |
| 4     | filter load                                                |
| 5     | activation load                                            |
| 6     | MMA issue, TMEM allocation                                 |

The filter is `[F, K]` with `K = tap * C + c`, which `pack_conv3d_filter`
builds from either FCQRS or QRSCF.
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
from structured_kernels.barriers import WarpGroupBarrier

from nn.conv.conv_utils import ConvSimdComputeFn, ConvSimdEpilogueFn

from .conv_epilogue import apply_conv_epilogue
from .conv3d_umma_common import (
    RingCursor,
    pack_conv3d_filter,
    stage_columns_swizzled,
)

comptime _SMEM_LIMIT = 232448
"""Dynamic shared memory of an SM100 CTA: 227 KiB."""


# ===----------------------------------------------------------------------=== #
# The depth conv
# ===----------------------------------------------------------------------=== #


struct Conv3dDepthUmma[
    act_type: DType,
    cin: Int,
    fout: Int,
    height: Int,
    width: Int,
    taps: Int,
    has_epilogue_fn: Bool = False,
    has_compute_fn: Bool = False,
]:
    """The depth conv (`taps x 1 x 1`, no padding) and its launch geometry.

    Parameters:
        act_type: Element type of the activation, filter and output.
        cin: Input channels.
        fout: Output channels.
        height: Plane height.
        width: Plane width.
        taps: Depth taps `Q`.
        has_epilogue_fn: An elementwise epilogue stores each output value.
        has_compute_fn: An elementwise epilogue transforms each output value.
    """

    comptime EB = size_of[Self.act_type]()
    comptime BM = 128
    comptime BN = 96
    """Output channels per tile: three 32-column staging boxes."""
    comptime SWIZZLE = TensorMapSwizzle.SWIZZLE_128B
    comptime BK = 128 // Self.EB
    """Channels per K step: one 128-byte swizzle row."""
    comptime NCB = Self.cin // Self.BK
    comptime PLANE = Self.height * Self.width
    comptime PTILES = ceildiv(Self.PLANE, Self.BM)
    comptime MPASS = 3
    """Output depths per pass."""
    comptime NJ = Self.MPASS + Self.taps - 1
    """Input planes one pass streams per channel block."""
    comptime UNITS = 10
    comptime STAGES = 6 if Self.EB == 4 else 8
    comptime FST = 6
    comptime SCST = 1
    comptime NACC = min(512 // Self.BN, Self.MPASS * Self.UNITS)

    comptime A_BYTES = Self.BM * Self.BK * Self.EB
    comptime B_BYTES = Self.BN * Self.BK * Self.EB
    comptime C_BYTES = Self.BM * 32 * Self.EB
    comptime A_ELEMS = Self.A_BYTES // Self.EB
    comptime B_ELEMS = Self.B_BYTES // Self.EB
    comptime C_ELEMS = Self.C_BYTES // Self.EB
    comptime A_OFF = 0
    comptime B_OFF = align_up(Self.A_OFF + Self.STAGES * Self.A_BYTES, 1024)
    comptime C_OFF = align_up(Self.B_OFF + Self.FST * Self.B_BYTES, 1024)
    comptime BAR_OFF = Self.C_OFF + 3 * Self.SCST * Self.C_BYTES

    comptime A_FULL = 0
    comptime A_EMPTY = Self.A_FULL + Self.STAGES
    comptime B_FULL = Self.A_EMPTY + Self.STAGES
    comptime B_EMPTY = Self.B_FULL + Self.FST
    comptime ACC_FULL = Self.B_EMPTY + Self.FST
    comptime ACC_EMPTY = Self.ACC_FULL + Self.NACC
    comptime TMEM_FREE = Self.ACC_EMPTY + Self.NACC
    comptime NBAR = Self.TMEM_FREE + 1
    comptime TMEM_ADDR_OFF = Self.BAR_OFF + Self.NBAR * 8
    comptime SMEM_BYTES = Self.TMEM_ADDR_OFF + 16

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
            Self.cin * Self.EB
        ) % 128 == 0, "a channel run is whole 128-byte swizzle spans"
        comptime assert Self.fout % Self.BN == 0, "the output tile divides F"
        comptime assert Self.EB == 2 or Self.EB == 4, "16-bit or tf32 operands"
        comptime assert Self.taps <= Self.FST, "a plane's taps fit the ring"
        comptime assert (
            Self.SMEM_BYTES <= _SMEM_LIMIT
        ), "shared memory exceeds the 227 KiB of an SM100 CTA"

    @staticmethod
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.NUM_THREADS)
        )
    )
    @__llvm_arg_metadata(act_desc, `nvvm.grid_constant`)
    @__llvm_arg_metadata(filt_desc, `nvvm.grid_constant`)
    @__llvm_arg_metadata(out_desc, `nvvm.grid_constant`)
    @__name(t"sm100_conv3d_depth_umma_{Self.act_type}_{Self.cin}x{Self.fout}")
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
        out_depth: Int32,
        ncta: Int32,
        epilogue_fn: EpilogueFnType,
        compute_fn: ComputeFnType,
    ):
        """Kernel entry point.

        Args:
            act_desc: Descriptor for the activation viewed `[D * PLANE, C]`
                as `(1, D * PLANE, 1, C)`, box `(1, BM, 1, BK)`.
            filt_desc: Descriptor for the `[F, K]` filter, box `(BN, BK)`.
            out_desc: Descriptor for the output viewed `(1, D_out, PLANE, F)`,
                box `(1, 1, BM, 32)`; unused with a fused epilogue.
            output: Output, NDHWC; written by the kernel when
                `has_compute_fn`.
            out_depth: Output depth.
            ncta: CTAs along x (the stride between a CTA's passes).
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
        var cty = block_idx.y

        if tid == 0:
            prefetch_tma_descriptor(
                Pointer(to=act_desc).unsafe_bitcast[NoneType]()
            )
            prefetch_tma_descriptor(
                Pointer(to=filt_desc).unsafe_bitcast[NoneType]()
            )
            comptime if not Self.fused:
                prefetch_tma_descriptor(
                    Pointer(to=out_desc).unsafe_bitcast[NoneType]()
                )
            comptime for i in range(Self.A_FULL, Self.ACC_EMPTY):
                bars.unsafe_offset(i)[].init(1)
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
                Int(ncta),
                cty,
                Int(out_depth),
                epilogue_fn,
                compute_fn,
            )
        elif warp == 4:
            Self._filter_load(filt_desc, sb, bars, cty)
        elif warp == 5:
            Self._act_load(act_desc, sa, bars, cta, Int(ncta))
        else:
            Self._mma(sa, sb, bars, tmem_slot)

    @staticmethod
    @always_inline
    def _filter_load(
        filt_desc: TMADescriptor,
        sb: MutPointer[Scalar[Self.act_type], _, address_space=.SHARED],
        bars: MutPointer[SharedMemBarrier, _, address_space=.SHARED],
        cty: Int,
    ):
        """Loads each channel block's `taps` filter boxes, once per pass."""
        var desc = Pointer(to=filt_desc).unsafe_bitcast[NoneType]()
        # The filter is the packing kernel's output.
        wait_on_dependent_grids()
        var ring = RingCursor[Self.FST]()
        for _ in range(Self.UNITS):
            for cb in range(Self.NCB):
                for q in range(Self.taps):
                    bars.unsafe_offset(Self.B_EMPTY + ring.slot())[].wait(
                        ring.phase ^ 1
                    )
                    if elect_one_sync():
                        var full = bars.unsafe_offset(Self.B_FULL + ring.slot())
                        full[].expect_bytes(Int32(Self.B_BYTES))
                        cp_async_bulk_tensor_shared_cluster_global[cta_group=1](
                            sb.unsafe_offset(ring.slot() * Self.B_ELEMS),
                            desc,
                            full,
                            ((q * Self.NCB + cb) * Self.BK, cty * Self.BN),
                        )
                    ring.advance()

    @staticmethod
    @always_inline
    def _act_load(
        act_desc: TMADescriptor,
        sa: MutPointer[Scalar[Self.act_type], _, address_space=.SHARED],
        bars: MutPointer[SharedMemBarrier, _, address_space=.SHARED],
        cta: Int,
        ncta: Int,
    ):
        """Streams each pass's input planes once per channel block."""
        var desc = Pointer(to=act_desc).unsafe_bitcast[NoneType]()
        # The activation is the predecessor's output.
        wait_on_dependent_grids()
        var ring = RingCursor[Self.STAGES]()
        for u in range(Self.UNITS):
            var cu = cta + u * ncta
            var dblock, ptile = divmod(cu, Self.PTILES)
            for cb in range(Self.NCB):
                for j in range(Self.NJ):
                    bars.unsafe_offset(Self.A_EMPTY + ring.slot())[].wait(
                        ring.phase ^ 1
                    )
                    if elect_one_sync():
                        var full = bars.unsafe_offset(Self.A_FULL + ring.slot())
                        full[].expect_bytes(Int32(Self.A_BYTES))
                        cp_async_bulk_tensor_shared_cluster_global[cta_group=1](
                            sa.unsafe_offset(ring.slot() * Self.A_ELEMS),
                            desc,
                            full,
                            (
                                cb * Self.BK,
                                0,
                                ptile * Self.BM
                                + (dblock * Self.MPASS + j) * Self.PLANE,
                                0,
                            ),
                        )
                    ring.advance()

    @staticmethod
    @always_inline
    def _mma(
        sa: MutPointer[Scalar[Self.act_type], _, address_space=.SHARED],
        sb: MutPointer[Scalar[Self.act_type], _, address_space=.SHARED],
        bars: MutPointer[SharedMemBarrier, _, address_space=.SHARED],
        tmem_slot: MutPointer[UInt32, _, address_space=.SHARED],
    ):
        """Allocates TMEM, issues every UMMA of every pass, and frees TMEM.

        Plane `j` of a channel block feeds tap `q` of output depth `j - q`:
        up to `taps` UMMAs under one activation slot, into different
        accumulators, against the block's `taps` filter boxes. Filter box
        `q` is waited at the first plane that uses it (`j = q`) and released
        after the last (`j = q + MPASS - 1`), so the ring refills it while the
        block's later planes run. Output depth `mt` completes at plane
        `mt + taps - 1` of the last channel block, so its accumulator is
        committed there and its epilogue starts while later depths still
        accumulate.
        """
        tcgen05_alloc[1](tmem_slot, UInt32(Self.TMEM_COLS))
        syncwarp()
        var tmem = tmem_slot[]
        Self.MmaEpiSync.arrive()

        var mma_op = Self.MmaOp()
        var a_desc = _create_mma_desc_k_major[Self.act_type, Self.SWIZZLE](sa)
        var b_desc = _create_mma_desc_k_major[Self.act_type, Self.SWIZZLE](sb)

        var ra = RingCursor[Self.STAGES]()
        var rb = RingCursor[Self.FST]()
        var acc = RingCursor[Self.NACC]()

        for _ in range(Self.UNITS):
            var acc_stage = Array[Int, Self.MPASS](uninitialized=True)
            comptime for mt in range(Self.MPASS):
                bars.unsafe_offset(Self.ACC_EMPTY + acc.slot())[].wait(
                    acc.phase ^ 1
                )
                acc_stage[mt] = acc.slot()
                acc.advance()
            tcgen05_fence_after()

            for cb in range(Self.NCB):
                var b_slots = Array[Int, Self.taps](uninitialized=True)
                comptime for j in range(Self.NJ):
                    comptime if j < Self.taps:
                        bars.unsafe_offset(Self.B_FULL + rb.slot())[].wait(
                            rb.phase
                        )
                        b_slots[j] = rb.slot()
                        rb.advance()
                    bars.unsafe_offset(Self.A_FULL + ra.slot())[].wait(ra.phase)
                    var a_slot = ra.slot()
                    ra.advance()
                    comptime for q in range(Self.taps):
                        comptime mt = j - q
                        comptime if mt >= 0 and mt < Self.MPASS:
                            if elect_one_sync():
                                Self.MmaOp().mma_from_desc(
                                    a_desc + a_slot * Self.A_BYTES,
                                    b_desc + b_slots[q] * Self.B_BYTES,
                                    tmem + UInt32(acc_stage[mt] * Self.BN),
                                    cb == 0 and q == 0,
                                )
                    if elect_one_sync():
                        mma_op.commit(bars.unsafe_offset(Self.A_EMPTY + a_slot))
                    comptime qr = j - (Self.MPASS - 1)
                    comptime if qr >= 0 and qr < Self.taps:
                        if elect_one_sync():
                            mma_op.commit(
                                bars.unsafe_offset(Self.B_EMPTY + b_slots[qr])
                            )
                    comptime if j >= Self.taps - 1:
                        comptime done = j - (Self.taps - 1)
                        comptime if done < Self.MPASS:
                            if cb == Self.NCB - 1:
                                if elect_one_sync():
                                    mma_op.commit(
                                        bars.unsafe_offset(
                                            Self.ACC_FULL + acc_stage[done]
                                        )
                                    )

        tcgen05_release_allocation_lock[1]()
        bars.unsafe_offset(Self.TMEM_FREE)[].wait(0)
        tcgen05_fence_after()
        tcgen05_dealloc[1](tmem, UInt32(Self.TMEM_COLS))

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
        out_depth: Int,
        epilogue_fn: EpilogueFnType,
        compute_fn: ComputeFnType,
    ):
        """Moves each accumulator TMEM -> registers -> smem -> TMA store, or
        applies the fused epilogue to the registers."""
        Self.MmaEpiSync.wait()
        var tmem = tmem_slot[]
        var desc = Pointer(to=out_desc).unsafe_bitcast[NoneType]()
        var acc = RingCursor[Self.NACC]()
        var out_ptr = output.unsafe_ptr().unsafe_origin_cast[
            MutUntrackedOrigin
        ]()

        for u in range(Self.UNITS):
            var cu = cta + u * ncta
            var dblock, ptile = divmod(cu, Self.PTILES)
            for mt in range(Self.MPASS):
                var depth = dblock * Self.MPASS + mt
                bars.unsafe_offset(Self.ACC_FULL + acc.slot())[].wait(acc.phase)
                tcgen05_fence_after()
                var taddr = tmem + UInt32(acc.slot() * Self.BN)

                comptime if Self.fused:
                    var pixel = ptile * Self.BM + tid
                    apply_conv_epilogue[
                        cols=96,
                        has_epilogue_fn=Self.has_epilogue_fn,
                        has_compute_fn=Self.has_compute_fn,
                        has_residual=False,
                        ndhwc=True,
                        round_first=True,
                    ](
                        taddr,
                        depth * Self.PLANE + pixel,
                        cty * Self.BN,
                        Self.fout,
                        out_ptr,
                        out_ptr,
                        0.0,
                        epilogue_fn,
                        compute_fn,
                        pixel < Self.PLANE and depth < out_depth,
                        Self.PLANE,
                        Self.width,
                        out_depth,
                    )
                else:
                    if tid == 0:
                        cp_async_bulk_wait_group[Int32(Self.SCST - 1)]()
                    Self.EpiSync.sync()
                    stage_columns_swizzled[Self.act_type, 3](
                        taddr, sc, Self.C_ELEMS, tid, True
                    )
                    fence_async_view_proxy()
                    Self.EpiSync.sync()
                    if tid == 0:
                        comptime for pc in range(3):
                            cp_async_bulk_tensor_global_shared_cta(
                                sc.unsafe_offset(pc * Self.C_ELEMS),
                                desc,
                                (
                                    cty * Self.BN + pc * 32,
                                    ptile * Self.BM,
                                    depth,
                                    0,
                                ),
                            )
                        cp_async_bulk_commit_group()

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


def conv3d_depth_umma_supported[
    act_type: DType, cin: Int, fout: Int, height: Int, width: Int, taps: Int
]() -> Bool:
    """Whether `launch_conv3d_depth_umma` serves a channel and plane pair.

    The pair is the VAE time conv the kernel was measured on, at each element
    type.

    Parameters:
        act_type: Element type of the activation, filter and output.
        cin: Input channels.
        fout: Output channels.
        height: Plane height.
        width: Plane width.
        taps: Depth taps.

    Returns:
        True for the measured shape.
    """
    return (
        (act_type == .float16 or act_type == .bfloat16 or act_type == .float32)
        and cin == 192
        and fout == 384
        and height == 60
        and width == 104
        and taps == 3
    )


def conv3d_depth_umma_fits[
    act_type: DType,
    cin: Int,
    fout: Int,
    height: Int,
    width: Int,
    taps: Int,
](depth: Int, ctx: DeviceContext) -> Bool:
    """Whether the depth conv's grid for `depth` input slices is one wave of
    CTAs, the only extent the kernel was measured on.

    Parameters:
        act_type: Element type of the activation, filter and output.
        cin: Input channels.
        fout: Output channels.
        height: Plane height.
        width: Plane width.
        taps: Depth taps.

    Args:
        depth: Input depth.
        ctx: The device the kernel would run on.

    Returns:
        True when every CTA of the grid is resident at once.
    """
    comptime K = Conv3dDepthUmma[act_type, cin, fout, height, width, taps]
    var passes = K.PTILES * ceildiv(depth - (taps - 1), K.MPASS)
    return ceildiv(passes, K.UNITS) * (fout // K.BN) <= (
        ctx.default_device_info.sm_count
    )


def launch_conv3d_depth_umma_packed[
    EpilogueFnType: ConvSimdEpilogueFn,
    ComputeFnType: ConvSimdComputeFn,
    packed_type: DType,
    //,
    act_type: DType,
    cin: Int,
    fout: Int,
    height: Int,
    width: Int,
    taps: Int,
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
    """Launches the depth conv kernel.

    The caller has checked `conv3d_depth_umma_supported`, a batch of one,
    dense row-major tensors and 16-byte aligned base pointers, and a grid of
    one wave (`conv3d_depth_umma_fits`), and keeps `packed` alive until the
    launch is enqueued.

    Parameters:
        EpilogueFnType: Type of `epilogue_fn` (inferred).
        ComputeFnType: Type of `compute_fn` (inferred).
        act_type: Element type of the activation, filter and output.
        cin: Input channels.
        fout: Output channels.
        height: Plane height.
        width: Plane width.
        taps: Depth taps.
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
    comptime K = Conv3dDepthUmma[
        act_type,
        cin,
        fout,
        height,
        width,
        taps,
        has_epilogue_fn,
        has_compute_fn,
    ]
    var depth = Int(input.dim[1]())
    var out_depth = depth - (taps - 1)
    var passes = K.PTILES * ceildiv(out_depth, K.MPASS)
    var ncta = ceildiv(passes, K.UNITS)
    var grid_y = fout // K.BN

    comptime assert (
        output.dtype == act_type and packed_type == act_type
    ), "activation, filter and output share one element type"

    comptime out_swizzle = TensorMapSwizzle.SWIZZLE_64B if K.EB == 2 else (
        TensorMapSwizzle.SWIZZLE_128B
    )
    var rows = depth * K.PLANE
    var act_desc = create_tma_descriptor[act_type, swizzle_mode=K.SWIZZLE](
        DeviceBuffer(
            ctx,
            input.unsafe_ptr(),
            1,
            owning=False,
        ),
        (1, rows, 1, cin),
        (rows * cin, cin, cin, 1),
        (1, K.BM, 1, K.BK),
    )
    var filt_desc = create_tma_descriptor[packed_type, swizzle_mode=K.SWIZZLE](
        DeviceBuffer(ctx, packed.unsafe_ptr(), 1, owning=False),
        (fout, taps * cin),
        (taps * cin, 1),
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
        (1, out_depth, K.PLANE, fout),
        (out_depth * K.PLANE * fout, K.PLANE * fout, fout, 1),
        (1, 1, K.BM, 32),
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
        Int32(out_depth),
        Int32(ncta),
        host_args=(epilogue_fn, compute_fn),
        grid_dim=(ncta, grid_y, 1),
        block_dim=K.NUM_THREADS,
        shared_mem_bytes=K.SMEM_BYTES,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(K.SMEM_BYTES)
        ),
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )


# ===----------------------------------------------------------------------=== #
# The 3x3x3 conv with 16 input channels
# ===----------------------------------------------------------------------=== #


struct Conv3dWindowUmma[
    act_type: DType,
    fout: Int,
    height: Int,
    width: Int,
    has_epilogue_fn: Bool = False,
    has_compute_fn: Bool = False,
]:
    """The 3x3x3 conv with padding 1 and 16 input channels, and its launch
    geometry.

    Parameters:
        act_type: Element type of the activation, filter and output.
        fout: Output channels.
        height: Plane height.
        width: Plane width, at most two tiles of `WT` columns.
        has_epilogue_fn: An elementwise epilogue stores each output value.
        has_compute_fn: An elementwise epilogue transforms each output value.
    """

    comptime EB = size_of[Self.act_type]()
    comptime CIN = 16
    comptime BM = 128
    comptime BN = 96 if Self.EB == 2 else 64
    """Output channels per tile: 96, or 64 at fp32, where 27 resident filter
    boxes of 96 rows do not fit next to the windows."""
    comptime NB = Self.BN // 32
    """32-column staging boxes per tile."""
    comptime SWIZ_BYTES = 32 if Self.EB == 2 else 64
    comptime SWIZZLE = TensorMapSwizzle.SWIZZLE_32B if Self.EB == 2 else (
        TensorMapSwizzle.SWIZZLE_64B
    )
    comptime BK = 16
    """Channels per tap: the UMMA K step is the whole slot."""
    comptime TAPS = 27
    comptime HT = 4
    comptime WT = 26
    """Output rows and columns of a tile."""
    comptime HP = Self.HT + 2
    comptime WP = Self.WT + 2
    comptime WR = Self.HP * Self.WP
    """Rows of a window slot: the padded block."""
    comptime TROWS = Self.HT * Self.WT
    """Rows of a staged output box."""
    comptime WBT = ceildiv(Self.width, Self.WT)
    comptime HBT = ceildiv(Self.height, Self.HT)
    comptime UNITS = 10 if Self.EB == 2 else 14
    comptime STAGES = 12 if Self.EB == 2 else 8
    comptime FST = 27
    comptime SCST = 1
    comptime NACC = min(512 // Self.BN, Self.UNITS)

    comptime A_BYTES = Self.WR * Self.BK * Self.EB
    comptime B_BYTES = Self.BN * Self.BK * Self.EB
    comptime C_BYTES = Self.TROWS * 32 * Self.EB
    comptime A_ELEMS = Self.A_BYTES // Self.EB
    comptime B_ELEMS = Self.B_BYTES // Self.EB
    comptime C_ELEMS = Self.C_BYTES // Self.EB
    comptime A_OFF = 0
    comptime B_OFF = align_up(Self.A_OFF + Self.STAGES * Self.A_BYTES, 1024)
    comptime C_OFF = align_up(Self.B_OFF + Self.FST * Self.B_BYTES, 1024)
    comptime BAR_OFF = Self.C_OFF + Self.NB * Self.SCST * Self.C_BYTES

    comptime A_FULL = 0
    comptime A_EMPTY = Self.A_FULL + Self.STAGES
    comptime B_FULL = Self.A_EMPTY + Self.STAGES
    comptime B_EMPTY = Self.B_FULL + Self.FST
    comptime ACC_FULL = Self.B_EMPTY + Self.FST
    comptime ACC_EMPTY = Self.ACC_FULL + Self.NACC
    comptime TMEM_FREE = Self.ACC_EMPTY + Self.NACC
    comptime NBAR = Self.TMEM_FREE + 1
    comptime TMEM_ADDR_OFF = Self.BAR_OFF + Self.NBAR * 8
    comptime SMEM_BYTES = Self.TMEM_ADDR_OFF + 16

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
        comptime assert Self.fout % Self.BN == 0, "the output tile divides F"
        comptime assert Self.EB == 2 or Self.EB == 4, "16-bit or tf32 operands"
        comptime assert (
            Self.width <= 2 * Self.WT
        ), "at most two tiles along the width"
        comptime assert (
            Self.SMEM_BYTES <= _SMEM_LIMIT
        ), "shared memory exceeds the 227 KiB of an SM100 CTA"

    @staticmethod
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.NUM_THREADS)
        )
    )
    @__llvm_arg_metadata(act_desc, `nvvm.grid_constant`)
    @__llvm_arg_metadata(filt_desc, `nvvm.grid_constant`)
    @__llvm_arg_metadata(out_desc, `nvvm.grid_constant`)
    @__name(t"sm100_conv3d_window_umma_{Self.act_type}_{Self.fout}")
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
        out_depth: Int32,
        ncta: Int32,
        epilogue_fn: EpilogueFnType,
        compute_fn: ComputeFnType,
    ):
        """Kernel entry point.

        Args:
            act_desc: Descriptor for the activation viewed `(D, H, W, C)`, box
                `(1, HP, WP, C)`.
            filt_desc: Descriptor for the `[F, K]` filter, box `(BN, 16)`.
            out_desc: Descriptor for the output viewed `(D, H, W, F)`, box
                `(1, HT, WT, 32)`; unused with a fused epilogue.
            output: Output, NDHWC; written by the kernel when
                `has_compute_fn`.
            out_depth: Output depth.
            ncta: CTAs along x (the stride between a CTA's tiles).
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
        var cty = block_idx.y

        if tid == 0:
            prefetch_tma_descriptor(
                Pointer(to=act_desc).unsafe_bitcast[NoneType]()
            )
            prefetch_tma_descriptor(
                Pointer(to=filt_desc).unsafe_bitcast[NoneType]()
            )
            comptime if not Self.fused:
                prefetch_tma_descriptor(
                    Pointer(to=out_desc).unsafe_bitcast[NoneType]()
                )
            comptime for i in range(Self.A_FULL, Self.ACC_EMPTY):
                bars.unsafe_offset(i)[].init(1)
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
                Int(ncta),
                cty,
                Int(out_depth),
                epilogue_fn,
                compute_fn,
            )
        elif warp == 4:
            Self._filter_load(filt_desc, sb, bars, cty)
        elif warp == 5:
            Self._act_load(act_desc, sa, bars, cta, Int(ncta))
        else:
            Self._mma(sa, sb, bars, tmem_slot)

    @staticmethod
    @always_inline
    def _filter_load(
        filt_desc: TMADescriptor,
        sb: MutPointer[Scalar[Self.act_type], _, address_space=.SHARED],
        bars: MutPointer[SharedMemBarrier, _, address_space=.SHARED],
        cty: Int,
    ):
        """Loads the 27 filter boxes once; they stay resident."""
        var desc = Pointer(to=filt_desc).unsafe_bitcast[NoneType]()
        # The filter is the packing kernel's output.
        wait_on_dependent_grids()
        comptime for t in range(Self.TAPS):
            if elect_one_sync():
                var full = bars.unsafe_offset(Self.B_FULL + t)
                full[].expect_bytes(Int32(Self.B_BYTES))
                cp_async_bulk_tensor_shared_cluster_global[cta_group=1](
                    sb.unsafe_offset(t * Self.B_ELEMS),
                    desc,
                    full,
                    (t * Self.BK, cty * Self.BN),
                )

    @staticmethod
    @always_inline
    def _act_load(
        act_desc: TMADescriptor,
        sa: MutPointer[Scalar[Self.act_type], _, address_space=.SHARED],
        bars: MutPointer[SharedMemBarrier, _, address_space=.SHARED],
        cta: Int,
        ncta: Int,
    ):
        """Loads each tile's three padded windows, one per depth tap."""
        var desc = Pointer(to=act_desc).unsafe_bitcast[NoneType]()
        # The activation is the predecessor's output.
        wait_on_dependent_grids()
        var ring = RingCursor[Self.STAGES]()
        for u in range(Self.UNITS):
            var cu = cta + u * ncta
            var dd, tile = divmod(cu, Self.WBT * Self.HBT)
            var htile, wtile = divmod(tile, Self.WBT)
            comptime for qd in range(3):
                bars.unsafe_offset(Self.A_EMPTY + ring.slot())[].wait(
                    ring.phase ^ 1
                )
                if elect_one_sync():
                    var full = bars.unsafe_offset(Self.A_FULL + ring.slot())
                    full[].expect_bytes(Int32(Self.A_BYTES))
                    cp_async_bulk_tensor_shared_cluster_global[cta_group=1](
                        sa.unsafe_offset(ring.slot() * Self.A_ELEMS),
                        desc,
                        full,
                        (
                            0,
                            wtile * Self.WT - 1,
                            htile * Self.HT - 1,
                            dd + qd - 1,
                        ),
                    )
                ring.advance()

    @staticmethod
    @always_inline
    def _mma(
        sa: MutPointer[Scalar[Self.act_type], _, address_space=.SHARED],
        sb: MutPointer[Scalar[Self.act_type], _, address_space=.SHARED],
        bars: MutPointer[SharedMemBarrier, _, address_space=.SHARED],
        tmem_slot: MutPointer[UInt32, _, address_space=.SHARED],
    ):
        """Allocates TMEM, issues nine UMMAs per window (one per in-plane tap,
        each reading the window `qh * WP + qw` rows on), and frees TMEM. The
        27 filter boxes are waited once."""
        tcgen05_alloc[1](tmem_slot, UInt32(Self.TMEM_COLS))
        syncwarp()
        var tmem = tmem_slot[]
        Self.MmaEpiSync.arrive()

        var mma_op = Self.MmaOp()
        var a_desc = _create_mma_desc_k_major[Self.act_type, Self.SWIZZLE](sa)
        var b_desc = _create_mma_desc_k_major[Self.act_type, Self.SWIZZLE](sb)
        comptime row_bytes = Self.BK * Self.EB

        var ra = RingCursor[Self.STAGES]()
        var acc = RingCursor[Self.NACC]()

        comptime for t in range(Self.TAPS):
            bars.unsafe_offset(Self.B_FULL + t)[].wait(0)

        for _ in range(Self.UNITS):
            bars.unsafe_offset(Self.ACC_EMPTY + acc.slot())[].wait(
                acc.phase ^ 1
            )
            tcgen05_fence_after()
            var taddr = tmem + UInt32(acc.slot() * Self.BN)
            comptime for qd in range(3):
                bars.unsafe_offset(Self.A_FULL + ra.slot())[].wait(ra.phase)
                var a_slot = ra.slot()
                ra.advance()
                comptime for qh in range(3):
                    comptime for qw in range(3):
                        comptime t = (qd * 3 + qh) * 3 + qw
                        if elect_one_sync():
                            mma_op.mma_from_desc(
                                a_desc
                                + a_slot * Self.A_BYTES
                                + (qh * Self.WP + qw) * row_bytes,
                                b_desc + t * Self.B_BYTES,
                                taddr,
                                t == 0,
                            )
                if elect_one_sync():
                    mma_op.commit(bars.unsafe_offset(Self.A_EMPTY + a_slot))
            if elect_one_sync():
                mma_op.commit(bars.unsafe_offset(Self.ACC_FULL + acc.slot()))
            acc.advance()

        tcgen05_release_allocation_lock[1]()
        bars.unsafe_offset(Self.TMEM_FREE)[].wait(0)
        tcgen05_fence_after()
        tcgen05_dealloc[1](tmem, UInt32(Self.TMEM_COLS))

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
        out_depth: Int,
        epilogue_fn: EpilogueFnType,
        compute_fn: ComputeFnType,
    ):
        """Moves each accumulator TMEM -> registers -> smem -> TMA store, or
        applies the fused epilogue to the registers.

        Accumulator row `p = h * WP + w` is output pixel `(h, w)` of the
        tile; rows with `h >= HT` or `w >= WT` hold throwaway values. The
        staging boxes hold only the `WT` real columns of each row, packed as
        `h * WT + w`, so one rank-4 TMA box stores them; the descriptor clips
        the ragged edge tiles.
        """
        Self.MmaEpiSync.wait()
        var tmem = tmem_slot[]
        var desc = Pointer(to=out_desc).unsafe_bitcast[NoneType]()
        var acc = RingCursor[Self.NACC]()
        var out_ptr = output.unsafe_ptr().unsafe_origin_cast[
            MutUntrackedOrigin
        ]()
        var h_lane, w_lane = divmod(tid, Self.WP)
        var live_lane = h_lane < Self.HT and w_lane < Self.WT

        for u in range(Self.UNITS):
            var cu = cta + u * ncta
            var dd, tile = divmod(cu, Self.WBT * Self.HBT)
            var htile, wtile = divmod(tile, Self.WBT)
            bars.unsafe_offset(Self.ACC_FULL + acc.slot())[].wait(acc.phase)
            tcgen05_fence_after()
            var taddr = tmem + UInt32(acc.slot() * Self.BN)

            comptime if Self.fused:
                var h = htile * Self.HT + h_lane
                var w = wtile * Self.WT + w_lane
                comptime for half in range(Self.BN // 32):
                    apply_conv_epilogue[
                        cols=32,
                        has_epilogue_fn=Self.has_epilogue_fn,
                        has_compute_fn=Self.has_compute_fn,
                        has_residual=False,
                        ndhwc=True,
                        round_first=True,
                    ](
                        taddr + UInt32(32 * half),
                        (dd * Self.height + h) * Self.width + w,
                        cty * Self.BN + 32 * half,
                        Self.fout,
                        out_ptr,
                        out_ptr,
                        0.0,
                        epilogue_fn,
                        compute_fn,
                        live_lane
                        and dd < out_depth
                        and h < Self.height
                        and w < Self.width,
                        Self.height * Self.width,
                        Self.width,
                        out_depth,
                    )
            else:
                if tid == 0:
                    cp_async_bulk_wait_group[Int32(Self.SCST - 1)]()
                Self.EpiSync.sync()
                stage_columns_swizzled[Self.act_type, Self.NB](
                    taddr,
                    sc,
                    Self.C_ELEMS,
                    h_lane * Self.WT + w_lane,
                    live_lane,
                )
                fence_async_view_proxy()
                Self.EpiSync.sync()
                if tid == 0:
                    comptime for pc in range(Self.NB):
                        cp_async_bulk_tensor_global_shared_cta(
                            sc.unsafe_offset(pc * Self.C_ELEMS),
                            desc,
                            (
                                cty * Self.BN + pc * 32,
                                wtile * Self.WT,
                                htile * Self.HT,
                                dd,
                            ),
                        )
                    cp_async_bulk_commit_group()

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


def conv3d_window_umma_supported[
    act_type: DType, cin: Int, fout: Int, height: Int, width: Int
]() -> Bool:
    """Whether `launch_conv3d_window_umma` serves a channel and plane pair.

    The pair is the VAE input conv the kernel was measured on, at each element
    type.

    Parameters:
        act_type: Element type of the activation, filter and output.
        cin: Input channels.
        fout: Output channels.
        height: Plane height.
        width: Plane width.

    Returns:
        True for the measured shape.
    """
    return (
        (act_type == .float16 or act_type == .bfloat16 or act_type == .float32)
        and cin == 16
        and fout == 384
        and height == 30
        and width == 52
    )


def conv3d_window_umma_fits[
    act_type: DType, fout: Int, height: Int, width: Int
](depth: Int, ctx: DeviceContext) -> Bool:
    """Whether the input conv's grid for `depth` input slices is one wave of
    CTAs, the only extent the kernel was measured on.

    Parameters:
        act_type: Element type of the activation, filter and output.
        fout: Output channels.
        height: Plane height.
        width: Plane width.

    Args:
        depth: Input depth.
        ctx: The device the kernel would run on.

    Returns:
        True when every CTA of the grid is resident at once.
    """
    comptime K = Conv3dWindowUmma[act_type, fout, height, width]
    return ceildiv(depth * K.WBT * K.HBT, K.UNITS) * (fout // K.BN) <= (
        ctx.default_device_info.sm_count
    )


def launch_conv3d_window_umma_packed[
    EpilogueFnType: ConvSimdEpilogueFn,
    ComputeFnType: ConvSimdComputeFn,
    packed_type: DType,
    //,
    act_type: DType,
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
    """Launches the 16-channel 3x3x3 kernel.

    The caller has checked `conv3d_window_umma_supported`, a batch of one,
    dense row-major tensors and 16-byte aligned base pointers, and a grid of
    one wave (`conv3d_window_umma_fits`), and keeps `packed` alive until the
    launch is enqueued.

    Parameters:
        EpilogueFnType: Type of `epilogue_fn` (inferred).
        ComputeFnType: Type of `compute_fn` (inferred).
        act_type: Element type of the activation, filter and output.
        fout: Output channels.
        height: Plane height.
        width: Plane width.
        has_epilogue_fn: `epilogue_fn` stores each output value.
        has_compute_fn: `compute_fn` transforms each output value.

    Args:
        input: Input activation, NDHWC, batch 1, 16 channels.
        packed: The filter packed by `pack_conv3d_filter`.
        output: Output, NDHWC, batch 1; as deep as the input.
        ctx: Device context for the launches.
        epilogue_fn: Storing epilogue, used when `has_epilogue_fn`.
        compute_fn: Transforming epilogue, used when `has_compute_fn`.

    Raises:
        If descriptor creation or a launch fails.
    """
    comptime K = Conv3dWindowUmma[
        act_type, fout, height, width, has_epilogue_fn, has_compute_fn
    ]
    var depth = Int(input.dim[1]())
    var ncta = ceildiv(depth * K.WBT * K.HBT, K.UNITS)
    var grid_y = fout // K.BN

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
        (depth, height, width, 16),
        (height * width * 16, width * 16, 16, 1),
        (1, K.HP, K.WP, K.BK),
    )
    var filt_desc = create_tma_descriptor[packed_type, swizzle_mode=K.SWIZZLE](
        DeviceBuffer(ctx, packed.unsafe_ptr(), 1, owning=False),
        (fout, 27 * 16),
        (27 * 16, 1),
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
        (depth, height, width, fout),
        (height * width * fout, width * fout, fout, 1),
        (1, K.HT, K.WT, 32),
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
        Int32(depth),
        Int32(ncta),
        host_args=(epilogue_fn, compute_fn),
        grid_dim=(ncta, grid_y, 1),
        block_dim=K.NUM_THREADS,
        shared_mem_bytes=K.SMEM_BYTES,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(K.SMEM_BYTES)
        ),
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )


def launch_conv3d_depth_umma[
    EpilogueFnType: ConvSimdEpilogueFn,
    ComputeFnType: ConvSimdComputeFn,
    //,
    act_type: DType,
    cin: Int,
    fout: Int,
    height: Int,
    width: Int,
    taps: Int,
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
) raises -> Bool:
    """Packs the filter and launches the depth conv kernel.

    The caller has checked `conv3d_depth_umma_supported`, a batch of one,
    dense row-major tensors and 16-byte aligned base pointers.

    Parameters:
        EpilogueFnType: Type of `epilogue_fn` (inferred).
        ComputeFnType: Type of `compute_fn` (inferred).
        act_type: Element type of the activation, filter and output.
        cin: Input channels.
        fout: Output channels.
        height: Plane height.
        width: Plane width.
        taps: Depth taps.
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

    Returns:
        False without launching if the grid would need more than one wave of
        CTAs, which the kernel was not measured on.

    Raises:
        If descriptor creation or a launch fails.
    """
    comptime assert (
        filter.dtype == act_type
    ), "activation, filter and output share one element type"
    if not conv3d_depth_umma_fits[act_type, cin, fout, height, width, taps](
        Int(input.dim[1]()), ctx
    ):
        return False
    var packed = pack_conv3d_filter[filter_is_fcrs, taps, cin](
        filter, fout, ctx
    )
    launch_conv3d_depth_umma_packed[
        act_type,
        cin,
        fout,
        height,
        width,
        taps,
        has_epilogue_fn,
        has_compute_fn,
    ](input, packed, output, ctx, epilogue_fn, compute_fn)
    # The descriptor holds the packed filter's address, which Mojo cannot see.
    _ = packed^
    return True


def launch_conv3d_window_umma[
    EpilogueFnType: ConvSimdEpilogueFn,
    ComputeFnType: ConvSimdComputeFn,
    //,
    act_type: DType,
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
) raises -> Bool:
    """Packs the filter and launches the 16-channel 3x3x3 kernel.

    The caller has checked `conv3d_window_umma_supported`, a batch of one,
    dense row-major tensors and 16-byte aligned base pointers.

    Parameters:
        EpilogueFnType: Type of `epilogue_fn` (inferred).
        ComputeFnType: Type of `compute_fn` (inferred).
        act_type: Element type of the activation, filter and output.
        fout: Output channels.
        height: Plane height.
        width: Plane width.
        filter_is_fcrs: The filter is FCQRS rather than QRSCF.
        has_epilogue_fn: `epilogue_fn` stores each output value.
        has_compute_fn: `compute_fn` transforms each output value.

    Args:
        input: Input activation, NDHWC, batch 1, 16 channels.
        filter: Filter, FCQRS or QRSCF.
        output: Output, NDHWC, batch 1; as deep as the input.
        ctx: Device context for the launches.
        epilogue_fn: Storing epilogue, used when `has_epilogue_fn`.
        compute_fn: Transforming epilogue, used when `has_compute_fn`.

    Returns:
        False without launching if the grid would need more than one wave of
        CTAs, which the kernel was not measured on.

    Raises:
        If descriptor creation or a launch fails.
    """
    comptime assert (
        filter.dtype == act_type
    ), "activation, filter and output share one element type"
    if not conv3d_window_umma_fits[act_type, fout, height, width](
        Int(input.dim[1]()), ctx
    ):
        return False
    var packed = pack_conv3d_filter[filter_is_fcrs, 27, 16](filter, fout, ctx)
    launch_conv3d_window_umma_packed[
        act_type, fout, height, width, has_epilogue_fn, has_compute_fn
    ](input, packed, output, ctx, epilogue_fn, compute_fn)
    # The descriptor holds the packed filter's address, which Mojo cannot see.
    _ = packed^
    return True
