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
"""The fp32 3x3 conv2d with three input channels (the VAE `conv_in`), on SIMT FMAs.

Target family: Blackwell sm_100a (B200). The reduction is only
`R * S * C = 27` deep, so a tensor-core tile has little to amortize its
pipeline hand-offs over, and cuDNN's exact-fp32 engine runs the same work on
the FMA pipes. This kernel skips the tile machinery and keeps the whole
filter in registers instead.

A lane owns the output channels `lane, lane + 32, lane + 64` and holds their
`3 * 27` weights for the life of the kernel. A warp walks a contiguous run of
output pixels in chunks of 32:

1. each lane gathers the 27 input values of one pixel (zero outside the
   image) into a row of a warp-private shared-memory patch buffer, so the
   padding and stride arithmetic is paid once per pixel and not once per
   output channel;
2. the warp then visits the 32 pixels in turn. A pixel's patch row is read
   back with broadcast 16-byte loads (every lane wants the same 27 values),
   each lane runs its 81 FMAs, and its three channels are stored. Consecutive
   lanes store consecutive channels, so every store is a full 128-byte line.

The grid is persistent: as many blocks as fit on the device at the residency
below. Each warp's pixel run has the same length to within one pixel, so the
work needs no scheduler and has no tail wave.

The accumulation is fp32 FMA in ascending `(r, s, c)` order, the same class
as cuDNN's exact-fp32 engine. The store epilogue runs once per output element
after the 27-term sum, so nothing of the closure is live across the FMA loop;
`nvvm.minctasm` caps the kernel at 128 registers so two blocks share an SM.

The envelope is `conv2d_fp32_small_c_supported`.
"""

from std.math import align_up, fma
from std.sys import align_of
from std.utils.static_tuple import StaticTuple

from layout import (
    TensorEngine,
    TensorLayout,
    TileTensor,
    row_major,
    stack_allocation as tt_stack_allocation,
)
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    grid_dim,
    lane_id,
    thread_idx,
)
from max.gpu.host import DeviceContext
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from max.gpu.sync import barrier, syncwarp

from nn.conv.conv_utils import ConvSimdEpilogueFn

comptime _C = 3
"""Input channels the kernel serves."""
comptime _R = 3
comptime _S = 3
"""Filter extent the kernel serves."""
comptime _NG = 3
"""Output channels per lane; the kernel serves `_NG * WARP_SIZE` channels."""
comptime _F = _NG * WARP_SIZE
comptime _K = _R * _S * _C
"""Reduction depth."""
comptime _PITCH = align_up(_K, 4)
"""Patch row length: `_K` rounded up to whole 16-byte vectors."""
comptime _BLOCK = 256
"""Threads per block; eight warps."""
comptime _WARPS = _BLOCK // WARP_SIZE
comptime _BLOCKS_PER_SM = 2
"""Residency the register budget is sized for: 65536 / (2 * 256) = 128."""


def conv2d_fp32_small_c_supported(
    in_channels: Int,
    out_channels: Int,
    filter_h: Int,
    filter_w: Int,
) -> Bool:
    """Whether the kernel serves a convolution.

    The geometry is fixed at compile time (3 -> 96 channels, 3x3); stride,
    dilation and padding are runtime values.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        filter_h: Filter height.
        filter_w: Filter width.

    Returns:
        True if the kernel serves the convolution.
    """
    return (
        in_channels == _C
        and out_channels == _F
        and filter_h == _R
        and filter_w == _S
    )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_BLOCK))
)
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(_BLOCKS_PER_SM))
@__name(t"conv2d_fp32_small_c_{input_type}_{output_type}")
def conv2d_fp32_small_c_kernel[
    input_layout: TensorLayout,
    filter_layout: TensorLayout,
    output_layout: TensorLayout,
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    input_origin: ImmOrigin,
    filter_origin: ImmOrigin,
    output_origin: MutOrigin,
    EpilogueFnType: ConvSimdEpilogueFn,
    InputEngine: TensorEngine,
    FilterEngine: TensorEngine,
    OutputEngine: TensorEngine,
    has_epilogue_fn: Bool,
](
    input: TileTensor[
        input_type, input_layout, input_origin, Engine=InputEngine
    ],
    filter: TileTensor[
        filter_type, filter_layout, filter_origin, Engine=FilterEngine
    ],
    output: TileTensor[
        output_type, output_layout, output_origin, Engine=OutputEngine
    ],
    stride_h: Int32,
    stride_w: Int32,
    dilation_h: Int32,
    dilation_w: Int32,
    pad_h: Int32,
    pad_w: Int32,
    epilogue_fn: EpilogueFnType,
):
    """Convolves an NHWC fp32 activation with an FCRS 3 -> 96 channel 3x3
    filter.

    Parameters:
        input_layout: Memory layout of the input tensor.
        filter_layout: Memory layout of the filter tensor.
        output_layout: Memory layout of the output tensor.
        input_type: Element type of the input tensor.
        filter_type: Element type of the filter tensor.
        output_type: Element type of the output tensor.
        input_origin: Origin of the input tensor.
        filter_origin: Origin of the filter tensor.
        output_origin: Origin of the output tensor.
        EpilogueFnType: Type of `epilogue_fn`.
        InputEngine: Storage engine of the input tensor.
        FilterEngine: Storage engine of the filter tensor.
        OutputEngine: Storage engine of the output tensor.
        has_epilogue_fn: `epilogue_fn` stores each output element in place of
            the kernel.

    Args:
        input: Input activation, NHWC.
        filter: Filter, FCRS.
        output: Output, NHWC.
        stride_h: Stride along the height.
        stride_w: Stride along the width.
        dilation_h: Dilation along the height.
        dilation_w: Dilation along the width.
        pad_h: Symmetric padding along the height.
        pad_w: Symmetric padding along the width.
        epilogue_fn: Storing epilogue, used when `has_epilogue_fn`.
    """
    comptime acc_type = DType.float32

    var in_h = Int(input.dim[1]())
    var in_w = Int(input.dim[2]())
    var out_h = Int(output.dim[1]())
    var out_w = Int(output.dim[2]())
    var total = Int(input.dim[0]()) * out_h * out_w
    var sh = Int(stride_h)
    var sw = Int(stride_w)
    var dh = Int(dilation_h)
    var dw = Int(dilation_w)
    var ph = Int(pad_h)
    var pw = Int(pad_w)

    var tid = thread_idx.x
    var lane = Int(lane_id())
    var wid = tid // WARP_SIZE

    # The filter may be the output of the grid launched just before this one
    # (a weight permute or cast in the same graph), so the wait precedes the
    # staging and not only the first activation load.
    wait_on_dependent_grids()

    var patch = tt_stack_allocation[
        dtype=acc_type, address_space=.SHARED, alignment=16
    ](row_major[_BLOCK, _PITCH]())
    var patch_v = patch.vectorize[1, 4]()

    # The filter, transposed to [K, F] so a lane's channel is the fast index
    # of the reads below. It borrows the patch buffer, which the gather does
    # not touch until every lane holds its weights. A row is one word longer
    # than F, which spreads the transposing stores over the banks.
    comptime assert _K * (_F + 1) <= _BLOCK * _PITCH, "filter fits the patch"
    var sfilt = TileTensor(patch.unsafe_ptr(), row_major[_K, _F + 1]())
    for i in range(tid, _F * _K, _BLOCK):
        var f, rem = divmod(i, _K)
        var c, rs = divmod(rem, _R * _S)
        var r, s = divmod(rs, _S)
        sfilt[rs * _C + c, f] = filter.load((f, c, r, s))[0].cast[acc_type]()

    barrier()

    var w = tt_stack_allocation[dtype=acc_type, address_space=.LOCAL](
        row_major[_NG, _K]()
    )
    comptime for g in range(_NG):
        comptime for k in range(_K):
            w[g, k] = sfilt[k, g * WARP_SIZE + lane]
    barrier()

    # This warp's pixel run: the runs differ in length by at most one.
    var gwarp = block_idx.x * _WARPS + wid
    var nwarps = grid_dim.x * _WARPS
    var lo = gwarp * total // nwarps
    var hi = (gwarp + 1) * total // nwarps

    var base = lo
    while base < hi:
        var cnt = min(WARP_SIZE, hi - base)

        # Gather: this lane's pixel into its own patch row, one 16-byte
        # vector at a time (a 28-word row pitch makes a scalar store a
        # four-way bank conflict; the vector store is not).
        if lane < cnt:
            var n, rem = divmod(base + lane, out_h * out_w)
            var ho, wo = divmod(rem, out_w)
            comptime for kv in range(_PITCH // 4):
                var vec = SIMD[acc_type, 4](0)
                comptime for e in range(4):
                    comptime k = kv * 4 + e
                    comptime if k < _K:
                        comptime r = k // (_S * _C)
                        comptime s = (k // _C) % _S
                        comptime c = k % _C
                        var hi_r = ho * sh - ph + r * dh
                        var wi_s = wo * sw - pw + s * dw
                        if (
                            hi_r >= 0
                            and hi_r < in_h
                            and wi_s >= 0
                            and wi_s < in_w
                        ):
                            vec[e] = input.load((n, hi_r, wi_s, c))[0].cast[
                                acc_type
                            ]()
                patch_v[tid, kv] = vec
        syncwarp()

        var n, rem = divmod(base, out_h * out_w)
        var ho, wo = divmod(rem, out_w)
        for j in range(cnt):
            var row = wid * WARP_SIZE + j
            var acc = tt_stack_allocation[dtype=acc_type, address_space=.LOCAL](
                row_major[1, _NG]()
            ).fill(0)
            comptime for kv in range(_PITCH // 4):
                var pv = patch_v[row, kv]
                comptime for e in range(4):
                    comptime k = kv * 4 + e
                    comptime if k < _K:
                        comptime for g in range(_NG):
                            acc[0, g] = fma(pv[e], w[g, k], acc[0, g])

            comptime for g in range(_NG):
                var co = g * WARP_SIZE + lane
                var val = acc[0, g].cast[output_type]()
                comptime if has_epilogue_fn:
                    epilogue_fn[alignment=align_of[output_type]()](
                        (n, ho, wo, co), val
                    )
                else:
                    output.store((n, ho, wo, co), val)

            wo += 1
            if wo == out_w:
                wo = 0
                ho += 1
                if ho == out_h:
                    ho = 0
                    n += 1
        syncwarp()
        base += cnt

    launch_dependent_grids()


def conv2d_fp32_small_c[
    EpilogueFnType: ConvSimdEpilogueFn,
    //,
    has_epilogue_fn: Bool,
](
    input: TileTensor[mut=True, address_space=.GENERIC, ...],
    filter: TileTensor[address_space=.GENERIC, ...],
    output: TileTensor[mut=True, address_space=.GENERIC, ...],
    stride_h: Int,
    stride_w: Int,
    dilation_h: Int,
    dilation_w: Int,
    pad_h: Int,
    pad_w: Int,
    ctx: DeviceContext,
    epilogue_fn: EpilogueFnType,
) raises:
    """Launches the kernel on an NHWC activation and an FCRS filter.

    The caller has checked `conv2d_fp32_small_c_supported`.

    Parameters:
        EpilogueFnType: Type of `epilogue_fn` (inferred).
        has_epilogue_fn: `epilogue_fn` stores each output element in place of
            the kernel.

    Args:
        input: Input activation, NHWC.
        filter: Filter, FCRS.
        output: Output, NHWC.
        stride_h: Stride along the height.
        stride_w: Stride along the width.
        dilation_h: Dilation along the height.
        dilation_w: Dilation along the width.
        pad_h: Symmetric padding along the height.
        pad_w: Symmetric padding along the width.
        ctx: Device context for the launch.
        epilogue_fn: Storing epilogue, used when `has_epilogue_fn`.

    Raises:
        If the launch fails.
    """
    comptime assert (
        input.dtype == .float32
        and filter.dtype == .float32
        and output.dtype == .float32
    ), "the kernel accumulates and stores fp32"
    comptime assert input.flat_rank == 4, "input must be rank 4 (NHWC)"
    comptime assert filter.flat_rank == 4, "filter must be rank 4 (FCRS)"
    comptime assert output.flat_rank == 4, "output must be rank 4 (NHWC)"

    comptime kernel = conv2d_fp32_small_c_kernel[
        input.LayoutType,
        filter.LayoutType,
        output.LayoutType,
        input.dtype,
        filter.dtype,
        output.dtype,
        ImmOrigin(input.origin),
        ImmOrigin(filter.origin),
        output.origin,
        EpilogueFnType,
        input.Engine,
        filter.Engine,
        output.Engine,
        has_epilogue_fn,
    ]
    var blocks = ctx.default_device_info.sm_count * _BLOCKS_PER_SM
    ctx.enqueue_function[kernel](
        input.as_imm(),
        filter.as_imm(),
        output,
        Int32(stride_h),
        Int32(stride_w),
        Int32(dilation_h),
        Int32(dilation_w),
        Int32(pad_h),
        Int32(pad_w),
        host_args=(epilogue_fn,),
        grid_dim=blocks,
        block_dim=_BLOCK,
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )
