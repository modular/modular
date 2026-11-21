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
"""Dispatch of the SM100 3-D conv kernels from `conv_gpu`.

Target family: Blackwell sm_100a (B200). Each kernel serves the exact channel
counts, spatial planes and element types it was measured to beat the previous
path on; every other convolution declines and keeps its path. The paths they
replace:

- An FCQRS filter ran cuDNN (`conv3d_cudnn`), with tf32 conversion allowed.
- A bf16 QRSCF filter ran the matmul route for the 1x1x1 conv, the Q-slice
  route for the 384-channel residual block and the time conv, and the
  im2col route for the input conv and the other residual blocks.
- An fp16 or fp32 QRSCF filter ran the naive kernel. fp32 QRSCF stays on it:
  the tensor-core kernels round fp32 operands to tf32.

The fused store and compute epilogues apply through `conv_gpu` for a QRSCF
filter only; `conv_gpu` still rejects an FCQRS filter with a store epilogue.

| kernel                  | convolution                                    |
|-------------------------|------------------------------------------------|
| `conv3d_pointwise`      | 1x1x1, 16 -> 16 channels, 30 x 52 (exact fp32 |
|                         | FMA); not a bf16 QRSCF filter: a tie with the  |
|                         | matmul route                                   |
| `conv3d_col_umma`       | 3x3x3, 96 -> 96 on 240 x 416, 192 -> 192 on    |
|                         | 120 x 208, 384 -> 384 on 30 x 52               |
| `conv3d_tap_umma`       | 3x1x1, 192 -> 384 on 60 x 104; 3x3x3,          |
|                         | 16 -> 384 on 30 x 52                           |

All of them take stride 1, dilation 1, one group and a batch of one. The
residual blocks take padding `(0, 1, 1)` (the causal convs arrive pre-padded
along the depth), the input conv `(1, 1, 1)` and the time conv none.

The tensor-core kernels also need a minimum number of output frames
(`_deep_enough`, up to six depending on the kernel and epilogue): with a few
frames they launch a handful of CTAs, and a fused permute-and-bias epilogue is
then slower than on the previous path.
"""

from max.gpu.host import DeviceContext
from max.gpu.host.info import B200

from layout import Coord, TileTensor
from nn.conv.conv_utils import (
    ConvSimdComputeFn,
    ConvSimdEpilogueFn,
    no_conv_simd_compute_fn,
    no_conv_simd_epilogue_fn,
)

from .conv3d_col_umma import (
    conv3d_col_umma_supported,
    launch_conv3d_col_umma,
)
from .conv3d_pointwise import dispatch_conv3d_pointwise_sm100
from .conv3d_tap_umma import (
    launch_conv3d_depth_umma,
    launch_conv3d_window_umma,
)


def _is_dense_aligned[dtype: DType](tensor: TileTensor[dtype, ...]) -> Bool:
    """Whether the tensor is dense row-major at run time and 32-byte aligned.

    `is_row_major` only says the stride types are row-major ones; a view with
    dynamic strides (a slice of a larger tensor) still has to be checked.
    The 32 bytes cover the widest access of the fused epilogue (eight fp32
    channels).
    """
    if not tensor.is_row_major or Int(tensor.unsafe_ptr()) % 32 != 0:
        return False
    var expected = 1
    comptime for k in range(tensor.flat_rank):
        comptime i = tensor.flat_rank - 1 - k
        var extent = Int(tensor.dim[i]())
        if extent != 1 and Int(tensor.dynamic_stride(i)) != expected:
            return False
        expected *= extent
    return True


def _deep_enough(
    out_depth: Int,
    plain_min: Int,
    fused_min: Int,
    fcqrs_min: Int,
    fused: Bool,
    filter_is_fcrs: Bool,
) -> Bool:
    """Whether a conv with this many output frames is past the depth where the
    kernel beats the previous path.

    A conv with few output frames launches few CTAs, each running the whole
    reduction, and a fused epilogue then adds a latency that no other CTA
    hides. Each minimum is the smallest output depth of a measured sweep (bf16,
    p50 of 30, worst of seeds 1/2/3, route on against off) from which the
    kernel is faster than the previous path by more than one 2.048 us graph
    tick at every deeper depth measured. A QRSCF filter replaces the Q-slice,
    im2col and naive routes (`plain_min`, `fused_min`); an FCQRS filter
    replaces cuDNN, which is much faster at a few frames (`fcqrs_min`). The
    sweep ran in bf16; fp16 and fp32 use the same minimums unmeasured.
    """
    if filter_is_fcrs:
        return out_depth >= fcqrs_min
    return out_depth >= (fused_min if fused else plain_min)


def dispatch_conv3d_sm100[
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    EpilogueFnType: ConvSimdEpilogueFn = type_of(no_conv_simd_epilogue_fn),
    ComputeFnType: ConvSimdComputeFn = type_of(no_conv_simd_compute_fn),
    //,
    filter_is_fcrs: Bool = False,
    has_epilogue_fn: Bool = False,
    has_compute_fn: Bool = False,
](
    input: TileTensor[mut=True, input_type, address_space=.GENERIC, ...],
    filter: TileTensor[filter_type, address_space=.GENERIC, ...],
    output: TileTensor[mut=True, output_type, address_space=.GENERIC, ...],
    stride: Coord[Int, Int, Int],
    dilation: Coord[Int, Int, Int],
    symmetric_padding: Coord[Int, Int, Int],
    num_groups: Int,
    ctx: DeviceContext,
    epilogue_fn: EpilogueFnType = no_conv_simd_epilogue_fn,
    compute_fn: ComputeFnType = no_conv_simd_compute_fn,
) raises -> Bool:
    """Tries to run a 3-D forward convolution on an SM100 kernel.

    Declines (returns `False`) on any other device, element type, filter or
    plane than the ones the kernels were measured on.

    Parameters:
        input_type: Element type of the input tensor (inferred).
        filter_type: Element type of the filter tensor (inferred).
        output_type: Element type of the output tensor (inferred).
        EpilogueFnType: Type of `epilogue_fn` (inferred).
        ComputeFnType: Type of `compute_fn` (inferred).
        filter_is_fcrs: The filter is FCQRS rather than QRSCF.
        has_epilogue_fn: `epilogue_fn` stores each output value.
        has_compute_fn: `compute_fn` transforms each output value.

    Args:
        input: Input activation, NDHWC.
        filter: Filter, FCQRS or QRSCF with a static shape.
        output: Output, NDHWC.
        stride: Per-axis stride.
        dilation: Per-axis dilation.
        symmetric_padding: Per-axis padding.
        num_groups: Convolution groups.
        ctx: Device context for the launches.
        epilogue_fn: Storing epilogue, used when `has_epilogue_fn`.
        compute_fn: Transforming epilogue, used when `has_compute_fn`.

    Returns:
        True if a kernel ran the convolution.
    """
    comptime assert input.flat_rank == 5, "input must be rank 5 (NDHWC)"
    comptime assert filter.flat_rank == 5, "filter must be rank 5"
    comptime assert output.flat_rank == 5, "output must be rank 5 (NDHWC)"
    comptime assert not (
        has_epilogue_fn and has_compute_fn
    ), "a store epilogue excludes the compute epilogue"

    comptime if (
        not (ctx.default_device_info == B200)
        or input_type != output_type
        or filter_type != input_type
        or not filter.shape_known
        or (
            input_type != .float32
            and input_type != .float16
            and input_type != .bfloat16
        )
    ):
        return False
    else:
        if (
            num_groups != 1
            or stride[0] != 1
            or stride[1] != 1
            or stride[2] != 1
            or dilation[0] != 1
            or dilation[1] != 1
            or dilation[2] != 1
            or Int(input.dim[0]()) != 1
            or Int(output.dim[0]()) != 1
            or not _is_dense_aligned(input)
            or not _is_dense_aligned(output)
            or not _is_dense_aligned(filter)
        ):
            return False

        if dispatch_conv3d_pointwise_sm100[
            filter_is_fcrs,
            has_epilogue_fn=has_epilogue_fn,
            has_compute_fn=has_compute_fn,
        ](
            input,
            filter,
            output,
            stride,
            dilation,
            symmetric_padding,
            num_groups,
            ctx,
            epilogue_fn,
            compute_fn,
        ):
            return True

        # The tensor-core arms round fp32 operands to tf32. An fp32 FCQRS
        # filter already ran on cuDNN with that conversion allowed; an fp32
        # QRSCF filter ran the exact fp32 naive kernel, so it keeps it.
        comptime if input_type == .float32 and not filter_is_fcrs:
            return False

        comptime fq = filter.static_shape[2 if filter_is_fcrs else 0]
        comptime fr = filter.static_shape[3 if filter_is_fcrs else 1]
        comptime fs = filter.static_shape[4 if filter_is_fcrs else 2]
        comptime fc = filter.static_shape[1 if filter_is_fcrs else 3]
        comptime ff = filter.static_shape[0 if filter_is_fcrs else 4]

        comptime if fq == 3 and fr == 3 and fs == 3:
            return _dispatch_3x3x3[
                input_type,
                fc,
                ff,
                filter_is_fcrs,
                has_epilogue_fn,
                has_compute_fn,
            ](
                input,
                filter,
                output,
                symmetric_padding,
                ctx,
                epilogue_fn,
                compute_fn,
            )
        elif fq == 3 and fr == 1 and fs == 1:
            return _dispatch_depth[
                input_type,
                fc,
                ff,
                filter_is_fcrs,
                has_epilogue_fn,
                has_compute_fn,
            ](
                input,
                filter,
                output,
                symmetric_padding,
                ctx,
                epilogue_fn,
                compute_fn,
            )
        return False


def _same_plane(
    input: TileTensor[...], output: TileTensor[...], height: Int, width: Int
) -> Bool:
    return (
        Int(input.dim[2]()) == height
        and Int(input.dim[3]()) == width
        and Int(output.dim[2]()) == height
        and Int(output.dim[3]()) == width
    )


def _dispatch_3x3x3[
    EpilogueFnType: ConvSimdEpilogueFn,
    ComputeFnType: ConvSimdComputeFn,
    //,
    dt: DType,
    cin: Int,
    fout: Int,
    filter_is_fcrs: Bool,
    has_epilogue_fn: Bool,
    has_compute_fn: Bool,
](
    input: TileTensor[mut=True, dt, address_space=.GENERIC, ...],
    filter: TileTensor[address_space=.GENERIC, ...],
    output: TileTensor[mut=True, address_space=.GENERIC, ...],
    symmetric_padding: Coord[Int, Int, Int],
    ctx: DeviceContext,
    epilogue_fn: EpilogueFnType,
    compute_fn: ComputeFnType,
) raises -> Bool:
    """The 3x3x3 kernels, by channel counts. The caller has checked the
    device, the element types and the unit strides, dilations and groups."""
    comptime assert (
        filter.dtype == dt and output.dtype == dt
    ), "activation, filter and output share one element type"
    if (
        Int(input.dim[4]()) != cin
        or Int(output.dim[4]()) != fout
        or Int(input.dim[1]()) < 3
    ):
        return False
    comptime fused = has_epilogue_fn or has_compute_fn
    var out_depth = Int(output.dim[1]())
    var pad_ok_col = (
        symmetric_padding[0] == 0
        and symmetric_padding[1] == 1
        and symmetric_padding[2] == 1
        and Int(output.dim[1]()) == Int(input.dim[1]()) - 2
    )

    # The residual blocks: 3x3x3, no padding along the depth.
    comptime if cin == 96 and fout == 96:
        if pad_ok_col and _same_plane(input, output, 240, 416):
            launch_conv3d_col_umma[
                dt,
                96,
                96,
                240,
                416,
                filter_is_fcrs,
                has_epilogue_fn,
                has_compute_fn,
            ](input, filter, output, ctx, epilogue_fn, compute_fn)
            return True
    elif cin == 192 and fout == 192:
        if (
            pad_ok_col
            and _same_plane(input, output, 120, 208)
            and _deep_enough(out_depth, 1, 4, 4, fused, filter_is_fcrs)
        ):
            launch_conv3d_col_umma[
                dt,
                192,
                192,
                120,
                208,
                filter_is_fcrs,
                has_epilogue_fn,
                has_compute_fn,
            ](input, filter, output, ctx, epilogue_fn, compute_fn)
            return True
    elif cin == 384 and fout == 384:
        if (
            pad_ok_col
            and _same_plane(input, output, 30, 52)
            and _deep_enough(out_depth, 1, 4, 21, fused, filter_is_fcrs)
        ):
            launch_conv3d_col_umma[
                dt,
                384,
                384,
                30,
                52,
                filter_is_fcrs,
                has_epilogue_fn,
                has_compute_fn,
            ](input, filter, output, ctx, epilogue_fn, compute_fn)
            return True
    # The input conv: 16 channels, padding 1 along every axis.
    elif cin == 16 and fout == 384:
        if (
            symmetric_padding[0] == 1
            and symmetric_padding[1] == 1
            and symmetric_padding[2] == 1
            and Int(output.dim[1]()) == Int(input.dim[1]())
            and _same_plane(input, output, 30, 52)
            and _deep_enough(out_depth, 4, 6, 6, fused, filter_is_fcrs)
        ):
            return launch_conv3d_window_umma[
                dt,
                384,
                30,
                52,
                filter_is_fcrs,
                has_epilogue_fn,
                has_compute_fn,
            ](input, filter, output, ctx, epilogue_fn, compute_fn)
    return False


def _dispatch_depth[
    EpilogueFnType: ConvSimdEpilogueFn,
    ComputeFnType: ConvSimdComputeFn,
    //,
    dt: DType,
    cin: Int,
    fout: Int,
    filter_is_fcrs: Bool,
    has_epilogue_fn: Bool,
    has_compute_fn: Bool,
](
    input: TileTensor[mut=True, dt, address_space=.GENERIC, ...],
    filter: TileTensor[address_space=.GENERIC, ...],
    output: TileTensor[mut=True, address_space=.GENERIC, ...],
    symmetric_padding: Coord[Int, Int, Int],
    ctx: DeviceContext,
    epilogue_fn: EpilogueFnType,
    compute_fn: ComputeFnType,
) raises -> Bool:
    """The time conv (`3 x 1 x 1`, no padding). The caller has checked the
    device, the element types and the unit strides, dilations and groups."""
    comptime assert (
        filter.dtype == dt and output.dtype == dt
    ), "activation, filter and output share one element type"
    comptime if cin == 192 and fout == 384:
        if (
            symmetric_padding[0] == 0
            and symmetric_padding[1] == 0
            and symmetric_padding[2] == 0
            and Int(input.dim[4]()) == cin
            and Int(output.dim[4]()) == fout
            and Int(input.dim[1]()) >= 3
            and Int(output.dim[1]()) == Int(input.dim[1]()) - 2
            and _same_plane(input, output, 60, 104)
            and _deep_enough(
                Int(output.dim[1]()),
                4,
                4,
                21,
                has_epilogue_fn or has_compute_fn,
                filter_is_fcrs,
            )
        ):
            return launch_conv3d_depth_umma[
                dt,
                192,
                384,
                60,
                104,
                3,
                filter_is_fcrs,
                has_epilogue_fn,
                has_compute_fn,
            ](input, filter, output, ctx, epilogue_fn, compute_fn)
    return False
