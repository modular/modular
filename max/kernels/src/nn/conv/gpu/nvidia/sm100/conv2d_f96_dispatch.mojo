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
"""Entry point that sends an fp16 conv2d on the 96 output channel family to
the SM100 `conv2d_f96_fprop` kernel.

Target family: Blackwell sm_100a. `conv_gpu` sends fp16 to cuDNN unless a
dedicated kernel serves the problem; this routes the 192 -> 96 and 96 -> 96
channel 3x3 shapes `conv2d_f96_supported` accepts. The kernel reads a KRSC
filter, so the caller's FCRS or RSCF filter is packed into a scratch buffer
first, the same pack the bf16 route does in `dispatch.mojo`.

The convolution's fused residual, storing epilogue and compute epilogue are
served too, with the closures' coordinates and arithmetic dtype chosen by
`nhwc_coords` and `round_first` (see `conv_epilogue.apply_conv_epilogue`).
"""

from std.collections import OptionalReg
from std.math import ceildiv

from layout import TileTensor, row_major
from linalg.utils import (
    ElementwiseComputeFn,
    ElementwiseEpilogueFn,
    no_compute_fn,
    no_epilogue_fn,
)
from max.gpu.host import DeviceContext
from max.gpu.primitives.grid_controls import PDLLevel, pdl_launch_attributes

from .conv2d_f96_fprop import conv2d_f96_fprop, conv2d_f96_supported
from .dedicated_fprop import dedicated_problem_shape
from .dispatch import _transpose_fcrs_to_krsc, _transpose_rscf_to_krsc


def conv2d_f96_dispatch[
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
    """Runs the convolution on the 96 output channel kernel if it serves the
    problem.

    Parameters:
        input_type: Element type of the activation.
        filter_type: Element type of the filter.
        output_type: Element type of the output.
        EpilogueFnType: Type of `epilogue_fn` (inferred).
        ComputeFnType: Type of `compute_fn` (inferred).
        filter_is_fcrs: Whether `filter` is FCRS `[F, C, R, S]` rather than
            RSCF `[R, S, C, F]`.
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
        True if the kernel ran; False, with nothing enqueued, if the problem
        or the operand layouts are outside what it serves.

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
        var problem = dedicated_problem_shape[filter_is_fcrs](
            input, filter, output, pad_h, pad_w
        )
        if not conv2d_f96_supported[input_type](problem, ctx):
            return False

        var filter_size = filter.num_elements()
        var krsc = ctx.enqueue_create_buffer[filter_type](filter_size)
        comptime transpose_block = 256
        var grid = ceildiv(filter_size, transpose_block)
        comptime if filter_is_fcrs:
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
                grid_dim=grid,
                block_dim=transpose_block,
                attributes=pdl_launch_attributes(PDLLevel.ON),
            )
        else:
            comptime pack = _transpose_rscf_to_krsc[
                filter_type,
                filter.LayoutType,
                ImmOrigin(filter.origin),
                filter.Engine,
            ]
            ctx.enqueue_function[pack](
                filter.as_imm(),
                krsc.unsafe_ptr(),
                Int32(problem.filter_h),
                Int32(problem.filter_w),
                Int32(problem.in_channels),
                Int32(problem.out_channels),
                grid_dim=grid,
                block_dim=transpose_block,
                attributes=pdl_launch_attributes(PDLLevel.ON),
            )
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
        var filt_tt = TileTensor(
            krsc,
            row_major(
                problem.out_channels,
                problem.filter_h,
                problem.filter_w,
                problem.in_channels,
            ),
        )
        conv2d_f96_fprop[
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
