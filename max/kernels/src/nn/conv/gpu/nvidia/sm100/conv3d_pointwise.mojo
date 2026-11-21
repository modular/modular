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
"""SM100 1x1x1 conv3d for 16 -> 16 channels (the VAE `post_quant_conv`).

Target family: Blackwell sm_100a (B200), fp32, fp16 and bf16, fp32 accumulate
on the FMA pipes.

The product is 16 x 16 per output row, so the kernel is bound by the replay
latency of a launch, not by arithmetic: a tensor-core kernel pays for its
barrier and TMEM setup before it has done any of the work, and the matmul
route pays for a library launch. This kernel is the thinnest chain that does
the job. `_THREADS_PER_ROW` threads share an output row; each loads the whole
16-channel row with 16-byte vectors, keeps it in registers, and computes
`_F / _THREADS_PER_ROW` outputs against a filter held in shared memory as
fp32. The filter staging runs before the grid-dependency wait so it overlaps
the previous launch; the activation is read after it, because the previous
node may be what produced it.

The filter is read in either layout: QRSCF `[1, 1, 1, C, F]` is already the
`[C, F]` the staging wants, and FCQRS `[F, C, 1, 1, 1]` is transposed by the
staging loop's index arithmetic.

The accumulation order is `c = 0 .. 15` in fp32 for every layout.
"""

from std.math import ceildiv, fma
from std.sys import align_of, size_of

from layout import (
    Coord,
    TensorEngine,
    TensorLayout,
    TileTensor,
    row_major,
)
from layout import stack_allocation as tt_stack_allocation
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.host.info import B200
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from max.gpu.sync import barrier

from nn.conv.conv_utils import (
    ConvSimdComputeFn,
    ConvSimdEpilogueFn,
    no_conv_simd_compute_fn,
    no_conv_simd_epilogue_fn,
)

comptime _C = 16
"""Input channels the kernel serves."""
comptime _F = 16
"""Output channels the kernel serves."""
comptime _THREADS_PER_ROW = 2
"""Threads that share one output row. Two halve each thread's FMA and store
chain; at fp32 (twice the bytes per row) one thread per row leaves a heavy
tail on the replay clock."""
comptime _FP = _F // _THREADS_PER_ROW
"""Output channels one thread computes."""
comptime _THREADS = 128
"""Threads per block."""
comptime _H_MEASURED = 30
comptime _W_MEASURED = 52
"""The spatial plane the kernel was measured on (the VAE `post_quant_conv`)."""


@__name(t"conv3d_pointwise_{input_type}_{output_type}")
def conv3d_pointwise_kernel[
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
    ComputeFnType: ConvSimdComputeFn,
    InputEngine: TensorEngine,
    FilterEngine: TensorEngine,
    OutputEngine: TensorEngine,
    filter_is_fcrs: Bool,
    has_epilogue_fn: Bool,
    has_compute_fn: Bool,
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
    rows: Int64,
    d_out: Int32,
    plane: Int32,
    out_w: Int32,
    epilogue_fn: EpilogueFnType,
    compute_fn: ComputeFnType,
):
    """Convolves an NDHWC activation with a 1x1x1, 16 -> 16 channel filter.

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
        ComputeFnType: Type of `compute_fn`.
        InputEngine: Storage engine of the input tensor.
        FilterEngine: Storage engine of the filter tensor.
        OutputEngine: Storage engine of the output tensor.
        filter_is_fcrs: The filter is FCQRS rather than QRSCF.
        has_epilogue_fn: `epilogue_fn` stores each output value in place of
            the kernel.
        has_compute_fn: `compute_fn` transforms each output value before the
            kernel stores it.

    Args:
        input: Input activation, NDHWC, dense and 16-byte aligned.
        filter: Filter, 16 -> 16 channels.
        output: Output, NDHWC, dense and 16-byte aligned.
        rows: Output rows, `N * D * H * W`.
        d_out: Output depth.
        plane: Output rows per depth slice, `H * W`.
        out_w: Output width.
        epilogue_fn: Storing epilogue, used when `has_epilogue_fn`.
        compute_fn: Transforming epilogue, used when `has_compute_fn`.
    """
    comptime WI = 16 // size_of[input_type]()
    comptime WO = 16 // size_of[output_type]()
    comptime assert _C % WI == 0 and _FP % WO == 0
    # The flat row offsets below index a dense row-major tensor.
    comptime assert input.is_row_major and output.is_row_major
    comptime assert filter.is_row_major

    var w = tt_stack_allocation[
        dtype=.float32, address_space=.SHARED, alignment=16
    ](row_major[_C, _F]())
    var w_ptr = w.unsafe_ptr()
    var filter_ptr = filter.unsafe_ptr()
    var input_ptr = input.unsafe_ptr()
    var output_ptr = output.unsafe_ptr()
    var tid = thread_idx.x
    for j in range(tid, _C * _F, block_dim.x):
        var value = filter_ptr.unsafe_load(j).cast[.float32]()
        comptime if filter_is_fcrs:
            var f, c = divmod(j, _C)
            w_ptr.unsafe_store(c * _F + f, value)
        else:
            w_ptr.unsafe_store(j, value)
    barrier()

    # The activation may be the output of the node launched just before.
    wait_on_dependent_grids()

    var gid = block_idx.x * block_dim.x + tid
    var row, lane = divmod(gid, _THREADS_PER_ROW)
    var f0 = lane * _FP
    if Int64(row) < rows:
        var x = SIMD[.float32, _C](0)
        comptime for h in range(_C // WI):
            var v = input_ptr.unsafe_load[width=WI, alignment=16](
                row * _C + h * WI
            )
            comptime for e in range(WI):
                x[h * WI + e] = v[e].cast[.float32]()
        var acc = SIMD[.float32, _FP](0)
        comptime for c in range(_C):
            acc = fma(
                SIMD[.float32, _FP](x[c]),
                w_ptr.unsafe_load[width=_FP, alignment=16](c * _F + f0),
                acc,
            )

        comptime for h in range(_FP // WO):
            var o = SIMD[output_type, WO](0)
            comptime for e in range(WO):
                o[e] = acc[h * WO + e].cast[output_type]()
            comptime if has_epilogue_fn or has_compute_fn:
                var n_d, in_plane = divmod(row, Int(plane))
                var n, d = divmod(n_d, Int(d_out))
                var y, x_w = divmod(in_plane, Int(out_w))
                var coords = (n, d, y, x_w, f0 + h * WO)
                comptime if has_epilogue_fn:
                    epilogue_fn[alignment=16](coords, o)
                else:
                    output_ptr.unsafe_store[alignment=16](
                        row * _F + f0 + h * WO,
                        compute_fn[alignment=16](coords, o),
                    )
            else:
                output_ptr.unsafe_store[alignment=16](row * _F + f0 + h * WO, o)

    # Release the dependent launch once every store of the block is done.
    barrier()
    launch_dependent_grids()


def dispatch_conv3d_pointwise_sm100[
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
    input: TileTensor[input_type, ...],
    filter: TileTensor[filter_type, ...],
    output: TileTensor[mut=True, output_type, ...],
    stride: Coord[Int, Int, Int],
    dilation: Coord[Int, Int, Int],
    symmetric_padding: Coord[Int, Int, Int],
    num_groups: Int,
    ctx: DeviceContext,
    epilogue_fn: EpilogueFnType = no_conv_simd_epilogue_fn,
    compute_fn: ComputeFnType = no_conv_simd_compute_fn,
) raises -> Bool:
    """Tries to run a 16 -> 16 channel 1x1x1 conv3d on the pointwise kernel.

    Declines (returns `False`) unless the device is a B200, the three tensors
    share one element type and are dense row-major, the filter is 1x1x1 with
    16 input and 16 output channels, stride and dilation are 1, there is no
    padding or grouping, and the spatial plane is the 30 x 52 one the kernel
    was measured on.

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
        filter: Filter, FCQRS or QRSCF.
        output: Output, NDHWC.
        stride: Per-axis stride; must be 1.
        dilation: Per-axis dilation; must be 1.
        symmetric_padding: Per-axis padding; must be 0.
        num_groups: Convolution groups; must be 1.
        ctx: Device context for the launch.
        epilogue_fn: Storing epilogue, used when `has_epilogue_fn`.
        compute_fn: Transforming epilogue, used when `has_compute_fn`.

    Returns:
        True if the kernel ran the convolution.
    """
    comptime assert input.flat_rank == 5, "input must be rank 5 (NDHWC)"
    comptime assert filter.flat_rank == 5, "filter must be rank 5"
    comptime assert output.flat_rank == 5, "output must be rank 5 (NDHWC)"
    comptime assert not (
        has_epilogue_fn and has_compute_fn
    ), "a store epilogue excludes the compute epilogue"

    comptime if (
        # A bf16 QRSCF filter already runs the matmul route in the same time.
        (input_type == .bfloat16 and not filter_is_fcrs)
        or not (ctx.default_device_info == B200)
        or input_type != output_type
        or filter_type != input_type
        or not input.is_row_major
        or not filter.is_row_major
        or not output.is_row_major
        or not filter.shape_known
    ):
        return False
    else:
        comptime fq = filter.static_shape[2 if filter_is_fcrs else 0]
        comptime fr = filter.static_shape[3 if filter_is_fcrs else 1]
        comptime fs = filter.static_shape[4 if filter_is_fcrs else 2]
        comptime fc = filter.static_shape[1 if filter_is_fcrs else 3]
        comptime ff = filter.static_shape[0 if filter_is_fcrs else 4]
        comptime if (fq != 1 or fr != 1 or fs != 1 or fc != _C or ff != _F):
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
                or symmetric_padding[0] != 0
                or symmetric_padding[1] != 0
                or symmetric_padding[2] != 0
                or Int(input.dim[4]()) != _C
                or Int(output.dim[4]()) != _F
                or Int(input.dim[2]()) != _H_MEASURED
                or Int(input.dim[3]()) != _W_MEASURED
                or Int(output.dim[2]()) != _H_MEASURED
                or Int(output.dim[3]()) != _W_MEASURED
                or Int(input.dim[1]()) != Int(output.dim[1]())
                or Int(input.dim[0]()) != Int(output.dim[0]())
            ):
                return False
            if (
                Int(input.unsafe_ptr()) % 16 != 0
                or Int(output.unsafe_ptr()) % 16 != 0
                or Int(filter.unsafe_ptr()) % align_of[Scalar[filter_type]]()
                != 0
            ):
                return False

            var d_out = Int(output.dim[1]())
            var rows = Int(output.dim[0]()) * d_out * _H_MEASURED * _W_MEASURED
            if rows == 0:
                return False

            comptime kernel = conv3d_pointwise_kernel[
                input.LayoutType,
                filter.LayoutType,
                output.LayoutType,
                input_type,
                filter_type,
                output_type,
                ImmOrigin(input.origin),
                ImmOrigin(filter.origin),
                output.origin,
                EpilogueFnType,
                ComputeFnType,
                input.Engine,
                filter.Engine,
                output.Engine,
                filter_is_fcrs,
                has_epilogue_fn,
                has_compute_fn,
            ]
            ctx.enqueue_function[kernel](
                input.as_imm(),
                filter.as_imm(),
                output,
                Int64(rows),
                Int32(d_out),
                Int32(_H_MEASURED * _W_MEASURED),
                Int32(_W_MEASURED),
                host_args=(epilogue_fn, compute_fn),
                grid_dim=ceildiv(rows * _THREADS_PER_ROW, _THREADS),
                block_dim=_THREADS,
                attributes=pdl_launch_attributes(PDLLevel.ON),
            )
            return True
