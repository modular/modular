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

from std.math import ceildiv

from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from layout import Idx, IntTuple, TensorLayout, TileTensor, row_major
from layout.tile_tensor import stack_allocation
from layout.tensor_engine import DefaultEngine
from layout.int_tuple import product
from layout._fillers import random
from nn.conv.conv import conv_gpu
from std.testing import assert_almost_equal, assert_true

from std.utils.index import IndexList
from std.utils.numerics import get_accum_type


@inline(.always)
def _get_b[
    dtype: DType
](out B: TileTensor[dtype, type_of(row_major[4, 4]()), MutUntrackedOrigin]):
    B = stack_allocation[dtype=dtype](row_major[4, 4]())
    # fmt:off
    B[0,0] = 1.0; B[0,1] =  0.0; B[0,2] = -1.0; B[0,3] =  0.0
    B[1,0] = 0.0; B[1,1] =  1.0; B[1,2] =  1.0; B[1,3] =  0.0
    B[2,0] = 0.0; B[2,1] = -1.0; B[2,2] =  1.0; B[2,3] =  0.0
    B[3,0] = 0.0; B[3,1] =  1.0; B[3,2] =  0.0; B[3,3] = -1.0
    # fmt:on


@inline(.always)
def _get_g[
    dtype: DType
](out G: TileTensor[dtype, type_of(row_major[4, 3]()), MutUntrackedOrigin]):
    G = stack_allocation[dtype=dtype](row_major[4, 3]())
    # fmt:off
    G[0,0] = 1.0; G[0,1] =  0.0; G[0,2] = 0.0
    G[1,0] = 0.5; G[1,1] =  0.5; G[1,2] = 0.5
    G[2,0] = 0.5; G[2,1] = -0.5; G[2,2] = 0.5
    G[3,0] = 0.0; G[3,1] =  0.0; G[3,2] = 1.0
    # fmt:on


@inline(.always)
def _get_a[
    dtype: DType
](out A: TileTensor[dtype, type_of(row_major[2, 4]()), MutUntrackedOrigin]):
    A = stack_allocation[dtype=dtype](row_major[2, 4]())
    # fmt:off
    A[0,0] = 1.0; A[0,1] = 1.0; A[0,2] =  1.0; A[0,3] =  0.0
    A[1,0] = 0.0; A[1,1] = 1.0; A[1,2] = -1.0; A[1,3] = -1.0
    # fmt:on


@inline(.always)
def matmul[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    c_layout: TensorLayout,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    //,
    transpose_b: Bool,
    s_type: DType = get_accum_type[c_type](),
](
    C: TileTensor[
        mut=True, c_type, c_layout, Engine=DefaultEngine[element_width=1], ...
    ],
    A: TileTensor[a_type, a_layout, Engine=DefaultEngine[element_width=1], ...],
    B: TileTensor[b_type, b_layout, Engine=DefaultEngine[element_width=1], ...],
):
    comptime assert C.rank == A.rank == B.rank == 2
    comptime assert C.flat_rank == A.flat_rank == B.flat_rank == 2
    comptime assert C.element_size == A.element_size == B.element_size == 1
    comptime assert C.all_dims_known and A.all_dims_known and B.all_dims_known
    comptime M = Int(c_layout.static_shape[0])
    comptime N = Int(c_layout.static_shape[1])
    comptime K = Int(a_layout.static_shape[1])

    comptime if transpose_b:
        for i in range(M):
            for j in range(N):
                var sum: SIMD[s_type, C.element_size] = 0
                for k in range(K):
                    sum += A[i, k].cast[s_type]() * B[j, k].cast[s_type]()
                C[i, j] = sum.cast[c_type]()
    else:
        for i in range(M):
            for j in range(N):
                var sum: SIMD[s_type, C.element_size] = 0
                for k in range(K):
                    sum += A[i, k].cast[s_type]() * B[k, j].cast[s_type]()
                C[i, j] = sum.cast[c_type]()


# Copy the strided NHWC tile into writable scratch for the in-place transforms.
@inline(.always)
def get_tile[
    dtype: DType, //, tile_size: Int
](
    input_tensor: TileTensor[dtype, Engine=DefaultEngine[element_width=1], ...],
    n: Int,
    h: Int,
    w: Int,
    c: Int,
) -> TileTensor[
    dtype,
    type_of(row_major[tile_size, tile_size]()),
    MutUntrackedOrigin,
]:
    comptime assert input_tensor.rank == input_tensor.flat_rank == 4
    comptime assert input_tensor.element_size == 1
    # Always inline so the scratch allocation lives in the caller's frame.
    var result = stack_allocation[dtype=dtype](
        row_major[tile_size, tile_size]()
    )

    for i in range(tile_size):
        for j in range(tile_size):
            result[i, j] = input_tensor[n, h + i, w + j, c]

    return result


# Each thread processes a 4x4 input tile to produce a 2x2 output tile.
# This test supports one input channel and one filter output channel.
def winograd_conv2d_gpu_nhwc[
    input_layout: TensorLayout,
    filter_layout: TensorLayout,
    output_layout: TensorLayout,
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    block_size: Int,
](
    input: TileTensor[input_type, input_layout, ImmutAnyOrigin],
    filter: TileTensor[
        filter_type,
        filter_layout,
        ImmutAnyOrigin,
    ],
    output: TileTensor[
        mut=True,
        output_type,
        output_layout,
        MutAnyOrigin,
    ],
    stride: IndexList[2],
    dilation: IndexList[2],
    padding: IndexList[
        4
    ],  # Format: [pad_h_before, pad_h_after, pad_w_before, pad_w_after]
):
    """Implements Winograd F(2x2, 3x3) convolution algorithm for GPU.
    Winograd convolution is an optimization that reduces amount of muls by
    using more adds. This is done by transforming the input and filter into a different form.
    The filters can be pre-transformed once and reused for different inputs (not implemented here).

    Each GPU thread processes a 4x4 input tile to produce a 2x2 output tile.

    Currently only supports:
    - 3x3 filters
    - Stride 1
    - Single input and output channel
    - Even input height and width
    - No padding
    - No dilation
    - NHWC input layout
    - RSCF filter layout
    """
    comptime assert input.rank == filter.rank == output.rank == 4
    comptime assert input.flat_rank == filter.flat_rank == output.flat_rank == 4
    comptime assert filter.all_dims_known

    # Dimensions
    var C_in = Int(input.dim[3]())  # input channels
    var C_out = Int(output.dim[3]())  # output channels
    var H_out = Int(output.dim[1]())
    var W_out = Int(output.dim[2]())

    # Get transformation matrices
    var b = _get_b[input_type]()
    var g = _get_g[input_type]()
    var a = _get_a[input_type]()

    # Thread indices
    var n = block_idx.z
    var h_out = (block_idx.x * block_dim.x + thread_idx.x) * 2
    var w_out = (block_idx.y * block_dim.y + thread_idx.y) * 2

    # Check bounds
    if h_out + 1 >= H_out or w_out + 1 >= W_out:
        return

    # Allocate scratch space
    var scratch = stack_allocation[dtype=input_type](row_major[4, 3]())
    var scratch_2 = stack_allocation[dtype=input_type](row_major[4, 4]())
    var scratch_3 = stack_allocation[dtype=input_type](row_major[2, 4]())
    var m = stack_allocation[dtype=output_type](row_major[4, 4]())
    var g_transformed = stack_allocation[dtype=input_type](row_major[4, 4]())

    # Transform the filter as G * filter * G^T, fixing its channel indices.
    var filter_slice = filter.slice[:, :, 0, 0]()
    matmul[False](scratch, g, filter_slice)
    matmul[True](g_transformed, scratch, g)

    # Process each output channel
    for c_out in range(C_out):
        var output_tile = stack_allocation[dtype=output_type](row_major[2, 2]())

        # Process each input channel
        for c_in in range(C_in):
            var input_tile = get_tile[4](
                input.as_unsafe_any_origin(), n, h_out, w_out, c_in
            )

            # Transform input as B * d * B^T.
            matmul[transpose_b=False](scratch_2, b, input_tile)
            matmul[transpose_b=True](input_tile, scratch_2, b)

            # Multiply the transformed input and filter elementwise.
            for ii in range(4):
                for jj in range(4):
                    m[ii, jj] = (
                        input_tile[ii, jj][0].cast[output_type]()
                        * g_transformed[ii, jj][0].cast[output_type]()
                    )

            # Transform output as A * m * A^T.
            matmul[transpose_b=False](scratch_3, a, m)
            matmul[transpose_b=True](output_tile, scratch_3, a)

            for di in range(2):
                for dj in range(2):
                    output[n, h_out + di, w_out + dj, c_out] = output_tile[
                        di, dj
                    ][0]


def winograd_conv2d_gpu_launcher[
    input_type: DType,
    filter_type: DType,
    output_type: DType,
](
    input: TileTensor[input_type, ...],
    filter: TileTensor[filter_type, ...],
    output: TileTensor[mut=True, output_type, ...],
    stride: IndexList[2],
    dilation: IndexList[2],
    padding: IndexList[
        4
    ],  # Format: [pad_h_before, pad_h_after, pad_w_before, pad_w_after]
    num_groups: Int,
    ctx: DeviceContext,
) raises:
    comptime block_size = 16

    assert_true(
        input.dim[1]() % 2 == 0 and input.dim[2]() % 2 == 0,
        "H and W must be even number",
    )
    assert_true(
        input.dim[1]() >= 4 and input.dim[2]() >= 4,
        "Input must be at least 4x4",
    )
    assert_true(
        filter.dim[0]() == 3 and filter.dim[1]() == 3,
        "Filter must be 3x3",
    )
    assert_true(stride[0] == 1 and stride[1] == 1, "Stride not implemented")
    assert_true(
        dilation[0] == 1 and dilation[1] == 1, "Dilation not implemented"
    )
    assert_true(
        padding[0] == 0
        and padding[1] == 0
        and padding[2] == 0
        and padding[3] == 0,
        "Padding not implemented",
    )
    assert_true(num_groups == 1, "Num groups not implemented")
    assert_true(
        Int(input.dim[3]()) == Int(filter.dim[2]()),
        "Input and filter channels must match",
    )
    assert_true(input.dim[3]() == 1, "Multiple input channels not implemented")

    var grid_dim_x = ceildiv(Int(output.dim[2]()), 2 * block_size)
    var grid_dim_y = ceildiv(Int(output.dim[1]()), 2 * block_size)
    var grid_dim_z = Int(input.dim[0]())

    comptime kernel = winograd_conv2d_gpu_nhwc[
        type_of(input.layout),
        type_of(filter.layout),
        type_of(output.layout),
        input_type,
        filter_type,
        output_type,
        block_size,
    ]

    ctx.enqueue_function[kernel](
        input.as_imm().as_unsafe_any_origin(),
        filter.as_imm().as_unsafe_any_origin(),
        output.as_unsafe_any_origin(),
        stride,
        dilation,
        padding,
        grid_dim=(grid_dim_x, grid_dim_y, grid_dim_z),
        block_dim=(block_size, block_size),
    )


@inline(.always)
def get_output_dim[
    input_dim: IntTuple,
    filter_dim: IntTuple,
    stride: IndexList[2],
    dilation: IndexList[2],
    pad: IndexList[
        4
    ],  # Format: [pad_h_before, pad_h_after, pad_w_before, pad_w_after]
]() -> IndexList[4]:
    comptime N = Int(input_dim[0])
    comptime H = Int(input_dim[1])
    comptime W = Int(input_dim[2])
    comptime C = Int(input_dim[3])

    comptime R = Int(filter_dim[0])
    comptime S = Int(filter_dim[1])
    comptime F = Int(filter_dim[3])

    # Extract padding values: pad format is [pad_h_before, pad_h_after, pad_w_before, pad_w_after]
    comptime pad_h = IndexList[2](pad[0], pad[1])
    comptime pad_w = IndexList[2](pad[2], pad[3])

    comptime HO = (
        H + pad_h[0] + pad_h[1] - dilation[0] * (R - 1) - 1
    ) // stride[0] + 1
    comptime WO = (
        W + pad_w[0] + pad_w[1] - dilation[1] * (S - 1) - 1
    ) // stride[1] + 1
    comptime output_dim = IndexList[4](N, HO, WO, F)
    return output_dim


def test_winograd_conv_gpu[
    dtype: DType,
    input_dim: IntTuple,
    filter_dim: IntTuple,
    stride: IndexList[2],
    dilation: IndexList[2],
    pad: IndexList[
        4
    ],  # Format: [pad_h_before, pad_h_after, pad_w_before, pad_w_after]
    num_groups: Int = 1,
](ctx: DeviceContext) raises:
    print("== test_conv_winograd_gpu")

    comptime output_dim = get_output_dim[
        input_dim, filter_dim, stride, dilation, pad
    ]()

    comptime input_tt_layout = row_major(
        (
            Idx[Int(input_dim[0])],
            Idx[Int(input_dim[1])],
            Idx[Int(input_dim[2])],
            Idx[Int(input_dim[3])],
        )
    )
    comptime filter_tt_layout = row_major(
        (
            Idx[Int(filter_dim[0])],
            Idx[Int(filter_dim[1])],
            Idx[Int(filter_dim[2])],
            Idx[Int(filter_dim[3])],
        )
    )
    comptime output_tt_layout = row_major(
        (
            Idx[Int(output_dim[0])],
            Idx[Int(output_dim[1])],
            Idx[Int(output_dim[2])],
            Idx[Int(output_dim[3])],
        )
    )

    # Create device buffers
    var input_device = ctx.enqueue_create_buffer[dtype](product(input_dim))
    var filter_device = ctx.enqueue_create_buffer[dtype](product(filter_dim))
    var output_device = ctx.enqueue_create_buffer[dtype](
        output_dim.flattened_length()
    )
    var output_ref_device = ctx.enqueue_create_buffer[dtype](
        output_dim.flattened_length()
    )

    # Initialize input and filter with random values on host
    with input_device.map_to_host() as input_host:
        var input_host_tt = TileTensor(input_host, input_tt_layout)
        random(input_host_tt)

    with filter_device.map_to_host() as filter_host:
        var filter_host_tt = TileTensor(filter_host, filter_tt_layout)
        random(filter_host_tt)

    var input_tt = TileTensor(input_device, input_tt_layout)
    var filter_tt = TileTensor(filter_device, filter_tt_layout)
    var output_tt = TileTensor(output_device, output_tt_layout)
    var output_ref_tt = TileTensor(output_ref_device, output_tt_layout)

    # Run reference convolution
    conv_gpu[dtype, dtype, dtype](
        input_tt,
        filter_tt,
        output_ref_tt,
        stride,
        dilation,
        pad,
        num_groups,
        ctx,
    )

    # Run winograd convolution
    winograd_conv2d_gpu_launcher[dtype, dtype, dtype](
        input_tt,
        filter_tt,
        output_tt,
        stride,
        dilation,
        pad,
        num_groups,
        ctx,
    )

    ctx.synchronize()

    # Verify results
    comptime atol = 1e-06 if dtype == DType.float32 else 1e-1
    comptime rtol = 1e-06 if dtype == DType.float32 else 1e-4

    with output_device.map_to_host() as output_host:
        with output_ref_device.map_to_host() as output_ref_host:
            for x in range(output_dim.flattened_length()):
                assert_almost_equal(
                    output_ref_host[x],
                    output_host[x],
                    atol=atol,
                    rtol=rtol,
                )


def main() raises:
    comptime dtype = DType.float32

    with DeviceContext() as ctx:
        test_winograd_conv_gpu[
            dtype=dtype,
            input_dim=IntTuple(1, 8, 8, 1),
            filter_dim=IntTuple(3, 3, 1, 1),
            stride=(1, 1),
            dilation=(1, 1),
            pad=(
                0,
                0,
                0,
                0,
            ),  # [pad_h_before, pad_h_after, pad_w_before, pad_w_after]
        ](ctx)

        test_winograd_conv_gpu[
            dtype=dtype,
            input_dim=IntTuple(32, 256, 256, 1),
            filter_dim=IntTuple(3, 3, 1, 1),
            stride=(1, 1),
            dilation=(1, 1),
            pad=(
                0,
                0,
                0,
                0,
            ),  # [pad_h_before, pad_h_after, pad_w_before, pad_w_after]
        ](ctx)

        test_winograd_conv_gpu[
            dtype=dtype,
            input_dim=IntTuple(1, 4, 16, 1),
            filter_dim=IntTuple(3, 3, 1, 1),
            stride=(1, 1),
            dilation=(1, 1),
            pad=(
                0,
                0,
                0,
                0,
            ),  # [pad_h_before, pad_h_after, pad_w_before, pad_w_after]
        ](ctx)

        test_winograd_conv_gpu[
            dtype=dtype,
            input_dim=IntTuple(1, 16, 4, 1),
            filter_dim=IntTuple(3, 3, 1, 1),
            stride=(1, 1),
            dilation=(1, 1),
            pad=(
                0,
                0,
                0,
                0,
            ),  # [pad_h_before, pad_h_after, pad_w_before, pad_w_after]
        ](ctx)

        test_winograd_conv_gpu[
            dtype=DType.bfloat16,
            input_dim=IntTuple(1, 32, 32, 1),
            filter_dim=IntTuple(3, 3, 1, 1),
            stride=(1, 1),
            dilation=(1, 1),
            pad=(
                0,
                0,
                0,
                0,
            ),  # [pad_h_before, pad_h_after, pad_w_before, pad_w_after]
        ](ctx)
