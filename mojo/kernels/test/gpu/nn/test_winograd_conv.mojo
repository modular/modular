# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

# TODO:
# - Think about GPU memory access patterns/hierarchu
# - Use shared memory for transformation matrices (B, G, A) to avoid redundant loads
# - Use shared memory for input tiles to reduce global memory bandwidth
# - Add proper grid dimension calculation instead of hardcoded values
# - Implement proper tiling/slicing for rank 4 LayoutTensor to avoid get_tile workaround
# - Add support for padding/strides in the Winograd convolution
# - Add bounds checking for input dimensions
# - Add test cases for odd sizes, likely broken

from math import ceildiv, isclose

from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from internal_utils import DeviceNDBuffer, HostNDBuffer, random
from layout import Layout, LayoutTensor
from layout._ndbuffer_stub import from_ndbuffer_row_major
from memory import UnsafePointer
from nn.conv import conv_gpu
from testing import assert_almost_equal, assert_true

from utils.index import Index, IndexList
from utils.numerics import get_accum_type


@always_inline
fn _get_b[
    type: DType
](out B: LayoutTensor[type, Layout.row_major(4, 4), MutableAnyOrigin]):
    B = __type_of(B).stack_allocation()
    # fmt:off
    B[0,0] = 1.0; B[0,1] =  0.0; B[0,2] = -1.0; B[0,3] =  0.0
    B[1,0] = 0.0; B[1,1] =  1.0; B[1,2] =  1.0; B[1,3] =  0.0
    B[2,0] = 0.0; B[2,1] = -1.0; B[2,2] =  1.0; B[2,3] =  0.0
    B[3,0] = 0.0; B[3,1] =  1.0; B[3,2] =  0.0; B[3,3] = -1.0
    # fmt:on


@always_inline
fn _get_g[
    type: DType
](out G: LayoutTensor[type, Layout.row_major(4, 3), MutableAnyOrigin]):
    G = __type_of(G).stack_allocation()
    # fmt:off
    G[0,0] = 1.0; G[0,1] =  0.0; G[0,2] = 0.0
    G[1,0] = 0.5; G[1,1] =  0.5; G[1,2] = 0.5
    G[2,0] = 0.5; G[2,1] = -0.5; G[2,2] = 0.5
    G[3,0] = 0.0; G[3,1] =  0.0; G[3,2] = 1.0
    # fmt:on


@always_inline
fn _get_a[
    type: DType
](out A: LayoutTensor[type, Layout.row_major(2, 4), MutableAnyOrigin]):
    A = __type_of(A).stack_allocation()
    # fmt:off
    A[0,0] = 1.0; A[0,1] = 1.0; A[0,2] =  1.0; A[0,3] =  0.0
    A[1,0] = 0.0; A[1,1] = 1.0; A[1,2] = -1.0; A[1,3] = -1.0
    # fmt:on


@always_inline
fn matmul[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    c_layout: Layout,
    a_layout: Layout,
    b_layout: Layout,
    element_layout: Layout, //,
    transpose_b: Bool,
    s_type: DType = get_accum_type[c_type](),
](
    C: LayoutTensor[
        c_type, c_layout, MutableAnyOrigin, element_layout=element_layout, **_
    ],
    A: LayoutTensor[
        a_type, a_layout, MutableAnyOrigin, element_layout=element_layout, **_
    ],
    B: LayoutTensor[
        b_type, b_layout, MutableAnyOrigin, element_layout=element_layout, **_
    ],
):
    alias M = Int(c_layout.shape[0])
    alias N = Int(c_layout.shape[1])
    alias K = Int(a_layout.shape[1])

    @parameter
    if transpose_b:
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


# TODO: Workaround because I have not found a way to slice/tile a rank 4 LayoutTensor
# to a rank 2 LayoutTensor
@always_inline
fn get_tile[
    type: DType, layout: Layout, //, tile_size: Int
](
    input_tensor: LayoutTensor[type, layout, MutableAnyOrigin],
    n: Int,
    h: Int,
    w: Int,
    c: Int,
) -> LayoutTensor[
    type, Layout.row_major(tile_size, tile_size), MutableAnyOrigin
]:
    # TODO: Issue because returning a stack variable? Workaround
    # with @always_inline
    var result = LayoutTensor[
        type, Layout.row_major(tile_size, tile_size), MutableAnyOrigin
    ].stack_allocation()

    for i in range(tile_size):
        for j in range(tile_size):
            result[i, j] = input_tensor[n, h + i, w + j, c]

    return result


# Each thread processes a 4x4 input tile to produce a 2x2 output tile.
# The thread accumulates contributions from all input channels for each output channel.
fn winograd_conv2d_gpu_nhwc[
    input_dim: DimList,
    filter_dim: DimList,
    output_dim: DimList,
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    block_size: Int,
](
    input: NDBuffer[input_type, 4, MutableAnyOrigin, input_dim],
    filter: NDBuffer[filter_type, 4, MutableAnyOrigin, filter_dim],
    output: NDBuffer[mut=True, output_type, 4, MutableAnyOrigin, output_dim],
    stride: IndexList[2],
    dilation: IndexList[2],
    padding: IndexList[2],
):
    """Implements Winograd F(2x2, 3x3) convolution algorithm for GPU.
    Winograd convolution is an optimization that reduces amount of muls by
    using more adds. This is done by transforming the input and filter into a different form.
    The filters can be pre-transformed once and reused for different inputs (not implemented here).

    Each GPU thread processes a 4x4 input tile to produce a 2x2 output tile.

    Currently only supports:
    - 3x3 filters
    - Stride 1
    - Single input channel
    - Even filter input sizes
    - No padding
    - No dilation
    - NHWC input layout
    - RSCF filter layout
    """

    # TODO: Avoid mixing NDBuffer and LayoutTensor?
    var input_tensor = from_ndbuffer_row_major(input)  # (N, H, W, C)
    var filter_tensor = from_ndbuffer_row_major(filter)  # (R, S, C, F)
    var output_tensor = from_ndbuffer_row_major(output)  # (N, H_out, W_out, F)

    # Dimensions
    var C_in = input.dim[3]()  # input channels
    var C_out = output.dim[3]()  # output channels
    var H_out = output.dim[1]()
    var W_out = output.dim[2]()

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
    var scratch = LayoutTensor[
        input_type, Layout.row_major(4, 3), MutableAnyOrigin
    ].stack_allocation()
    var scratch_2 = LayoutTensor[
        input_type, Layout.row_major(4, 4), MutableAnyOrigin
    ].stack_allocation()
    var scratch_3 = LayoutTensor[
        input_type, Layout.row_major(2, 4), MutableAnyOrigin
    ].stack_allocation()
    var m = LayoutTensor[
        output_type, Layout.row_major(4, 4), MutableAnyOrigin
    ].stack_allocation()
    var g_transformed = LayoutTensor[
        input_type, Layout.row_major(4, 4), MutableAnyOrigin
    ].stack_allocation()

    # Pre-transform filter (G^T * filter * G)
    var filter_slice = filter_tensor.slice[:, :, slice_indices= (0, 1)](
        offsets=(0)
    )
    matmul[False](scratch, g, filter_slice)
    matmul[True](g_transformed, scratch, g)

    # Process each output channel
    for c_out in range(C_out):
        var output_tile = LayoutTensor[
            output_type, Layout.row_major(2, 2), MutableAnyOrigin
        ].stack_allocation()

        # Process each input channel
        for c_in in range(C_in):
            # 1. Get input tile

            # TODO: Can we do something like this instead?
            # var input_tile = input_tensor.tile[1,1,4,4](c_out, c_in)
            var input_tile = get_tile[4](input_tensor, n, h_out, w_out, c_in)

            # 2. Transform input (B^T * d * B)
            matmul[transpose_b=False](scratch_2, b, input_tile)
            matmul[transpose_b=True](input_tile, scratch_2, b)

            # 3. Element-wise multiply with transformed filter and accumulate
            # TODO: Can we do this instead? just need to figure out the casting of dtypes
            # m = input_tile * g_transformed
            for ii in range(4):
                for jj in range(4):
                    m[ii, jj] = (
                        input_tile[ii, jj].cast[output_type]()
                        * g_transformed[ii, jj].cast[output_type]()
                    )

            # 4. Transform output (A^T * m * A)
            matmul[transpose_b=False](scratch_3, a, m)
            matmul[transpose_b=True](output_tile, scratch_3, a)

            # 5. Store result
            for di in range(2):
                for dj in range(2):
                    output_tensor[
                        n, h_out + di, w_out + dj, c_out
                    ] = output_tile[di, dj]


fn winograd_conv2d_gpu_launcher[
    input_rank: Int,
    filter_rank: Int,
    input_dim: DimList,
    filter_dim: DimList,
    output_dim: DimList,
    input_type: DType,
    filter_type: DType,
    output_type: DType,
](
    input: NDBuffer[input_type, input_rank, MutableAnyOrigin, input_dim],
    filter: NDBuffer[filter_type, filter_rank, MutableAnyOrigin, filter_dim],
    output: NDBuffer[
        mut=True, output_type, input_rank, MutableAnyOrigin, output_dim
    ],
    stride: IndexList[2],
    dilation: IndexList[2],
    padding: IndexList[2],
    num_groups: Int,
    ctx: DeviceContext,
) raises:
    alias block_size = 16

    # TODO: Is assert_true the right way to do this?
    assert_true(
        input_dim.get[1]() % 2 == 0 and input_dim.get[2]() % 2 == 0,
        "H and W must be even number",
    )
    assert_true(
        input_dim.get[1]() >= 4 and input_dim.get[2]() >= 4,
        "Input must be at least 4x4",
    )
    assert_true(
        filter_dim.get[0]() == 3 and filter_dim.get[1]() == 3,
        "Filter must be 3x3",
    )
    assert_true(stride[0] == 1 and stride[1] == 1, "Stride not implemented")
    assert_true(
        dilation[0] == 1 and dilation[1] == 1, "Dilation not implemented"
    )
    assert_true(padding[0] == 0 and padding[1] == 0, "Padding not implemented")
    assert_true(num_groups == 1, "Num groups not implemented")
    assert_true(
        input_dim.get[3]() == filter_dim.get[2](),
        "Input and filter channels must match",
    )
    assert_true(
        input_dim.get[3]() == 1, "Multiple input channels not implemented"
    )

    var grid_dim_x = ceildiv(output_dim.get[2](), 2 * block_size)
    var grid_dim_y = ceildiv(output_dim.get[1](), 2 * block_size)
    var grid_dim_z = input_dim.get[0]()

    alias kernel = winograd_conv2d_gpu_nhwc[
        input_dim,
        filter_dim,
        output_dim,
        input_type,
        filter_type,
        output_type,
        block_size,
    ]

    ctx.enqueue_function[kernel](
        input,
        filter,
        output,
        stride,
        dilation,
        padding,
        grid_dim=(grid_dim_x, grid_dim_y, grid_dim_z),
        block_dim=(block_size, block_size),
    )


@always_inline
fn get_output_dim[
    input_dim: DimList,
    filter_dim: DimList,
    stride: IndexList[2],
    dilation: IndexList[2],
    pad: IndexList[2],
]() -> DimList:
    alias N = input_dim.get[0]()
    alias H = input_dim.get[1]()
    alias W = input_dim.get[2]()
    alias C = input_dim.get[3]()

    alias R = filter_dim.get[0]()
    alias S = filter_dim.get[1]()
    alias F = filter_dim.get[3]()

    alias pad_h = IndexList[2](pad[0], pad[0])
    alias pad_w = IndexList[2](pad[1], pad[1])

    alias HO = (H + pad_h[0] + pad_h[1] - dilation[0] * (R - 1) - 1) // stride[
        0
    ] + 1
    alias WO = (W + pad_w[0] + pad_w[1] - dilation[1] * (S - 1) - 1) // stride[
        1
    ] + 1
    alias output_dim = DimList(N, HO, WO, F)
    return output_dim


fn test_winograd_conv_gpu[
    type: DType,
    input_dim: DimList,
    filter_dim: DimList,
    stride: IndexList[2],
    dilation: IndexList[2],
    pad: IndexList[2],
    num_groups: Int = 1,
](ctx: DeviceContext) raises:
    print("== test_conv_winograd_gpu")

    alias output_dim = get_output_dim[
        input_dim, filter_dim, stride, dilation, pad
    ]()

    var host_input = HostNDBuffer[type, 4, input_dim]()
    var host_filter = HostNDBuffer[type, 4, filter_dim]()

    var device_output = DeviceNDBuffer[type, 4, output_dim](ctx=ctx)
    var device_output_ref = DeviceNDBuffer[type, 4, output_dim](ctx=ctx)

    random(host_filter.tensor)
    random(host_input.tensor)

    var device_input = host_input.copy_to_device(ctx)
    var device_filter = host_filter.copy_to_device(ctx)

    conv_gpu[4, 4, input_dim, filter_dim, output_dim, type, type, type](
        device_input.tensor,
        device_filter.tensor,
        device_output_ref.tensor,
        stride,
        dilation,
        pad,
        num_groups,
        ctx,
    )

    var host_output_ref = device_output_ref.copy_from_device(ctx)

    winograd_conv2d_gpu_launcher[
        4, 4, input_dim, filter_dim, output_dim, type, type, type
    ](
        device_input.tensor,
        device_filter.tensor,
        device_output.tensor,
        stride,
        dilation,
        pad,
        num_groups,
        ctx,
    )

    var host_output = device_output.copy_from_device(ctx)

    # TODO: Should tolerances really this high for BFloat16?
    alias atol = 1e-06 if type == DType.float32 else 1e-1
    alias rtol = 1e-06 if type == DType.float32 else 1e-4

    for x in range(output_dim.product().get()):
        assert_almost_equal(
            host_output_ref.tensor.data[x],
            host_output.tensor.data[x],
            atol=atol,
            rtol=rtol,
        )


fn main() raises:
    alias dtype = DType.float32

    with DeviceContext() as ctx:
        test_winograd_conv_gpu[
            type=dtype,
            input_dim = DimList(1, 8, 8, 1),
            filter_dim = DimList(3, 3, 1, 1),
            stride = IndexList[2](1, 1),
            dilation = IndexList[2](1, 1),
            pad = IndexList[2](0, 0),
        ](ctx)

        test_winograd_conv_gpu[
            type=dtype,
            input_dim = DimList(32, 256, 256, 1),
            filter_dim = DimList(3, 3, 1, 1),
            stride = IndexList[2](1, 1),
            dilation = IndexList[2](1, 1),
            pad = IndexList[2](0, 0),
        ](ctx)

        test_winograd_conv_gpu[
            type=dtype,
            input_dim = DimList(1, 4, 16, 1),
            filter_dim = DimList(3, 3, 1, 1),
            stride = IndexList[2](1, 1),
            dilation = IndexList[2](1, 1),
            pad = IndexList[2](0, 0),
        ](ctx)

        test_winograd_conv_gpu[
            type=dtype,
            input_dim = DimList(1, 16, 4, 1),
            filter_dim = DimList(3, 3, 1, 1),
            stride = IndexList[2](1, 1),
            dilation = IndexList[2](1, 1),
            pad = IndexList[2](0, 0),
        ](ctx)

        test_winograd_conv_gpu[
            type = DType.bfloat16,
            input_dim = DimList(1, 32, 32, 1),
            filter_dim = DimList(3, 3, 1, 1),
            stride = IndexList[2](1, 1),
            dilation = IndexList[2](1, 1),
            pad = IndexList[2](0, 0),
        ](ctx)
