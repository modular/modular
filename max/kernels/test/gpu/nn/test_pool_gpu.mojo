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

from max.gpu.host import DeviceContext
from layout._fillers import arange
from layout import TileTensor, row_major
from layout.coord import ComptimeInt, Coord
from layout.tile_layout import Layout
from nn.pool import (
    PoolMethod,
    avg_pool,
    avg_pool_gpu,
    max_pool,
    max_pool_gpu,
)
from std.testing import assert_almost_equal


def main() raises:
    with DeviceContext() as ctx:
        test_max_pool_2d(ctx)
        test_avg_pool_2d(ctx)
        test_avg_pool_2d_with_padding_gpu[True](ctx)
        test_avg_pool_2d_with_padding_gpu[False](ctx)
        test_maxpool_2d_ceil_gpu(ctx)
        test_average_pool_2d_ceil_excludeBound_gpu(ctx)
        test_average_pool_2d_ceil_includeBound_gpu(ctx)
        test_max_pool_pad_dilation_2d_gpu(ctx)
        # Channel counts that are a multiple of the vector width take the
        # vectorized GPU path; the rest fall back to the scalar kernel.
        test_pool_gpu_vectorized_channels[.bfloat16, 5, 7, 16, 3, 1, 1, True](
            ctx
        )
        test_pool_gpu_vectorized_channels[.bfloat16, 5, 7, 16, 3, 1, 1, False](
            ctx
        )
        test_pool_gpu_vectorized_channels[.float32, 7, 9, 8, 3, 2, 0, True](ctx)
        test_pool_gpu_vectorized_channels[.float32, 7, 9, 8, 3, 2, 0, False](
            ctx
        )
        # Ragged channel counts (C=3 and C=4 at a width of 8) must stay
        # bit-identical to the CPU path through the scalar fallback.
        test_pool_gpu_vectorized_channels[.bfloat16, 6, 6, 3, 2, 2, 0, True](
            ctx
        )
        test_pool_gpu_vectorized_channels[.bfloat16, 6, 6, 3, 2, 2, 0, False](
            ctx
        )
        test_pool_gpu_vectorized_channels[.bfloat16, 4, 4, 4, 2, 2, 0, False](
            ctx
        )
        # Average pooling over a padded window, once excluding and once
        # counting the padded elements.
        test_pool_gpu_vectorized_channels[
            .bfloat16, 5, 7, 16, 3, 1, 1, False, True
        ](ctx)
        test_pool_gpu_vectorized_channels[
            .bfloat16, 5, 7, 24, 3, 1, 1, False, False
        ](ctx)
        # Non-unit channel stride on both operands: the vector path would walk
        # consecutive physical elements instead of consecutive channels.
        test_pool_gpu_strided_channels[True](ctx)
        test_pool_gpu_strided_channels[False](ctx)


def test_max_pool_2d(ctx: DeviceContext) raises:
    print("== test_max_pool_2d")

    # output should have form
    # ([[[[ 30.,  31.],
    #    [ 36.,  37.]],
    #   [[ 58.,  59.],
    #    [ 64.,  65.]]],
    #  [[[ 100.,  101.],
    #    [ 106., 107.]],
    #   [[128., 129.],
    #    [134., 135.]]]])

    pool(PoolMethod.MAX, ctx)


def test_avg_pool_2d(ctx: DeviceContext) raises:
    print("== test_avg_pool_2d")

    # output should have form
    # ([[[[  15.5,  16.0],
    #    [ 21.0,  22.0]],
    #   [[ 43.0,  44.0],
    #    [ 49.0,  50.0]]],
    #  [[[ 85.0,  86.0],
    #    [ 91.0,  92.0]],
    #   [[113.0, 114.0],
    #    [119.0, 120.0]]]])

    pool(PoolMethod.AVG, ctx)


def test_maxpool_2d_ceil_gpu(ctx: DeviceContext) raises:
    print("== test_max_pool_2d_ceil_gpu")
    pool_ceil_test(PoolMethod.MAX, ctx)


def test_average_pool_2d_ceil_excludeBound_gpu(ctx: DeviceContext) raises:
    print("== test_average_pool_2d_ceil_excludeBound_gpu")
    pool_ceil_test(PoolMethod.AVG, ctx)


def test_average_pool_2d_ceil_includeBound_gpu(ctx: DeviceContext) raises:
    print("== test_average_pool_2d_ceil_includeBound_gpu")
    pool_ceil_test[True, True](PoolMethod.AVG, ctx)


def pool[
    count_boundary: Bool = False
](pool_method: PoolMethod, ctx: DeviceContext) raises:
    comptime in_layout = row_major[2, 5, 7, 2]()
    comptime out_layout = row_major[2, 2, 2, 2]()
    comptime in_size = in_layout.product()
    comptime out_size = out_layout.product()

    # Create host buffers
    var in_host_buffer = ctx.enqueue_create_host_buffer[.float32](in_size)
    var out_host_buffer = ctx.enqueue_create_host_buffer[.float32](out_size)
    var ref_host_buffer = ctx.enqueue_create_host_buffer[.float32](out_size)
    ctx.synchronize()

    # Create TileTensors from host buffers
    var input_tensor = TileTensor(in_host_buffer, in_layout)
    var output_tensor = TileTensor(out_host_buffer, out_layout)
    var h_output_ref = TileTensor(ref_host_buffer, out_layout)

    arange(input_tensor)
    for i in range(out_size):
        out_host_buffer[i] = 0
        ref_host_buffer[i] = 0

    # Create parameter tensors
    var paddings_stack = Array[Int32, 4](fill=0)
    var paddings_tensor = TileTensor(paddings_stack, row_major[4]())

    var filter_stack: Array[Int32, _] = [3, 2]
    var filter_tensor = TileTensor(filter_stack, row_major[2]())

    var stride_stack: Array[Int32, _] = [2, 3]
    var stride_tensor = TileTensor(stride_stack, row_major[2]())

    var dilation_stack = Array[Int32, 2](fill=1)
    var dilation_tensor = TileTensor(dilation_stack, row_major[2]())

    # Create device buffers
    var d_input_buffer = ctx.enqueue_create_buffer[.float32](in_size)
    var d_output_buffer = ctx.enqueue_create_buffer[.float32](out_size)

    # Create device TileTensors
    var d_input = TileTensor(d_input_buffer, in_layout)
    var d_output = TileTensor(d_output_buffer, out_layout)

    # Copy data to device
    ctx.enqueue_copy(d_input_buffer, in_host_buffer)
    ctx.enqueue_copy(d_output_buffer, out_host_buffer)

    if pool_method == PoolMethod.MAX:
        max_pool_gpu[int_type=DType.int32](
            ctx,
            d_input,
            filter_tensor,
            stride_tensor,
            dilation_tensor,
            paddings_tensor,
            d_output,
        )
        max_pool[int_type=DType.int32](
            input_tensor,
            filter_tensor,
            stride_tensor,
            dilation_tensor,
            paddings_tensor,
            h_output_ref,
        )
    else:
        avg_pool_gpu[int_type=DType.int32, count_boundary=count_boundary](
            ctx,
            d_input,
            filter_tensor,
            stride_tensor,
            dilation_tensor,
            paddings_tensor,
            d_output,
        )
        avg_pool[int_type=DType.int32, count_boundary=count_boundary](
            input_tensor,
            filter_tensor,
            stride_tensor,
            dilation_tensor,
            paddings_tensor,
            h_output_ref,
        )

    # Copy data back to host
    ctx.enqueue_copy(out_host_buffer, d_output_buffer)
    ctx.synchronize()

    # Ensure the GPU and CPU results are the same
    assert_allclose(h_output_ref, output_tensor)


def pool_ceil_test[
    count_boundary: Bool = False, ceil_mode: Bool = True
](pool_method: PoolMethod, ctx: DeviceContext) raises:
    comptime in_layout = row_major[1, 4, 4, 1]()
    comptime out_layout = row_major[1, 2, 2, 1]()
    comptime in_size = in_layout.product()
    comptime out_size = out_layout.product()

    # Create host buffers
    var in_host_buffer = ctx.enqueue_create_host_buffer[.float32](in_size)
    var out_host_buffer = ctx.enqueue_create_host_buffer[.float32](out_size)
    var ref_host_buffer = ctx.enqueue_create_host_buffer[.float32](out_size)
    ctx.synchronize()

    # Create TileTensors from host buffers
    var input_tensor = TileTensor(in_host_buffer, in_layout)
    var output_tensor = TileTensor(out_host_buffer, out_layout)
    var h_output_ref = TileTensor(ref_host_buffer, out_layout)

    arange(input_tensor)
    for i in range(out_size):
        out_host_buffer[i] = 0
        ref_host_buffer[i] = 0

    # Create parameter tensors
    var paddings_stack = Array[Int32, 4](fill=0)
    var paddings_tensor = TileTensor(paddings_stack, row_major[4]())

    var filter_stack = Array[Int32, 2](fill=3)
    var filter_tensor = TileTensor(filter_stack, row_major[2]())

    var stride_stack = Array[Int32, 2](fill=2)
    var stride_tensor = TileTensor(stride_stack, row_major[2]())

    var dilation_stack = Array[Int32, 2](fill=1)
    var dilation_tensor = TileTensor(dilation_stack, row_major[2]())

    # Create device buffers
    var d_input_buffer = ctx.enqueue_create_buffer[.float32](in_size)
    var d_output_buffer = ctx.enqueue_create_buffer[.float32](out_size)

    # Create device TileTensors
    var d_input = TileTensor(d_input_buffer, in_layout)
    var d_output = TileTensor(d_output_buffer, out_layout)

    # Copy data to device
    ctx.enqueue_copy(d_input_buffer, in_host_buffer)
    ctx.enqueue_copy(d_output_buffer, out_host_buffer)

    if pool_method == PoolMethod.MAX:
        max_pool_gpu[int_type=DType.int32](
            ctx,
            d_input,
            filter_tensor,
            stride_tensor,
            dilation_tensor,
            paddings_tensor,
            d_output,
            ceil_mode,
        )
        max_pool[int_type=DType.int32](
            input_tensor,
            filter_tensor,
            stride_tensor,
            dilation_tensor,
            paddings_tensor,
            h_output_ref,
            ceil_mode,
        )
    else:
        avg_pool_gpu[int_type=DType.int32, count_boundary=count_boundary](
            ctx,
            d_input,
            filter_tensor,
            stride_tensor,
            dilation_tensor,
            paddings_tensor,
            d_output,
            ceil_mode,
        )
        avg_pool[int_type=DType.int32, count_boundary=count_boundary](
            input_tensor,
            filter_tensor,
            stride_tensor,
            dilation_tensor,
            paddings_tensor,
            h_output_ref,
            ceil_mode,
        )

    # Copy data back to host
    ctx.enqueue_copy(out_host_buffer, d_output_buffer)
    ctx.synchronize()

    # Ensure the GPU and CPU results are the same
    assert_allclose(h_output_ref, output_tensor)


def test_avg_pool_2d_with_padding_gpu[
    count_boundary: Bool = False
](ctx: DeviceContext) raises:
    print("== test_avg_pool_2d_with_padding_gpu:", count_boundary)

    comptime in_layout = row_major[1, 7, 7, 1]()
    comptime out_layout = row_major[1, 7, 7, 1]()
    comptime in_size = in_layout.product()
    comptime out_size = out_layout.product()

    # Create host buffers
    var in_host_buffer = ctx.enqueue_create_host_buffer[.float32](in_size)
    var out_host_buffer = ctx.enqueue_create_host_buffer[.float32](out_size)
    var ref_host_buffer = ctx.enqueue_create_host_buffer[.float32](out_size)
    ctx.synchronize()

    # Create TileTensors from host buffers
    var input_tensor = TileTensor(in_host_buffer, in_layout)
    var output_tensor = TileTensor(out_host_buffer, out_layout)
    var h_output_ref = TileTensor(ref_host_buffer, out_layout)

    arange(input_tensor)
    for i in range(out_size):
        out_host_buffer[i] = 0
        ref_host_buffer[i] = 0

    # Create parameter tensors
    var paddings_stack = Array[Int32, 4](fill=1)
    var paddings_tensor = TileTensor(paddings_stack, row_major[4]())

    var filter_stack = Array[Int32, 2](fill=3)
    var filter_tensor = TileTensor(filter_stack, row_major[2]())

    var stride_stack = Array[Int32, 2](fill=1)
    var stride_tensor = TileTensor(stride_stack, row_major[2]())

    var dilation_stack = Array[Int32, 2](fill=1)
    var dilation_tensor = TileTensor(dilation_stack, row_major[2]())

    # Create device buffers
    var d_input_buffer = ctx.enqueue_create_buffer[.float32](in_size)
    var d_output_buffer = ctx.enqueue_create_buffer[.float32](out_size)

    # Create device TileTensors
    var d_input = TileTensor(d_input_buffer, in_layout)
    var d_output = TileTensor(d_output_buffer, out_layout)

    # Copy data to device
    ctx.enqueue_copy(d_input_buffer, in_host_buffer)
    ctx.enqueue_copy(d_output_buffer, out_host_buffer)

    avg_pool_gpu[int_type=DType.int32, count_boundary=count_boundary](
        ctx,
        d_input,
        filter_tensor,
        stride_tensor,
        dilation_tensor,
        paddings_tensor,
        d_output,
    )
    avg_pool[int_type=DType.int32, count_boundary=count_boundary](
        input_tensor,
        filter_tensor,
        stride_tensor,
        dilation_tensor,
        paddings_tensor,
        h_output_ref,
    )

    # Copy data back to host
    ctx.enqueue_copy(out_host_buffer, d_output_buffer)
    ctx.synchronize()

    # Ensure the GPU and CPU results are the same
    assert_allclose(h_output_ref, output_tensor)


def test_max_pool_pad_dilation_2d_gpu(ctx: DeviceContext) raises:
    print("== test_max_pool_pad_dilation_2d_gpu")

    comptime in_layout = row_major[1, 4, 4, 1]()
    comptime out_layout = row_major[1, 1, 3, 1]()
    comptime in_size = in_layout.product()
    comptime out_size = out_layout.product()

    # Create host buffers
    var in_host_buffer = ctx.enqueue_create_host_buffer[.float32](in_size)
    var out_host_buffer = ctx.enqueue_create_host_buffer[.float32](out_size)
    var ref_host_buffer = ctx.enqueue_create_host_buffer[.float32](out_size)
    ctx.synchronize()

    # Create TileTensors from host buffers
    var input_tensor = TileTensor(in_host_buffer, in_layout)
    var output_tensor = TileTensor(out_host_buffer, out_layout)
    var h_output_ref = TileTensor(ref_host_buffer, out_layout)

    arange(input_tensor)
    for i in range(out_size):
        out_host_buffer[i] = 0
        ref_host_buffer[i] = 0

    # Create parameter tensors
    var paddings_stack: Array[Int32, _] = [0, 0, 2, 0]
    var paddings_tensor = TileTensor(paddings_stack, row_major[4]())

    var filter_stack = Array[Int32, 2](fill=2)
    var filter_tensor = TileTensor(filter_stack, row_major[2]())

    var stride_stack = Array[Int32, 2](fill=1)
    var stride_tensor = TileTensor(stride_stack, row_major[2]())

    var dilation_stack = Array[Int32, 2](fill=3)
    var dilation_tensor = TileTensor(dilation_stack, row_major[2]())

    # Create device buffers
    var d_input_buffer = ctx.enqueue_create_buffer[.float32](in_size)
    var d_output_buffer = ctx.enqueue_create_buffer[.float32](out_size)

    # Create device TileTensors
    var d_input = TileTensor(d_input_buffer, in_layout)
    var d_output = TileTensor(d_output_buffer, out_layout)

    # Copy data to device
    ctx.enqueue_copy(d_input_buffer, in_host_buffer)
    ctx.enqueue_copy(d_output_buffer, out_host_buffer)

    max_pool_gpu[int_type=DType.int32](
        ctx,
        d_input,
        filter_tensor,
        stride_tensor,
        dilation_tensor,
        paddings_tensor,
        d_output,
    )
    max_pool[int_type=DType.int32](
        input_tensor,
        filter_tensor,
        stride_tensor,
        dilation_tensor,
        paddings_tensor,
        h_output_ref,
    )

    # Copy data back to host
    ctx.enqueue_copy(out_host_buffer, d_output_buffer)
    ctx.synchronize()

    # Ensure the GPU and CPU results are the same
    assert_allclose(h_output_ref, output_tensor)


def test_pool_gpu_strided_channels[is_max: Bool](ctx: DeviceContext) raises:
    """Checks the pool kernels on tensors whose channel stride is not 1.

    A vector covers `simd_width` consecutive *physical* elements, which is only
    the channel axis when channels are contiguous. A strided channel axis (here
    an element-interleaved view) therefore has to take the scalar kernel; the
    reference is a plain host loop over the strided offsets, because the CPU
    pool kernel makes the same contiguity assumption at its own vector width.

    Parameters:
        is_max: Whether to test max pooling (average pooling otherwise).
    """
    print("== test_pool_gpu_strided_channels:", is_max)
    comptime H = 5
    comptime W = 7
    comptime C = 16
    comptime K = 2
    comptime S = 2
    comptime out_h = (H - K) // S + 1
    comptime out_w = (W - K) // S + 1
    # Element-interleaved views: every other physical element is a hole.
    comptime in_channel_stride = 2
    comptime out_channel_stride = 2
    comptime in_size = H * W * C * in_channel_stride
    comptime out_size = out_h * out_w * C * out_channel_stride

    comptime InLayout = Layout[
        shape_types=Coord[
            ComptimeInt[1], ComptimeInt[H], ComptimeInt[W], ComptimeInt[C]
        ].element_types,
        stride_types=Coord[
            ComptimeInt[H * W * C * in_channel_stride],
            ComptimeInt[W * C * in_channel_stride],
            ComptimeInt[C * in_channel_stride],
            ComptimeInt[in_channel_stride],
        ].element_types,
    ]
    comptime OutLayout = Layout[
        shape_types=Coord[
            ComptimeInt[1],
            ComptimeInt[out_h],
            ComptimeInt[out_w],
            ComptimeInt[C],
        ].element_types,
        stride_types=Coord[
            ComptimeInt[out_h * out_w * C * out_channel_stride],
            ComptimeInt[out_w * C * out_channel_stride],
            ComptimeInt[C * out_channel_stride],
            ComptimeInt[out_channel_stride],
        ].element_types,
    ]

    var in_host_buffer = ctx.enqueue_create_host_buffer[.float32](in_size)
    var out_host_buffer = ctx.enqueue_create_host_buffer[.float32](out_size)
    ctx.synchronize()
    for i in range(in_size):
        in_host_buffer[i] = Float32(i + 1)
    for i in range(out_size):
        out_host_buffer[i] = 0

    var input_tensor = TileTensor(
        in_host_buffer,
        InLayout(
            Coord[
                ComptimeInt[1],
                ComptimeInt[H],
                ComptimeInt[W],
                ComptimeInt[C],
            ](),
            Coord[
                ComptimeInt[H * W * C * in_channel_stride],
                ComptimeInt[W * C * in_channel_stride],
                ComptimeInt[C * in_channel_stride],
                ComptimeInt[in_channel_stride],
            ](),
        ),
    )
    var output_tensor = TileTensor(
        out_host_buffer,
        OutLayout(
            Coord[
                ComptimeInt[1],
                ComptimeInt[out_h],
                ComptimeInt[out_w],
                ComptimeInt[C],
            ](),
            Coord[
                ComptimeInt[out_h * out_w * C * out_channel_stride],
                ComptimeInt[out_w * C * out_channel_stride],
                ComptimeInt[C * out_channel_stride],
                ComptimeInt[out_channel_stride],
            ](),
        ),
    )

    # Create parameter tensors
    var paddings_stack = Array[Int32, 4](fill=0)
    var paddings_tensor = TileTensor(paddings_stack, row_major[4]())

    var filter_stack = Array[Int32, 2](fill=K)
    var filter_tensor = TileTensor(filter_stack, row_major[2]())

    var stride_stack = Array[Int32, 2](fill=S)
    var stride_tensor = TileTensor(stride_stack, row_major[2]())

    var dilation_stack = Array[Int32, 2](fill=1)
    var dilation_tensor = TileTensor(dilation_stack, row_major[2]())

    # Create device buffers
    var d_input_buffer = ctx.enqueue_create_buffer[.float32](in_size)
    var d_output_buffer = ctx.enqueue_create_buffer[.float32](out_size)
    var d_input = TileTensor(d_input_buffer, input_tensor.layout)
    var d_output = TileTensor(d_output_buffer, output_tensor.layout)

    ctx.enqueue_copy(d_input_buffer, in_host_buffer)
    ctx.enqueue_copy(d_output_buffer, out_host_buffer)

    if is_max:
        max_pool_gpu[int_type=DType.int32](
            ctx,
            d_input,
            filter_tensor,
            stride_tensor,
            dilation_tensor,
            paddings_tensor,
            d_output,
        )
    else:
        avg_pool_gpu[int_type=DType.int32](
            ctx,
            d_input,
            filter_tensor,
            stride_tensor,
            dilation_tensor,
            paddings_tensor,
            d_output,
        )

    ctx.enqueue_copy(out_host_buffer, d_output_buffer)
    ctx.synchronize()

    # Reference: pool over the strided offsets with plain host arithmetic.
    for i in range(out_h):
        for j in range(out_w):
            for c in range(C):
                var expected = in_host_buffer[
                    ((i * S) * W + j * S) * C * in_channel_stride
                    + c * in_channel_stride
                ]
                if not is_max:
                    # The average pool accumulates the whole window, so start
                    # from zero rather than from the window's first element.
                    expected = Scalar[.float32](0)
                for ki in range(K):
                    for kj in range(K):
                        var v = in_host_buffer[
                            ((i * S + ki) * W + j * S + kj)
                            * C
                            * in_channel_stride
                            + c * in_channel_stride
                        ]
                        if is_max:
                            expected = max(expected, v)
                        else:
                            expected = expected + v
                if not is_max:
                    expected = expected / Scalar[.float32](K * K)
                var out_idx = (
                    i * out_w + j
                ) * C * out_channel_stride + c * out_channel_stride
                if out_host_buffer[out_idx] != expected:
                    print(
                        "mismatch at",
                        i,
                        j,
                        c,
                        ": expected",
                        expected,
                        "got",
                        out_host_buffer[out_idx],
                    )
                    raise Error(
                        "strided-channel pool output does not match the"
                        " reference"
                    )


def test_pool_gpu_vectorized_channels[
    dtype: DType,
    H: Int,
    W: Int,
    C: Int,
    K: Int,
    S: Int,
    P: Int,
    is_max: Bool,
    count_boundary: Bool = False,
](ctx: DeviceContext) raises:
    """Checks the channel-vectorized GPU pool against the CPU reference.

    The GPU stencil hands each thread `simd_width` adjacent channels of one
    output point, so the channel count decides whether a call takes the
    vectorized path or the scalar fallback. Both must match the CPU path
    bit for bit, on odd spatial sizes and on channel counts that are not a
    multiple of the vector width.

    Parameters:
        dtype: Data type of the input and output tensors.
        H: Input height.
        W: Input width.
        C: Input channel count.
        K: Pool window size (square window).
        S: Pool stride (square stride).
        P: Symmetric padding per side.
        is_max: Whether to test max pooling (average pooling otherwise).
        count_boundary: Whether the average pool counts padded elements.
    """
    print(
        "== test_pool_gpu_vectorized_channels:",
        String(dtype),
        "C=",
        C,
        "K=",
        K,
        "S=",
        S,
        "P=",
        P,
        "is_max=",
        is_max,
    )

    comptime out_h = (H + 2 * P - K) // S + 1
    comptime out_w = (W + 2 * P - K) // S + 1
    comptime in_layout = row_major[1, H, W, C]()
    comptime out_layout = row_major[1, out_h, out_w, C]()
    comptime in_size = in_layout.product()
    comptime out_size = out_layout.product()

    # Create host buffers
    var in_host_buffer = ctx.enqueue_create_host_buffer[dtype](in_size)
    var out_host_buffer = ctx.enqueue_create_host_buffer[dtype](out_size)
    var ref_host_buffer = ctx.enqueue_create_host_buffer[dtype](out_size)
    ctx.synchronize()

    # Create TileTensors from host buffers
    var input_tensor = TileTensor(in_host_buffer, in_layout)
    var h_output_ref = TileTensor(ref_host_buffer, out_layout)

    arange(input_tensor)
    for i in range(out_size):
        out_host_buffer[i] = 0
        ref_host_buffer[i] = 0

    # Create parameter tensors
    var paddings_stack: Array[Int32, _] = [
        Int32(P),
        Int32(P),
        Int32(P),
        Int32(P),
    ]
    var paddings_tensor = TileTensor(paddings_stack, row_major[4]())

    var filter_stack = Array[Int32, 2](fill=Int32(K))
    var filter_tensor = TileTensor(filter_stack, row_major[2]())

    var stride_stack = Array[Int32, 2](fill=Int32(S))
    var stride_tensor = TileTensor(stride_stack, row_major[2]())

    var dilation_stack = Array[Int32, 2](fill=1)
    var dilation_tensor = TileTensor(dilation_stack, row_major[2]())

    # Create device buffers
    var d_input_buffer = ctx.enqueue_create_buffer[dtype](in_size)
    var d_output_buffer = ctx.enqueue_create_buffer[dtype](out_size)
    var d_input = TileTensor(d_input_buffer, in_layout)
    var d_output = TileTensor(d_output_buffer, out_layout)

    ctx.enqueue_copy(d_input_buffer, in_host_buffer)
    ctx.enqueue_copy(d_output_buffer, out_host_buffer)

    if is_max:
        max_pool_gpu[int_type=DType.int32](
            ctx,
            d_input,
            filter_tensor,
            stride_tensor,
            dilation_tensor,
            paddings_tensor,
            d_output,
        )
        max_pool[int_type=DType.int32](
            input_tensor,
            filter_tensor,
            stride_tensor,
            dilation_tensor,
            paddings_tensor,
            h_output_ref,
        )
    else:
        avg_pool_gpu[int_type=DType.int32, count_boundary=count_boundary](
            ctx,
            d_input,
            filter_tensor,
            stride_tensor,
            dilation_tensor,
            paddings_tensor,
            d_output,
        )
        avg_pool[int_type=DType.int32, count_boundary=count_boundary](
            input_tensor,
            filter_tensor,
            stride_tensor,
            dilation_tensor,
            paddings_tensor,
            h_output_ref,
        )

    # Copy data back to host
    ctx.enqueue_copy(out_host_buffer, d_output_buffer)
    ctx.synchronize()

    # Bit-identical, not just close: vectorizing along channels must not change
    # the arithmetic for any channel.
    for i in range(out_size):
        if out_host_buffer[i] != ref_host_buffer[i]:
            print(
                "mismatch at",
                i,
                ": cpu=",
                ref_host_buffer[i],
                "gpu=",
                out_host_buffer[i],
            )
            raise Error("GPU and CPU pool results are not bit-identical")


def assert_allclose[
    dtype: DType,
](
    h_output_ref: TileTensor[dtype, ...],
    h_output_gpu: TileTensor[dtype, ...],
) raises:
    var ref_flat = h_output_ref.reshape(
        row_major(h_output_ref.layout.product())
    )
    var gpu_flat = h_output_gpu.reshape(
        row_major(h_output_gpu.layout.product())
    )
    try:
        for i in range(h_output_ref.layout.product()):
            assert_almost_equal(
                ref_flat.load[width=1]((i,)),
                gpu_flat.load[width=1]((i,)),
            )
    except e:
        print(e)
        print("left: ", h_output_ref)
        print("right: ", h_output_gpu)
        raise Error("GPU and CPU results are not the same")
