# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from gpu.host import DeviceContext, HostBuffer
from layout._fillers import arange
from layout._layout import row_major
from layout._tile_tensor import TileTensor
from nn.pool import (
    PoolMethod,
    avg_pool,
    avg_pool_gpu,
    max_pool,
    max_pool_gpu,
)
from testing import assert_almost_equal

from utils.index import IndexList


def main():
    with DeviceContext() as ctx:
        test_max_pool_2d(ctx)
        test_avg_pool_2d(ctx)
        test_avg_pool_2d_with_padding_gpu[True](ctx)
        test_avg_pool_2d_with_padding_gpu[False](ctx)
        test_maxpool_2d_ceil_gpu(ctx)
        test_average_pool_2d_ceil_excludeBound_gpu(ctx)
        test_average_pool_2d_ceil_includeBound_gpu(ctx)
        test_max_pool_pad_dilation_2d_gpu(ctx)


fn test_max_pool_2d(ctx: DeviceContext) raises:
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


fn test_avg_pool_2d(ctx: DeviceContext) raises:
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


fn test_maxpool_2d_ceil_gpu(ctx: DeviceContext) raises:
    print("== test_max_pool_2d_ceil_gpu")
    pool_ceil_test(PoolMethod.MAX, ctx)


fn test_average_pool_2d_ceil_excludeBound_gpu(ctx: DeviceContext) raises:
    print("== test_average_pool_2d_ceil_excludeBound_gpu")
    pool_ceil_test(PoolMethod.AVG, ctx)


fn test_average_pool_2d_ceil_includeBound_gpu(ctx: DeviceContext) raises:
    print("== test_average_pool_2d_ceil_includeBound_gpu")
    pool_ceil_test[True, True](PoolMethod.AVG, ctx)


fn pool[
    count_boundary: Bool = False
](pool_method: PoolMethod, ctx: DeviceContext) raises:
    comptime in_layout = row_major[2, 5, 7, 2]()
    comptime out_layout = row_major[2, 2, 2, 2]()
    comptime in_size = in_layout.size()
    comptime out_size = out_layout.size()

    # Create host buffers
    var in_host_buffer = ctx.enqueue_create_host_buffer[DType.float32](in_size)
    var out_host_buffer = ctx.enqueue_create_host_buffer[DType.float32](
        out_size
    )
    var ref_host_buffer = ctx.enqueue_create_host_buffer[DType.float32](
        out_size
    )
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
    var paddings_stack = InlineArray[Scalar[DType.int32], 4](uninitialized=True)
    var paddings_tensor = TileTensor(paddings_stack, row_major[4]())
    paddings_tensor[0] = 0
    paddings_tensor[1] = 0
    paddings_tensor[2] = 0
    paddings_tensor[3] = 0

    var filter_stack = InlineArray[Scalar[DType.int32], 2](uninitialized=True)
    var filter_tensor = TileTensor(filter_stack, row_major[2]())
    filter_tensor[0] = 3
    filter_tensor[1] = 2

    var stride_stack = InlineArray[Scalar[DType.int32], 2](uninitialized=True)
    var stride_tensor = TileTensor(stride_stack, row_major[2]())
    stride_tensor[0] = 2
    stride_tensor[1] = 3

    var dilation_stack = InlineArray[Scalar[DType.int32], 2](uninitialized=True)
    var dilation_tensor = TileTensor(dilation_stack, row_major[2]())
    dilation_tensor[0] = 1
    dilation_tensor[1] = 1

    # Create device buffers
    var d_input_buffer = ctx.enqueue_create_buffer[DType.float32](in_size)
    var d_output_buffer = ctx.enqueue_create_buffer[DType.float32](out_size)

    # Create device TileTensors
    var d_input = TileTensor(d_input_buffer.unsafe_ptr(), in_layout)
    var d_output = TileTensor(d_output_buffer.unsafe_ptr(), out_layout)

    # Copy data to device
    ctx.enqueue_copy(d_input_buffer, in_host_buffer)
    ctx.enqueue_copy(d_output_buffer, out_host_buffer)

    if pool_method == PoolMethod.MAX:
        max_pool_gpu[int_type = DType.int32](
            ctx,
            d_input,
            filter_tensor,
            stride_tensor,
            dilation_tensor,
            paddings_tensor,
            d_output,
        )
        max_pool[int_type = DType.int32](
            input_tensor,
            filter_tensor,
            stride_tensor,
            dilation_tensor,
            paddings_tensor,
            h_output_ref,
        )
    else:
        avg_pool_gpu[int_type = DType.int32, count_boundary=count_boundary](
            ctx,
            d_input,
            filter_tensor,
            stride_tensor,
            dilation_tensor,
            paddings_tensor,
            d_output,
        )
        avg_pool[int_type = DType.int32, count_boundary=count_boundary](
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


fn pool_ceil_test[
    count_boundary: Bool = False, ceil_mode: Bool = True
](pool_method: PoolMethod, ctx: DeviceContext) raises:
    comptime in_layout = row_major[1, 4, 4, 1]()
    comptime out_layout = row_major[1, 2, 2, 1]()
    comptime in_size = in_layout.size()
    comptime out_size = out_layout.size()

    # Create host buffers
    var in_host_buffer = ctx.enqueue_create_host_buffer[DType.float32](in_size)
    var out_host_buffer = ctx.enqueue_create_host_buffer[DType.float32](
        out_size
    )
    var ref_host_buffer = ctx.enqueue_create_host_buffer[DType.float32](
        out_size
    )
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
    var paddings_stack = InlineArray[Scalar[DType.int32], 4](uninitialized=True)
    var paddings_tensor = TileTensor(paddings_stack, row_major[4]())
    paddings_tensor[0] = 0
    paddings_tensor[1] = 0
    paddings_tensor[2] = 0
    paddings_tensor[3] = 0

    var filter_stack = InlineArray[Scalar[DType.int32], 2](uninitialized=True)
    var filter_tensor = TileTensor(filter_stack, row_major[2]())
    filter_tensor[0] = 3
    filter_tensor[1] = 3

    var stride_stack = InlineArray[Scalar[DType.int32], 2](uninitialized=True)
    var stride_tensor = TileTensor(stride_stack, row_major[2]())
    stride_tensor[0] = 2
    stride_tensor[1] = 2

    var dilation_stack = InlineArray[Scalar[DType.int32], 2](uninitialized=True)
    var dilation_tensor = TileTensor(dilation_stack, row_major[2]())
    dilation_tensor[0] = 1
    dilation_tensor[1] = 1

    # Create device buffers
    var d_input_buffer = ctx.enqueue_create_buffer[DType.float32](in_size)
    var d_output_buffer = ctx.enqueue_create_buffer[DType.float32](out_size)

    # Create device TileTensors
    var d_input = TileTensor(d_input_buffer.unsafe_ptr(), in_layout)
    var d_output = TileTensor(d_output_buffer.unsafe_ptr(), out_layout)

    # Copy data to device
    ctx.enqueue_copy(d_input_buffer, in_host_buffer)
    ctx.enqueue_copy(d_output_buffer, out_host_buffer)

    if pool_method == PoolMethod.MAX:
        max_pool_gpu[int_type = DType.int32](
            ctx,
            d_input,
            filter_tensor,
            stride_tensor,
            dilation_tensor,
            paddings_tensor,
            d_output,
            ceil_mode,
        )
        max_pool[int_type = DType.int32](
            input_tensor,
            filter_tensor,
            stride_tensor,
            dilation_tensor,
            paddings_tensor,
            h_output_ref,
            ceil_mode,
        )
    else:
        avg_pool_gpu[int_type = DType.int32, count_boundary=count_boundary](
            ctx,
            d_input,
            filter_tensor,
            stride_tensor,
            dilation_tensor,
            paddings_tensor,
            d_output,
            ceil_mode,
        )
        avg_pool[int_type = DType.int32, count_boundary=count_boundary](
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


fn test_avg_pool_2d_with_padding_gpu[
    count_boundary: Bool = False
](ctx: DeviceContext) raises:
    print("== test_avg_pool_2d_with_padding_gpu:", count_boundary)

    comptime in_layout = row_major[1, 7, 7, 1]()
    comptime out_layout = row_major[1, 7, 7, 1]()
    comptime in_size = in_layout.size()
    comptime out_size = out_layout.size()

    # Create host buffers
    var in_host_buffer = ctx.enqueue_create_host_buffer[DType.float32](in_size)
    var out_host_buffer = ctx.enqueue_create_host_buffer[DType.float32](
        out_size
    )
    var ref_host_buffer = ctx.enqueue_create_host_buffer[DType.float32](
        out_size
    )
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
    var paddings_stack = InlineArray[Scalar[DType.int32], 4](uninitialized=True)
    var paddings_tensor = TileTensor(paddings_stack, row_major[4]())
    paddings_tensor[0] = 1
    paddings_tensor[1] = 1
    paddings_tensor[2] = 1
    paddings_tensor[3] = 1

    var filter_stack = InlineArray[Scalar[DType.int32], 2](uninitialized=True)
    var filter_tensor = TileTensor(filter_stack, row_major[2]())
    filter_tensor[0] = 3
    filter_tensor[1] = 3

    var stride_stack = InlineArray[Scalar[DType.int32], 2](uninitialized=True)
    var stride_tensor = TileTensor(stride_stack, row_major[2]())
    stride_tensor[0] = 1
    stride_tensor[1] = 1

    var dilation_stack = InlineArray[Scalar[DType.int32], 2](uninitialized=True)
    var dilation_tensor = TileTensor(dilation_stack, row_major[2]())
    dilation_tensor[0] = 1
    dilation_tensor[1] = 1

    # Create device buffers
    var d_input_buffer = ctx.enqueue_create_buffer[DType.float32](in_size)
    var d_output_buffer = ctx.enqueue_create_buffer[DType.float32](out_size)

    # Create device TileTensors
    var d_input = TileTensor(d_input_buffer.unsafe_ptr(), in_layout)
    var d_output = TileTensor(d_output_buffer.unsafe_ptr(), out_layout)

    # Copy data to device
    ctx.enqueue_copy(d_input_buffer, in_host_buffer)
    ctx.enqueue_copy(d_output_buffer, out_host_buffer)

    avg_pool_gpu[int_type = DType.int32, count_boundary=count_boundary](
        ctx,
        d_input,
        filter_tensor,
        stride_tensor,
        dilation_tensor,
        paddings_tensor,
        d_output,
    )
    avg_pool[int_type = DType.int32, count_boundary=count_boundary](
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


fn test_max_pool_pad_dilation_2d_gpu(ctx: DeviceContext) raises:
    print("== test_max_pool_pad_dilation_2d_gpu")

    comptime in_layout = row_major[1, 4, 4, 1]()
    comptime out_layout = row_major[1, 1, 3, 1]()
    comptime in_size = in_layout.size()
    comptime out_size = out_layout.size()

    # Create host buffers
    var in_host_buffer = ctx.enqueue_create_host_buffer[DType.float32](in_size)
    var out_host_buffer = ctx.enqueue_create_host_buffer[DType.float32](
        out_size
    )
    var ref_host_buffer = ctx.enqueue_create_host_buffer[DType.float32](
        out_size
    )
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
    var paddings_stack = InlineArray[Scalar[DType.int32], 4](uninitialized=True)
    var paddings_tensor = TileTensor(paddings_stack, row_major[4]())
    paddings_tensor[0] = 0
    paddings_tensor[1] = 0
    paddings_tensor[2] = 2
    paddings_tensor[3] = 0

    var filter_stack = InlineArray[Scalar[DType.int32], 2](uninitialized=True)
    var filter_tensor = TileTensor(filter_stack, row_major[2]())
    filter_tensor[0] = 2
    filter_tensor[1] = 2

    var stride_stack = InlineArray[Scalar[DType.int32], 2](uninitialized=True)
    var stride_tensor = TileTensor(stride_stack, row_major[2]())
    stride_tensor[0] = 1
    stride_tensor[1] = 1

    var dilation_stack = InlineArray[Scalar[DType.int32], 2](uninitialized=True)
    var dilation_tensor = TileTensor(dilation_stack, row_major[2]())
    dilation_tensor[0] = 3
    dilation_tensor[1] = 3

    # Create device buffers
    var d_input_buffer = ctx.enqueue_create_buffer[DType.float32](in_size)
    var d_output_buffer = ctx.enqueue_create_buffer[DType.float32](out_size)

    # Create device TileTensors
    var d_input = TileTensor(d_input_buffer.unsafe_ptr(), in_layout)
    var d_output = TileTensor(d_output_buffer.unsafe_ptr(), out_layout)

    # Copy data to device
    ctx.enqueue_copy(d_input_buffer, in_host_buffer)
    ctx.enqueue_copy(d_output_buffer, out_host_buffer)

    max_pool_gpu[int_type = DType.int32](
        ctx,
        d_input,
        filter_tensor,
        stride_tensor,
        dilation_tensor,
        paddings_tensor,
        d_output,
    )
    max_pool[int_type = DType.int32](
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


fn assert_allclose[
    dtype: DType,
](
    h_output_ref: TileTensor[dtype, ...],
    h_output_gpu: TileTensor[dtype, ...],
) raises:
    try:
        for i in range(h_output_ref.layout.size()):
            assert_almost_equal(h_output_ref.ptr[i], h_output_gpu.ptr[i])
    except e:
        print(e)
        print("left: ", h_output_ref)
        print("right: ", h_output_gpu)
        raise Error("GPU and CPU results are not the same")
