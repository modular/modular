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

from gpu.host import DeviceContext
from memory import UnsafePointer
from layout import (
    LayoutTensor,
    Layout,
    RuntimeLayout,
    UNKNOWN_VALUE,
)
from layout._fillers import arange
from nn.pool import (
    PoolMethod,
    avg_pool,
    avg_pool_gpu,
    max_pool,
    max_pool_gpu,
)
from testing import assert_almost_equal

from utils.index import IndexList


fn main() raises:
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


# CHECK-LABEL: test_avg_pool_2d
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


fn test_maxpool_2d_ceil(ctx: DeviceContext) raises:
    print("== test_max_pool_2d_ceil")
    pool_ceil_test(PoolMethod.MAX, ctx)


fn test_average_pool_2d_ceil_excludeBound(ctx: DeviceContext) raises:
    print("== test_average_pool_2d_ceil_excludeBound")
    pool_ceil_test(PoolMethod.AVG, ctx)


fn test_average_pool_2d_ceil_includeBound(ctx: DeviceContext) raises:
    print("== test_average_pool_2d_ceil_includeBound")
    pool_ceil_test[True, True](PoolMethod.AVG, ctx)


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
    alias in_layout = Layout.row_major(2, 5, 7, 2)
    alias out_layout = Layout.row_major(2, 2, 2, 2)

    var in_heap = List[Float32](capacity=in_layout.size())
    var input_tensor = LayoutTensor[DType.float32, in_layout](in_heap)
    arange(input_tensor)

    var out_heap = List[Float32](capacity=out_layout.size())
    var output_tensor = LayoutTensor[DType.float32, out_layout](out_heap).fill(
        0
    )

    var h_output_ref_ptr = UnsafePointer[Float32].alloc(Int(out_layout.size()))
    var h_output_ref = LayoutTensor[DType.float32, Layout.row_major[4]()](
        h_output_ref_ptr,
        RuntimeLayout[
            Layout.row_major[4](), element_type = output_tensor.layout_int_type
        ].row_major(
            IndexList[4, element_type = output_tensor.layout_int_type](
                2, 2, 2, 2
            )
        ),
    ).fill(0)

    var paddings = List[Int32](0, 0, 0, 0)
    var filter = List[Int32](3, 2)
    var stride = List[Int32](2, 3)
    var dilation = List[Int32](1, 1)

    var paddings_tensor = LayoutTensor[
        DType.int32, Layout.row_major(UNKNOWN_VALUE)
    ](
        paddings,
        RuntimeLayout[
            Layout.row_major(UNKNOWN_VALUE), element_type = DType.int32
        ].row_major(IndexList[1, element_type = DType.int32](4)),
    )
    var filter_tensor = LayoutTensor[
        DType.int32, Layout.row_major(UNKNOWN_VALUE)
    ](
        filter,
        RuntimeLayout[
            Layout.row_major(UNKNOWN_VALUE), element_type = DType.int32
        ].row_major(IndexList[1, element_type = DType.int32](2)),
    )
    var stride_tensor = LayoutTensor[
        DType.int32, Layout.row_major(UNKNOWN_VALUE)
    ](
        stride,
        RuntimeLayout[
            Layout.row_major(UNKNOWN_VALUE), element_type = DType.int32
        ].row_major(IndexList[1, element_type = DType.int32](2)),
    )
    var dilation_tensor = LayoutTensor[
        DType.int32, Layout.row_major(UNKNOWN_VALUE)
    ](
        dilation,
        RuntimeLayout[
            Layout.row_major(UNKNOWN_VALUE), element_type = DType.int32
        ].row_major(IndexList[1, element_type = DType.int32](2)),
    )

    # Copy data to device
    var d_input_buffer = ctx.enqueue_create_buffer[DType.float32](
        in_layout.size()
    )
    var d_output_buffer = ctx.enqueue_create_buffer[DType.float32](
        out_layout.size()
    )
    var d_input = LayoutTensor[DType.float32, in_layout](d_input_buffer)
    var d_output = LayoutTensor[DType.float32, out_layout](d_output_buffer)

    ctx.enqueue_copy(d_input_buffer, input_tensor.ptr)
    ctx.enqueue_copy(d_output_buffer, output_tensor.ptr)

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
    ctx.enqueue_copy(output_tensor.ptr, d_output_buffer)
    ctx.synchronize()

    # Ensure the GPU and CPU results are the same
    assert_allclose(h_output_ref, output_tensor)

    h_output_ref_ptr.free()


fn pool_ceil_test[
    count_boundary: Bool = False, ceil_mode: Bool = True
](pool_method: PoolMethod, ctx: DeviceContext) raises:
    alias in_layout = Layout.row_major(1, 4, 4, 1)
    alias out_layout = Layout.row_major(1, 2, 2, 1)

    var in_heap = List[Float32](capacity=in_layout.size())
    var input_tensor = LayoutTensor[DType.float32, in_layout](in_heap)
    arange(input_tensor)

    var out_heap = List[Float32](capacity=out_layout.size())
    var output_tensor = LayoutTensor[DType.float32, out_layout](out_heap).fill(
        0
    )

    var h_output_ref_ptr = UnsafePointer[Float32].alloc(out_layout.size())
    var h_output_ref = LayoutTensor[DType.float32, out_layout](
        h_output_ref_ptr,
        RuntimeLayout[
            out_layout, element_type = output_tensor.layout_int_type
        ].row_major(
            IndexList[4, element_type = output_tensor.layout_int_type](
                1, 2, 2, 1
            )
        ),
    ).fill(0)

    var paddings = List[Int32](0, 0, 0, 0)
    var filter = List[Int32](3, 3)
    var stride = List[Int32](2, 2)
    var dilation = List[Int32](1, 1)

    var paddings_tensor = LayoutTensor[
        DType.int32, Layout.row_major(UNKNOWN_VALUE)
    ](
        paddings,
        RuntimeLayout[
            Layout.row_major(UNKNOWN_VALUE),
            element_type = input_tensor.layout_int_type,
        ].row_major(
            IndexList[1, element_type = input_tensor.layout_int_type](4)
        ),
    )
    var filter_tensor = LayoutTensor[
        DType.int32, Layout.row_major(UNKNOWN_VALUE)
    ](
        filter,
        RuntimeLayout[
            Layout.row_major(UNKNOWN_VALUE),
            element_type = input_tensor.layout_int_type,
        ].row_major(
            IndexList[1, element_type = input_tensor.layout_int_type](2)
        ),
    )
    var stride_tensor = LayoutTensor[
        DType.int32, Layout.row_major(UNKNOWN_VALUE)
    ](
        stride,
        RuntimeLayout[
            Layout.row_major(UNKNOWN_VALUE),
            element_type = input_tensor.layout_int_type,
        ].row_major(
            IndexList[1, element_type = input_tensor.layout_int_type](2)
        ),
    )
    var dilation_tensor = LayoutTensor[
        DType.int32, Layout.row_major(UNKNOWN_VALUE)
    ](
        dilation,
        RuntimeLayout[
            Layout.row_major(UNKNOWN_VALUE),
            element_type = input_tensor.layout_int_type,
        ].row_major(
            IndexList[1, element_type = input_tensor.layout_int_type](2)
        ),
    )

    # Copy data to device
    var d_input_buffer = ctx.enqueue_create_buffer[DType.float32](
        in_layout.size()
    )
    var d_output_buffer = ctx.enqueue_create_buffer[DType.float32](
        in_layout.size()
    )
    var d_input = LayoutTensor[DType.float32, in_layout](d_input_buffer)
    var d_output = LayoutTensor[DType.float32, out_layout](d_output_buffer)

    ctx.enqueue_copy(d_input_buffer, input_tensor.ptr)
    ctx.enqueue_copy(d_output_buffer, output_tensor.ptr)

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
    ctx.enqueue_copy(output_tensor.ptr, d_output_buffer)
    ctx.synchronize()

    # Ensure the GPU and CPU results are the same
    assert_allclose(h_output_ref, output_tensor)

    h_output_ref_ptr.free()


fn test_avg_pool_2d_with_padding_gpu[
    count_boundary: Bool = False
](ctx: DeviceContext) raises:
    print("== test_avg_pool_2d_with_padding_gpu:", count_boundary)

    alias in_layout = Layout.row_major(1, 7, 7, 1)
    alias out_layout = Layout.row_major(1, 7, 7, 1)

    var in_heap = List[Float32](capacity=in_layout.size())
    var input_tensor = LayoutTensor[DType.float32, in_layout](in_heap)
    arange(input_tensor)

    var out_heap = List[Float32](capacity=out_layout.size())
    var output_tensor = LayoutTensor[DType.float32, out_layout](out_heap).fill(
        0
    )

    var h_output_ref_ptr = UnsafePointer[Float32].alloc(out_layout.size())
    var h_output_ref = LayoutTensor[DType.float32, out_layout](
        h_output_ref_ptr,
        RuntimeLayout[
            out_layout, element_type = output_tensor.layout_int_type
        ].row_major(
            IndexList[4, element_type = output_tensor.layout_int_type](
                1, 7, 7, 1
            )
        ),
    ).fill(0)

    var paddings = List[Int32](1, 1, 1, 1)
    var filter = List[Int32](3, 3)
    var stride = List[Int32](1, 1)
    var dilation = List[Int32](1, 1)

    var paddings_tensor = LayoutTensor[
        DType.int32, Layout.row_major(UNKNOWN_VALUE)
    ](
        paddings,
        RuntimeLayout[
            Layout.row_major(UNKNOWN_VALUE),
            element_type = input_tensor.layout_int_type,
        ].row_major(
            IndexList[1, element_type = input_tensor.layout_int_type](4)
        ),
    )
    var filter_tensor = LayoutTensor[
        DType.int32, Layout.row_major(UNKNOWN_VALUE)
    ](
        filter,
        RuntimeLayout[
            Layout.row_major(UNKNOWN_VALUE),
            element_type = input_tensor.layout_int_type,
        ].row_major(
            IndexList[1, element_type = input_tensor.layout_int_type](2)
        ),
    )
    var stride_tensor = LayoutTensor[
        DType.int32, Layout.row_major(UNKNOWN_VALUE)
    ](
        stride,
        RuntimeLayout[
            Layout.row_major(UNKNOWN_VALUE),
            element_type = input_tensor.layout_int_type,
        ].row_major(
            IndexList[1, element_type = input_tensor.layout_int_type](2)
        ),
    )
    var dilation_tensor = LayoutTensor[
        DType.int32, Layout.row_major(UNKNOWN_VALUE)
    ](
        dilation,
        RuntimeLayout[
            Layout.row_major(UNKNOWN_VALUE),
            element_type = input_tensor.layout_int_type,
        ].row_major(
            IndexList[1, element_type = input_tensor.layout_int_type](2)
        ),
    )

    # Copy data to device
    var d_input_buffer = ctx.enqueue_create_buffer[DType.float32](
        in_layout.size()
    )
    var d_output_buffer = ctx.enqueue_create_buffer[DType.float32](
        in_layout.size()
    )
    var d_input = LayoutTensor[DType.float32, in_layout](d_input_buffer)
    var d_output = LayoutTensor[DType.float32, out_layout](d_output_buffer)

    ctx.enqueue_copy(d_input_buffer, input_tensor.ptr)
    ctx.enqueue_copy(d_output_buffer, output_tensor.ptr)

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
    ctx.enqueue_copy(output_tensor.ptr, d_output_buffer)
    ctx.synchronize()

    # Ensure the GPU and CPU results are the same
    assert_allclose(h_output_ref, output_tensor)

    h_output_ref_ptr.free()


fn test_max_pool_pad_dilation_2d_gpu(ctx: DeviceContext) raises:
    print("== test_max_pool_pad_dilation_2d_gpu")

    alias in_layout = Layout.row_major(1, 4, 4, 1)
    alias out_layout = Layout.row_major(1, 1, 3, 1)

    var in_heap = List[Float32](capacity=in_layout.size())
    var input_tensor = LayoutTensor[DType.float32, in_layout](in_heap)
    arange(input_tensor)

    var out_heap = List[Float32](capacity=out_layout.size())
    var output_tensor = LayoutTensor[DType.float32, out_layout](out_heap).fill(
        0
    )

    var h_output_ref_ptr = UnsafePointer[Float32].alloc(out_layout.size())
    var h_output_ref = LayoutTensor[DType.float32, out_layout](
        h_output_ref_ptr,
        RuntimeLayout[
            out_layout,
            element_type = output_tensor.layout_int_type,
        ].row_major(
            IndexList[4, element_type = output_tensor.layout_int_type](
                1, 1, 3, 1
            )
        ),
    ).fill(0)

    var paddings = List[Int32](0, 0, 2, 0)
    var filter = List[Int32](2, 2)
    var stride = List[Int32](1, 1)
    var dilation = List[Int32](3, 3)

    var paddings_tensor = LayoutTensor[
        DType.int32, Layout.row_major(UNKNOWN_VALUE)
    ](
        paddings,
        RuntimeLayout[
            Layout.row_major(UNKNOWN_VALUE),
            element_type = input_tensor.layout_int_type,
        ].row_major(
            IndexList[1, element_type = input_tensor.layout_int_type](4)
        ),
    )
    var filter_tensor = LayoutTensor[
        DType.int32, Layout.row_major(UNKNOWN_VALUE)
    ](
        filter,
        RuntimeLayout[
            Layout.row_major(UNKNOWN_VALUE),
            element_type = input_tensor.layout_int_type,
        ].row_major(
            IndexList[1, element_type = input_tensor.layout_int_type](2)
        ),
    )
    var stride_tensor = LayoutTensor[
        DType.int32, Layout.row_major(UNKNOWN_VALUE)
    ](
        stride,
        RuntimeLayout[
            Layout.row_major(UNKNOWN_VALUE),
            element_type = input_tensor.layout_int_type,
        ].row_major(
            IndexList[1, element_type = input_tensor.layout_int_type](2)
        ),
    )
    var dilation_tensor = LayoutTensor[
        DType.int32, Layout.row_major(UNKNOWN_VALUE)
    ](
        dilation,
        RuntimeLayout[
            Layout.row_major(UNKNOWN_VALUE),
            element_type = input_tensor.layout_int_type,
        ].row_major(
            IndexList[1, element_type = input_tensor.layout_int_type](2)
        ),
    )

    # Copy data to device
    var d_input_buffer = ctx.enqueue_create_buffer[DType.float32](
        in_layout.size()
    )
    var d_output_buffer = ctx.enqueue_create_buffer[DType.float32](
        in_layout.size()
    )
    var d_input = LayoutTensor[DType.float32, in_layout](d_input_buffer)
    var d_output = LayoutTensor[DType.float32, out_layout](d_output_buffer)

    ctx.enqueue_copy(d_input_buffer, input_tensor.ptr)
    ctx.enqueue_copy(d_output_buffer, output_tensor.ptr)

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
    ctx.enqueue_copy(output_tensor.ptr, d_output_buffer)
    ctx.synchronize()

    # Ensure the GPU and CPU results are the same
    assert_allclose(h_output_ref, output_tensor)

    h_output_ref_ptr.free()


fn assert_allclose[
    dtype: DType,
](
    h_output_ref: LayoutTensor[dtype, **_],
    h_output_gpu: LayoutTensor[dtype, **_],
) raises:
    try:
        for i in range(h_output_ref.size()):
            assert_almost_equal(h_output_ref.ptr[i], h_output_gpu.ptr[i])
    except e:
        print(e)
        print("left: ", h_output_ref)
        print("right: ", h_output_gpu)
        raise Error("GPU and CPU results are not the same")
