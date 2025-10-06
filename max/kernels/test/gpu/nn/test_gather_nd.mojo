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

from buffer.dimlist import DimList
from gpu.host import DeviceContext
from internal_utils import DeviceNDBuffer, HostNDBuffer
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from nn.gather_scatter import _gather_nd_impl, gather_nd_shape
from testing import assert_equal, assert_true

from utils import IndexList


def execute_gather_nd_test[
    data_type: DType, //,
    batch_dims: Int,
](
    data_host: HostNDBuffer[data_type, **_],
    indices_host: HostNDBuffer,
    expected_output: HostNDBuffer[data_type, **_],
    ctx: DeviceContext,
):
    # create device-side buffers and copy data to them
    var data_device = DeviceNDBuffer[
        data_host.dtype, data_host.rank, data_host.shape
    ](
        data_host.tensor.get_shape(),
        ctx=ctx,
    )
    var indices_device = DeviceNDBuffer[
        indices_host.dtype, indices_host.rank, indices_host.shape
    ](
        indices_host.tensor.get_shape(),
        ctx=ctx,
    )
    var actual_output_host = HostNDBuffer[
        expected_output.dtype, expected_output.rank, expected_output.shape
    ](
        expected_output.tensor.get_shape(),
    )
    var actual_output_device = DeviceNDBuffer[
        expected_output.dtype, expected_output.rank, expected_output.shape
    ](
        expected_output.tensor.get_shape(),
        ctx=ctx,
    )
    ctx.enqueue_copy(data_device.buffer, data_host.tensor.data)
    ctx.enqueue_copy(indices_device.buffer, indices_host.tensor.data)

    var data_device_tensor = data_device.to_layout_tensor()
    var indices_device_tensor = indices_device.to_layout_tensor()
    var actual_output_device_tensor = actual_output_device.to_layout_tensor()

    alias data_layout = Layout.row_major[data_device_tensor.rank]()
    alias indices_layout = Layout.row_major[indices_device_tensor.rank]()
    alias actual_output_layout = Layout.row_major[
        actual_output_device_tensor.rank
    ]()

    # execute the kernel
    _gather_nd_impl[batch_dims, target="gpu"](
        LayoutTensor[data_device_tensor.dtype, data_layout](
            data_device_tensor.ptr,
            RuntimeLayout[data_layout].row_major(
                data_device_tensor.runtime_layout.shape.value.canonicalize()
            ),
        ),
        LayoutTensor[indices_device_tensor.dtype, indices_layout](
            indices_device_tensor.ptr,
            RuntimeLayout[indices_layout].row_major(
                indices_device_tensor.runtime_layout.shape.value.canonicalize()
            ),
        ),
        LayoutTensor[actual_output_device_tensor.dtype, actual_output_layout](
            actual_output_device_tensor.ptr,
            RuntimeLayout[actual_output_layout].row_major(
                actual_output_device_tensor.runtime_layout.shape.value.canonicalize()
            ),
        ),
        ctx,
    )

    # copy the output back to host
    ctx.enqueue_copy(
        actual_output_host.tensor.data, actual_output_device.buffer
    )
    ctx.synchronize()

    # check that our shapes are consistent and that the contents of the output are consistent
    assert_true(
        actual_output_host.tensor.dynamic_shape
        == expected_output.tensor.dynamic_shape
    )
    for i in range(actual_output_host.tensor.num_elements()):
        assert_equal(
            actual_output_host.tensor.data[i], expected_output.tensor.data[i]
        )

    _ = data_device^
    _ = indices_device^
    _ = actual_output_host^
    _ = actual_output_device^


fn test_gather_nd_eg1(ctx: DeviceContext) raises:
    # Example 1
    alias batch_dims = 0
    alias data_rank = 2
    alias data_type = DType.int32
    alias data_layout = Layout.row_major[data_rank]()
    var data = HostNDBuffer[data_type, data_rank, DimList(2, 2)](
        IndexList[data_rank](2, 2)
    )

    var data_tensor = data.to_layout_tensor()

    data_tensor[0, 0] = 0
    data_tensor[0, 1] = 1
    data_tensor[1, 0] = 2
    data_tensor[1, 1] = 3

    alias indices_rank = 2
    alias indices_layout = Layout.row_major[indices_rank]()
    var indices = HostNDBuffer[DType.int64, indices_rank, DimList(2, 2)](
        IndexList[indices_rank](2, 2)
    )
    var indices_tensor = indices.to_layout_tensor()

    indices_tensor[0, 0] = 0
    indices_tensor[0, 1] = 0
    indices_tensor[1, 0] = 1
    indices_tensor[1, 1] = 1

    alias output_rank = 1
    alias layout_2d = Layout.row_major[2]()
    var output_shape = gather_nd_shape[
        output_rank,
        data_type,
        DType.int64,
        batch_dims,
    ](
        LayoutTensor[data_tensor.dtype, data_layout](
            data_tensor.ptr,
            RuntimeLayout[data_layout].row_major(
                data_tensor.runtime_layout.shape.value.canonicalize()
            ),
        ),
        LayoutTensor[indices_tensor.dtype, indices_layout](
            indices_tensor.ptr,
            RuntimeLayout[indices_layout].row_major(
                indices_tensor.runtime_layout.shape.value.canonicalize()
            ),
        ),
    )

    var expected_output = HostNDBuffer[data_type, output_rank](output_shape)
    var expected_output_tensor = expected_output.to_layout_tensor()
    expected_output_tensor[0] = 0
    expected_output_tensor[1,] = 3
    execute_gather_nd_test[batch_dims](data, indices, expected_output, ctx)


fn test_gather_nd_eg2(ctx: DeviceContext) raises:
    # Example 2
    alias batch_dims = 0
    alias data_rank = 2
    alias data_type = DType.int8
    alias data_layout = Layout.row_major[data_rank]()

    var data = HostNDBuffer[data_type, data_rank, DimList(2, 2)](
        IndexList[data_rank](2, 2)
    )
    var data_tensor = data.to_layout_tensor()
    data_tensor[0, 0] = 0
    data_tensor[0, 1] = 1
    data_tensor[1, 0] = 2
    data_tensor[1, 1] = 3

    alias indices_rank = 2
    alias indices_layout = Layout.row_major[indices_rank]()
    var indices = HostNDBuffer[DType.int64, indices_rank, DimList(2, 1)](
        IndexList[indices_rank](2, 1)
    )
    var indices_tensor = indices.to_layout_tensor()

    indices_tensor[0, 0] = 1
    indices_tensor[1, 0] = 0

    alias output_rank = 2
    var output_shape = gather_nd_shape[
        output_rank,
        data_type,
        DType.int64,
        batch_dims,
    ](
        LayoutTensor[data_tensor.dtype, data_layout](
            data_tensor.ptr,
            RuntimeLayout[data_layout].row_major(
                data_tensor.runtime_layout.shape.value.canonicalize(),
            ),
        ),
        LayoutTensor[indices_tensor.dtype, indices_layout](
            indices_tensor.ptr,
            RuntimeLayout[indices_layout].row_major(
                indices_tensor.runtime_layout.shape.value.canonicalize(),
            ),
        ),
    )

    var expected_output = HostNDBuffer[data_type, output_rank](output_shape)
    var expected_tensor = expected_output.to_layout_tensor()
    expected_tensor[0, 0] = 2
    expected_tensor[0, 1] = 3
    expected_tensor[1, 0] = 0
    expected_tensor[1, 1] = 1
    execute_gather_nd_test[batch_dims](data, indices, expected_output, ctx)


fn test_gather_nd_eg3(ctx: DeviceContext) raises:
    # Example 3
    alias batch_dims = 0
    alias data_rank = 3
    alias data_type = DType.float32
    alias data_layout = Layout.row_major[data_rank]()
    var data = HostNDBuffer[data_type, data_rank, DimList(2, 2, 2)](
        IndexList[data_rank](2, 2, 2)
    )
    var data_tensor = data.to_layout_tensor()

    data_tensor[0, 0, 0] = 0
    data_tensor[0, 0, 1] = 1
    data_tensor[0, 1, 0] = 2
    data_tensor[0, 1, 1] = 3
    data_tensor[1, 0, 0] = 4
    data_tensor[1, 0, 1] = 5
    data_tensor[1, 1, 0] = 6
    data_tensor[1, 1, 1] = 7

    alias indices_rank = 2
    alias indices_layout = Layout.row_major[indices_rank]()
    var indices = HostNDBuffer[DType.int64, indices_rank, DimList(2, 2)](
        IndexList[indices_rank](2, 2)
    )
    var indices_tensor = indices.to_layout_tensor()

    indices_tensor[0, 0] = 0
    indices_tensor[0, 1] = 1
    indices_tensor[1, 0] = 1
    indices_tensor[1, 1] = 0

    alias output_rank = 2
    var output_shape = gather_nd_shape[
        output_rank,
        data_type,
        DType.int64,
        batch_dims,
    ](
        LayoutTensor[data_tensor.dtype, data_layout](
            data_tensor.ptr,
            RuntimeLayout[data_layout].row_major(
                data_tensor.runtime_layout.shape.value.canonicalize()
            ),
        ),
        LayoutTensor[indices_tensor.dtype, indices_layout](
            indices_tensor.ptr,
            RuntimeLayout[indices_layout].row_major(
                indices_tensor.runtime_layout.shape.value.canonicalize()
            ),
        ),
    )

    var expected_output = HostNDBuffer[data_type, output_rank](output_shape)
    var expected_output_tensor = expected_output.to_layout_tensor()
    expected_output_tensor[0, 0] = 2
    expected_output_tensor[0, 1] = 3
    expected_output_tensor[1, 0] = 4
    expected_output_tensor[1, 1] = 5

    execute_gather_nd_test[batch_dims](data, indices, expected_output, ctx)


fn test_gather_nd_eg4(ctx: DeviceContext) raises:
    # Example 4
    alias batch_dims = 0
    alias data_rank = 3
    alias data_type = DType.int8
    alias data_layout = Layout.row_major[data_rank]()
    var data = HostNDBuffer[data_type, data_rank, DimList(2, 2, 2)](
        IndexList[data_rank](2, 2, 2)
    )
    var data_tensor = data.to_layout_tensor()

    data_tensor[0, 0, 0] = 0
    data_tensor[0, 0, 1] = 1
    data_tensor[0, 1, 0] = 2
    data_tensor[0, 1, 1] = 3
    data_tensor[1, 0, 0] = 4
    data_tensor[1, 0, 1] = 5
    data_tensor[1, 1, 0] = 6
    data_tensor[1, 1, 1] = 7

    alias indices_rank = 3
    alias indices_layout = Layout.row_major[indices_rank]()
    var indices = HostNDBuffer[DType.int64, indices_rank, DimList(2, 1, 2)](
        IndexList[indices_rank](2, 1, 2)
    )
    var indices_tensor = indices.to_layout_tensor()

    indices_tensor[0, 0, 0] = 0
    indices_tensor[0, 0, 1] = 1
    indices_tensor[1, 0, 0] = 1
    indices_tensor[1, 0, 1] = 0

    alias output_rank = 3
    var output_shape = gather_nd_shape[
        output_rank,
        data_type,
        DType.int64,
        batch_dims,
    ](
        LayoutTensor[data_tensor.dtype, data_layout](
            data_tensor.ptr,
            RuntimeLayout[data_layout].row_major(
                data_tensor.runtime_layout.shape.value.canonicalize()
            ),
        ),
        LayoutTensor[indices_tensor.dtype, indices_layout](
            indices_tensor.ptr,
            RuntimeLayout[indices_layout].row_major(
                indices_tensor.runtime_layout.shape.value.canonicalize()
            ),
        ),
    )

    var expected_output = HostNDBuffer[data_type, output_rank](output_shape)
    var expected_output_tensor = expected_output.to_layout_tensor()
    expected_output_tensor[0, 0, 0] = 2
    expected_output_tensor[0, 0, 1] = 3
    expected_output_tensor[1, 0, 0] = 4
    expected_output_tensor[1, 0, 1] = 5

    execute_gather_nd_test[batch_dims](data, indices, expected_output, ctx)


fn test_gather_nd_eg5(ctx: DeviceContext) raises:
    # Example 5
    alias batch_dims = 1
    alias data_rank = 3
    alias data_type = DType.int32
    alias data_layout = Layout.row_major[data_rank]()
    var data = HostNDBuffer[data_type, data_rank, DimList(2, 2, 2)](
        IndexList[data_rank](2, 2, 2)
    )
    var data_tensor = data.to_layout_tensor()

    data_tensor[0, 0, 0] = 0
    data_tensor[0, 0, 1] = 1
    data_tensor[0, 1, 0] = 2
    data_tensor[0, 1, 1] = 3
    data_tensor[1, 0, 0] = 4
    data_tensor[1, 0, 1] = 5
    data_tensor[1, 1, 0] = 6
    data_tensor[1, 1, 1] = 7

    alias indices_rank = 2
    alias indices_layout = Layout.row_major[indices_rank]()
    var indices = HostNDBuffer[DType.int64, indices_rank, DimList(2, 1)](
        IndexList[indices_rank](2, 1)
    )
    var indices_tensor = indices.to_layout_tensor()

    indices_tensor[0, 0] = 1
    indices_tensor[1, 0] = 0

    alias output_rank = 2
    var output_shape = gather_nd_shape[
        output_rank,
        data_type,
        DType.int64,
        batch_dims,
    ](
        LayoutTensor[data_tensor.dtype, data_layout](
            data_tensor.ptr,
            RuntimeLayout[data_layout].row_major(
                data_tensor.runtime_layout.shape.value.canonicalize()
            ),
        ),
        LayoutTensor[indices_tensor.dtype, indices_layout](
            indices_tensor.ptr,
            RuntimeLayout[indices_layout].row_major(
                indices_tensor.runtime_layout.shape.value.canonicalize()
            ),
        ),
    )

    var expected_output = HostNDBuffer[data_type, output_rank](output_shape)
    var expected_output_tensor = expected_output.to_layout_tensor()
    expected_output_tensor[0, 0] = 2
    expected_output_tensor[0, 1] = 3
    expected_output_tensor[1, 0] = 4
    expected_output_tensor[1, 1] = 5

    execute_gather_nd_test[batch_dims](data, indices, expected_output, ctx)


fn test_gather_nd_eg6(ctx: DeviceContext) raises:
    # Example 6
    alias batch_dims = 2
    alias data_rank = 3
    alias data_type = DType.int8
    alias data_layout = Layout.row_major[data_rank]()
    var data = HostNDBuffer[data_type, data_rank, DimList(2, 3, 4)](
        IndexList[data_rank](2, 3, 4)
    )
    var data_tensor = data.to_layout_tensor()

    data_tensor[0, 0, 0] = 1
    data_tensor[0, 0, 1] = 2
    data_tensor[0, 0, 2] = 3
    data_tensor[0, 0, 3] = 4

    data_tensor[0, 1, 0] = 5
    data_tensor[0, 1, 1] = 6
    data_tensor[0, 1, 2] = 7
    data_tensor[0, 1, 3] = 8

    data_tensor[0, 2, 0] = 9
    data_tensor[0, 2, 1] = 10
    data_tensor[0, 2, 2] = 11
    data_tensor[0, 2, 3] = 12

    data_tensor[1, 0, 0] = 13
    data_tensor[1, 0, 1] = 14
    data_tensor[1, 0, 2] = 15
    data_tensor[1, 0, 3] = 16

    data_tensor[1, 1, 0] = 17
    data_tensor[1, 1, 1] = 18
    data_tensor[1, 1, 2] = 19
    data_tensor[1, 1, 3] = 20

    data_tensor[1, 2, 0] = 21
    data_tensor[1, 2, 1] = 22
    data_tensor[1, 2, 2] = 23
    data_tensor[1, 2, 3] = 24

    alias indices_rank = 4
    alias indices_layout = Layout.row_major[indices_rank]()
    var indices = HostNDBuffer[DType.int64, indices_rank, DimList(2, 3, 1, 1)](
        IndexList[indices_rank](
            2,
            3,
            1,
            1,
        )
    )
    var indices_tensor = indices.to_layout_tensor()

    indices_tensor[0, 0, 0, 0] = 1
    indices_tensor[0, 1, 0, 0] = 0
    indices_tensor[0, 2, 0, 0] = 2
    indices_tensor[1, 0, 0, 0] = 0
    indices_tensor[1, 1, 0, 0] = 2
    indices_tensor[1, 2, 0, 0] = 2

    alias output_rank = 3
    var output_shape = gather_nd_shape[
        output_rank,
        data_type,
        DType.int64,
        batch_dims,
    ](
        LayoutTensor[data_tensor.dtype, data_layout](
            data_tensor.ptr,
            RuntimeLayout[data_layout].row_major(
                data_tensor.runtime_layout.shape.value.canonicalize()
            ),
        ),
        LayoutTensor[indices_tensor.dtype, indices_layout](
            indices_tensor.ptr,
            RuntimeLayout[indices_layout].row_major(
                indices_tensor.runtime_layout.shape.value.canonicalize()
            ),
        ),
    )

    var expected_output = HostNDBuffer[data_type, output_rank](output_shape)
    var expected_output_tensor = expected_output.to_layout_tensor()
    expected_output_tensor[0, 0, 0] = 2
    expected_output_tensor[0, 1, 0] = 5
    expected_output_tensor[0, 2, 0] = 11
    expected_output_tensor[1, 0, 0] = 13
    expected_output_tensor[1, 1, 0] = 19
    expected_output_tensor[1, 2, 0] = 23

    execute_gather_nd_test[batch_dims](data, indices, expected_output, ctx)


fn test_gather_nd_eg7(ctx: DeviceContext) raises:
    # Example 7
    alias batch_dims = 0
    alias data_rank = 3
    alias data_type = DType.int8
    alias data_layout = Layout.row_major[data_rank]()
    var data = HostNDBuffer[data_type, data_rank, DimList(2, 2, 2)](
        IndexList[data_rank](2, 2, 2)
    )
    var data_tensor = data.to_layout_tensor()

    data_tensor[0, 0, 0] = 0
    data_tensor[0, 0, 1] = 1
    data_tensor[0, 1, 0] = 2
    data_tensor[0, 1, 1] = 3
    data_tensor[1, 0, 0] = 4
    data_tensor[1, 0, 1] = 5
    data_tensor[1, 1, 0] = 6
    data_tensor[1, 1, 1] = 7

    alias indices_rank = 3
    alias indices_layout = Layout.row_major[indices_rank]()
    var indices = HostNDBuffer[DType.int64, indices_rank, DimList(2, 1, 1)](
        IndexList[indices_rank](
            2,
            1,
            1,
        )
    )
    var indices_tensor = indices.to_layout_tensor()

    indices_tensor[0, 0, 0] = 0
    indices_tensor[1, 0, 0] = 1

    alias output_rank = 4
    var output_shape = gather_nd_shape[
        output_rank,
        data_type,
        DType.int64,
        batch_dims,
    ](
        LayoutTensor[data_tensor.dtype, data_layout](
            data_tensor.ptr,
            RuntimeLayout[data_layout].row_major(
                data_tensor.runtime_layout.shape.value.canonicalize()
            ),
        ),
        LayoutTensor[indices_tensor.dtype, indices_layout](
            indices_tensor.ptr,
            RuntimeLayout[indices_layout].row_major(
                indices_tensor.runtime_layout.shape.value.canonicalize()
            ),
        ),
    )

    var expected_output = HostNDBuffer[data_type, output_rank](output_shape)
    var expected_output_tensor = expected_output.to_layout_tensor()
    expected_output_tensor[0, 0, 0, 0] = 0
    expected_output_tensor[0, 0, 0, 1] = 1
    expected_output_tensor[0, 0, 1, 0] = 2
    expected_output_tensor[0, 0, 1, 1] = 3
    expected_output_tensor[1, 0, 0, 0] = 4
    expected_output_tensor[1, 0, 0, 1] = 5
    expected_output_tensor[1, 0, 1, 0] = 6
    expected_output_tensor[1, 0, 1, 1] = 7

    execute_gather_nd_test[batch_dims](data, indices, expected_output, ctx)


fn test_gather_nd_eg8(ctx: DeviceContext) raises:
    # Example 2
    alias batch_dims = 0
    alias data_rank = 2
    alias data_type = DType.int8
    alias data_layout = Layout.row_major[data_rank]()
    var data = HostNDBuffer[data_type, data_rank, DimList(2, 3)](
        IndexList[data_rank](2, 3)
    )
    var data_tensor = data.to_layout_tensor()

    data_tensor[0, 0] = 0
    data_tensor[0, 1] = 1
    data_tensor[0, 2] = 2
    data_tensor[1, 0] = 3
    data_tensor[1, 1] = 4
    data_tensor[1, 2] = 5

    alias indices_rank = 2
    alias indices_layout = Layout.row_major[indices_rank]()
    var indices = HostNDBuffer[DType.int64, indices_rank, DimList(2, 1)](
        IndexList[indices_rank](2, 1)
    )
    var indices_tensor = indices.to_layout_tensor()

    indices_tensor[0, 0] = 1
    indices_tensor[1, 0] = 0

    alias output_rank = 2
    var output_shape = gather_nd_shape[
        output_rank,
        data_type,
        DType.int64,
        batch_dims,
    ](
        LayoutTensor[data_tensor.dtype, data_layout](
            data_tensor.ptr,
            RuntimeLayout[data_layout].row_major(
                data_tensor.runtime_layout.shape.value.canonicalize()
            ),
        ),
        LayoutTensor[indices_tensor.dtype, indices_layout](
            indices_tensor.ptr,
            RuntimeLayout[indices_layout].row_major(
                indices_tensor.runtime_layout.shape.value.canonicalize()
            ),
        ),
    )

    var expected_output = HostNDBuffer[data_type, output_rank](output_shape)
    var expected_output_tensor = expected_output.to_layout_tensor()
    expected_output_tensor[0, 0] = 3
    expected_output_tensor[0, 1] = 4
    expected_output_tensor[0, 2] = 5
    expected_output_tensor[1, 0] = 0
    expected_output_tensor[1, 1] = 1
    expected_output_tensor[1, 2] = 2

    execute_gather_nd_test[batch_dims](data, indices, expected_output, ctx)


def main():
    """
    Note: Examples 1-5 are from:
    https://github.com/onnx/onnx/blob/main/docs/Operators.md#GatherND.
    """

    with DeviceContext() as ctx:
        test_gather_nd_eg1(ctx)
        test_gather_nd_eg2(ctx)
        test_gather_nd_eg3(ctx)
        test_gather_nd_eg4(ctx)
        test_gather_nd_eg5(ctx)
        test_gather_nd_eg6(ctx)
        test_gather_nd_eg7(ctx)
        test_gather_nd_eg8(ctx)
