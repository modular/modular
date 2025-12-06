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

from gpu.host import DeviceBuffer, DeviceContext
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from nn.gather_scatter import _gather_nd_impl, gather_nd_shape
from testing import assert_equal

from utils import IndexList


def execute_gather_nd_test[
    data_type: DType,
    indices_type: DType,
    batch_dims: Int,
    data_layout: Layout,
    indices_layout: Layout,
    output_layout: Layout,
](
    data_device: DeviceBuffer[data_type],
    data_runtime: RuntimeLayout[data_layout],
    indices_device: DeviceBuffer[indices_type],
    indices_runtime: RuntimeLayout[indices_layout],
    expected_output_device: DeviceBuffer[data_type],
    output_runtime: RuntimeLayout[output_layout],
    ctx: DeviceContext,
):
    # Create output device buffer
    var actual_output_device = ctx.enqueue_create_buffer[data_type](
        output_runtime.size()
    )

    # Create layout tensors for GPU operations
    var data_tensor = LayoutTensor[data_type, data_layout](
        data_device, data_runtime
    )
    var indices_tensor = LayoutTensor[indices_type, indices_layout](
        indices_device, indices_runtime
    )
    var actual_output_tensor = LayoutTensor[data_type, output_layout](
        actual_output_device, output_runtime
    )

    # execute the kernel
    _gather_nd_impl[batch_dims, target="gpu"](
        data_tensor,
        indices_tensor,
        actual_output_tensor,
        ctx,
    )
    ctx.synchronize()

    # check that the contents of the output are consistent
    with actual_output_device.map_to_host() as actual_host:
        with expected_output_device.map_to_host() as expected_host:
            for i in range(output_runtime.size()):
                assert_equal(actual_host[i], expected_host[i])


fn test_gather_nd_eg1(ctx: DeviceContext) raises:
    # Example 1
    comptime batch_dims = 0
    comptime data_type = DType.int32
    comptime indices_type = DType.int64

    comptime data_layout = Layout.row_major(2, 2)
    var data_shape = IndexList[2](2, 2)
    var data_runtime = RuntimeLayout[data_layout].row_major(data_shape)

    comptime indices_layout = Layout.row_major(2, 2)
    var indices_shape = IndexList[2](2, 2)
    var indices_runtime = RuntimeLayout[indices_layout].row_major(indices_shape)

    # Create device buffers
    var data_device = ctx.enqueue_create_buffer[data_type](
        data_shape.flattened_length()
    )
    var indices_device = ctx.enqueue_create_buffer[indices_type](
        indices_shape.flattened_length()
    )

    # Initialize data on host
    with data_device.map_to_host() as data_host:
        var data_tensor = LayoutTensor[data_type, data_layout](
            data_host, data_runtime
        )
        data_tensor[0, 0] = 0
        data_tensor[0, 1] = 1
        data_tensor[1, 0] = 2
        data_tensor[1, 1] = 3

    with indices_device.map_to_host() as indices_host:
        var indices_tensor = LayoutTensor[indices_type, indices_layout](
            indices_host, indices_runtime
        )
        indices_tensor[0, 0] = 0
        indices_tensor[0, 1] = 0
        indices_tensor[1, 0] = 1
        indices_tensor[1, 1] = 1

    # Compute output shape
    var data_tensor = LayoutTensor[data_type, data_layout](
        data_device, data_runtime
    )
    var indices_tensor = LayoutTensor[indices_type, indices_layout](
        indices_device, indices_runtime
    )

    comptime output_rank = 1
    var output_shape = gather_nd_shape[
        output_rank,
        data_type,
        indices_type,
        batch_dims,
    ](data_tensor, indices_tensor)

    comptime output_layout = Layout.row_major(UNKNOWN_VALUE)
    var output_runtime = RuntimeLayout[output_layout].row_major(output_shape)

    var expected_output_device = ctx.enqueue_create_buffer[data_type](
        output_shape.flattened_length()
    )
    with expected_output_device.map_to_host() as expected_host:
        var expected_tensor = LayoutTensor[data_type, output_layout](
            expected_host, output_runtime
        )
        expected_tensor[0] = 0
        expected_tensor[1] = 3

    execute_gather_nd_test[
        data_type,
        indices_type,
        batch_dims,
        data_layout,
        indices_layout,
        output_layout,
    ](
        data_device,
        data_runtime,
        indices_device,
        indices_runtime,
        expected_output_device,
        output_runtime,
        ctx,
    )


fn test_gather_nd_eg2(ctx: DeviceContext) raises:
    # Example 2
    comptime batch_dims = 0
    comptime data_type = DType.int8
    comptime indices_type = DType.int64

    comptime data_layout = Layout.row_major(2, 2)
    var data_shape = IndexList[2](2, 2)
    var data_runtime = RuntimeLayout[data_layout].row_major(data_shape)

    comptime indices_layout = Layout.row_major(2, 1)
    var indices_shape = IndexList[2](2, 1)
    var indices_runtime = RuntimeLayout[indices_layout].row_major(indices_shape)

    var data_device = ctx.enqueue_create_buffer[data_type](
        data_shape.flattened_length()
    )
    var indices_device = ctx.enqueue_create_buffer[indices_type](
        indices_shape.flattened_length()
    )

    with data_device.map_to_host() as data_host:
        var data_tensor = LayoutTensor[data_type, data_layout](
            data_host, data_runtime
        )
        data_tensor[0, 0] = 0
        data_tensor[0, 1] = 1
        data_tensor[1, 0] = 2
        data_tensor[1, 1] = 3

    with indices_device.map_to_host() as indices_host:
        var indices_tensor = LayoutTensor[indices_type, indices_layout](
            indices_host, indices_runtime
        )
        indices_tensor[0, 0] = 1
        indices_tensor[1, 0] = 0

    var data_tensor = LayoutTensor[data_type, data_layout](
        data_device, data_runtime
    )
    var indices_tensor = LayoutTensor[indices_type, indices_layout](
        indices_device, indices_runtime
    )

    comptime output_rank = 2
    var output_shape = gather_nd_shape[
        output_rank,
        data_type,
        indices_type,
        batch_dims,
    ](data_tensor, indices_tensor)

    comptime output_layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    var output_runtime = RuntimeLayout[output_layout].row_major(output_shape)

    var expected_output_device = ctx.enqueue_create_buffer[data_type](
        output_shape.flattened_length()
    )
    with expected_output_device.map_to_host() as expected_host:
        var expected_tensor = LayoutTensor[data_type, output_layout](
            expected_host, output_runtime
        )
        expected_tensor[0, 0] = 2
        expected_tensor[0, 1] = 3
        expected_tensor[1, 0] = 0
        expected_tensor[1, 1] = 1

    execute_gather_nd_test[
        data_type,
        indices_type,
        batch_dims,
        data_layout,
        indices_layout,
        output_layout,
    ](
        data_device,
        data_runtime,
        indices_device,
        indices_runtime,
        expected_output_device,
        output_runtime,
        ctx,
    )


fn test_gather_nd_eg3(ctx: DeviceContext) raises:
    # Example 3
    comptime batch_dims = 0
    comptime data_type = DType.float32
    comptime indices_type = DType.int64

    comptime data_layout = Layout.row_major(2, 2, 2)
    var data_shape = IndexList[3](2, 2, 2)
    var data_runtime = RuntimeLayout[data_layout].row_major(data_shape)

    comptime indices_layout = Layout.row_major(2, 2)
    var indices_shape = IndexList[2](2, 2)
    var indices_runtime = RuntimeLayout[indices_layout].row_major(indices_shape)

    var data_device = ctx.enqueue_create_buffer[data_type](
        data_shape.flattened_length()
    )
    var indices_device = ctx.enqueue_create_buffer[indices_type](
        indices_shape.flattened_length()
    )

    with data_device.map_to_host() as data_host:
        var data_tensor = LayoutTensor[data_type, data_layout](
            data_host, data_runtime
        )
        data_tensor[0, 0, 0] = 0
        data_tensor[0, 0, 1] = 1
        data_tensor[0, 1, 0] = 2
        data_tensor[0, 1, 1] = 3
        data_tensor[1, 0, 0] = 4
        data_tensor[1, 0, 1] = 5
        data_tensor[1, 1, 0] = 6
        data_tensor[1, 1, 1] = 7

    with indices_device.map_to_host() as indices_host:
        var indices_tensor = LayoutTensor[indices_type, indices_layout](
            indices_host, indices_runtime
        )
        indices_tensor[0, 0] = 0
        indices_tensor[0, 1] = 1
        indices_tensor[1, 0] = 1
        indices_tensor[1, 1] = 0

    var data_tensor = LayoutTensor[data_type, data_layout](
        data_device, data_runtime
    )
    var indices_tensor = LayoutTensor[indices_type, indices_layout](
        indices_device, indices_runtime
    )

    comptime output_rank = 2
    var output_shape = gather_nd_shape[
        output_rank,
        data_type,
        indices_type,
        batch_dims,
    ](data_tensor, indices_tensor)

    comptime output_layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    var output_runtime = RuntimeLayout[output_layout].row_major(output_shape)

    var expected_output_device = ctx.enqueue_create_buffer[data_type](
        output_shape.flattened_length()
    )
    with expected_output_device.map_to_host() as expected_host:
        var expected_tensor = LayoutTensor[data_type, output_layout](
            expected_host, output_runtime
        )
        expected_tensor[0, 0] = 2
        expected_tensor[0, 1] = 3
        expected_tensor[1, 0] = 4
        expected_tensor[1, 1] = 5

    execute_gather_nd_test[
        data_type,
        indices_type,
        batch_dims,
        data_layout,
        indices_layout,
        output_layout,
    ](
        data_device,
        data_runtime,
        indices_device,
        indices_runtime,
        expected_output_device,
        output_runtime,
        ctx,
    )


fn test_gather_nd_eg4(ctx: DeviceContext) raises:
    # Example 4
    comptime batch_dims = 0
    comptime data_type = DType.int8
    comptime indices_type = DType.int64

    comptime data_layout = Layout.row_major(2, 2, 2)
    var data_shape = IndexList[3](2, 2, 2)
    var data_runtime = RuntimeLayout[data_layout].row_major(data_shape)

    comptime indices_layout = Layout.row_major(2, 1, 2)
    var indices_shape = IndexList[3](2, 1, 2)
    var indices_runtime = RuntimeLayout[indices_layout].row_major(indices_shape)

    var data_device = ctx.enqueue_create_buffer[data_type](
        data_shape.flattened_length()
    )
    var indices_device = ctx.enqueue_create_buffer[indices_type](
        indices_shape.flattened_length()
    )

    with data_device.map_to_host() as data_host:
        var data_tensor = LayoutTensor[data_type, data_layout](
            data_host, data_runtime
        )
        data_tensor[0, 0, 0] = 0
        data_tensor[0, 0, 1] = 1
        data_tensor[0, 1, 0] = 2
        data_tensor[0, 1, 1] = 3
        data_tensor[1, 0, 0] = 4
        data_tensor[1, 0, 1] = 5
        data_tensor[1, 1, 0] = 6
        data_tensor[1, 1, 1] = 7

    with indices_device.map_to_host() as indices_host:
        var indices_tensor = LayoutTensor[indices_type, indices_layout](
            indices_host, indices_runtime
        )
        indices_tensor[0, 0, 0] = 0
        indices_tensor[0, 0, 1] = 1
        indices_tensor[1, 0, 0] = 1
        indices_tensor[1, 0, 1] = 0

    var data_tensor = LayoutTensor[data_type, data_layout](
        data_device, data_runtime
    )
    var indices_tensor = LayoutTensor[indices_type, indices_layout](
        indices_device, indices_runtime
    )

    comptime output_rank = 3
    var output_shape = gather_nd_shape[
        output_rank,
        data_type,
        indices_type,
        batch_dims,
    ](data_tensor, indices_tensor)

    comptime output_layout = Layout.row_major(
        UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE
    )
    var output_runtime = RuntimeLayout[output_layout].row_major(output_shape)

    var expected_output_device = ctx.enqueue_create_buffer[data_type](
        output_shape.flattened_length()
    )
    with expected_output_device.map_to_host() as expected_host:
        var expected_tensor = LayoutTensor[data_type, output_layout](
            expected_host, output_runtime
        )
        expected_tensor[0, 0, 0] = 2
        expected_tensor[0, 0, 1] = 3
        expected_tensor[1, 0, 0] = 4
        expected_tensor[1, 0, 1] = 5

    execute_gather_nd_test[
        data_type,
        indices_type,
        batch_dims,
        data_layout,
        indices_layout,
        output_layout,
    ](
        data_device,
        data_runtime,
        indices_device,
        indices_runtime,
        expected_output_device,
        output_runtime,
        ctx,
    )


fn test_gather_nd_eg5(ctx: DeviceContext) raises:
    # Example 5
    comptime batch_dims = 1
    comptime data_type = DType.int32
    comptime indices_type = DType.int64

    comptime data_layout = Layout.row_major(2, 2, 2)
    var data_shape = IndexList[3](2, 2, 2)
    var data_runtime = RuntimeLayout[data_layout].row_major(data_shape)

    comptime indices_layout = Layout.row_major(2, 1)
    var indices_shape = IndexList[2](2, 1)
    var indices_runtime = RuntimeLayout[indices_layout].row_major(indices_shape)

    var data_device = ctx.enqueue_create_buffer[data_type](
        data_shape.flattened_length()
    )
    var indices_device = ctx.enqueue_create_buffer[indices_type](
        indices_shape.flattened_length()
    )

    with data_device.map_to_host() as data_host:
        var data_tensor = LayoutTensor[data_type, data_layout](
            data_host, data_runtime
        )
        data_tensor[0, 0, 0] = 0
        data_tensor[0, 0, 1] = 1
        data_tensor[0, 1, 0] = 2
        data_tensor[0, 1, 1] = 3
        data_tensor[1, 0, 0] = 4
        data_tensor[1, 0, 1] = 5
        data_tensor[1, 1, 0] = 6
        data_tensor[1, 1, 1] = 7

    with indices_device.map_to_host() as indices_host:
        var indices_tensor = LayoutTensor[indices_type, indices_layout](
            indices_host, indices_runtime
        )
        indices_tensor[0, 0] = 1
        indices_tensor[1, 0] = 0

    var data_tensor = LayoutTensor[data_type, data_layout](
        data_device, data_runtime
    )
    var indices_tensor = LayoutTensor[indices_type, indices_layout](
        indices_device, indices_runtime
    )

    comptime output_rank = 2
    var output_shape = gather_nd_shape[
        output_rank,
        data_type,
        indices_type,
        batch_dims,
    ](data_tensor, indices_tensor)

    comptime output_layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    var output_runtime = RuntimeLayout[output_layout].row_major(output_shape)

    var expected_output_device = ctx.enqueue_create_buffer[data_type](
        output_shape.flattened_length()
    )
    with expected_output_device.map_to_host() as expected_host:
        var expected_tensor = LayoutTensor[data_type, output_layout](
            expected_host, output_runtime
        )
        expected_tensor[0, 0] = 2
        expected_tensor[0, 1] = 3
        expected_tensor[1, 0] = 4
        expected_tensor[1, 1] = 5

    execute_gather_nd_test[
        data_type,
        indices_type,
        batch_dims,
        data_layout,
        indices_layout,
        output_layout,
    ](
        data_device,
        data_runtime,
        indices_device,
        indices_runtime,
        expected_output_device,
        output_runtime,
        ctx,
    )


fn test_gather_nd_eg6(ctx: DeviceContext) raises:
    # Example 6
    comptime batch_dims = 2
    comptime data_type = DType.int8
    comptime indices_type = DType.int64

    comptime data_layout = Layout.row_major(2, 3, 4)
    var data_shape = IndexList[3](2, 3, 4)
    var data_runtime = RuntimeLayout[data_layout].row_major(data_shape)

    comptime indices_layout = Layout.row_major(2, 3, 1, 1)
    var indices_shape = IndexList[4](2, 3, 1, 1)
    var indices_runtime = RuntimeLayout[indices_layout].row_major(indices_shape)

    var data_device = ctx.enqueue_create_buffer[data_type](
        data_shape.flattened_length()
    )
    var indices_device = ctx.enqueue_create_buffer[indices_type](
        indices_shape.flattened_length()
    )

    with data_device.map_to_host() as data_host:
        var data_tensor = LayoutTensor[data_type, data_layout](
            data_host, data_runtime
        )
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

    with indices_device.map_to_host() as indices_host:
        var indices_tensor = LayoutTensor[indices_type, indices_layout](
            indices_host, indices_runtime
        )
        indices_tensor[0, 0, 0, 0] = 1
        indices_tensor[0, 1, 0, 0] = 0
        indices_tensor[0, 2, 0, 0] = 2
        indices_tensor[1, 0, 0, 0] = 0
        indices_tensor[1, 1, 0, 0] = 2
        indices_tensor[1, 2, 0, 0] = 2

    var data_tensor = LayoutTensor[data_type, data_layout](
        data_device, data_runtime
    )
    var indices_tensor = LayoutTensor[indices_type, indices_layout](
        indices_device, indices_runtime
    )

    comptime output_rank = 3
    var output_shape = gather_nd_shape[
        output_rank,
        data_type,
        indices_type,
        batch_dims,
    ](data_tensor, indices_tensor)

    comptime output_layout = Layout.row_major(
        UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE
    )
    var output_runtime = RuntimeLayout[output_layout].row_major(output_shape)

    var expected_output_device = ctx.enqueue_create_buffer[data_type](
        output_shape.flattened_length()
    )
    with expected_output_device.map_to_host() as expected_host:
        var expected_tensor = LayoutTensor[data_type, output_layout](
            expected_host, output_runtime
        )
        expected_tensor[0, 0, 0] = 2
        expected_tensor[0, 1, 0] = 5
        expected_tensor[0, 2, 0] = 11
        expected_tensor[1, 0, 0] = 13
        expected_tensor[1, 1, 0] = 19
        expected_tensor[1, 2, 0] = 23

    execute_gather_nd_test[
        data_type,
        indices_type,
        batch_dims,
        data_layout,
        indices_layout,
        output_layout,
    ](
        data_device,
        data_runtime,
        indices_device,
        indices_runtime,
        expected_output_device,
        output_runtime,
        ctx,
    )


fn test_gather_nd_eg7(ctx: DeviceContext) raises:
    # Example 7
    comptime batch_dims = 0
    comptime data_type = DType.int8
    comptime indices_type = DType.int64

    comptime data_layout = Layout.row_major(2, 2, 2)
    var data_shape = IndexList[3](2, 2, 2)
    var data_runtime = RuntimeLayout[data_layout].row_major(data_shape)

    comptime indices_layout = Layout.row_major(2, 1, 1)
    var indices_shape = IndexList[3](2, 1, 1)
    var indices_runtime = RuntimeLayout[indices_layout].row_major(indices_shape)

    var data_device = ctx.enqueue_create_buffer[data_type](
        data_shape.flattened_length()
    )
    var indices_device = ctx.enqueue_create_buffer[indices_type](
        indices_shape.flattened_length()
    )

    with data_device.map_to_host() as data_host:
        var data_tensor = LayoutTensor[data_type, data_layout](
            data_host, data_runtime
        )
        data_tensor[0, 0, 0] = 0
        data_tensor[0, 0, 1] = 1
        data_tensor[0, 1, 0] = 2
        data_tensor[0, 1, 1] = 3
        data_tensor[1, 0, 0] = 4
        data_tensor[1, 0, 1] = 5
        data_tensor[1, 1, 0] = 6
        data_tensor[1, 1, 1] = 7

    with indices_device.map_to_host() as indices_host:
        var indices_tensor = LayoutTensor[indices_type, indices_layout](
            indices_host, indices_runtime
        )
        indices_tensor[0, 0, 0] = 0
        indices_tensor[1, 0, 0] = 1

    var data_tensor = LayoutTensor[data_type, data_layout](
        data_device, data_runtime
    )
    var indices_tensor = LayoutTensor[indices_type, indices_layout](
        indices_device, indices_runtime
    )

    comptime output_rank = 4
    var output_shape = gather_nd_shape[
        output_rank,
        data_type,
        indices_type,
        batch_dims,
    ](data_tensor, indices_tensor)

    comptime output_layout = Layout.row_major(
        UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE
    )
    var output_runtime = RuntimeLayout[output_layout].row_major(output_shape)

    var expected_output_device = ctx.enqueue_create_buffer[data_type](
        output_shape.flattened_length()
    )
    with expected_output_device.map_to_host() as expected_host:
        var expected_tensor = LayoutTensor[data_type, output_layout](
            expected_host, output_runtime
        )
        expected_tensor[0, 0, 0, 0] = 0
        expected_tensor[0, 0, 0, 1] = 1
        expected_tensor[0, 0, 1, 0] = 2
        expected_tensor[0, 0, 1, 1] = 3
        expected_tensor[1, 0, 0, 0] = 4
        expected_tensor[1, 0, 0, 1] = 5
        expected_tensor[1, 0, 1, 0] = 6
        expected_tensor[1, 0, 1, 1] = 7

    execute_gather_nd_test[
        data_type,
        indices_type,
        batch_dims,
        data_layout,
        indices_layout,
        output_layout,
    ](
        data_device,
        data_runtime,
        indices_device,
        indices_runtime,
        expected_output_device,
        output_runtime,
        ctx,
    )


fn test_gather_nd_eg8(ctx: DeviceContext) raises:
    # Example 8
    comptime batch_dims = 0
    comptime data_type = DType.int8
    comptime indices_type = DType.int64

    comptime data_layout = Layout.row_major(2, 3)
    var data_shape = IndexList[2](2, 3)
    var data_runtime = RuntimeLayout[data_layout].row_major(data_shape)

    comptime indices_layout = Layout.row_major(2, 1)
    var indices_shape = IndexList[2](2, 1)
    var indices_runtime = RuntimeLayout[indices_layout].row_major(indices_shape)

    var data_device = ctx.enqueue_create_buffer[data_type](
        data_shape.flattened_length()
    )
    var indices_device = ctx.enqueue_create_buffer[indices_type](
        indices_shape.flattened_length()
    )

    with data_device.map_to_host() as data_host:
        var data_tensor = LayoutTensor[data_type, data_layout](
            data_host, data_runtime
        )
        data_tensor[0, 0] = 0
        data_tensor[0, 1] = 1
        data_tensor[0, 2] = 2
        data_tensor[1, 0] = 3
        data_tensor[1, 1] = 4
        data_tensor[1, 2] = 5

    with indices_device.map_to_host() as indices_host:
        var indices_tensor = LayoutTensor[indices_type, indices_layout](
            indices_host, indices_runtime
        )
        indices_tensor[0, 0] = 1
        indices_tensor[1, 0] = 0

    var data_tensor = LayoutTensor[data_type, data_layout](
        data_device, data_runtime
    )
    var indices_tensor = LayoutTensor[indices_type, indices_layout](
        indices_device, indices_runtime
    )

    comptime output_rank = 2
    var output_shape = gather_nd_shape[
        output_rank,
        data_type,
        indices_type,
        batch_dims,
    ](data_tensor, indices_tensor)

    comptime output_layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    var output_runtime = RuntimeLayout[output_layout].row_major(output_shape)

    var expected_output_device = ctx.enqueue_create_buffer[data_type](
        output_shape.flattened_length()
    )
    with expected_output_device.map_to_host() as expected_host:
        var expected_tensor = LayoutTensor[data_type, output_layout](
            expected_host, output_runtime
        )
        expected_tensor[0, 0] = 3
        expected_tensor[0, 1] = 4
        expected_tensor[0, 2] = 5
        expected_tensor[1, 0] = 0
        expected_tensor[1, 1] = 1
        expected_tensor[1, 2] = 2

    execute_gather_nd_test[
        data_type,
        indices_type,
        batch_dims,
        data_layout,
        indices_layout,
        output_layout,
    ](
        data_device,
        data_runtime,
        indices_device,
        indices_runtime,
        expected_output_device,
        output_runtime,
        ctx,
    )


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
