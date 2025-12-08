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
from layout import LayoutTensor, Layout, RuntimeLayout, UNKNOWN_VALUE
from nn.gather_scatter import gather

from utils import IndexList


# CHECK-LABEL: test_gather
fn test_gather(ctx: DeviceContext) raises:
    print("== test_gather")

    @no_inline
    @parameter
    fn _test_gather[indices_type: DType]() raises:
        comptime num_rows = 16
        comptime row_size = 4
        comptime num_indices = 16

        # Input tensor layout [num_rows, row_size]
        comptime input_layout = Layout.row_major(num_rows, row_size)
        var input_shape = IndexList[2](num_rows, row_size)
        var input_runtime = RuntimeLayout[input_layout].row_major(input_shape)

        # Indices tensor layout [num_indices]
        comptime indices_layout = Layout.row_major(num_indices)
        var indices_shape = IndexList[1](num_indices)
        var indices_runtime = RuntimeLayout[indices_layout].row_major(
            indices_shape
        )

        # Output tensor layout [num_indices, row_size]
        comptime output_layout = Layout.row_major(num_indices, row_size)
        var output_shape = IndexList[2](num_indices, row_size)
        var output_runtime = RuntimeLayout[output_layout].row_major(
            output_shape
        )

        # Create device buffers
        var input_device = ctx.enqueue_create_buffer[DType.float32](
            input_shape.flattened_length()
        )
        var indices_device = ctx.enqueue_create_buffer[indices_type](
            indices_shape.flattened_length()
        )
        var output_device = ctx.enqueue_create_buffer[DType.float32](
            output_shape.flattened_length()
        )

        # Initialize input data
        with input_device.map_to_host() as input_host:
            var input_tensor = LayoutTensor[DType.float32, input_layout](
                input_host, input_runtime
            )
            for i in range(num_rows):
                for j in range(row_size):
                    input_tensor[i, j] = Float32(i)

        # Initialize indices
        with indices_device.map_to_host() as indices_host:
            var indices_tensor = LayoutTensor[indices_type, indices_layout](
                indices_host, indices_runtime
            )
            for i in range(num_indices):
                indices_tensor[i] = i // 2
            indices_tensor[0] = -1
            indices_tensor[1] = -num_rows

        # Create layout tensors for GPU operations
        var input_tensor = LayoutTensor[DType.float32, input_layout](
            input_device, input_runtime
        )
        var indices_tensor = LayoutTensor[indices_type, indices_layout](
            indices_device, indices_runtime
        )
        var output_tensor = LayoutTensor[DType.float32, output_layout](
            output_device, output_runtime
        )

        gather[axis=0, target="gpu"](
            output_tensor,
            input_tensor,
            indices_tensor,
            context=ctx,
        )
        ctx.synchronize()

        # Read back and print results
        with output_device.map_to_host() as output_host:
            var output_tensor_host = LayoutTensor[DType.float32, output_layout](
                output_host, output_runtime
            )
            print(output_tensor_host[0, 0])
            print(output_tensor_host[1, 0])
            print(output_tensor_host[2, 0])
            print(output_tensor_host[6, 0])
            print(output_tensor_host[15, 0])

    # CHECK: 15.0
    # CHECK: 0.0
    # CHECK-NEXT: 1.0
    # CHECK-NEXT: 3.0
    # CHECK-NEXT: 7.0
    _test_gather[DType.int32]()
    # CHECK: 0.0
    # CHECK-NEXT: 1.0
    # CHECK-NEXT: 3.0
    # CHECK-NEXT: 7.0
    _test_gather[DType.int64]()


def main():
    with DeviceContext() as ctx:
        test_gather(ctx)
