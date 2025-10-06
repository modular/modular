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

from sys.info import size_of

from gpu.host import DeviceContext
from layout import LayoutTensor, Layout, RuntimeLayout, UNKNOWN_VALUE
from nn.gather_scatter import gather

from utils.index import Index


# CHECK-LABEL: test_gather
fn test_gather(ctx: DeviceContext) raises:
    print("== test_gather")

    @no_inline
    @parameter
    fn _test_gather[indices_type: DType]() raises:
        alias num_rows = 16
        alias row_size = 4

        alias layout_2d = Layout.row_major[2]()
        var input_host_ptr = UnsafePointer[Float32].alloc(num_rows * row_size)
        var input_host = LayoutTensor[
            DType.float32,
            Layout.row_major(num_rows, row_size),
        ](input_host_ptr)
        for i in range(num_rows):
            for j in range(row_size):
                input_host[i, j] = Float32(i)
        var input_device_ptr = ctx.enqueue_create_buffer[DType.float32](
            input_host.size() * size_of[DType.float32]()
        )
        ctx.enqueue_copy(input_device_ptr, input_host.ptr)
        var input_device = LayoutTensor[
            DType.float32,
            Layout.row_major(num_rows, row_size),
        ](input_device_ptr.unsafe_ptr())

        alias num_indices = 16
        var indices_host_ptr = UnsafePointer[Scalar[indices_type]].alloc(
            num_indices
        )
        var indices_host = LayoutTensor[
            indices_type,
            Layout.row_major(num_indices),
        ](indices_host_ptr)
        var indices_device_ptr = ctx.enqueue_create_buffer[indices_type](
            indices_host.size() * size_of[indices_type]()
        )
        var indices_device = LayoutTensor[
            indices_type,
            Layout.row_major(num_indices),
        ](indices_device_ptr.unsafe_ptr())

        for i in range(num_indices):
            indices_host[i] = i // 2
        indices_host[0] = -1
        indices_host[1] = -num_rows

        ctx.enqueue_copy(indices_device_ptr, indices_host.ptr)

        # create output
        var output_host_ptr = UnsafePointer[Float32].alloc(
            num_indices * row_size
        )
        var output_host = LayoutTensor[
            DType.float32,
            Layout.row_major(num_indices, row_size),
        ](output_host_ptr)
        var output_device_ptr = ctx.enqueue_create_buffer[DType.float32](
            output_host.size() * size_of[DType.float32]()
        )
        var output_device = LayoutTensor[
            DType.float32,
            Layout.row_major(num_indices, row_size),
        ](output_device_ptr.unsafe_ptr())

        alias output_layout = Layout.row_major[output_device.rank]()
        alias input_layout = Layout.row_major[input_device.rank]()
        alias indices_layout = Layout.row_major[indices_device.rank]()

        gather[axis=0, target="gpu"](
            LayoutTensor[output_device.dtype, output_layout](
                output_device.ptr,
                RuntimeLayout[output_layout].row_major(
                    output_device.runtime_layout.shape.value.canonicalize()
                ),
            ),
            LayoutTensor[input_device.dtype, input_layout](
                input_device.ptr,
                RuntimeLayout[input_layout].row_major(
                    input_device.runtime_layout.shape.value.canonicalize()
                ),
            ),
            LayoutTensor[indices_device.dtype, indices_layout](
                indices_device.ptr,
                RuntimeLayout[indices_layout].row_major(
                    indices_device.runtime_layout.shape.value.canonicalize()
                ),
            ),
            context=ctx,
        )
        ctx.synchronize()

        ctx.enqueue_copy(output_host.ptr, output_device_ptr)

        print(output_host[0, 0])
        print(output_host[1, 0])
        print(output_host[2, 0])
        print(output_host[6, 0])
        print(output_host[15, 0])

        input_host_ptr.free()
        indices_host_ptr.free()
        output_host_ptr.free()

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
