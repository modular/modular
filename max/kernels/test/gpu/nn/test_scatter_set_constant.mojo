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

from layout import Layout, LayoutTensor
from gpu.host import DeviceContext
from runtime.asyncrt import DeviceContextPtr
from internal_utils import TestTensor, assert_equal
from nn.gather_scatter import scatter_set_constant
from memory import UnsafePointer


fn test_scatter_set_constant(ctx: DeviceContext) raises:
    # TODO not sure why this doesn't work with InlineArray?
    var data_ptr = UnsafePointer[Float32].alloc(3 * 3)
    var data = LayoutTensor[
        DType.float32, Layout.row_major(3, 3), MutableAnyOrigin
    ](data_ptr,).fill(0.0)
    var data_ptr_gpu = ctx.enqueue_create_buffer[DType.float32](3 * 3)
    ctx.enqueue_copy(data_ptr_gpu, data_ptr)

    var data_gpu = LayoutTensor[
        DType.float32, Layout.row_major(3, 3), MutableAnyOrigin
    ](
        data_ptr_gpu,
    )

    var indices = LayoutTensor[DType.int32, Layout.row_major(4, 2)](
        InlineArray[Scalar[DType.int32], 4 * 2](uninitialized=True),
    )

    indices[0, 0] = 0
    indices[0, 1] = 1
    indices[1, 0] = 1
    indices[1, 1] = 2
    indices[2, 0] = 1
    indices[2, 1] = 3
    indices[3, 0] = 2
    indices[3, 1] = 0

    var indices_ptr_gpu = ctx.enqueue_create_buffer[DType.int32](4 * 2)
    ctx.enqueue_copy(indices_ptr_gpu, indices.ptr)
    var indices_gpu = LayoutTensor[DType.int32, Layout.row_major(4, 2)](
        indices_ptr_gpu,
    )

    var fill_value: Float32 = 5.0
    var expected_output_ptr = UnsafePointer[Float32].alloc(3 * 3)
    var expected_output = LayoutTensor[
        DType.float32, Layout.row_major(3, 3), MutableAnyOrigin
    ](expected_output_ptr,).fill(0.0)

    expected_output[0, 1] = 5.0
    expected_output[1, 2] = 5.0
    expected_output[1, 3] = 5.0
    expected_output[2, 0] = 5.0

    var ctx_ptr = DeviceContextPtr(ctx)

    scatter_set_constant[target="gpu",](
        data_gpu, indices_gpu, fill_value, ctx_ptr
    )

    ctx.enqueue_copy(data_ptr, data_ptr_gpu)

    for i in range(3):
        for j in range(3):
            if data[i, j] != expected_output[i, j]:
                raise "data[" + String(i) + ", " + String(j) + "] = " + String(
                    data[i, j]
                ) + " != " + String(expected_output[i, j])

    data_ptr.free()
    expected_output_ptr.free()


fn main() raises:
    with DeviceContext() as ctx:
        test_scatter_set_constant(ctx)
