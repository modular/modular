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
from layout import LayoutTensor, Layout, RuntimeLayout, UNKNOWN_VALUE
from nn.gather_scatter import _gather_nd_impl, gather_nd_shape

from utils import IndexList


def execute_gather_nd_test[
    data_type: DType, //,
    batch_dims: Int,
](
    data_host: HostNDBuffer[data_type, **_],
    indices_host: HostNDBuffer,
    ctx: DeviceContext,
):
    var data_host_tensor = data_host.to_layout_tensor()
    var indices_host_tensor = indices_host.to_layout_tensor()
    # create device-side buffers and copy data to them
    var data_device = DeviceNDBuffer[
        data_host.dtype, data_host.rank, data_host.shape
    ](
        data_host.tensor.get_shape(),
        ctx=ctx,
    )
    var data_device_tensor = data_device.to_layout_tensor()
    alias data_layout = Layout.row_major[data_device_tensor.rank]()
    var indices_device = DeviceNDBuffer[
        indices_host.dtype, indices_host.rank, indices_host.shape
    ](
        indices_host.tensor.get_shape(),
        ctx=ctx,
    )
    var indices_device_tensor = indices_device.to_layout_tensor()
    alias indices_layout = Layout.row_major[indices_device_tensor.rank]()
    alias output_rank = 1

    var output_shape = gather_nd_shape[
        output_rank,
        data_type,
        indices_host.dtype,
        batch_dims,
    ](
        LayoutTensor[data_host_tensor.dtype, data_layout](
            data_host_tensor.ptr,
            RuntimeLayout[data_layout].row_major(
                data_host_tensor.runtime_layout.shape.value.canonicalize()
            ),
        ),
        LayoutTensor[indices_host_tensor.dtype, indices_layout](
            indices_host_tensor.ptr,
            RuntimeLayout[indices_layout].row_major(
                indices_host_tensor.runtime_layout.shape.value.canonicalize()
            ),
        ),
    )

    var actual_output_device = DeviceNDBuffer[
        data_host.dtype,
        output_shape.size,
    ](
        output_shape,
        ctx=ctx,
    )
    var actual_output_tensor = actual_output_device.to_layout_tensor()
    alias actual_output_layout = Layout.row_major[actual_output_tensor.rank]()
    ctx.enqueue_copy(data_device.buffer, data_host_tensor.ptr)
    ctx.enqueue_copy(indices_device.buffer, indices_host_tensor.ptr)

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
        LayoutTensor[actual_output_tensor.dtype, actual_output_layout](
            actual_output_tensor.ptr,
            RuntimeLayout[actual_output_layout].row_major(
                actual_output_tensor.runtime_layout.shape.value.canonicalize()
            ),
        ),
        ctx,
    )
    # Give the kernel an opportunity to raise the error before finishing the test.
    ctx.synchronize()

    _ = data_device^
    _ = indices_device^
    _ = actual_output_device^


fn test_gather_nd_oob(ctx: DeviceContext) raises:
    # Example 1
    alias batch_dims = 0
    alias data_rank = 2
    alias data_type = DType.int32
    var data = HostNDBuffer[data_type, data_rank, DimList(2, 2)](
        IndexList[data_rank](2, 2)
    )
    alias data_layout = Layout.row_major[data_rank]()
    var data_tensor = data.to_layout_tensor()

    data_tensor[0, 0] = 0
    data_tensor[0, 1] = 1
    data_tensor[1, 0] = 2
    data_tensor[1, 1] = 3

    alias indices_rank = 2
    var indices = HostNDBuffer[DType.int64, indices_rank, DimList(2, 2)](
        IndexList[indices_rank](2, 2)
    )
    var indices_tensor = indices.to_layout_tensor()

    indices_tensor[0, 0] = 0
    indices_tensor[0, 1] = 0
    indices_tensor[1, 0] = 1
    indices_tensor[1, 1] = 100  # wildly out of bounds

    execute_gather_nd_test[batch_dims](data, indices, ctx)
    ctx.synchronize()


def main():
    with DeviceContext() as ctx:
        # CHECK: {{.*}}data index out of bounds{{.*}}
        test_gather_nd_oob(ctx)
