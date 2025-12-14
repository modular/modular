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

from random import random_ui64

from gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from nn.index_tensor import _index_tensor_impl
from testing import assert_equal, assert_true

from utils import IndexList


def execute_index_tensor_test[
    data_type: DType, //,
    batch_dims: Int,
](
    data_device: LayoutTensor[
        data_type, *_, address_space = AddressSpace.GENERIC, **_
    ],
    indices_device: LayoutTensor[*_, address_space = AddressSpace.GENERIC, **_],
    expected_output_device: LayoutTensor[
        data_type, *_, address_space = AddressSpace.GENERIC, **_
    ],
    expected_output_device_buffer: DeviceBuffer[data_type],
    ctx: DeviceContext,
):
    # execute the kernel
    comptime data_dyn_layout = Layout.row_major[data_device.rank]()
    comptime indices_dyn_layout = Layout.row_major[indices_device.rank]()
    comptime output_dyn_layout = Layout.row_major[expected_output_device.rank]()
    var actual_output_device = ctx.enqueue_create_buffer[
        expected_output_device.dtype
    ](expected_output_device.size())
    var actual_output_tensor = LayoutTensor[
        actual_output_device.dtype, output_dyn_layout
    ](
        actual_output_device,
        RuntimeLayout[output_dyn_layout].row_major(
            expected_output_device.runtime_layout.shape.value.canonicalize()
        ),
    )
    _index_tensor_impl[batch_dims, target="gpu"](
        LayoutTensor[data_device.dtype, data_dyn_layout](
            data_device.ptr,
            RuntimeLayout[data_dyn_layout].row_major(
                data_device.runtime_layout.shape.value.canonicalize()
            ),
        ),
        LayoutTensor[indices_device.dtype, indices_dyn_layout](
            indices_device.ptr,
            RuntimeLayout[indices_dyn_layout].row_major(
                indices_device.runtime_layout.shape.value.canonicalize()
            ),
        ),
        actual_output_tensor,
        ctx,
    )

    ctx.synchronize()

    # check that our shapes are consistent and that the contents of the output are consistent
    assert_true(
        rebind[IndexList[expected_output_device.rank]](
            actual_output_tensor.runtime_layout.shape.value.canonicalize()
        )
        == rebind[IndexList[expected_output_device.rank]](
            expected_output_device.runtime_layout.shape.value.canonicalize()
        )
    )
    with actual_output_device.map_to_host() as actual_output_host:
        with expected_output_device_buffer.map_to_host() as expected_output_host:
            for i in range(len(actual_output_host)):
                assert_equal(actual_output_host[i], expected_output_host[i])


fn test_index_tensor_DLRM(ctx: DeviceContext) raises:
    print("== test_index_tensor_DLRM")

    comptime input_type = DType.int32
    comptime dim_0 = 4096
    comptime dim_1 = 9
    comptime dim_2 = 9

    comptime batch_dims = 1
    comptime index_len = 45

    comptime input_rank = 3
    comptime indices_rank = 2
    comptime output_rank = 2

    # dim_0 x dim_1 x dim_2 input tensor.
    comptime input_layout = Layout.row_major(dim_0, dim_1, dim_2)
    var input = ctx.enqueue_create_buffer[input_type](input_layout.size())
    var input_tensor = LayoutTensor[input_type, input_layout](input)

    # Initialize with sequential data for test purposes.
    with input.map_to_host() as input_host:
        for i in range(dim_0 * dim_1 * dim_2):
            input_host[i] = i

    # We have a 2D tensor of shape (2,index_len).
    comptime indices_layout = Layout.row_major(index_len, 2)
    var indices = ctx.enqueue_create_buffer[DType.uint64](indices_layout.size())
    with indices.map_to_host() as indices_host:
        var indices_host_tensor = LayoutTensor[DType.uint64, indices_layout](
            indices_host
        )
        for i in range(index_len):
            indices_host_tensor[i, 0] = random_ui64(0, dim_1 - 1)
            indices_host_tensor[i, 1] = random_ui64(0, dim_1 - 1)
    var indices_tensor = LayoutTensor[DType.uint64, indices_layout](indices)

    # The 2D tensor is used as coordinates to dimensions 1 and 2 in the
    # dim_0 x dim_1 x dim_1 input tensor. Dimension 0 is preserved.
    # output[x, n] = input[x, Y[n, 0], Y[n, 1]],
    # where x = [0, input.dim(0)), n = [0, indices.dim(0))

    # Reference output of shape dim_0 x index_len.
    comptime output_layout = Layout.row_major(dim_0, index_len)
    var ref_output = ctx.enqueue_create_buffer[input_type](output_layout.size())
    with ref_output.map_to_host() as ref_output_host:
        with input.map_to_host() as input_host:
            with indices.map_to_host() as indices_host:
                var indices_tensor_host = LayoutTensor[
                    DType.uint64, indices_layout
                ](indices_host)
                var input_tensor_host = LayoutTensor[input_type, input_layout](
                    input_host
                )
                var ref_output_host_tensor = LayoutTensor[
                    input_type, output_layout
                ](ref_output_host)
                for i in range(Int(input_layout.shape[0])):
                    for j in range(index_len):
                        ref_output_host_tensor[i, j] = input_tensor_host[
                            i,
                            Int(indices_tensor_host[j, 0]),
                            Int(indices_tensor_host[j, 1]),
                        ]
    var ref_output_tensor = LayoutTensor[input_type, output_layout](ref_output)
    execute_index_tensor_test[batch_dims](
        input_tensor,
        indices_tensor,
        ref_output_tensor.get_immutable(),
        ref_output,
        ctx,
    )


fn test_index_tensor_DLRM_batch(ctx: DeviceContext) raises:
    print("== test_index_tensor_DLRM_batch")

    comptime input_type = DType.int32

    comptime dim_0 = 2
    comptime dim_1 = 2
    comptime dim_2 = 3
    comptime dim_3 = 4

    comptime batch_dims = 2
    comptime index_len = 5

    comptime input_rank = 4
    comptime indices_rank = 2
    comptime output_rank = 3

    # dim_0 x dim_1 x dim_2 x dim_3 input tensor.
    comptime input_layout = Layout.row_major(dim_0, dim_1, dim_2, dim_3)
    var input = ctx.enqueue_create_buffer[input_type](input_layout.size())
    var input_tensor = LayoutTensor[input_type, input_layout](input)

    # Initialize with sequential data for test purposes.
    with input.map_to_host() as input_host:
        for i in range(dim_0 * dim_1 * dim_2 * dim_3):
            input_host[i] = i

    # We have a 2D tensor of shape (index_len, 2).
    comptime indices_layout = Layout.row_major(index_len, 2)
    var indices = ctx.enqueue_create_buffer[DType.uint64](indices_layout.size())
    with indices.map_to_host() as indices_host:
        var indices_host_tensor = LayoutTensor[DType.uint64, indices_layout](
            indices_host
        )
        for i in range(index_len):
            indices_host_tensor[i, 0] = random_ui64(0, dim_2 - 1)
            indices_host_tensor[i, 1] = random_ui64(0, dim_3 - 1)
    var indices_tensor = LayoutTensor[DType.uint64, indices_layout](indices)

    # The 2D tensor is used as coordinates to dimensions 2 and 3 in the
    # dim_0 x dim_1 x dim_2 x dim_3 input tensor. Dimension 0, 1 is preserved.
    # output[x, y, n] = input[x, y, Z[n, 0], Z[n, 1]],
    # where x = [0, input.dim(0)), y = [0, input.dim(1)) and n = [0, indices.dim(0))

    # Reference output of shape dim_0 x dim_1 x index_len.
    comptime output_layout = Layout.row_major(dim_0, dim_1, index_len)
    var ref_output = ctx.enqueue_create_buffer[input_type](output_layout.size())
    with ref_output.map_to_host() as ref_output_host:
        with input.map_to_host() as input_host:
            with indices.map_to_host() as indices_host:
                var indices_tensor_host = LayoutTensor[
                    DType.uint64, indices_layout
                ](indices_host)
                var input_tensor_host = LayoutTensor[input_type, input_layout](
                    input_host
                )
                var ref_output_host_tensor = LayoutTensor[
                    input_type, output_layout
                ](ref_output_host)
                for i in range(Int(input_layout.shape[0])):
                    for j in range(Int(input_layout.shape[1])):
                        for k in range(index_len):
                            ref_output_host_tensor[i, j, k] = input_tensor_host[
                                i,
                                j,
                                Int(indices_tensor_host[k, 0]),
                                Int(indices_tensor_host[k, 1]),
                            ]
    var ref_output_tensor = LayoutTensor[input_type, output_layout](ref_output)
    execute_index_tensor_test[batch_dims](
        input_tensor,
        indices_tensor,
        ref_output_tensor.get_immutable(),
        ref_output,
        ctx,
    )


def main():
    with DeviceContext() as ctx:
        test_index_tensor_DLRM(ctx)
        test_index_tensor_DLRM_batch(ctx)
