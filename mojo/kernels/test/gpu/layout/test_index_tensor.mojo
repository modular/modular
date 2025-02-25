# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug-no-assert %s

from random import random_ui64

from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.host import DeviceContext
from internal_utils import DeviceNDBuffer, HostNDBuffer
from memory import stack_allocation
from nn.index_tensor import _index_tensor_impl, index_tensor_shape
from testing import assert_equal, assert_true

from utils import IndexList


def execute_index_tensor_test[
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
        data_host.type, data_host.rank, data_host.shape
    ](
        data_host.tensor.get_shape(),
        ctx=ctx,
    )
    var indices_device = DeviceNDBuffer[
        indices_host.type, indices_host.rank, indices_host.shape
    ](
        indices_host.tensor.get_shape(),
        ctx=ctx,
    )
    var actual_output_host = HostNDBuffer[
        expected_output.type, expected_output.rank, expected_output.shape
    ](
        expected_output.tensor.get_shape(),
    )
    var actual_output_device = DeviceNDBuffer[
        expected_output.type, expected_output.rank, expected_output.shape
    ](
        expected_output.tensor.get_shape(),
        ctx=ctx,
    )
    ctx.enqueue_copy(data_device.buffer, data_host.tensor.data)
    ctx.enqueue_copy(indices_device.buffer, indices_host.tensor.data)

    # execute the kernel
    _index_tensor_impl[batch_dims, target="gpu"](
        data_device.tensor.make_dims_unknown(),
        indices_device.tensor.make_dims_unknown(),
        actual_output_device.tensor.make_dims_unknown(),
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


fn test_index_tensor_DLRM(ctx: DeviceContext) raises:
    print("== test_index_tensor_DLRM")

    alias input_type = DType.int32
    alias dim_0 = 4096
    alias dim_1 = 9
    alias dim_2 = 9

    alias batch_dims = 1
    alias index_len = 45

    alias input_rank = 3
    alias indices_rank = 2
    alias output_rank = 2

    # dim_0 x dim_1 x dim_2 input tensor.
    var input = HostNDBuffer[
        input_type, input_rank, DimList(dim_0, dim_1, dim_2)
    ](IndexList[input_rank](dim_0, dim_1, dim_2))

    # Initialize with sequential data for test purposes.
    for i in range(dim_0 * dim_1 * dim_2):
        input.tensor.data[i] = i

    # We have a 2D tensor of shape (2,index_len).
    var indices = HostNDBuffer[
        DType.uint64, indices_rank, DimList(index_len, 2)
    ](IndexList[indices_rank](index_len, 2))
    for i in range(index_len):
        indices.tensor[IndexList[indices_rank](i, 0)] = random_ui64(
            0, dim_1 - 1
        )
        indices.tensor[IndexList[indices_rank](i, 1)] = random_ui64(
            0, dim_1 - 1
        )

    # The 2D tensor is used as coordinates to dimensions 1 and 2 in the
    # dim_0 x dim_1 x dim_1 input tensor. Dimension 0 is preserved.
    # output[x, n] = input[x, Y[n, 0], Y[n, 1]],
    # where x = [0, input.dim(0)), n = [0, indices.dim(0))

    # Reference output of shape dim_0 x index_len.
    var ref_output = HostNDBuffer[
        input_type, output_rank, DimList(dim_0, index_len)
    ](IndexList[output_rank](dim_0, index_len))
    for i in range(input.tensor.get_shape()[0]):
        for j in range(indices.tensor.get_shape()[0]):
            ref_output.tensor[IndexList[output_rank](i, j)] = input.tensor[
                IndexList[input_rank](
                    i,
                    indices.tensor[j, 0].__int__(),
                    indices.tensor[j, 1].__int__(),
                )
            ]
    execute_index_tensor_test[batch_dims](input, indices, ref_output, ctx)


fn test_index_tensor_DLRM_batch(ctx: DeviceContext) raises:
    print("== test_index_tensor_DLRM_batch")

    alias input_type = DType.int32

    alias dim_0 = 2
    alias dim_1 = 2
    alias dim_2 = 3
    alias dim_3 = 4

    alias batch_dims = 2
    alias index_len = 5

    alias input_rank = 4
    alias indices_rank = 2
    alias output_rank = 3

    # dim_0 x dim_1 x dim_2 x dim_3 input tensor.
    var input = HostNDBuffer[
        input_type, input_rank, DimList(dim_0, dim_1, dim_2, dim_3)
    ](IndexList[input_rank](dim_0, dim_1, dim_2, dim_3))

    # Initialize with sequential data for test purposes.
    for i in range(dim_0 * dim_1 * dim_2 * dim_3):
        input.tensor.data[i] = i

    # We have a 2D tensor of shape (2,index_len).
    var indices = HostNDBuffer[
        DType.uint64, indices_rank, DimList(index_len, 2)
    ](IndexList[indices_rank](index_len, 2))
    for i in range(index_len):
        indices.tensor[IndexList[indices_rank](i, 0)] = random_ui64(
            0, dim_1 - 1
        )
        indices.tensor[IndexList[indices_rank](i, 1)] = random_ui64(
            0, dim_1 - 1
        )

    # The 2D tensor is used as coordinates to dimensions 2 and 3 in the
    # dim_0 x dim_1 x dim_2 x dim_3 input tensor. Dimension 0, 1 is preserved.
    # output[x, y, n] = input[x, y, Z[n, 0], Z[n, 1]],
    # where x = [0, input.dim(0)), y = [0, input.dim(1)) and n = [0, indices.dim(0))

    # Reference output of shape dim_0 x dim_1 x index_len.
    var ref_output = HostNDBuffer[
        input_type, output_rank, DimList(dim_0, dim_1, index_len)
    ](IndexList[output_rank](dim_0, dim_1, index_len))
    for i in range(input.tensor.get_shape()[0]):
        for j in range(input.tensor.get_shape()[1]):
            for k in range(indices.tensor.get_shape()[0]):
                ref_output.tensor[
                    IndexList[output_rank](i, j, k)
                ] = input.tensor[
                    IndexList[input_rank](
                        i,
                        j,
                        indices.tensor[k, 0].__int__(),
                        indices.tensor[k, 1].__int__(),
                    )
                ]
    execute_index_tensor_test[batch_dims](input, indices, ref_output, ctx)


def main():
    with DeviceContext() as ctx:
        test_index_tensor_DLRM(ctx)
        test_index_tensor_DLRM_batch(ctx)
