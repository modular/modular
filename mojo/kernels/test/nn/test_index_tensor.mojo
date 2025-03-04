# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from random import random_ui64

from buffer import NDBuffer
from buffer.dimlist import DimList
from nn.gather_scatter import gather, gather_nd, gather_nd_shape, gather_shape
from nn.index_tensor import (
    _index_tensor_1d,
    _index_tensor_impl,
    index_tensor_shape,
)
from collections import InlineArray
from runtime.asyncrt import DeviceContextPtr

from utils import IndexList
from utils.index import Index


# TODO: It is like example 5 ONNX.
# CHECK-LABEL: test_index_tensor_DLRM
# CHECK: Results match
fn test_index_tensor_DLRM() raises:
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
    var input = NDBuffer[
        input_type, input_rank, DimList(dim_0, dim_1, dim_2)
    ].stack_allocation()
    # Initialize with sequential data for test purposes.
    for i in range(dim_0 * dim_1 * dim_2):
        input.data[i] = i

    # We have two 1D tensors with index_len elements each.

    # index_len-element input tensor.
    var index_a = NDBuffer[
        DType.uint64, 1, DimList(index_len)
    ].stack_allocation()
    # Initialize with random values within [0-dim_1) since it points do dim_1 of
    # input.
    for i in range(index_len):
        index_a.data[i] = random_ui64(0, dim_1 - 1)

    # index_len-element input tensor.
    var index_b = NDBuffer[
        DType.uint64, 1, DimList(index_len)
    ].stack_allocation()
    # Initialize with random values within [0-dim_2) since it points do dim_2 of
    # input.
    for i in range(index_len):
        index_b.data[i] = random_ui64(0, dim_2 - 1)

    # The two 1D tensors are used as coordinates to dimensions 1 and 2 in the
    # dim_0 x dim_1 x dim_1 input tensor. Dimension 0 is preserved.
    # output[x, n] = input[x, Y[n], Z[n]],
    # where x = [0, input.dim(0)), n = [0, index_a.dim(0))

    # Reference output of shape dim_0 x index_len.
    var ref_output = NDBuffer[
        input_type, output_rank, DimList(dim_0, index_len)
    ].stack_allocation()
    for i in range(input.dim(0)):
        for j in range(index_a.dim(0)):
            ref_output[IndexList[output_rank](i, j)] = input[
                IndexList[input_rank](
                    i, index_a[j].__int__(), index_b[j].__int__()
                )
            ]

    # Convert index_a, index_b (each of 1D size index_len) to a
    # 2D index_len x 2 indices NDBuffer.
    # TODO: This needs to be part of the OP itself.
    var indices = NDBuffer[
        DType.uint64, indices_rank, DimList(index_len, 2)
    ].stack_allocation()
    for i in range(index_len):
        indices[IndexList[indices_rank](i, 0)] = index_a[i]
        indices[IndexList[indices_rank](i, 1)] = index_b[i]

    var output_shape = index_tensor_shape[
        input_rank,
        indices_rank,
        output_rank,
        input_type,
        DType.uint64,
        batch_dims,
    ](input.make_dims_unknown(), indices.make_dims_unknown())

    var output_data_data = InlineArray[Scalar[input_type], dim_0 * index_len](
        unsafe_uninitialized=True
    )
    var output_data_buffer = NDBuffer[input_type, output_rank](
        output_data_data.unsafe_ptr(), output_shape
    )

    _index_tensor_1d[batch_dims](
        input.make_dims_unknown(),
        indices.make_dims_unknown(),
        output_data_buffer,
    )

    for i in range(input.dim(0)):
        for j in range(index_a.dim(0)):
            if (
                output_data_buffer[IndexList[output_rank](i, j)]
                != ref_output[IndexList[output_rank](i, j)]
            ):
                print("Results mismatch")
                return

    print("Results match")


# Example with batch_dim = 2 (i.e., result[:, :, indexA, indexB])
# CHECK-LABEL: test_index_tensor_DLRM_batch
# CHECK: Results match
fn test_index_tensor_DLRM_batch() raises:
    print("== test_index_tensor_DLRM_batch")

    alias input_type = DType.int32

    alias dim_0 = 2
    alias dim_1 = 2
    alias dim_3 = 3
    alias dim_4 = 4

    alias batch_dims = 2
    alias index_len = 5

    alias input_rank = 4
    alias indices_rank = 2
    alias output_rank = 3

    # dim_0 x dim_1 x dim_3 x dim_4 input tensor.
    var input = NDBuffer[
        input_type,
        input_rank,
        DimList(dim_0, dim_1, dim_3, dim_4),
    ].stack_allocation()
    # Initialize with sequential data for test purposes.
    for i in range(dim_0 * dim_1 * dim_3 * dim_4):
        input.data[i] = i

    # We have two 1D tensors with index_len elements each.

    # index_len-element input tensor.
    var index_a = NDBuffer[
        DType.uint64, 1, DimList(index_len)
    ].stack_allocation()
    # Initialize with random values within [0-dim_3)
    for i in range(index_len):
        index_a.data[i] = random_ui64(0, dim_3 - 1)

    # index_len-element input tensor.
    var index_b = NDBuffer[
        DType.uint64, 1, DimList(index_len)
    ].stack_allocation()
    # Initialize with random values within [0-dim_4)
    for i in range(index_len):
        index_b.data[i] = random_ui64(0, dim_4 - 1)

    # The two 1D tensors are used as coordinates to dimensions 1 and 2 in the
    # dim_0 x dim_1 x dim_1 input tensor. Dimension 0 is preserved.
    # output[x, y, n] = input[x, y, Y[n], Z[n]],
    # where x = [0, input.dim(0)), y = [0, input.dim(1)),
    # n = [0, index_a.dim(0))

    # Reference output of shape dim_0 x index_len

    var ref_output = NDBuffer[
        input_type, output_rank, DimList(dim_0, dim_1, index_len)
    ].stack_allocation()
    for i in range(input.dim(0)):
        for j in range(input.dim(1)):
            for k in range(index_a.dim(0)):
                ref_output[IndexList[output_rank](i, j, k)] = input[
                    IndexList[input_rank](
                        i, j, index_a[k].__int__(), index_b[k].__int__()
                    )
                ]

    # Convert index_a, index_b (each of 1D size index_len) to a 2D index_len x 2
    # indices NDBuffer.
    var indices = NDBuffer[
        DType.uint64, indices_rank, DimList(index_len, 2)
    ].stack_allocation()
    for i in range(index_len):
        indices[IndexList[indices_rank](i, 0)] = index_a[i]
        indices[IndexList[indices_rank](i, 1)] = index_b[i]

    var output_shape = index_tensor_shape[
        input_rank,
        indices_rank,
        output_rank,
        input_type,
        DType.uint64,
        batch_dims,
    ](input.make_dims_unknown(), indices.make_dims_unknown())

    var output_data_data = InlineArray[
        Scalar[input_type], dim_0 * dim_1 * index_len
    ](unsafe_uninitialized=True)
    var output_data_buffer = NDBuffer[input_type, output_rank](
        output_data_data.unsafe_ptr(), output_shape
    )

    _index_tensor_impl[batch_dims](
        input.make_dims_unknown(),
        indices.make_dims_unknown(),
        output_data_buffer,
    )

    for i in range(input.dim(0)):
        for j in range(input.dim(1)):
            for k in range(index_a.dim(0)):
                if (
                    output_data_buffer[IndexList[output_rank](i, j, k)]
                    != ref_output[IndexList[output_rank](i, j, k)]
                ):
                    print("Results mismatch")
                    return

    print("Results match")


# TODO: It is like example 3 ONNX gather_nd.
# CHECK-LABEL: test_index_tensor_CLIPVIT
# CHECK: Results match
fn test_index_tensor_CLIPVIT() raises:
    print("== test_index_tensor_CLIPVIT")

    alias input_type = DType.int32
    alias dim_0 = 2
    alias dim_1 = 2
    alias dim_2 = 768

    alias batch_dims = 0
    alias index_len = 2

    alias input_rank = 3
    alias indices_rank = 2
    alias output_rank = 2

    # dim_0 x dim_1 x dim_2 input tensor.
    var input = NDBuffer[
        input_type, input_rank, DimList(dim_0, dim_1, dim_2)
    ].stack_allocation()
    # Initialize with sequential data for test purposes.
    for i in range(dim_0 * dim_1 * dim_2):
        input.data[i] = i

    # We have two 2D tensors with 1 element each.

    # 1-element input tensor.
    var index_a = NDBuffer[
        DType.uint64, 1, DimList(index_len)
    ].stack_allocation()
    # Initialize with [0,1]
    index_a.data[0] = 0
    index_a.data[1] = 1

    # 1-element input tensor.
    var index_b = NDBuffer[
        DType.uint64, 1, DimList(index_len)
    ].stack_allocation()
    # Initialize with [1,0]
    index_b.data[0] = 1
    index_b.data[1] = 0

    # Reference output of shape dim_0 x dim_2

    var ref_output = NDBuffer[
        input_type, output_rank, DimList(dim_0, dim_2)
    ].stack_allocation()

    for j in range(dim_2):
        ref_output[IndexList[output_rank](0, j)] = input[
            IndexList[input_rank](index_a[0].__int__(), index_a[1].__int__(), j)
        ]
    for j in range(dim_2):
        ref_output[IndexList[output_rank](1, j)] = input[
            IndexList[input_rank](index_b[0].__int__(), index_b[1].__int__(), j)
        ]

    # TODO:
    # See how I need to convert separate indices to combined indices ndbuffer
    # to be as input to gather_nd.
    # See if it works with 2D indices case.
    # See if it works with non-contiguous case.

    # Convert index_a, index_b (each of 1D size 2) to a 2D indices_len x 2 indices NDBuffer
    var indices = NDBuffer[
        DType.uint64, indices_rank, DimList(index_len, 2)
    ].stack_allocation()
    indices[IndexList[indices_rank](0, 0)] = index_a[0]
    indices[IndexList[indices_rank](0, 1)] = index_b[0]
    indices[IndexList[indices_rank](1, 0)] = index_a[1]
    indices[IndexList[indices_rank](1, 1)] = index_b[1]
    # TODO: Or index_a[0], index_a[1] and index_b[0], index_b[1]???

    var output_shape = gather_nd_shape[
        input_rank,
        indices_rank,
        output_rank,
        input_type,
        DType.uint64,
        0,
    ](input.make_dims_unknown(), indices.make_dims_unknown())

    var output_data_data = InlineArray[Scalar[input_type], dim_0 * dim_2](
        unsafe_uninitialized=True
    )
    var output_data_buffer = NDBuffer[input_type, output_rank](
        output_data_data.unsafe_ptr(), output_shape
    )

    # TODO: index_tensor works too. For batch_dims = 0 only.
    gather_nd[
        input_type,
        DType.uint64,
        input_rank,
        indices_rank,
        output_rank,
        batch_dims,
        target="cpu",
    ](
        input.make_dims_unknown(),
        indices.make_dims_unknown(),
        output_data_buffer,
        DeviceContextPtr(),
    )

    for i in range(dim_0):
        for j in range(dim_2):
            if (
                output_data_buffer[IndexList[output_rank](i, j)]
                != ref_output[IndexList[output_rank](i, j)]
            ):
                print("Results mismatch")
                return

    print("Results match")


# CHECK-LABEL: test_index_tensor_llama2_mistral
# CHECK: Results match
fn test_index_tensor_llama2_mistral() raises:
    print("== test_index_tensor_llama2_mistral")

    alias input_type = DType.int32
    alias index_type = DType.uint64
    alias dim_0 = 257
    alias dim_1 = 128

    alias batch_dims = 0
    alias index_dim_0 = 1
    alias index_dim_1 = 1

    alias input_rank = 2
    alias index_rank = 2
    alias output_rank = 3

    # dim_0 x dim_1 input tensor.
    var input = NDBuffer[
        input_type, input_rank, DimList(dim_0, dim_1)
    ].stack_allocation()
    # Initialize with sequential data for test purposes.
    for i in range(dim_0 * dim_1):
        input.data[i] = i

    # We have one 2D tensor with index_len elements each.

    # index_len-element input tensor.
    var index_a = NDBuffer[
        index_type, index_rank, DimList(index_dim_0, index_dim_1)
    ].stack_allocation()
    # Initialize with one.
    for i in range(index_dim_0):
        for j in range(index_dim_1):
            index_a[IndexList[index_rank](i, j)] = 1

    # This is effectively a gather operation.

    # Reference output of shape index_dim_0 x index_dim_1 x dim_1.
    var ref_output = NDBuffer[
        input_type, output_rank, DimList(index_dim_0, index_dim_1, dim_1)
    ].stack_allocation()
    for i in range(index_dim_0):
        for j in range(index_dim_1):
            for k in range(dim_1):
                ref_output[IndexList[output_rank](i, j, k)] = input[
                    IndexList[input_rank](index_a[i, j].__int__(), k)
                ]

    var output_shape = gather_shape[
        output_rank, input_rank, index_rank, input_type, index_type
    ](input.make_dims_unknown(), index_a.make_dims_unknown(), 0)

    var output_data_data = InlineArray[
        Scalar[input_type], index_dim_0 * index_dim_1 * dim_1
    ](unsafe_uninitialized=True)
    var output_data_buffer = NDBuffer[input_type, output_rank](
        output_data_data.unsafe_ptr(), output_shape
    )

    gather[axis=0](
        output_data_buffer,
        input.make_dims_unknown(),
        index_a.make_dims_unknown(),
    )

    for i in range(index_dim_0):
        for j in range(index_dim_1):
            for k in range(dim_1):
                if (
                    output_data_buffer[IndexList[output_rank](i, j, k)]
                    != ref_output[IndexList[output_rank](i, j, k)]
                ):
                    print("Results mismatch")

    print("Results match")


def main():
    test_index_tensor_DLRM()
    test_index_tensor_DLRM_batch()
    test_index_tensor_CLIPVIT()
    test_index_tensor_llama2_mistral()
