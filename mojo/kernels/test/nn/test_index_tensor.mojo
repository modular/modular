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
# RUN: %mojo-no-debug %s | FileCheck %s

from collections import InlineArray
from random import random_ui64

from buffer import NDBuffer
from buffer.dimlist import DimList
from nn.gather_scatter import gather, gather_nd, gather_nd_shape, gather_shape
from nn.index_tensor import (
    _index_tensor_1d,
    _index_tensor_impl,
    advanced_indexing_getitem,
    advanced_indexing_getitem_shape,
    advanced_indexing_setitem_inplace,
    index_tensor_shape,
)
from runtime.asyncrt import DeviceContextPtr

from utils import IndexList, StaticTuple
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
    alias input_shape = DimList(dim_0, dim_1, dim_2)
    var input_stack = InlineArray[
        Scalar[input_type], Int(input_shape.product())
    ](uninitialized=True)
    var input = NDBuffer[input_type, input_rank, _, input_shape](input_stack)
    # Initialize with sequential data for test purposes.
    for i in range(dim_0 * dim_1 * dim_2):
        input.data[i] = i

    # We have two 1D tensors with index_len elements each.

    # index_len-element input tensor.
    var a_stack = InlineArray[UInt64, index_len](uninitialized=True)
    var index_a = NDBuffer[DType.uint64, 1, _, DimList(index_len)](a_stack)
    # Initialize with random values within [0-dim_1) since it points do dim_1 of
    # input.
    for i in range(index_len):
        index_a.data[i] = random_ui64(0, dim_1 - 1)

    # index_len-element input tensor.
    var b_stack = InlineArray[UInt64, index_len](uninitialized=True)
    var index_b = NDBuffer[DType.uint64, 1, _, DimList(index_len)](b_stack)
    # Initialize with random values within [0-dim_2) since it points do dim_2 of
    # input.
    for i in range(index_len):
        index_b.data[i] = random_ui64(0, dim_2 - 1)

    # The two 1D tensors are used as coordinates to dimensions 1 and 2 in the
    # dim_0 x dim_1 x dim_1 input tensor. Dimension 0 is preserved.
    # output[x, n] = input[x, Y[n], Z[n]],
    # where x = [0, input.dim(0)), n = [0, index_a.dim(0))

    # Reference output of shape dim_0 x index_len.
    alias ref_shape = DimList(dim_0, index_len)
    var ref_stack = InlineArray[Scalar[input_type], Int(ref_shape.product())](
        uninitialized=True
    )
    var ref_output = NDBuffer[input_type, output_rank, _, ref_shape](ref_stack)
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
    var indices_stack = InlineArray[UInt64, index_len * 2](uninitialized=True)
    var indices = NDBuffer[
        DType.uint64, indices_rank, _, DimList(index_len, 2)
    ](indices_stack)
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

    var output_data_stack = InlineArray[Scalar[input_type], dim_0 * index_len](
        uninitialized=True
    )
    var output_data_buffer = NDBuffer[input_type, output_rank](
        output_data_stack, output_shape
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
    alias input_shape = DimList(dim_0, dim_1, dim_3, dim_4)
    var input_stack = InlineArray[
        Scalar[input_type], Int(input_shape.product())
    ](uninitialized=True)
    var input = NDBuffer[
        input_type,
        input_rank,
        _,
        DimList(dim_0, dim_1, dim_3, dim_4),
    ](input_stack)
    # Initialize with sequential data for test purposes.
    for i in range(dim_0 * dim_1 * dim_3 * dim_4):
        input.data[i] = i

    # We have two 1D tensors with index_len elements each.

    # index_len-element input tensor.
    var a_stack = InlineArray[UInt64, index_len](uninitialized=True)
    var index_a = NDBuffer[DType.uint64, 1, _, DimList(index_len)](a_stack)
    # Initialize with random values within [0-dim_3)
    for i in range(index_len):
        index_a.data[i] = random_ui64(0, dim_3 - 1)

    # index_len-element input tensor.
    var b_stack = InlineArray[UInt64, index_len](uninitialized=True)
    var index_b = NDBuffer[DType.uint64, 1, _, DimList(index_len)](b_stack)
    # Initialize with random values within [0-dim_4)
    for i in range(index_len):
        index_b.data[i] = random_ui64(0, dim_4 - 1)

    # The two 1D tensors are used as coordinates to dimensions 1 and 2 in the
    # dim_0 x dim_1 x dim_1 input tensor. Dimension 0 is preserved.
    # output[x, y, n] = input[x, y, Y[n], Z[n]],
    # where x = [0, input.dim(0)), y = [0, input.dim(1)),
    # n = [0, index_a.dim(0))

    # Reference output of shape dim_0 x index_len
    alias ref_shape = DimList(dim_0, dim_1, index_len)
    var ref_stack = InlineArray[Scalar[input_type], Int(ref_shape.product())](
        uninitialized=True
    )
    var ref_output = NDBuffer[input_type, output_rank, _, ref_shape](ref_stack)
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
    var indices_stack = InlineArray[UInt64, index_len * 2](uninitialized=True)
    var indices = NDBuffer[
        DType.uint64, indices_rank, _, DimList(index_len, 2)
    ](indices_stack)
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

    var output_data_stack = InlineArray[
        Scalar[input_type], dim_0 * dim_1 * index_len
    ](uninitialized=True)
    var output_data_buffer = NDBuffer[input_type, output_rank](
        output_data_stack, output_shape
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
    alias input_shape = DimList(dim_0, dim_1, dim_2)
    var input_stack = InlineArray[
        Scalar[input_type], Int(input_shape.product())
    ](uninitialized=True)
    var input = NDBuffer[input_type, input_rank, _, input_shape](input_stack)
    # Initialize with sequential data for test purposes.
    for i in range(dim_0 * dim_1 * dim_2):
        input.data[i] = i

    # We have two 2D tensors with 1 element each.

    # 1-element input tensor.
    var a_stack = InlineArray[UInt64, index_len](uninitialized=True)
    var index_a = NDBuffer[DType.uint64, 1, _, DimList(index_len)](a_stack)
    # Initialize with [0,1]
    index_a.data[0] = 0
    index_a.data[1] = 1

    # 1-element input tensor.
    var b_stack = InlineArray[UInt64, index_len](uninitialized=True)
    var index_b = NDBuffer[DType.uint64, 1, _, DimList(index_len)](b_stack)
    # Initialize with [1,0]
    index_b.data[0] = 1
    index_b.data[1] = 0

    # Reference output of shape dim_0 x dim_2

    alias ref_shape = DimList(dim_0, dim_2)
    var ref_stack = InlineArray[Scalar[input_type], Int(ref_shape.product())](
        uninitialized=True
    )
    var ref_output = NDBuffer[input_type, output_rank, _, ref_shape](ref_stack)

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
    var indices_stack = InlineArray[UInt64, index_len * 2](uninitialized=True)
    var indices = NDBuffer[
        DType.uint64, indices_rank, _, DimList(index_len, 2)
    ](indices_stack)
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

    var output_data_stack = InlineArray[Scalar[input_type], dim_0 * dim_2](
        uninitialized=True
    )
    var output_data_buffer = NDBuffer[input_type, output_rank](
        output_data_stack, output_shape
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
    alias input_shape = DimList(dim_0, dim_1)
    var input_stack = InlineArray[
        Scalar[input_type], Int(input_shape.product())
    ](uninitialized=True)
    var input = NDBuffer[input_type, input_rank, _, input_shape](input_stack)
    # Initialize with sequential data for test purposes.
    for i in range(dim_0 * dim_1):
        input.data[i] = i

    # We have one 2D tensor with index_len elements each.

    # index_len-element input tensor.
    alias index_shape = DimList(index_dim_0, index_dim_1)
    var a_stack = InlineArray[UInt64, Int(index_shape.product())](
        uninitialized=True
    )
    var index_a = NDBuffer[index_type, index_rank, _, index_shape](a_stack)
    # Initialize with one.
    for i in range(index_dim_0):
        for j in range(index_dim_1):
            index_a[IndexList[index_rank](i, j)] = 1

    # This is effectively a gather operation.

    # Reference output of shape index_dim_0 x index_dim_1 x dim_1.
    alias ref_shape = DimList(index_dim_0, index_dim_1, dim_1)
    var ref_stack = InlineArray[Scalar[input_type], Int(ref_shape.product())](
        uninitialized=True
    )
    var ref_output = NDBuffer[input_type, output_rank, _, ref_shape](ref_stack)
    for i in range(index_dim_0):
        for j in range(index_dim_1):
            for k in range(dim_1):
                ref_output[IndexList[output_rank](i, j, k)] = input[
                    IndexList[input_rank](index_a[i, j].__int__(), k)
                ]

    var output_shape = gather_shape[
        output_rank, input_rank, index_rank, input_type, index_type
    ](input.make_dims_unknown(), index_a.make_dims_unknown(), 0)

    var output_data_stack = InlineArray[
        Scalar[input_type], index_dim_0 * index_dim_1 * dim_1
    ](uninitialized=True)
    var output_data_buffer = NDBuffer[input_type, output_rank](
        output_data_stack, output_shape
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


# CHECK-LABEL: test_advanced_indexing_getitem
# Matches equivalent numpy: input[:, :, index_a, index_b]
# CHECK{LITERAL}: NDBuffer([[[[1, 8, 15],
# CHECK{LITERAL}: [22, 24, 1]],
# CHECK{LITERAL}: [[31, 38, 45],
# CHECK{LITERAL}: [52, 54, 31]],
# CHECK{LITERAL}: [[61, 68, 75],
# CHECK{LITERAL}: [82, 84, 61]],
# CHECK{LITERAL}: [[91, 98, 105],
# CHECK{LITERAL}: [112, 114, 91]],
# CHECK{LITERAL}: [[121, 128, 135],
# CHECK{LITERAL}: [142, 144, 121]],
# CHECK{LITERAL}: [[151, 158, 165],
# CHECK{LITERAL}: [172, 174, 151]]]], dtype=int32, shape=2x3x2x3)
fn test_advanced_indexing_getitem() raises:
    print("== test_advanced_indexing_getitem")

    # Initialize input with sequential data for test purposes.
    alias input_type = DType.int32
    alias input_rank = 4
    alias input_shape = IndexList[input_rank](2, 3, 5, 6)
    var input_stack = InlineArray[
        Scalar[input_type], Int(input_shape.flattened_length())
    ](uninitialized=True)
    var input_buffer = NDBuffer[input_type, input_rank](
        input_stack, input_shape
    )
    for i in range(input_shape.flattened_length()):
        input_buffer.data[i] = i

    # Create tensors for indexing in a somewhat predictable pattern
    alias index_rank = 2
    alias index_shape = IndexList[index_rank](2, 3)
    alias index_type = DType.uint64
    var a_stack = InlineArray[
        Scalar[index_type], Int(index_shape.flattened_length())
    ](uninitialized=True)
    var index_a = NDBuffer[index_type, index_rank](a_stack, index_shape)
    var b_stack = InlineArray[
        Scalar[index_type], Int(index_shape.flattened_length())
    ](uninitialized=True)
    var index_b = NDBuffer[index_type, index_rank](b_stack, index_shape)
    for i in range(index_shape.flattened_length()):
        index_a.data[i] = i % 5
        index_b.data[i] = (i + 1) % 5
    var indices = StaticTuple[
        NDBuffer[index_type, index_rank, MutableAnyOrigin], 2
    ](index_a, index_b)

    # Create output tensor
    alias output_rank = input_rank + index_rank - num_index_tensors
    alias ref_shape = IndexList[output_rank](2, 3, 2, 3)
    alias start_axis = 2
    alias num_index_tensors = 2
    alias output_shape = advanced_indexing_getitem_shape[
        start_axis=start_axis, num_index_tensors=num_index_tensors
    ](input_shape, index_shape)
    var output_data_stack = InlineArray[
        Scalar[input_type], output_shape.flattened_length()
    ](uninitialized=True)
    var output_data_buffer = NDBuffer[input_type, output_rank](
        output_data_stack, output_shape
    )

    @parameter
    @always_inline
    fn input_tensor_fn[
        width: Int
    ](idx: IndexList[input_rank]) capturing -> SIMD[input_type, width]:
        return input_buffer.load[width=width](idx)

    @always_inline
    @parameter
    fn indices_fn[
        indices_index: Int,
    ](coordinates: IndexList[index_rank]) capturing -> SIMD[index_type, 1]:
        return indices[indices_index].load[width=1](coordinates)

    advanced_indexing_getitem[
        input_rank=input_rank,
        start_axis=start_axis,
        num_index_tensors=num_index_tensors,
        target="cpu",
        single_thread_blocking_override=False,
        trace_description="test_advanced_indexing_getitem",
        input_tensor_fn=input_tensor_fn,
        indices_fn=indices_fn,
    ](
        output_data_buffer,
        input_buffer.get_strides(),
        DeviceContextPtr(),
    )

    # Print to filecheck
    print(output_data_buffer)

    # Keep data alive
    _ = (indices[0], indices[1])
    _ = input_buffer


# CHECK-LABEL: test_advanced_indexing_setitem_inplace
# Matches equivalent numpy: input[:, :, index_a, index_b] = updates
# CHECK{LITERAL}: NDBuffer([[[[0, 1, 0, 0],
# CHECK{LITERAL}: [0, 0, 2, 0],
# CHECK{LITERAL}: [0, 0, 0, 3],
# CHECK{LITERAL}: [4, 0, 0, 0]],
# CHECK{LITERAL}: [[0, 5, 0, 0],
# CHECK{LITERAL}: [0, 0, 6, 0],
# CHECK{LITERAL}: [0, 0, 0, 7],
# CHECK{LITERAL}: [8, 0, 0, 0]],
# CHECK{LITERAL}: [[0, 9, 0, 0],
# CHECK{LITERAL}: [0, 0, 10, 0],
# CHECK{LITERAL}: [0, 0, 0, 11],
# CHECK{LITERAL}: [12, 0, 0, 0]],
# CHECK{LITERAL}: [[0, 13, 0, 0],
# CHECK{LITERAL}: [0, 0, 14, 0],
# CHECK{LITERAL}: [0, 0, 0, 15],
# CHECK{LITERAL}: [16, 0, 0, 0]]]], dtype=int32, shape=2x2x4x4)
fn test_advanced_indexing_setitem_inplace() raises:
    print("== test_advanced_indexing_setitem_inplace")

    # Create input vector
    alias input_type = DType.int32
    alias input_rank = 4
    alias input_shape = IndexList[input_rank](2, 2, 4, 4)
    var input_stack = InlineArray[
        Scalar[input_type], Int(input_shape.flattened_length())
    ](uninitialized=True)
    var input_buffer = NDBuffer[input_type, input_rank](
        input_stack, input_shape
    )

    # Initialize with all zeros to make it obvious where we write to
    for i in range(input_shape.flattened_length()):
        input_buffer.data[i] = 0

    # Create indexing tensors, ensure no pair of indices point to the same
    # location in `input` to avoid nondeterministic behavior.
    alias index_rank = 2
    alias num_index_tensors = 2
    alias index_shape = IndexList[index_rank](2, 2)
    alias index_type = DType.uint64

    var a_stack = InlineArray[
        Scalar[index_type], Int(index_shape.flattened_length())
    ](uninitialized=True)
    var index_a = NDBuffer[index_type, index_rank](a_stack, index_shape)
    var b_stack = InlineArray[
        Scalar[index_type], Int(index_shape.flattened_length())
    ](uninitialized=True)
    var index_b = NDBuffer[index_type, index_rank](b_stack, index_shape)
    for i in range(index_shape.flattened_length()):
        index_a.data[i] = i % 4
        index_b.data[i] = (i + 1) % 4
    var indices = StaticTuple[
        NDBuffer[index_type, index_rank, MutableAnyOrigin], 2
    ](index_a, index_b)

    # Create the updates list and set it sequential data to make it easy to read
    alias updates_rank = 4
    alias updates_shape = IndexList[updates_rank](2, 2, 2, 2)
    var updates_stack = InlineArray[
        Scalar[input_type], Int(updates_shape.flattened_length())
    ](uninitialized=True)
    var updates = NDBuffer[input_type, updates_rank](
        updates_stack, updates_shape
    )
    for i in range(updates_shape.flattened_length()):
        updates.data[i] = 1 + i

    @parameter
    @always_inline
    fn updates_tensor_fn[
        width: Int
    ](idx: IndexList[updates_rank]) capturing -> SIMD[input_type, width]:
        return updates.load[width=width](idx)

    @always_inline
    @parameter
    fn indices_fn[
        indices_index: Int,
    ](coordinates: IndexList[index_rank]) capturing -> SIMD[index_type, 1]:
        return indices[indices_index].load[width=1](coordinates)

    alias start_axis = 2
    advanced_indexing_setitem_inplace[
        input_rank=input_rank,
        index_rank=index_rank,
        start_axis=start_axis,
        num_index_tensors=num_index_tensors,
        target="cpu",
        single_thread_blocking_override=False,
        trace_description="test_advanced_indexing_setitem_inplace",
        updates_tensor_fn=updates_tensor_fn,
        indices_fn=indices_fn,
    ](
        input_buffer,
        indices[0].dynamic_shape,
        updates.get_strides(),
        DeviceContextPtr(),
    )

    print(input_buffer)

    # Keep data alive
    _ = (indices[0], indices[1])
    _ = updates


def main():
    test_index_tensor_DLRM()
    test_index_tensor_DLRM_batch()
    test_index_tensor_CLIPVIT()
    test_index_tensor_llama2_mistral()
    test_advanced_indexing_getitem()
    test_advanced_indexing_setitem_inplace()
