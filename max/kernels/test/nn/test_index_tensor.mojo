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

from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
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
from testing import assert_equal

from utils import IndexList, StaticTuple


# TODO: It is like example 5 ONNX.
# CHECK-LABEL: test_index_tensor_DLRM
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
    alias input_shape = IndexList[3](dim_0, dim_1, dim_2)
    alias input_layout = Layout.row_major(input_shape)
    var input_stack = InlineArray[Scalar[input_type], input_layout.size()](
        uninitialized=True
    )
    var input = LayoutTensor[input_type, input_layout](input_stack)
    # Initialize with sequential data for test purposes.
    for i in range(dim_0 * dim_1 * dim_2):
        input.ptr[i] = i

    # We have two 1D tensors with index_len elements each.

    # index_len-element input tensor.
    var a_stack = InlineArray[UInt64, index_len](uninitialized=True)
    var index_a = LayoutTensor[DType.uint64, Layout(index_len)](a_stack)
    # Initialize with random values within [0-dim_1) since it points do dim_1 of
    # input.
    for i in range(index_len):
        index_a.ptr[i] = random_ui64(0, dim_1 - 1)

    # index_len-element input tensor.
    var b_stack = InlineArray[UInt64, index_len](uninitialized=True)
    var index_b = LayoutTensor[DType.uint64, Layout(index_len)](b_stack)
    # Initialize with random values within [0-dim_2) since it points do dim_2 of
    # input.
    for i in range(index_len):
        index_b.ptr[i] = random_ui64(0, dim_2 - 1)

    # The two 1D tensors are used as coordinates to dimensions 1 and 2 in the
    # dim_0 x dim_1 x dim_1 input tensor. Dimension 0 is preserved.
    # output[x, n] = input[x, Y[n], Z[n]],
    # where x = [0, input.dim(0)), n = [0, index_a.dim(0))

    # Reference output of shape dim_0 x index_len.
    alias ref_shape = IndexList[2](dim_0, index_len)
    alias ref_layout = Layout.row_major(ref_shape)
    var ref_stack = InlineArray[Scalar[input_type], ref_layout.size()](
        uninitialized=True
    )
    var ref_output = LayoutTensor[input_type, ref_layout](ref_stack)
    for i in range(input.dim(0)):
        for j in range(index_a.dim(0)):
            ref_output[i, j] = input[i, Int(index_a[j]), Int(index_b[j])]

    # Convert index_a, index_b (each of 1D size index_len) to a
    # 2D index_len x 2 indices NDBuffer.
    # TODO: This needs to be part of the OP itself.
    var indices_stack = InlineArray[UInt64, index_len * 2](uninitialized=True)
    var indices = LayoutTensor[DType.uint64, Layout.row_major(index_len, 2)](
        indices_stack
    )
    for i in range(index_len):
        indices[i, 0] = index_a[i]
        indices[i, 1] = index_b[i]

    alias input_dyn_layout = Layout.row_major[input.rank]()
    alias indices_dyn_layout = Layout.row_major[indices.rank]()
    var output_shape = index_tensor_shape[
        output_rank,
        input_type,
        DType.uint64,
        batch_dims,
    ](
        LayoutTensor[input.dtype, input_dyn_layout](
            input.ptr,
            RuntimeLayout[input_dyn_layout].row_major(input_shape),
        ),
        LayoutTensor[indices.dtype, indices_dyn_layout](
            indices.ptr,
            RuntimeLayout[indices_dyn_layout].row_major(
                IndexList[2](index_len, 2)
            ),
        ),
    )

    var output_data_stack = InlineArray[Scalar[input_type], dim_0 * index_len](
        uninitialized=True
    )
    alias output_layout = Layout.row_major[output_rank]()
    var output_data_buffer = LayoutTensor[input_type, output_layout](
        output_data_stack, RuntimeLayout[output_layout].row_major(output_shape)
    )

    _index_tensor_1d[batch_dims](
        LayoutTensor[input.dtype, input_dyn_layout](
            input.ptr,
            RuntimeLayout[input_dyn_layout].row_major(input_shape),
        ),
        LayoutTensor[indices.dtype, indices_dyn_layout](
            indices.ptr,
            RuntimeLayout[indices_dyn_layout].row_major(
                IndexList[2](index_len, 2)
            ),
        ),
        output_data_buffer,
    )

    for i in range(input.dim(0)):
        for j in range(index_a.dim(0)):
            assert_equal(output_data_buffer[i, j], ref_output[i, j])


# Example with batch_dim = 2 (i.e., result[:, :, indexA, indexB])
# CHECK-LABEL: test_index_tensor_DLRM_batch
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
    alias input_shape = IndexList[4](dim_0, dim_1, dim_3, dim_4)
    alias input_layout = Layout.row_major(input_shape)
    var input_stack = InlineArray[Scalar[input_type], input_layout.size()](
        uninitialized=True
    )
    var input = LayoutTensor[input_type, input_layout](input_stack)
    # Initialize with sequential data for test purposes.
    for i in range(dim_0 * dim_1 * dim_3 * dim_4):
        input.ptr[i] = i

    # We have two 1D tensors with index_len elements each.

    # index_len-element input tensor.
    var a_stack = InlineArray[UInt64, index_len](uninitialized=True)
    var index_a = LayoutTensor[DType.uint64, Layout(index_len)](a_stack)
    # Initialize with random values within [0-dim_3)
    for i in range(index_len):
        index_a.ptr[i] = random_ui64(0, dim_3 - 1)

    # index_len-element input tensor.
    var b_stack = InlineArray[UInt64, index_len](uninitialized=True)
    var index_b = LayoutTensor[DType.uint64, Layout(index_len)](b_stack)
    # Initialize with random values within [0-dim_4)
    for i in range(index_len):
        index_b.ptr[i] = random_ui64(0, dim_4 - 1)

    # The two 1D tensors are used as coordinates to dimensions 1 and 2 in the
    # dim_0 x dim_1 x dim_1 input tensor. Dimension 0 is preserved.
    # output[x, y, n] = input[x, y, Y[n], Z[n]],
    # where x = [0, input.dim(0)), y = [0, input.dim(1)),
    # n = [0, index_a.dim(0))

    # Reference output of shape dim_0 x index_len
    alias ref_shape = IndexList[3](dim_0, dim_1, index_len)
    alias ref_layout = Layout.row_major(ref_shape)
    var ref_stack = InlineArray[Scalar[input_type], ref_layout.size()](
        uninitialized=True
    )
    var ref_output = LayoutTensor[input_type, ref_layout](ref_stack)
    for i in range(input.dim(0)):
        for j in range(input.dim(1)):
            for k in range(index_a.dim(0)):
                ref_output[i, j, k] = input[
                    i, j, Int(index_a[k]), Int(index_b[k])
                ]

    # Convert index_a, index_b (each of 1D size index_len) to a 2D index_len x 2
    # indices NDBuffer.
    var indices_stack = InlineArray[UInt64, index_len * 2](uninitialized=True)
    var indices = LayoutTensor[DType.uint64, Layout.row_major(index_len, 2)](
        indices_stack
    )
    for i in range(index_len):
        indices[i, 0] = index_a[i]
        indices[i, 1] = index_b[i]

    alias input_dyn_layout = Layout.row_major[input.rank]()
    alias indices_dyn_layout = Layout.row_major[indices.rank]()
    var output_shape = index_tensor_shape[
        output_rank,
        input_type,
        DType.uint64,
        batch_dims,
    ](
        LayoutTensor[input.dtype, input_dyn_layout](
            input.ptr,
            RuntimeLayout[input_dyn_layout].row_major(input_shape),
        ),
        LayoutTensor[indices.dtype, indices_dyn_layout](
            indices.ptr,
            RuntimeLayout[indices_dyn_layout].row_major(
                IndexList[2](index_len, 2)
            ),
        ),
    )

    var output_data_stack = InlineArray[
        Scalar[input_type], dim_0 * dim_1 * index_len
    ](uninitialized=True)
    alias output_layout = Layout.row_major[output_rank]()
    var output_data_buffer = LayoutTensor[input_type, output_layout](
        output_data_stack, RuntimeLayout[output_layout].row_major(output_shape)
    )

    _index_tensor_impl[batch_dims](
        LayoutTensor[input.dtype, input_dyn_layout](
            input.ptr,
            RuntimeLayout[input_dyn_layout].row_major(input_shape),
        ),
        LayoutTensor[indices.dtype, indices_dyn_layout](
            indices.ptr,
            RuntimeLayout[indices_dyn_layout].row_major(
                IndexList[2](index_len, 2)
            ),
        ),
        output_data_buffer,
    )

    for i in range(input.dim(0)):
        for j in range(input.dim(1)):
            for k in range(index_a.dim(0)):
                assert_equal(output_data_buffer[i, j, k], ref_output[i, j, k])


# TODO: It is like example 3 ONNX gather_nd.
# CHECK-LABEL: test_index_tensor_CLIPVIT
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
    alias input_shape = IndexList[3](dim_0, dim_1, dim_2)
    alias input_layout = Layout.row_major(input_shape)
    var input_stack = InlineArray[Scalar[input_type], input_layout.size()](
        uninitialized=True
    )
    var input = LayoutTensor[input_type, input_layout](input_stack)
    # Initialize with sequential data for test purposes.
    for i in range(dim_0 * dim_1 * dim_2):
        input.ptr[i] = i

    # We have two 2D tensors with 1 element each.

    # 1-element input tensor.
    var a_stack = InlineArray[UInt64, index_len](uninitialized=True)
    var index_a = LayoutTensor[DType.uint64, Layout(index_len)](a_stack)
    # Initialize with [0,1]
    index_a.ptr[0] = 0
    index_a.ptr[1] = 1

    # 1-element input tensor.
    var b_stack = InlineArray[UInt64, index_len](uninitialized=True)
    var index_b = LayoutTensor[DType.uint64, Layout(index_len)](b_stack)
    # Initialize with [1,0]
    index_b.ptr[0] = 1
    index_b.ptr[1] = 0

    # Reference output of shape dim_0 x dim_2

    alias ref_shape = IndexList[2](dim_0, dim_2)
    alias ref_layout = Layout.row_major(ref_shape)
    var ref_stack = InlineArray[Scalar[input_type], ref_layout.size()](
        uninitialized=True
    )
    var ref_output = LayoutTensor[input_type, ref_layout](ref_stack)

    for j in range(dim_2):
        ref_output[0, j] = input[Int(index_a[0]), Int(index_a[1]), j]
    for j in range(dim_2):
        ref_output[1, j] = input[Int(index_b[0]), Int(index_b[1]), j]

    # TODO:
    # See how I need to convert separate indices to combined indices ndbuffer
    # to be as input to gather_nd.
    # See if it works with 2D indices case.
    # See if it works with non-contiguous case.

    # Convert index_a, index_b (each of 1D size 2) to a 2D indices_len x 2 indices NDBuffer
    var indices_stack = InlineArray[UInt64, index_len * 2](uninitialized=True)
    var indices = LayoutTensor[DType.uint64, Layout.row_major(index_len, 2)](
        indices_stack
    )
    indices[0, 0] = index_a[0]
    indices[0, 1] = index_b[0]
    indices[1, 0] = index_a[1]
    indices[1, 1] = index_b[1]
    # TODO: Or index_a[0], index_a[1] and index_b[0], index_b[1]???

    alias input_dyn_layout = Layout.row_major[input.rank]()
    alias indices_dyn_layout = Layout.row_major[indices.rank]()
    var output_shape = gather_nd_shape[
        output_rank,
        input_type,
        DType.uint64,
        0,
    ](
        LayoutTensor[input.dtype, input_dyn_layout](
            input.ptr,
            RuntimeLayout[input_dyn_layout].row_major(input_shape),
        ),
        LayoutTensor[indices.dtype, indices_dyn_layout](
            indices.ptr,
            RuntimeLayout[indices_dyn_layout].row_major(
                IndexList[2](index_len, 2)
            ),
        ),
    )

    var output_data_stack = InlineArray[Scalar[input_type], dim_0 * dim_2](
        uninitialized=True
    )
    alias output_layout = Layout.row_major[output_rank]()
    var output_data_buffer = LayoutTensor[input_type, output_layout](
        output_data_stack, RuntimeLayout[output_layout].row_major(output_shape)
    )

    # TODO: index_tensor works too. For batch_dims = 0 only.
    gather_nd[input_type, DType.uint64, batch_dims, target="cpu"](
        LayoutTensor[input.dtype, input_dyn_layout](
            input.ptr,
            RuntimeLayout[input_dyn_layout].row_major(input_shape),
        ),
        LayoutTensor[indices.dtype, indices_dyn_layout](
            indices.ptr,
            RuntimeLayout[indices_dyn_layout].row_major(
                IndexList[2](index_len, 2)
            ),
        ),
        output_data_buffer,
        DeviceContextPtr(),
    )

    for i in range(dim_0):
        for j in range(dim_2):
            assert_equal(output_data_buffer[i, j], ref_output[i, j])


# CHECK-LABEL: test_index_tensor_llama2_mistral
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
    alias input_shape = IndexList[2](dim_0, dim_1)
    alias input_layout = Layout.row_major(input_shape)
    var input_stack = InlineArray[Scalar[input_type], input_layout.size()](
        uninitialized=True
    )
    var input = LayoutTensor[input_type, input_layout](input_stack)
    # Initialize with sequential data for test purposes.
    for i in range(dim_0 * dim_1):
        input.ptr[i] = i

    # We have one 2D tensor with index_len elements each.

    # index_len-element input tensor.
    alias index_shape = IndexList[2](index_dim_0, index_dim_1)
    alias index_layout = Layout.row_major(index_shape)
    var a_stack = InlineArray[UInt64, index_layout.size()](uninitialized=True)
    var index_a = LayoutTensor[index_type, index_layout](a_stack)
    # Initialize with one.
    for i in range(index_dim_0):
        for j in range(index_dim_1):
            index_a[i, j] = 1

    # This is effectively a gather operation.

    # Reference output of shape index_dim_0 x index_dim_1 x dim_1.
    alias ref_shape = IndexList[3](index_dim_0, index_dim_1, dim_1)
    alias ref_layout = Layout.row_major(ref_shape)
    var ref_stack = InlineArray[Scalar[input_type], ref_layout.size()](
        uninitialized=True
    )
    var ref_output = LayoutTensor[input_type, ref_layout](ref_stack)
    for i in range(index_dim_0):
        for j in range(index_dim_1):
            for k in range(dim_1):
                ref_output[i, j, k] = input[Int(index_a[i, j]), k]

    alias input_dyn_layout = Layout.row_major[input.rank]()
    alias indices_dyn_layout = Layout.row_major[index_a.rank]()
    var output_shape = gather_shape[output_rank, input_type, index_type](
        LayoutTensor[input.dtype, input_dyn_layout](
            input.ptr,
            RuntimeLayout[input_dyn_layout].row_major(input_shape),
        ),
        LayoutTensor[index_a.dtype, indices_dyn_layout](
            index_a.ptr,
            RuntimeLayout[indices_dyn_layout].row_major(index_shape),
        ),
        0,
    )

    var output_data_stack = InlineArray[
        Scalar[input_type], index_dim_0 * index_dim_1 * dim_1
    ](uninitialized=True)
    alias output_layout = Layout.row_major[output_rank]()
    var output_data_buffer = LayoutTensor[input_type, output_layout](
        output_data_stack, RuntimeLayout[output_layout].row_major(output_shape)
    )

    gather[axis=0](
        output_data_buffer,
        LayoutTensor[input.dtype, input_dyn_layout](
            input.ptr,
            RuntimeLayout[input_dyn_layout].row_major(input_shape),
        ),
        LayoutTensor[index_a.dtype, indices_dyn_layout](
            index_a.ptr,
            RuntimeLayout[indices_dyn_layout].row_major(index_shape),
        ),
    )

    for i in range(index_dim_0):
        for j in range(index_dim_1):
            for k in range(dim_1):
                assert_equal(output_data_buffer[i, j, k], ref_output[i, j, k])


# CHECK-LABEL: test_advanced_indexing_getitem
# Matches equivalent numpy: input[:, :, index_a, index_b]
fn test_advanced_indexing_getitem() raises:
    print("== test_advanced_indexing_getitem")

    # Initialize input with sequential data for test purposes.
    alias input_type = DType.int32
    alias input_rank = 4
    alias input_shape = IndexList[input_rank](2, 3, 5, 6)
    var input_stack = InlineArray[
        Scalar[input_type], Int(input_shape.flattened_length())
    ](uninitialized=True)
    alias input_layout = Layout.row_major[input_rank]()
    var input_buffer = LayoutTensor[input_type, input_layout](
        input_stack, RuntimeLayout[input_layout].row_major(input_shape)
    )
    for i in range(input_shape.flattened_length()):
        input_buffer.ptr[i] = i

    # Create tensors for indexing in a somewhat predictable pattern
    alias index_rank = 2
    alias index_shape = IndexList[index_rank](2, 3)
    alias index_type = DType.uint64
    var a_stack = InlineArray[
        Scalar[index_type], Int(index_shape.flattened_length())
    ](uninitialized=True)
    var b_stack = InlineArray[
        Scalar[index_type], Int(index_shape.flattened_length())
    ](uninitialized=True)
    alias index_layout = Layout.row_major[index_rank]()
    var index_a = LayoutTensor[index_type, index_layout, MutableAnyOrigin](
        a_stack, RuntimeLayout[index_layout].row_major(index_shape)
    )
    var index_b = LayoutTensor[index_type, index_layout, MutableAnyOrigin](
        b_stack, RuntimeLayout[index_layout].row_major(index_shape)
    )
    for i in range(index_shape.flattened_length()):
        index_a.ptr[i] = i % 5
        index_b.ptr[i] = (i + 1) % 5
    var indices = StaticTuple[
        LayoutTensor[index_type, index_layout, MutableAnyOrigin], 2
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
    alias output_layout = Layout.row_major[output_rank]()
    var output_data_buffer = LayoutTensor[input_type, output_layout](
        output_data_stack, RuntimeLayout[output_layout].row_major(output_shape)
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
    ](coordinates: IndexList[index_rank]) capturing -> Scalar[index_type]:
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
        rebind[IndexList[input_rank]](
            input_buffer.runtime_layout.stride.value.canonicalize()
        ),
        DeviceContextPtr(),
    )

    var output_stack = InlineArray[
        Scalar[input_type], Int(output_shape.flattened_length())
    ](uninitialized=True)
    var reference_output = LayoutTensor[input_type, output_layout](
        output_stack, RuntimeLayout[output_layout].row_major(output_shape)
    )

    reference_output[0, 0, 0, 0] = 1
    reference_output[0, 0, 0, 1] = 8
    reference_output[0, 0, 0, 2] = 15
    reference_output[0, 0, 1, 0] = 22
    reference_output[0, 0, 1, 1] = 24
    reference_output[0, 0, 1, 2] = 1

    reference_output[0, 1, 0, 0] = 31
    reference_output[0, 1, 0, 1] = 38
    reference_output[0, 1, 0, 2] = 45
    reference_output[0, 1, 1, 0] = 52
    reference_output[0, 1, 1, 1] = 54
    reference_output[0, 1, 1, 2] = 31

    reference_output[0, 2, 0, 0] = 61
    reference_output[0, 2, 0, 1] = 68
    reference_output[0, 2, 0, 2] = 75
    reference_output[0, 2, 1, 0] = 82
    reference_output[0, 2, 1, 1] = 84
    reference_output[0, 2, 1, 2] = 61

    reference_output[0, 3, 0, 0] = 91
    reference_output[0, 3, 0, 1] = 98
    reference_output[0, 3, 0, 2] = 105
    reference_output[0, 3, 1, 0] = 112
    reference_output[0, 3, 1, 1] = 114
    reference_output[0, 3, 1, 2] = 91

    reference_output[0, 4, 0, 0] = 121
    reference_output[0, 4, 0, 1] = 128
    reference_output[0, 4, 0, 2] = 135
    reference_output[0, 4, 1, 0] = 142
    reference_output[0, 4, 1, 1] = 144
    reference_output[0, 4, 1, 2] = 121

    reference_output[0, 5, 0, 0] = 151
    reference_output[0, 5, 0, 1] = 158
    reference_output[0, 5, 0, 2] = 165
    reference_output[0, 5, 1, 0] = 172
    reference_output[0, 5, 1, 1] = 174
    reference_output[0, 5, 1, 2] = 151

    for i in range(output_shape.flattened_length()):
        assert_equal(output_data_buffer.ptr[i], reference_output.ptr[i])
    _ = b_stack^
    _ = a_stack^


# CHECK-LABEL: test_advanced_indexing_setitem_inplace
# Matches equivalent numpy: input[:, :, index_a, index_b] = updates
fn test_advanced_indexing_setitem_inplace() raises:
    print("== test_advanced_indexing_setitem_inplace")

    # Create input vector
    alias input_type = DType.int32
    alias input_rank = 4
    alias input_shape = IndexList[input_rank](2, 2, 4, 4)
    var input_stack = InlineArray[
        Scalar[input_type], Int(input_shape.flattened_length())
    ](uninitialized=True)
    alias input_layout = Layout.row_major[input_rank]()
    var input_buffer = LayoutTensor[input_type, input_layout](
        input_stack, RuntimeLayout[input_layout].row_major(input_shape)
    ).fill(0)

    # Create indexing tensors, ensure no pair of indices point to the same
    # location in `input` to avoid nondeterministic behavior.
    alias index_rank = 2
    alias num_index_tensors = 2
    alias index_shape = IndexList[index_rank](2, 2)
    alias index_type = DType.uint64

    var a_stack = InlineArray[
        Scalar[index_type], Int(index_shape.flattened_length())
    ](uninitialized=True)
    var b_stack = InlineArray[
        Scalar[index_type], Int(index_shape.flattened_length())
    ](uninitialized=True)
    alias index_layout = Layout.row_major[index_rank]()
    var index_a = LayoutTensor[index_type, index_layout, MutableAnyOrigin](
        a_stack, RuntimeLayout[index_layout].row_major(index_shape)
    )
    var index_b = LayoutTensor[index_type, index_layout, MutableAnyOrigin](
        b_stack, RuntimeLayout[index_layout].row_major(index_shape)
    )
    for i in range(index_shape.flattened_length()):
        index_a.ptr[i] = i % 4
        index_b.ptr[i] = (i + 1) % 4
    var indices = StaticTuple[
        LayoutTensor[index_type, index_layout, MutableAnyOrigin], 2
    ](index_a, index_b)

    # Create the updates list and set it sequential data to make it easy to read
    alias updates_rank = 4
    alias updates_shape = IndexList[updates_rank](2, 2, 2, 2)
    var updates_stack = InlineArray[
        Scalar[input_type], Int(updates_shape.flattened_length())
    ](uninitialized=True)
    alias updates_layout = Layout.row_major[updates_rank]()
    var updates = LayoutTensor[input_type, updates_layout](
        updates_stack, RuntimeLayout[updates_layout].row_major(updates_shape)
    )
    for i in range(updates_shape.flattened_length()):
        updates.ptr[i] = 1 + i

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
    ](coordinates: IndexList[index_rank]) capturing -> Scalar[index_type]:
        return indices[indices_index].load[width=1](coordinates)

    alias start_axis = 2
    advanced_indexing_setitem_inplace[
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
        rebind[IndexList[index_rank]](
            indices[0].runtime_layout.shape.value.canonicalize()
        ),
        rebind[IndexList[updates_rank]](
            updates.runtime_layout.stride.value.canonicalize()
        ),
        DeviceContextPtr(),
    )

    var output_stack = InlineArray[
        Scalar[input_type], Int(input_shape.flattened_length())
    ](uninitialized=True)

    var reference_output = LayoutTensor[input_type, input_layout](
        output_stack, RuntimeLayout[input_layout].row_major(input_shape)
    ).fill(0)

    reference_output[0, 0, 0, 1] = 1
    reference_output[0, 0, 1, 2] = 2
    reference_output[0, 0, 2, 3] = 3
    reference_output[0, 0, 3, 0] = 4

    reference_output[0, 1, 0, 1] = 5
    reference_output[0, 1, 1, 2] = 6
    reference_output[0, 1, 2, 3] = 7
    reference_output[0, 1, 3, 0] = 8

    reference_output[1, 0, 0, 1] = 9
    reference_output[1, 0, 1, 2] = 10
    reference_output[1, 0, 2, 3] = 11
    reference_output[1, 0, 3, 0] = 12

    reference_output[1, 1, 0, 1] = 13
    reference_output[1, 1, 1, 2] = 14
    reference_output[1, 1, 2, 3] = 15
    reference_output[1, 1, 3, 0] = 16

    for i in range(input_shape.flattened_length()):
        assert_equal(input_buffer.ptr[i], reference_output.ptr[i])

    _ = a_stack^
    _ = b_stack^
    _ = updates_stack^
    _ = input_stack^


def main():
    test_index_tensor_DLRM()
    test_index_tensor_DLRM_batch()
    test_index_tensor_CLIPVIT()
    test_index_tensor_llama2_mistral()
    test_advanced_indexing_getitem()
    test_advanced_indexing_setitem_inplace()
