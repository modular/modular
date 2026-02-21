# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

from layout._coord import Coord
from layout._layout import row_major
from layout._tile_tensor import TileTensor
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
    comptime input_layout = row_major[dim_0, dim_1, dim_2]()
    var input_stack = InlineArray[Scalar[input_type], input_layout.product()](
        uninitialized=True
    )
    var input = TileTensor(input_stack, input_layout)
    # Initialize with sequential data for test purposes.
    for i in range(dim_0 * dim_1 * dim_2):
        input_stack[i] = Int32(i)

    # We have two 1D tensors with index_len elements each.

    # index_len-element input tensor.
    var a_stack = InlineArray[UInt64, index_len](uninitialized=True)
    var index_a = TileTensor(a_stack, row_major[index_len]())
    # Initialize with random values within [0-dim_1) since it points do dim_1 of
    # input.
    for i in range(index_len):
        a_stack[i] = random_ui64(0, dim_1 - 1)

    # index_len-element input tensor.
    var b_stack = InlineArray[UInt64, index_len](uninitialized=True)
    var index_b = TileTensor(b_stack, row_major[index_len]())
    # Initialize with random values within [0-dim_2) since it points do dim_2 of
    # input.
    for i in range(index_len):
        b_stack[i] = random_ui64(0, dim_2 - 1)

    # The two 1D tensors are used as coordinates to dimensions 1 and 2 in the
    # dim_0 x dim_1 x dim_1 input tensor. Dimension 0 is preserved.
    # output[x, n] = input[x, Y[n], Z[n]],
    # where x = [0, input.dim(0)), n = [0, index_a.dim(0))

    # Reference output of shape dim_0 x index_len.
    comptime ref_layout = row_major[dim_0, index_len]()
    var ref_stack = InlineArray[Scalar[input_type], ref_layout.product()](
        uninitialized=True
    )
    var ref_output = TileTensor(ref_stack, ref_layout)
    for i in range(dim_0):
        for j in range(index_len):
            ref_output[i, j] = input[i, index_a[j], index_b[j]]

    # Convert index_a, index_b (each of 1D size index_len) to a
    # 2D index_len x 2 indices TileTensor.
    # TODO: This needs to be part of the OP itself.
    var indices_stack = InlineArray[UInt64, index_len * 2](uninitialized=True)
    var indices = TileTensor(indices_stack, row_major[index_len, 2]())
    for i in range(index_len):
        indices[i, 0] = index_a[i]
        indices[i, 1] = index_b[i]

    var input_dyn = input.make_dynamic[DType.int64]()
    var indices_dyn = indices.make_dynamic[DType.int64]()
    var output_shape = index_tensor_shape[
        output_rank,
        input_type,
        DType.uint64,
        batch_dims,
    ](input_dyn, indices_dyn)

    var output_data_stack = InlineArray[Scalar[input_type], dim_0 * index_len](
        uninitialized=True
    )
    comptime output_static_layout = row_major[dim_0, index_len]()
    var output_data_buffer = TileTensor(output_data_stack, output_static_layout)
    var output_dyn = output_data_buffer.make_dynamic[DType.int64]()

    _index_tensor_1d[batch_dims](input_dyn, indices_dyn, output_dyn)

    for i in range(dim_0):
        for j in range(index_len):
            assert_equal(output_data_buffer[i, j], ref_output[i, j])


# Example with batch_dim = 2 (i.e., result[:, :, indexA, indexB])
# CHECK-LABEL: test_index_tensor_DLRM_batch
fn test_index_tensor_DLRM_batch() raises:
    print("== test_index_tensor_DLRM_batch")

    comptime input_type = DType.int32

    comptime dim_0 = 2
    comptime dim_1 = 2
    comptime dim_3 = 3
    comptime dim_4 = 4

    comptime batch_dims = 2
    comptime index_len = 5

    comptime input_rank = 4
    comptime indices_rank = 2
    comptime output_rank = 3

    # dim_0 x dim_1 x dim_3 x dim_4 input tensor.
    comptime input_layout = row_major[dim_0, dim_1, dim_3, dim_4]()
    var input_stack = InlineArray[Scalar[input_type], input_layout.product()](
        uninitialized=True
    )
    var input = TileTensor(input_stack, input_layout)
    # Initialize with sequential data for test purposes.
    for i in range(dim_0 * dim_1 * dim_3 * dim_4):
        input_stack[i] = Int32(i)

    # We have two 1D tensors with index_len elements each.

    # index_len-element input tensor.
    var a_stack = InlineArray[UInt64, index_len](uninitialized=True)
    var index_a = TileTensor(a_stack, row_major[index_len]())
    # Initialize with random values within [0-dim_3)
    for i in range(index_len):
        a_stack[i] = random_ui64(0, dim_3 - 1)

    # index_len-element input tensor.
    var b_stack = InlineArray[UInt64, index_len](uninitialized=True)
    var index_b = TileTensor(b_stack, row_major[index_len]())
    # Initialize with random values within [0-dim_4)
    for i in range(index_len):
        b_stack[i] = random_ui64(0, dim_4 - 1)

    # The two 1D tensors are used as coordinates to dimensions 1 and 2 in the
    # dim_0 x dim_1 x dim_1 input tensor. Dimension 0 is preserved.
    # output[x, y, n] = input[x, y, Y[n], Z[n]],
    # where x = [0, input.dim(0)), y = [0, input.dim(1)),
    # n = [0, index_a.dim(0))

    # Reference output of shape dim_0 x index_len
    comptime ref_layout = row_major[dim_0, dim_1, index_len]()
    var ref_stack = InlineArray[Scalar[input_type], ref_layout.product()](
        uninitialized=True
    )
    var ref_output = TileTensor(ref_stack, ref_layout)
    for i in range(dim_0):
        for j in range(dim_1):
            for k in range(index_len):
                ref_output[i, j, k] = input[i, j, index_a[k], index_b[k]]

    # Convert index_a, index_b (each of 1D size index_len) to a 2D index_len x 2
    # indices TileTensor.
    var indices_stack = InlineArray[UInt64, index_len * 2](uninitialized=True)
    var indices = TileTensor(indices_stack, row_major[index_len, 2]())
    for i in range(index_len):
        indices[i, 0] = index_a[i]
        indices[i, 1] = index_b[i]

    var input_dyn = input.make_dynamic[DType.int64]()
    var indices_dyn = indices.make_dynamic[DType.int64]()
    var output_shape = index_tensor_shape[
        output_rank,
        input_type,
        DType.uint64,
        batch_dims,
    ](input_dyn, indices_dyn)

    var output_data_stack = InlineArray[
        Scalar[input_type], dim_0 * dim_1 * index_len
    ](uninitialized=True)
    comptime output_static_layout = row_major[dim_0, dim_1, index_len]()
    var output_data_buffer = TileTensor(output_data_stack, output_static_layout)
    var output_dyn = output_data_buffer.make_dynamic[DType.int64]()

    _index_tensor_impl[batch_dims](input_dyn, indices_dyn, output_dyn)

    for i in range(dim_0):
        for j in range(dim_1):
            for k in range(index_len):
                assert_equal(output_data_buffer[i, j, k], ref_output[i, j, k])


# TODO: It is like example 3 ONNX gather_nd.
# CHECK-LABEL: test_index_tensor_CLIPVIT
fn test_index_tensor_CLIPVIT() raises:
    print("== test_index_tensor_CLIPVIT")

    comptime input_type = DType.int32
    comptime dim_0 = 2
    comptime dim_1 = 2
    comptime dim_2 = 768

    comptime batch_dims = 0
    comptime index_len = 2

    comptime input_rank = 3
    comptime indices_rank = 2
    comptime output_rank = 2

    # dim_0 x dim_1 x dim_2 input tensor.
    comptime input_layout = row_major[dim_0, dim_1, dim_2]()
    var input_stack = InlineArray[Scalar[input_type], input_layout.product()](
        uninitialized=True
    )
    var input = TileTensor(input_stack, input_layout)
    # Initialize with sequential data for test purposes.
    for i in range(dim_0 * dim_1 * dim_2):
        input_stack[i] = Int32(i)

    # We have two 2D tensors with 1 element each.

    # 1-element input tensor.
    var a_stack = InlineArray[UInt64, index_len](uninitialized=True)
    var index_a = TileTensor(a_stack, row_major[index_len]())
    # Initialize with [0,1]
    a_stack[0] = 0
    a_stack[1] = 1

    # 1-element input tensor.
    var b_stack = InlineArray[UInt64, index_len](uninitialized=True)
    var index_b = TileTensor(b_stack, row_major[index_len]())
    # Initialize with [1,0]
    b_stack[0] = 1
    b_stack[1] = 0

    # Reference output of shape dim_0 x dim_2

    comptime ref_layout = row_major[dim_0, dim_2]()
    var ref_stack = InlineArray[Scalar[input_type], ref_layout.product()](
        uninitialized=True
    )
    var ref_output = TileTensor(ref_stack, ref_layout)

    for j in range(dim_2):
        ref_output[0, j] = input[Int(index_a[0]), Int(index_a[1]), j]
    for j in range(dim_2):
        ref_output[1, j] = input[Int(index_b[0]), Int(index_b[1]), j]

    # TODO:
    # See how I need to convert separate indices to
    # combined indices TileTensor
    # to be as input to gather_nd.
    # See if it works with 2D indices case.
    # See if it works with non-contiguous case.

    # Convert index_a, index_b (each of 1D size 2) to a
    # 2D indices_len x 2 indices TileTensor
    var indices_stack = InlineArray[UInt64, index_len * 2](uninitialized=True)
    var indices = TileTensor(indices_stack, row_major[index_len, 2]())
    indices[0, 0] = index_a[0]
    indices[0, 1] = index_b[0]
    indices[1, 0] = index_a[1]
    indices[1, 1] = index_b[1]
    # TODO: Or index_a[0], index_a[1] and index_b[0], index_b[1]???

    var input_dyn = input.make_dynamic[DType.int64]()
    var indices_dyn = indices.make_dynamic[DType.int64]()
    var output_shape = gather_nd_shape[
        output_rank,
        input_type,
        DType.uint64,
        0,
    ](
        input_dyn,
        indices_dyn,
    )

    var output_data_stack = InlineArray[Scalar[input_type], dim_0 * dim_2](
        uninitialized=True
    )
    comptime output_static_layout = row_major[dim_0, dim_2]()
    var output_data_buffer = TileTensor(output_data_stack, output_static_layout)
    var output_dyn = output_data_buffer.make_dynamic[DType.int64]()

    # TODO: index_tensor works too. For batch_dims = 0 only.
    gather_nd[input_type, DType.uint64, batch_dims, target="cpu"](
        input_dyn,
        indices_dyn,
        output_dyn,
        DeviceContextPtr(),
    )

    for i in range(dim_0):
        for j in range(dim_2):
            assert_equal(output_data_buffer[i, j], ref_output[i, j])


# CHECK-LABEL: test_index_tensor_llama2_mistral
fn test_index_tensor_llama2_mistral() raises:
    print("== test_index_tensor_llama2_mistral")

    comptime input_type = DType.int32
    comptime index_type = DType.uint64
    comptime dim_0 = 257
    comptime dim_1 = 128

    comptime batch_dims = 0
    comptime index_dim_0 = 1
    comptime index_dim_1 = 1

    comptime input_rank = 2
    comptime index_rank = 2
    comptime output_rank = 3

    # dim_0 x dim_1 input tensor.
    comptime input_layout = row_major[dim_0, dim_1]()
    var input_stack = InlineArray[Scalar[input_type], input_layout.product()](
        uninitialized=True
    )
    var input = TileTensor(input_stack, input_layout)
    # Initialize with sequential data for test purposes.
    for i in range(dim_0 * dim_1):
        input_stack[i] = Int32(i)

    # We have one 2D tensor with index_len elements each.

    # index_len-element input tensor.
    comptime index_layout = row_major[index_dim_0, index_dim_1]()
    var a_stack = InlineArray[UInt64, index_layout.product()](
        uninitialized=True
    )
    var index_a = TileTensor(a_stack, index_layout)
    # Initialize with one.
    for i in range(index_dim_0):
        for j in range(index_dim_1):
            index_a[i, j] = 1

    # This is effectively a gather operation.

    # Reference output of shape index_dim_0 x index_dim_1 x dim_1.
    comptime ref_layout = row_major[index_dim_0, index_dim_1, dim_1]()
    var ref_stack = InlineArray[Scalar[input_type], ref_layout.product()](
        uninitialized=True
    )
    var ref_output = TileTensor(ref_stack, ref_layout)
    for i in range(index_dim_0):
        for j in range(index_dim_1):
            for k in range(dim_1):
                ref_output[i, j, k] = input[Int(index_a[i, j]), k]

    var input_dyn = input.make_dynamic[DType.int64]()
    var index_a_dyn = index_a.make_dynamic[DType.int64]()
    var output_shape = gather_shape[output_rank, input_type, index_type](
        input_dyn,
        index_a_dyn,
        0,
    )

    var output_data_stack = InlineArray[
        Scalar[input_type], index_dim_0 * index_dim_1 * dim_1
    ](uninitialized=True)
    comptime output_static_layout = row_major[index_dim_0, index_dim_1, dim_1]()
    var output_data_buffer = TileTensor(output_data_stack, output_static_layout)
    var output_dyn = output_data_buffer.make_dynamic[DType.int64]()

    gather[axis=0](
        output_dyn,
        input_dyn,
        index_a_dyn,
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
    comptime input_type = DType.int32
    comptime input_rank = 4
    comptime input_shape = IndexList[input_rank](2, 3, 5, 6)
    var input_stack = InlineArray[
        Scalar[input_type], input_shape.flattened_length()
    ](uninitialized=True)
    comptime input_static_layout = row_major[2, 3, 5, 6]()
    var input_buffer = TileTensor(input_stack, input_static_layout)
    var input_dyn = input_buffer.make_dynamic[DType.int64]()
    for i in range(input_shape.flattened_length()):
        input_stack[i] = Int32(i)

    # Create tensors for indexing in a somewhat predictable pattern
    comptime index_rank = 2
    comptime index_shape = IndexList[index_rank](2, 3)
    comptime index_type = DType.uint64
    var a_stack = InlineArray[
        Scalar[index_type], index_shape.flattened_length()
    ](uninitialized=True)
    var b_stack = InlineArray[
        Scalar[index_type], index_shape.flattened_length()
    ](uninitialized=True)
    comptime index_static_layout = row_major[2, 3]()
    var index_a = TileTensor(a_stack, index_static_layout)
    var index_b = TileTensor(b_stack, index_static_layout)
    var _index_a_dyn = index_a.make_dynamic[DType.int64]()
    var _index_b_dyn = index_b.make_dynamic[DType.int64]()
    for i in range(index_shape.flattened_length()):
        a_stack[i] = UInt64(i % 5)
        b_stack[i] = UInt64((i + 1) % 5)
    # Create output tensor
    comptime output_rank = input_rank + index_rank - num_index_tensors
    comptime ref_shape = IndexList[output_rank](2, 3, 2, 3)
    comptime start_axis = 2
    comptime num_index_tensors = 2
    comptime output_shape = advanced_indexing_getitem_shape[
        start_axis=start_axis, num_index_tensors=num_index_tensors
    ](input_shape, index_shape)
    var output_data_stack = InlineArray[
        Scalar[input_type], output_shape.flattened_length()
    ](uninitialized=True)
    comptime output_static_layout = row_major[2, 3, 2, 3]()
    var output_data_buffer = TileTensor(output_data_stack, output_static_layout)
    var output_dyn = output_data_buffer.make_dynamic[DType.int64]()

    @parameter
    @always_inline
    fn input_tensor_fn[
        width: Int
    ](idx: IndexList[input_rank]) capturing -> SIMD[input_type, width]:
        return input_dyn.load[width=width](Coord(idx))

    @always_inline
    @parameter
    fn indices_fn[
        indices_index: Int,
    ](coordinates: IndexList[index_rank]) capturing -> Scalar[index_type]:
        comptime if indices_index == 0:
            return _index_a_dyn.load[width=1](Coord(coordinates))
        else:
            return _index_b_dyn.load[width=1](Coord(coordinates))

    # Build input strides IndexList manually from layout
    var in_strides = IndexList[input_rank](
        Int(input_dyn.dynamic_stride(0)),
        Int(input_dyn.dynamic_stride(1)),
        Int(input_dyn.dynamic_stride(2)),
        Int(input_dyn.dynamic_stride(3)),
    )

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
        output_dyn,
        in_strides,
        DeviceContextPtr(),
    )

    var output_stack = InlineArray[
        Scalar[input_type], output_shape.flattened_length()
    ](uninitialized=True)
    var reference_output = TileTensor(output_stack, output_static_layout)

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

    reference_output[1, 0, 0, 0] = 91
    reference_output[1, 0, 0, 1] = 98
    reference_output[1, 0, 0, 2] = 105
    reference_output[1, 0, 1, 0] = 112
    reference_output[1, 0, 1, 1] = 114
    reference_output[1, 0, 1, 2] = 91

    reference_output[1, 1, 0, 0] = 121
    reference_output[1, 1, 0, 1] = 128
    reference_output[1, 1, 0, 2] = 135
    reference_output[1, 1, 1, 0] = 142
    reference_output[1, 1, 1, 1] = 144
    reference_output[1, 1, 1, 2] = 121

    reference_output[1, 2, 0, 0] = 151
    reference_output[1, 2, 0, 1] = 158
    reference_output[1, 2, 0, 2] = 165
    reference_output[1, 2, 1, 0] = 172
    reference_output[1, 2, 1, 1] = 174
    reference_output[1, 2, 1, 2] = 151

    for i in range(output_shape.flattened_length()):
        assert_equal(output_data_stack[i], output_stack[i])
    _ = b_stack^
    _ = a_stack^


# CHECK-LABEL: test_advanced_indexing_setitem_inplace
# Matches equivalent numpy: input[:, :, index_a, index_b] = updates
fn test_advanced_indexing_setitem_inplace() raises:
    print("== test_advanced_indexing_setitem_inplace")

    # Create input vector
    comptime input_type = DType.int32
    comptime input_rank = 4
    comptime input_shape = IndexList[input_rank](2, 2, 4, 4)
    var input_stack = InlineArray[
        Scalar[input_type], input_shape.flattened_length()
    ](uninitialized=True)
    comptime input_static_layout = row_major[2, 2, 4, 4]()
    var input_buffer = TileTensor(input_stack, input_static_layout)
    # Fill with zeros
    for i in range(input_shape.flattened_length()):
        input_stack[i] = 0
    var input_dyn = input_buffer.make_dynamic[DType.int64]()

    # Create indexing tensors, ensure no pair of indices point to the same
    # location in `input` to avoid nondeterministic behavior.
    comptime index_rank = 2
    comptime num_index_tensors = 2
    comptime index_shape = IndexList[index_rank](2, 2)
    comptime index_type = DType.uint64

    var a_stack = InlineArray[
        Scalar[index_type], index_shape.flattened_length()
    ](uninitialized=True)
    var b_stack = InlineArray[
        Scalar[index_type], index_shape.flattened_length()
    ](uninitialized=True)
    comptime index_static_layout = row_major[2, 2]()
    var index_a = TileTensor(a_stack, index_static_layout)
    var index_b = TileTensor(b_stack, index_static_layout)
    var _index_a_dyn = index_a.make_dynamic[DType.int64]()
    var _index_b_dyn = index_b.make_dynamic[DType.int64]()
    for i in range(index_shape.flattened_length()):
        a_stack[i] = UInt64(i % 4)
        b_stack[i] = UInt64((i + 1) % 4)

    # Create the updates list and set it sequential data to make it easy to read
    comptime updates_rank = 4
    comptime updates_shape = IndexList[updates_rank](2, 2, 2, 2)
    var updates_stack = InlineArray[
        Scalar[input_type], updates_shape.flattened_length()
    ](uninitialized=True)
    comptime updates_static_layout = row_major[2, 2, 2, 2]()
    var updates = TileTensor(updates_stack, updates_static_layout)
    var updates_dyn = updates.make_dynamic[DType.int64]()
    for i in range(updates_shape.flattened_length()):
        updates_stack[i] = Int32(1 + i)

    @parameter
    @always_inline
    fn updates_tensor_fn[
        width: Int
    ](idx: IndexList[updates_rank]) capturing -> SIMD[input_type, width]:
        return updates_dyn.load[width=width](Coord(idx))

    @always_inline
    @parameter
    fn indices_fn[
        indices_index: Int,
    ](coordinates: IndexList[index_rank]) capturing -> Scalar[index_type]:
        comptime if indices_index == 0:
            return _index_a_dyn.load[width=1](Coord(coordinates))
        else:
            return _index_b_dyn.load[width=1](Coord(coordinates))

    # Build index shape and updates strides manually
    var idx_shape = IndexList[index_rank](
        Int(_index_a_dyn.dim(0)), Int(_index_a_dyn.dim(1))
    )
    var upd_strides = IndexList[updates_rank](
        Int(updates_dyn.dynamic_stride(0)),
        Int(updates_dyn.dynamic_stride(1)),
        Int(updates_dyn.dynamic_stride(2)),
        Int(updates_dyn.dynamic_stride(3)),
    )

    comptime start_axis = 2
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
        input_dyn,
        idx_shape,
        upd_strides,
        DeviceContextPtr(),
    )

    var output_stack = InlineArray[
        Scalar[input_type], input_shape.flattened_length()
    ](uninitialized=True)
    var reference_output = TileTensor(output_stack, input_static_layout)
    # Fill with zeros
    for i in range(input_shape.flattened_length()):
        output_stack[i] = 0

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
        assert_equal(input_stack[i], output_stack[i])

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
