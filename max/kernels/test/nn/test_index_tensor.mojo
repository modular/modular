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

from max.gpu.host import DeviceContext
from std.random import random_ui64

from layout import TileTensor, row_major
from nn.gather_scatter import gather, gather_nd, gather_nd_shape, gather_shape
from nn.index_tensor import (
    _index_tensor_1d,
    _index_tensor_impl,
    index_tensor_shape,
)
from std.math import align_up

from std.sys import simd_width_of
from std.testing import assert_equal


# TODO: It is like example 5 ONNX.
# CHECK-LABEL: test_index_tensor_DLRM
def test_index_tensor_DLRM() raises:
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
    # Initialize with sequential data for test purposes.
    var input_stack = Array[
        Scalar[input_type],
        align_up(input_layout.product(), simd_width_of[input_type]()),
    ](fill_with=lambda (i: Int) -> Scalar[input_type]: Int32(i))
    var input = TileTensor(input_stack, input_layout)

    # We have two 1D tensors with index_len elements each.

    # index_len-element input tensor.
    # Initialize with random values within [0-dim_1) since it points do dim_1 of
    # input.
    var a_stack = Array[
        UInt64, align_up(index_len, simd_width_of[DType.uint64]())
    ](fill_with=lambda (_i: Int) -> UInt64: random_ui64(0, dim_1 - 1))
    var index_a = TileTensor(a_stack, row_major[index_len]())

    # index_len-element input tensor.
    # Initialize with random values within [0-dim_2) since it points do dim_2 of
    # input.
    var b_stack = Array[
        UInt64, align_up(index_len, simd_width_of[DType.uint64]())
    ](fill_with=lambda (_i: Int) -> UInt64: random_ui64(0, dim_2 - 1))
    var index_b = TileTensor(b_stack, row_major[index_len]())

    # The two 1D tensors are used as coordinates to dimensions 1 and 2 in the
    # dim_0 x dim_1 x dim_1 input tensor. Dimension 0 is preserved.
    # output[x, n] = input[x, Y[n], Z[n]],
    # where x = [0, input.dim(0)), n = [0, index_a.dim(0))

    # Reference output of shape dim_0 x index_len.
    comptime ref_layout = row_major[dim_0, index_len]()
    var ref_stack = Array[
        Scalar[input_type],
        align_up(ref_layout.product(), simd_width_of[input_type]()),
    ](fill=0)
    var ref_output = TileTensor(ref_stack, ref_layout)
    for i in range(dim_0):
        for j in range(index_len):
            ref_output[i, j] = input[i, index_a[j], index_b[j]]

    # Convert index_a, index_b (each of 1D size index_len) to a
    # 2D index_len x 2 indices TileTensor.
    # TODO: This needs to be part of the OP itself.
    var indices_stack = Array[
        UInt64, align_up(index_len * 2, simd_width_of[DType.uint64]())
    ](fill=0)
    var indices = TileTensor(indices_stack, row_major[index_len, 2]())
    for i in range(index_len):
        indices[i, 0] = index_a[i]
        indices[i, 1] = index_b[i]

    var input_dyn = input.make_dynamic[.int64]()
    var indices_dyn = indices.make_dynamic[.int64]()
    var output_shape = index_tensor_shape[
        output_rank,
        input_type,
        .uint64,
        batch_dims,
    ](input_dyn, indices_dyn)

    var output_data_stack = Array[
        Scalar[input_type],
        align_up(dim_0 * index_len, simd_width_of[input_type]()),
    ](fill=0)
    comptime output_static_layout = row_major[dim_0, index_len]()
    var output_data_buffer = TileTensor(output_data_stack, output_static_layout)
    var output_dyn = output_data_buffer.make_dynamic[.int64]()

    _index_tensor_1d[batch_dims](input_dyn, indices_dyn, output_dyn)

    for i in range(dim_0):
        for j in range(index_len):
            assert_equal(output_data_buffer[i, j], ref_output[i, j])


# Example with batch_dim = 2 (i.e., result[:, :, indexA, indexB])
# CHECK-LABEL: test_index_tensor_DLRM_batch
def test_index_tensor_DLRM_batch() raises:
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
    # Initialize with sequential data for test purposes.
    var input_stack = Array[
        Scalar[input_type],
        align_up(input_layout.product(), simd_width_of[input_type]()),
    ](fill_with=lambda (i: Int) -> Scalar[input_type]: Int32(i))
    var input = TileTensor(input_stack, input_layout)

    # We have two 1D tensors with index_len elements each.

    # index_len-element input tensor.
    # Initialize with random values within [0-dim_3)
    var a_stack = Array[
        UInt64, align_up(index_len, simd_width_of[DType.uint64]())
    ](fill_with=lambda (_i: Int) -> UInt64: random_ui64(0, dim_3 - 1))
    var index_a = TileTensor(a_stack, row_major[index_len]())

    # index_len-element input tensor.
    # Initialize with random values within [0-dim_4)
    var b_stack = Array[
        UInt64, align_up(index_len, simd_width_of[DType.uint64]())
    ](fill_with=lambda (_i: Int) -> UInt64: random_ui64(0, dim_4 - 1))
    var index_b = TileTensor(b_stack, row_major[index_len]())

    # The two 1D tensors are used as coordinates to dimensions 1 and 2 in the
    # dim_0 x dim_1 x dim_1 input tensor. Dimension 0 is preserved.
    # output[x, y, n] = input[x, y, Y[n], Z[n]],
    # where x = [0, input.dim(0)), y = [0, input.dim(1)),
    # n = [0, index_a.dim(0))

    # Reference output of shape dim_0 x index_len
    comptime ref_layout = row_major[dim_0, dim_1, index_len]()
    var ref_stack = Array[
        Scalar[input_type],
        align_up(ref_layout.product(), simd_width_of[input_type]()),
    ](fill=0)
    var ref_output = TileTensor(ref_stack, ref_layout)
    for i in range(dim_0):
        for j in range(dim_1):
            for k in range(index_len):
                ref_output[i, j, k] = input[i, j, index_a[k], index_b[k]]

    # Convert index_a, index_b (each of 1D size index_len) to a 2D index_len x 2
    # indices TileTensor.
    var indices_stack = Array[
        UInt64, align_up(index_len * 2, simd_width_of[DType.uint64]())
    ](fill=0)
    var indices = TileTensor(indices_stack, row_major[index_len, 2]())
    for i in range(index_len):
        indices[i, 0] = index_a[i]
        indices[i, 1] = index_b[i]

    var input_dyn = input.make_dynamic[.int64]()
    var indices_dyn = indices.make_dynamic[.int64]()
    var output_shape = index_tensor_shape[
        output_rank,
        input_type,
        .uint64,
        batch_dims,
    ](input_dyn, indices_dyn)

    var output_data_stack = Array[
        Scalar[input_type],
        align_up(dim_0 * dim_1 * index_len, simd_width_of[input_type]()),
    ](fill=0)
    comptime output_static_layout = row_major[dim_0, dim_1, index_len]()
    var output_data_buffer = TileTensor(output_data_stack, output_static_layout)
    var output_dyn = output_data_buffer.make_dynamic[.int64]()

    _index_tensor_impl[batch_dims](input_dyn, indices_dyn, output_dyn)

    for i in range(dim_0):
        for j in range(dim_1):
            for k in range(index_len):
                assert_equal(output_data_buffer[i, j, k], ref_output[i, j, k])


# TODO: It is like example 3 ONNX gather_nd.
# CHECK-LABEL: test_index_tensor_CLIPVIT
def test_index_tensor_CLIPVIT() raises:
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
    # Initialize with sequential data for test purposes.
    var input_stack = Array[
        Scalar[input_type],
        align_up(input_layout.product(), simd_width_of[input_type]()),
    ](fill_with=lambda (i: Int) -> Scalar[input_type]: Int32(i))
    var input = TileTensor(input_stack, input_layout)

    # We have two 2D tensors with 1 element each.

    # 1-element input tensor.
    # Initialize with [0,1]
    var a_stack = Array[
        UInt64, align_up(index_len, simd_width_of[DType.uint64]())
    ](fill_with=lambda (i: Int) -> UInt64: UInt64(i))
    var index_a = TileTensor(a_stack, row_major[index_len]())

    # 1-element input tensor.
    # Initialize with [1,0]
    var b_stack = Array[
        UInt64, align_up(index_len, simd_width_of[DType.uint64]())
    ](fill_with=lambda (i: Int) -> UInt64: 1 if i == 0 else 0)
    var index_b = TileTensor(b_stack, row_major[index_len]())

    # Reference output of shape dim_0 x dim_2

    comptime ref_layout = row_major[dim_0, dim_2]()
    var ref_stack = Array[
        Scalar[input_type],
        align_up(ref_layout.product(), simd_width_of[input_type]()),
    ](
        fill_with=lambda (idx: Int) -> Scalar[input_type]: (
            input[Int(index_a[0]), Int(index_a[1]), idx % dim_2] if idx // dim_2
            == 0 else input[Int(index_b[0]), Int(index_b[1]), idx % dim_2]
        )
    )
    var ref_output = TileTensor(ref_stack, ref_layout)

    # TODO:
    # See how I need to convert separate indices to
    # combined indices TileTensor
    # to be as input to gather_nd.
    # See if it works with 2D indices case.
    # See if it works with non-contiguous case.

    # Convert index_a, index_b (each of 1D size 2) to a
    # 2D indices_len x 2 indices TileTensor
    var indices_stack = Array[
        UInt64, align_up(index_len * 2, simd_width_of[DType.uint64]())
    ](fill=0)
    var indices = TileTensor(indices_stack, row_major[index_len, 2]())
    indices[0, 0] = index_a[0]
    indices[0, 1] = index_b[0]
    indices[1, 0] = index_a[1]
    indices[1, 1] = index_b[1]
    # TODO: Or index_a[0], index_a[1] and index_b[0], index_b[1]???

    var input_dyn = input.make_dynamic[.int64]()
    var indices_dyn = indices.make_dynamic[.int64]()
    var output_shape = gather_nd_shape[
        output_rank,
        input_type,
        .uint64,
        0,
    ](
        input_dyn,
        indices_dyn,
    )

    var output_data_stack = Array[
        Scalar[input_type],
        align_up(dim_0 * dim_2, simd_width_of[input_type]()),
    ](fill=0)
    comptime output_static_layout = row_major[dim_0, dim_2]()
    var output_data_buffer = TileTensor(output_data_stack, output_static_layout)
    var output_dyn = output_data_buffer.make_dynamic[.int64]()

    # TODO: index_tensor works too. For batch_dims = 0 only.
    gather_nd[batch_dims, target="cpu"](
        input_dyn,
        indices_dyn,
        output_dyn,
        DeviceContext(api="cpu"),
    )

    for i in range(dim_0):
        for j in range(dim_2):
            assert_equal(output_data_buffer[i, j], ref_output[i, j])


# CHECK-LABEL: test_index_tensor_llama2_mistral
def test_index_tensor_llama2_mistral() raises:
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
    # Initialize with sequential data for test purposes.
    var input_stack = Array[
        Scalar[input_type],
        align_up(input_layout.product(), simd_width_of[input_type]()),
    ](fill_with=lambda (i: Int) -> Scalar[input_type]: Int32(i))
    var input = TileTensor(input_stack, input_layout)

    # We have one 2D tensor with index_len elements each.

    # index_len-element input tensor.
    # Initialize with one.
    comptime index_layout = row_major[index_dim_0, index_dim_1]()
    var a_stack = Array[
        UInt64, align_up(index_layout.product(), simd_width_of[DType.uint64]())
    ](fill=1)
    var index_a = TileTensor(a_stack, index_layout)

    # This is effectively a gather operation.

    # Reference output of shape index_dim_0 x index_dim_1 x dim_1.
    comptime ref_layout = row_major[index_dim_0, index_dim_1, dim_1]()
    var ref_stack = Array[
        Scalar[input_type],
        align_up(ref_layout.product(), simd_width_of[input_type]()),
    ](fill=0)
    var ref_output = TileTensor(ref_stack, ref_layout)
    for i in range(index_dim_0):
        for j in range(index_dim_1):
            for k in range(dim_1):
                ref_output[i, j, k] = input[Int(index_a[i, j]), k]

    var input_dyn = input.make_dynamic[.int64]()
    var index_a_dyn = index_a.make_dynamic[.int64]()
    var output_shape = gather_shape[output_rank, input_type, index_type](
        input_dyn,
        index_a_dyn,
        0,
    )

    var output_data_stack = Array[
        Scalar[input_type],
        align_up(
            index_dim_0 * index_dim_1 * dim_1, simd_width_of[input_type]()
        ),
    ](fill=0)
    comptime output_static_layout = row_major[index_dim_0, index_dim_1, dim_1]()
    var output_data_buffer = TileTensor(output_data_stack, output_static_layout)
    var output_dyn = output_data_buffer.make_dynamic[.int64]()

    gather[axis=0](
        output_dyn,
        input_dyn,
        index_a_dyn,
        context=DeviceContext(api="cpu"),
    )

    for i in range(index_dim_0):
        for j in range(index_dim_1):
            for k in range(dim_1):
                assert_equal(output_data_buffer[i, j, k], ref_output[i, j, k])


def main() raises:
    test_index_tensor_DLRM()
    test_index_tensor_DLRM_batch()
    test_index_tensor_CLIPVIT()
    test_index_tensor_llama2_mistral()
