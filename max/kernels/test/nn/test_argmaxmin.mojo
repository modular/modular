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

from algorithm import sum
from algorithm.reduction import _reduce_generator, max, min
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from builtin.math import max as _max
from builtin.math import min as _min
from memory import UnsafePointer
from nn.argmaxmin import argmax, argmin

from utils.index import Index, IndexList, StaticTuple


# CHECK-LABEL: test_argn
fn test_argn() raises:
    print("== test_argn")

    alias size = 93

    var vector_stack = InlineArray[Int32, size](uninitialized=True)
    var vector = LayoutTensor[DType.int32, Layout.row_major(size)](vector_stack)
    var output_stack = InlineArray[Scalar[DType.index], 1](uninitialized=True)
    var output = LayoutTensor[DType.index, Layout.row_major(1)](output_stack)

    for i in range(size):
        vector[i] = i

    argmax(
        LayoutTensor[DType.int32, Layout.row_major(UNKNOWN_VALUE)](
            vector_stack,
            RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
                IndexList[1](size)
            ),
        ),
        0,
        LayoutTensor[DType.index, Layout.row_major(UNKNOWN_VALUE)](
            output_stack,
            RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
                IndexList[1](1)
            ),
        ),
    )

    # CHECK: argmax = 92
    print("argmax = ", output[0])

    argmin(
        LayoutTensor[DType.int32, Layout.row_major(UNKNOWN_VALUE),](
            vector_stack,
            RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
                IndexList[1](size)
            ),
        ),
        0,
        LayoutTensor[DType.index, Layout.row_major(UNKNOWN_VALUE)](
            output_stack,
            RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
                IndexList[1](1)
            ),
        ),
    )

    # CHECK: argmin = 0
    print("argmin = ", output[0])


# CHECK-LABEL: test_argn_2
fn test_argn_2() raises:
    print("== test_argn_2")

    alias batch_size = 4
    alias size = 91

    var vector_stack = InlineArray[Float32, batch_size * size](
        uninitialized=True
    )
    var vector = LayoutTensor[
        DType.float32, Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    ](
        vector_stack,
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)].row_major(
            IndexList[2](batch_size, size)
        ),
    )
    var output_stack = InlineArray[Scalar[DType.index], batch_size](
        uninitialized=True
    )
    var output = LayoutTensor[
        DType.index, Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    ](
        output_stack,
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)].row_major(
            IndexList[2](batch_size, 1)
        ),
    )

    for i in range(batch_size):
        for j in range(size):
            vector[i, j] = j

    argmax(vector, 1, output)

    # CHECK: argmax = 90
    # CHECK: argmax = 90
    # CHECK: argmax = 90
    # CHECK: argmax = 90
    for i in range(batch_size):
        print("argmax = ", output[i, 0])

    argmin(vector, 1, output)

    # CHECK: argmin = 0
    # CHECK: argmin = 0
    # CHECK: argmin = 0
    # CHECK: argmin = 0
    for i in range(batch_size):
        print("argmin = ", output[i, 0])


# CHECK-LABEL: test_argn_2_test_2
fn test_argn_2_test_2() raises:
    print("== test_argn_2_test_2")

    alias batch_size = 2
    alias size = 3

    var vector_stack = InlineArray[Float32, batch_size * size](
        uninitialized=True
    )
    var vector = LayoutTensor[
        DType.float32,
        Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE),
    ](
        vector_stack,
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)].row_major(
            IndexList[2](batch_size, size)
        ),
    )
    var output_stack = InlineArray[Scalar[DType.index], batch_size](
        uninitialized=True
    )
    var output = LayoutTensor[
        DType.index,
        Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE),
    ](
        output_stack,
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)].row_major(
            IndexList[2](batch_size, 1)
        ),
    )

    for i in range(batch_size):
        for j in range(size):
            vector[i, j] = i * size + j
            if i % 2:
                vector[i, j] *= -1

    argmax(vector, 1, output)

    # CHECK: argmax = 2
    # CHECK: argmax = 0
    for i in range(batch_size):
        print("argmax = ", output[i, 0])

    argmin(vector, 1, output)

    # CHECK: argmin = 0
    # CHECK: argmin = 2
    for i in range(batch_size):
        print("argmin = ", output[i, 0])


# CHECK-LABEL: test_argn_2_neg_axis
fn test_argn_2_neg_axis() raises:
    print("== test_argn_2_neg_axis")

    alias batch_size = 2
    alias size = 3

    var vector_stack = InlineArray[Float32, batch_size * size](
        uninitialized=True
    )
    var vector = LayoutTensor[
        DType.float32,
        Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE),
    ](
        vector_stack,
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)].row_major(
            IndexList[2](batch_size, size)
        ),
    )
    var output_stack = InlineArray[Scalar[DType.index], batch_size](
        uninitialized=True
    )
    var output = LayoutTensor[
        DType.index,
        Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE),
    ](
        output_stack,
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)].row_major(
            IndexList[2](batch_size, 1)
        ),
    )

    for i in range(batch_size):
        for j in range(size):
            vector[i, j] = i * size + j
            if i % 2:
                vector[i, j] *= -1

    argmax(vector, -1, output)

    # CHECK: argmax = 2
    # CHECK: argmax = 0
    for i in range(batch_size):
        print("argmax = ", output[i, 0])

    argmin(vector, -1, output)

    # CHECK: argmin = 0
    # CHECK: argmin = 2
    for i in range(batch_size):
        print("argmin = ", output[i, 0])


# CHECK-LABEL: test_argn_test_zeros
fn test_argn_test_zeros() raises:
    print("== test_argn_test_zeros")

    alias batch_size = 1
    alias size = 16

    var vector_stack = InlineArray[Float32, batch_size * size](
        uninitialized=True
    )
    var vector = LayoutTensor[
        DType.float32,
        Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE),
    ](
        vector_stack,
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)].row_major(
            IndexList[2](batch_size, size)
        ),
    )
    var output_stack = InlineArray[Scalar[DType.index], batch_size](
        uninitialized=True
    )
    var output = LayoutTensor[
        DType.index,
        Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE),
    ](
        output_stack,
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)].row_major(
            IndexList[2](batch_size, 1)
        ),
    )

    for i in range(batch_size):
        for j in range(size):
            vector[i, j] = 0

    argmax(vector, 1, output)

    # CHECK: argmax = 0
    for i in range(batch_size):
        print("argmax = ", output[i, 0])

    argmin(vector, 1, output)

    # CHECK: argmin = 0
    for i in range(batch_size):
        print("argmin = ", output[i, 0])


# CHECK-LABEL: test_argn_test_identity
fn test_argn_test_identity() raises:
    print("== test_argn_test_identity")

    alias batch_size = 3
    alias size = 5

    var vector_stack = InlineArray[Int64, batch_size * size](uninitialized=True)
    var vector = LayoutTensor[
        DType.int64,
        Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE),
    ](
        vector_stack,
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)].row_major(
            IndexList[2](batch_size, size)
        ),
    )
    var output_stack = InlineArray[Scalar[DType.index], batch_size](
        uninitialized=True
    )
    var output = LayoutTensor[
        DType.index,
        Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE),
    ](
        output_stack,
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)].row_major(
            IndexList[2](batch_size, 1)
        ),
    )

    for i in range(batch_size):
        for j in range(size):
            vector[i, j] = 0

    vector[1, 4] = 1
    vector[2, 3] = 1
    vector[2, 4] = 1

    argmax(vector, 1, output)

    # CHECK: argmax = 0
    print("argmax = ", output[0, 0])
    # CHECK: argmax = 4
    print("argmax = ", output[1, 0])
    # CHECK: argmax = 3
    print("argmax = ", output[2, 0])

    argmin(vector, 1, output)

    # CHECK: argmin = 0
    # CHECK: argmin = 0
    # CHECK: argmin = 0
    for i in range(batch_size):
        print("argmin = ", output[i, 0])


# CHECK-LABEL: test_argn_3d_identity
fn test_argn_3d_identity() raises:
    print("== test_argn_3d_identity")

    alias batch_size = 2
    alias seq_len = 2
    alias hidden_dim = 5

    alias vector_shape = Layout.row_major(batch_size, seq_len, hidden_dim)
    var vector_stack = InlineArray[Int64, vector_shape.size()](
        uninitialized=True
    )
    var vector = LayoutTensor[
        DType.int64,
        Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE),
    ](
        vector_stack,
        RuntimeLayout[
            Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE)
        ].row_major(IndexList[3](batch_size, seq_len, hidden_dim)),
    ).fill(
        0
    )

    var output_stack = InlineArray[Scalar[DType.index], batch_size * seq_len](
        uninitialized=True
    )
    var output = LayoutTensor[
        DType.index,
        Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE),
    ](
        output_stack,
        RuntimeLayout[
            Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE)
        ].row_major(IndexList[3](batch_size, seq_len, 1)),
    ).fill(
        0
    )

    vector[0, 1, 4] = 1
    vector[1, 0, 1] = 1
    vector[1, 0, 2] = 1
    vector[1, 1, 3] = 1

    argmax(vector, 2, output)

    # CHECK: argmax = 0
    print("argmax = ", output[0, 0, 0])
    # CHECK: argmax = 4
    print("argmax = ", output[0, 1, 0])
    # CHECK: argmax = 1
    print("argmax = ", output[1, 0, 0])
    # CHECK: argmax = 3
    print("argmax = ", output[1, 1, 0])

    argmin(vector, 2, output)

    # CHECK: argmin = 0
    # CHECK: argmin = 0
    # CHECK: argmin = 0
    # CHECK: argmin = 0
    for i in range(batch_size):
        for j in range(seq_len):
            print("argmin = ", output[i, j, 0])


fn test_argn_less_than_simd() raises:
    print("== test_argn_less_than_simd")

    alias batch_size = 2
    alias hidden_dim = 3  # assumes simd_width of 4

    var vector_stack = InlineArray[Int64, batch_size * hidden_dim](
        uninitialized=True
    )
    var vector = LayoutTensor[
        DType.int64, Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    ](
        vector_stack,
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)].row_major(
            IndexList[2](batch_size, hidden_dim)
        ),
    ).fill(
        0
    )

    var output_stack = InlineArray[Scalar[DType.index], batch_size](
        uninitialized=True
    )
    var output = LayoutTensor[
        DType.index, Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    ](
        output_stack,
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)].row_major(
            IndexList[2](batch_size, 1)
        ),
    ).fill(
        0
    )

    vector[0, 0] = 0
    vector[0, 1] = 1
    vector[0, 2] = 2
    vector[1, 0] = 5
    vector[1, 1] = 4
    vector[1, 2] = 3

    argmax(vector, 1, output)

    # CHECK: argmax = 2
    print("argmax = ", output[0, 0])
    # CHECK: argmax = 0
    print("argmax = ", output[1, 0])

    argmin(vector, 1, output)

    # CHECK: argmin = 0
    print("argmin = ", output[0, 0])
    # CHECK: argmin = 2
    print("argmin = ", output[1, 0])


# CHECK-LABEL: test_argn_simd_edge_case
fn test_argn_simd_index_order() raises:
    print("== test_argn_simd_edge_case")

    # Checks the case where the maximal value is found in two simd_chunks, where
    # the index of the maximal value in the second simd_chunk is earlier than in the first.
    # ex:
    #   simd_width = 4
    #   [0, 0, 1, 0, 0, 1, 0, 0, 0]
    #   <--------->  <-------->  <>
    #          ^        ^
    alias size = 17

    var vector_stack = InlineArray[Int32, size](uninitialized=True)
    var vector = LayoutTensor[DType.int32, Layout.row_major(UNKNOWN_VALUE)](
        vector_stack,
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
            IndexList[1](size)
        ),
    ).fill(0)
    var output_stack = InlineArray[Scalar[DType.index], 1](uninitialized=True)
    var output = LayoutTensor[DType.index, Layout.row_major(UNKNOWN_VALUE)](
        output_stack,
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
            IndexList[1](1)
        ),
    )

    vector[5] = 1
    vector[4] = -1
    vector[8] = -1
    vector[9] = 1

    argmax(vector, 0, output)

    # CHECK: argmax = 5
    print("argmax = ", output[0])

    argmin(vector, 0, output)

    # CHECK: argmin = 4
    print("argmin = ", output[0])


# CHECK-LABEL: test_argn_parallelize
fn test_argn_parallelize() raises:
    print("== test_argn_parallelize")

    # Checks argn's performance when the size of the NDBuffer exceeds the threshold to enable parallelism
    alias batch_size = 8
    alias hidden_dim = 16384

    var input_ptr = UnsafePointer[Float32].alloc(batch_size * hidden_dim)
    var input = LayoutTensor[
        DType.float32, Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    ](
        input_ptr,
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)].row_major(
            IndexList[2](batch_size, hidden_dim)
        ),
    ).fill(
        0
    )

    var output_stack = InlineArray[Scalar[DType.index], batch_size](
        uninitialized=True
    )
    var output = LayoutTensor[
        DType.index, Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    ](
        output_stack,
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)].row_major(
            IndexList[2](batch_size, 1)
        ),
    )

    input[0, 10] = 100
    input[0, 100] = -100
    input[1, 20] = -100
    input[1, 200] = 100
    input[2, 30] = 100
    input[2, 300] = -100
    input[3, 40] = -100
    input[3, 400] = 100
    input[4, 10] = 100
    input[4, 100] = -100
    input[5, 20] = 100
    input[5, 200] = -100
    input[6, 30] = 100
    input[6, 300] = -100
    input[7, 40] = -100
    input[7, 400] = 100

    argmax(input, 1, output)

    # CHECK: argmax = 10
    print("argmax = ", output[0, 0])
    # CHECK: argmax = 200
    print("argmax = ", output[1, 0])
    # CHECK: argmax = 30
    print("argmax = ", output[2, 0])
    # CHECK: argmax = 400
    print("argmax = ", output[3, 0])
    # CHECK: argmax = 10
    print("argmax = ", output[4, 0])
    # CHECK: argmax = 20
    print("argmax = ", output[5, 0])
    # CHECK: argmax = 30
    print("argmax = ", output[6, 0])
    # CHECK: argmax = 400
    print("argmax = ", output[7, 0])

    argmin(input, 1, output)

    # CHECK: argmin = 100
    print("argmin = ", output[0, 0])
    # CHECK: argmin = 20
    print("argmin = ", output[1, 0])
    # CHECK: argmin = 300
    print("argmin = ", output[2, 0])
    # CHECK: argmin = 40
    print("argmin = ", output[3, 0])
    # CHECK: argmin = 100
    print("argmin = ", output[4, 0])
    # CHECK: argmin = 200
    print("argmin = ", output[5, 0])
    # CHECK: argmin = 300
    print("argmin = ", output[6, 0])
    # CHECK: argmin = 40
    print("argmin = ", output[7, 0])

    input_ptr.free()


fn main() raises:
    test_argn()
    test_argn_2()
    test_argn_2_test_2()
    test_argn_2_neg_axis()
    test_argn_test_zeros()
    test_argn_test_identity()
    test_argn_3d_identity()
    test_argn_less_than_simd()
    test_argn_simd_index_order()
    test_argn_parallelize()
