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
from nn.repeat_interleave import _collapse_dims_around_axis, repeat_interleave

from utils.index import IndexList


fn test_collapse_dims_around_axis() raises:
    # CHECK-LABEL: test_collapse_dims_around_axis
    print("test_collapse_dims_around_axis")

    # CHECK: (1, 100, 1)
    print(_collapse_dims_around_axis(IndexList[1](100), 0))

    # CHECK: (1, 17, 23)
    print(_collapse_dims_around_axis(IndexList[2](17, 23), 0))

    # CHECK: (17, 23, 1)
    print(_collapse_dims_around_axis(IndexList[2](17, 23), 1))

    # CHECK: (1, 2, 16)
    print(_collapse_dims_around_axis(IndexList[5](2, 2, 2, 2, 2), 0))

    # CHECK: (4, 2, 4)
    print(_collapse_dims_around_axis(IndexList[5](2, 2, 2, 2, 2), 2))

    # CHECK: (16, 2, 1)
    print(_collapse_dims_around_axis(IndexList[5](2, 2, 2, 2, 2), 4))


fn test_repeat_interleave_1d() raises:
    # CHECK-LABEL: test_repeat_interleave_1d
    print("test_repeat_interleave_1d")

    alias rank = 1
    alias type = DType.float32

    var input_stack = InlineArray[Scalar[type], 4](uninitialized=True)
    var input = LayoutTensor[
        type,
        Layout.row_major(4),
    ](input_stack)

    input[0] = 0
    input[1] = 1
    input[2] = 2
    input[3] = 3

    # rank_repeats is always 1
    alias rank_repeats = 1
    # type_repeats is always DType.int64
    alias type_repeats = DType.int64

    var repeats_stack = InlineArray[Scalar[type_repeats], 4](uninitialized=True)
    var repeats = LayoutTensor[
        type_repeats,
        Layout.row_major(4),
    ](repeats_stack)

    repeats[0] = 1
    repeats[1] = 2
    repeats[2] = 3
    repeats[3] = 4

    var output_stack = InlineArray[Scalar[type], 10](uninitialized=True)
    var output = LayoutTensor[
        mut=True,
        type,
        Layout.row_major(10),
    ](output_stack)

    repeat_interleave(input, repeats, 0, output)

    # CHECK: 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0
    print()
    for i in range(10):
        print(output[i], ", ", sep="", end="")
    print()


fn test_repeat_interleave_1d_broadcast_repeats() raises:
    # CHECK-LABEL: test_repeat_interleave_1d_broadcast_repeats
    print("test_repeat_interleave_1d_broadcast_repeats")

    alias rank = 1
    alias type = DType.float32

    var input_stack = InlineArray[Scalar[type], 4](uninitialized=True)
    var input = LayoutTensor[
        type,
        Layout.row_major(4),
    ](input_stack)

    input[0] = 0
    input[1] = 1
    input[2] = 2
    input[3] = 3

    # rank_repeats is always 1
    alias rank_repeats = 1
    # type_repeats is always DType.int64
    alias type_repeats = DType.int64

    var repeats_stack = InlineArray[Scalar[type_repeats], 1](uninitialized=True)
    var repeats = LayoutTensor[
        type_repeats,
        Layout.row_major(1),
    ](repeats_stack)

    repeats[0] = 2

    var output_stack = InlineArray[Scalar[type], 8](uninitialized=True)
    var output = LayoutTensor[
        type,
        Layout.row_major(8),
    ](output_stack)

    repeat_interleave(input, repeats, 0, output)

    # CHECK: 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0,
    print()
    for i in range(8):
        print(output[i], ", ", sep="", end="")
    print()


fn test_repeat_interleave_2d_axis_0() raises:
    # CHECK-LABEL: test_repeat_interleave_2d_axis_0
    print("test_repeat_interleave_2d_axis_0")

    alias rank = 2
    alias type = DType.float32

    var input_stack = InlineArray[Scalar[type], 4](uninitialized=True)
    var input = LayoutTensor[
        type,
        Layout.row_major(2, 2),
    ](input_stack)

    input[0, 0] = 0
    input[0, 1] = 1
    input[1, 0] = 2
    input[1, 1] = 3

    # rank_repeats is always 1
    alias rank_repeats = 1
    # type_repeats is always DType.int64
    alias type_repeats = DType.int64

    var repeats_stack = InlineArray[Scalar[type_repeats], 4](uninitialized=True)
    var repeats = LayoutTensor[
        type_repeats,
        Layout.row_major(2),
    ](repeats_stack)

    repeats[0] = 2
    repeats[1] = 3

    # Result is 2x5
    var output_stack = InlineArray[Scalar[type], 10](uninitialized=True)
    var output = LayoutTensor[
        type,
        Layout.row_major(5, 2),
    ](output_stack)

    repeat_interleave(input, repeats, 0, output)

    # CHECK: 0.0, 1.0,
    # CHECK: 0.0, 1.0,
    # CHECK: 2.0, 3.0,
    # CHECK: 2.0, 3.0,
    # CHECK: 2.0, 3.0,
    print()
    for i in range(5):
        for j in range(2):
            print(output[i, j], ", ", sep="", end="")
        print()
    print()


fn test_repeat_interleave_2d_axis_1() raises:
    # CHECK-LABEL: test_repeat_interleave_2d_axis_1
    print("test_repeat_interleave_2d_axis_1")

    alias rank = 2
    alias type = DType.float32

    var input_stack = InlineArray[Scalar[type], 4](uninitialized=True)
    var input = LayoutTensor[
        type,
        Layout.row_major(2, 2),
    ](input_stack)

    input[0, 0] = 0
    input[0, 1] = 1
    input[1, 0] = 2
    input[1, 1] = 3

    # rank_repeats is always 1
    alias rank_repeats = 1
    # type_repeats is always DType.int64
    alias type_repeats = DType.int64

    var repeats_stack = InlineArray[Scalar[type_repeats], 2](uninitialized=True)
    var repeats = LayoutTensor[
        type_repeats,
        Layout.row_major(2),
    ](repeats_stack)

    repeats[0] = 2
    repeats[1] = 3

    # Result is 2x5
    var output_stack = InlineArray[Scalar[type], 10](uninitialized=True)
    var output = LayoutTensor[
        type,
        Layout.row_major(2, 5),
    ](output_stack)

    repeat_interleave(input, repeats, 1, output)

    # CHECK: 0.0, 0.0, 1.0, 1.0, 1.0
    # CHECK: 2.0, 2.0, 3.0, 3.0, 3.0
    print()
    for i in range(2):
        for j in range(5):
            print(output[i, j], ", ", sep="", end="")
        print()
    print()


fn test_repeat_interleave_3d() raises:
    # CHECK-LABEL: test_repeat_interleave_3d
    print("test_repeat_interleave_3d")

    alias rank = 3
    alias type = DType.float32

    var input_stack = InlineArray[Scalar[type], 8](uninitialized=True)
    var input = LayoutTensor[
        type,
        Layout.row_major(2, 2, 2),
    ](input_stack)

    input[0, 0, 0] = 0
    input[0, 0, 1] = 1
    input[0, 1, 0] = 2
    input[0, 1, 1] = 3

    input[1, 0, 0] = 4
    input[1, 0, 1] = 5
    input[1, 1, 0] = 6
    input[1, 1, 1] = 7

    # rank_repeats is always 1
    alias rank_repeats = 1
    # type_repeats is always DType.int64
    alias type_repeats = DType.int64

    var repeats_stack = InlineArray[Scalar[type_repeats], 2](uninitialized=True)
    var repeats = LayoutTensor[
        type_repeats,
        Layout.row_major(2),
    ](repeats_stack)

    repeats[0] = 2
    repeats[1] = 3

    # Result is 2x5
    var output_stack = InlineArray[Scalar[type], 20](uninitialized=True)
    var output = LayoutTensor[
        type,
        Layout.row_major(2, 5, 2),
    ](output_stack)

    repeat_interleave(input, repeats, 1, output)

    # CHECK: 0.0, 1.0,
    # CHECK: 0.0, 1.0,
    # CHECK: 2.0, 3.0,
    # CHECK: 2.0, 3.0,
    # CHECK: 2.0, 3.0,
    # CHECK: =====
    # CHECK: 4.0, 5.0,
    # CHECK: 4.0, 5.0,
    # CHECK: 6.0, 7.0,
    # CHECK: 6.0, 7.0,
    # CHECK: 6.0, 7.0,
    # CHECK: =====

    print()
    for i in range(2):
        for j in range(5):
            for k in range(2):
                print(output[i, j, k], ", ", sep="", end="")
            print()
        print("=====")
    print()


fn main() raises:
    test_collapse_dims_around_axis()
    test_repeat_interleave_1d()
    test_repeat_interleave_1d_broadcast_repeats()
    test_repeat_interleave_2d_axis_0()
    test_repeat_interleave_2d_axis_1()
    test_repeat_interleave_3d()
