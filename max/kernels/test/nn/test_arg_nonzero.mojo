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

from nn.arg_nonzero import arg_nonzero, arg_nonzero_shape
from testing import assert_equal
from layout import LayoutTensor, Layout, RuntimeLayout, UNKNOWN_VALUE

from utils import IndexList


# CHECK-LABEL: test_where_size
def test_where_size():
    print("== test_where_size")
    alias rank = 3
    alias values_shape = Layout.row_major(3, 2, 1)
    var values_stack = InlineArray[Float32, values_shape.size()](
        uninitialized=True
    )
    var values = LayoutTensor[DType.float32, values_shape](values_stack)

    values[0, 0, 0] = 1.0
    values[0, 1, 0] = 2.0
    values[1, 0, 0] = 0.0
    values[1, 1, 0] = 0.0
    values[2, 0, 0] = 0.0
    values[2, 1, 0] = -3.0

    alias layout_unknown = Layout.row_major(
        UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE
    )
    var output_shape = arg_nonzero_shape[DType.float32, True](
        LayoutTensor[DType.float32, layout_unknown,](
            values_stack,
            RuntimeLayout[layout_unknown].row_major(
                IndexList[3](3, 2, 1),
            ),
        )
    )

    assert_equal(output_shape[0], 3)
    assert_equal(output_shape[1], 3)


# CHECK-LABEL: test_where_size_bool
def test_where_size_bool():
    print("== test_where_size_bool")
    alias rank = 3
    alias values_shape = Layout.row_major(3, 2, 1)
    var values_stack = InlineArray[Scalar[DType.bool], values_shape.size()](
        uninitialized=True
    )
    var values = LayoutTensor[DType.bool, values_shape](values_stack)

    values[0, 0, 0] = True
    values[0, 1, 0] = True
    values[1, 0, 0] = False
    values[1, 1, 0] = False
    values[2, 0, 0] = Scalar[DType.bool](False)
    values[2, 1, 0] = Scalar[DType.bool](True)

    alias layout_unknown = Layout.row_major(
        UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE
    )
    var output_shape = arg_nonzero_shape[DType.bool, True](
        LayoutTensor[DType.bool, layout_unknown,](
            values_stack,
            RuntimeLayout[layout_unknown].row_major(
                IndexList[3](3, 2, 1),
            ),
        )
    )

    assert_equal(output_shape[0], 3)
    assert_equal(output_shape[1], 3)


# CHECK-LABEL: test_where
def test_where():
    print("== test_where")
    alias rank = 3
    alias values_shape = Layout.row_major(3, 2, 1)
    var values_stack = InlineArray[Float32, values_shape.size()](
        uninitialized=True
    )
    var values = LayoutTensor[DType.float32, values_shape](values_stack)

    values[0, 0, 0] = 1.0
    values[0, 1, 0] = 2.0
    values[1, 0, 0] = 0.0
    values[1, 1, 0] = 0.0
    values[2, 0, 0] = 0.0
    values[2, 1, 0] = -3.0

    var computed_stack = InlineArray[Scalar[DType.index], 9](uninitialized=True)
    var computed_outputs = LayoutTensor[
        DType.index,
        Layout.row_major(3, 3),
    ](computed_stack)

    var golden_stack = InlineArray[Scalar[DType.index], 9](uninitialized=True)
    var golden_outputs = LayoutTensor[
        DType.index,
        Layout.row_major(3, 3),
    ](golden_stack)

    golden_outputs[0, 0] = 0
    golden_outputs[0, 1] = 0
    golden_outputs[0, 2] = 0
    golden_outputs[1, 0] = 0
    golden_outputs[1, 1] = 1
    golden_outputs[1, 2] = 0
    golden_outputs[2, 0] = 2
    golden_outputs[2, 1] = 1
    golden_outputs[2, 2] = 0

    alias layout_unknown_3d = Layout.row_major(
        UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE
    )
    alias layout_unknown_2d = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    arg_nonzero(
        LayoutTensor[DType.float32, layout_unknown_3d](
            values_stack,
            RuntimeLayout[layout_unknown_3d].row_major(
                IndexList[3](3, 2, 1),
            ),
        ),
        LayoutTensor[DType.index, layout_unknown_2d](
            computed_stack,
            RuntimeLayout[layout_unknown_2d].row_major(
                IndexList[2](3, 3),
            ),
        ),
    )

    for i in range(3):
        for j in range(3):
            assert_equal(computed_outputs[i, j], golden_outputs[i, j])


# CHECK-LABEL: test_where_1d
def test_where_1d():
    print("== test_where_1d")
    alias num_elements = 12
    alias num_indices = 6

    var values_stack = InlineArray[Float32, num_elements](uninitialized=True)
    var values = LayoutTensor[DType.float32, Layout.row_major(num_elements)](
        values_stack
    )

    values[0] = 0.0
    values[1] = 1.0
    values[2] = 0.0
    values[3] = 1.0
    values[4] = 0.0
    values[5] = 1.0
    values[6] = 0.0
    values[7] = 1.0
    values[8] = 0.0
    values[9] = 1.0
    values[10] = 0.0
    values[11] = 1.0

    var computed_stack = InlineArray[Scalar[DType.index], num_indices](
        uninitialized=True
    )
    var computed_outputs = LayoutTensor[
        DType.index,
        Layout.row_major(num_indices, 1),
    ](computed_stack)

    var golden_stack = InlineArray[Scalar[DType.index], num_indices](
        uninitialized=True
    )
    var golden_outputs = LayoutTensor[
        DType.index,
        Layout.row_major(num_indices),
    ](golden_stack)

    golden_outputs[0] = 1
    golden_outputs[1] = 3
    golden_outputs[2] = 5
    golden_outputs[3] = 7
    golden_outputs[4] = 9
    golden_outputs[5] = 11

    alias layout_unknown_1d = Layout.row_major(UNKNOWN_VALUE)
    alias layout_unknown_2d = Layout.row_major(UNKNOWN_VALUE, 1)

    arg_nonzero(
        LayoutTensor[DType.float32, layout_unknown_1d](
            values_stack,
            RuntimeLayout[layout_unknown_1d].row_major(
                IndexList[1](num_elements),
            ),
        ),
        LayoutTensor[DType.index, layout_unknown_2d](
            computed_stack,
            RuntimeLayout[layout_unknown_2d].row_major(
                IndexList[2](num_indices, 1),
            ),
        ),
    )

    for i in range(num_indices):
        assert_equal(computed_outputs[i, 0], golden_outputs[i])


# CHECK-LABEL: test_where_bool
def test_where_bool():
    print("== test_where_bool")
    alias rank = 3
    alias values_shape = Layout.row_major(3, 2, 1)
    var values_stack = InlineArray[
        Scalar[DType.bool], Int(values_shape.size())
    ](uninitialized=True)
    var values = LayoutTensor[DType.bool, values_shape](values_stack)

    values[0, 0, 0] = True
    values[0, 1, 0] = True
    values[1, 0, 0] = False
    values[1, 1, 0] = False
    values[2, 0, 0] = False
    values[2, 1, 0] = True

    var computed_stack = InlineArray[Scalar[DType.index], 9](uninitialized=True)
    var computed_outputs = LayoutTensor[
        DType.index,
        Layout.row_major(3, 3),
    ](computed_stack)

    var golden_stack = InlineArray[Scalar[DType.index], 9](uninitialized=True)
    var golden_outputs = LayoutTensor[
        DType.index,
        Layout.row_major(3, 3),
    ](golden_stack)

    golden_outputs[0, 0] = 0
    golden_outputs[0, 1] = 0
    golden_outputs[0, 2] = 0
    golden_outputs[1, 0] = 0
    golden_outputs[1, 1] = 1
    golden_outputs[1, 2] = 0
    golden_outputs[2, 0] = 2
    golden_outputs[2, 1] = 1
    golden_outputs[2, 2] = 0

    alias layout_unknown_3d = Layout.row_major(
        UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE
    )
    alias layout_unknown_2d = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)

    arg_nonzero(
        LayoutTensor[DType.bool, layout_unknown_3d](
            values_stack,
            RuntimeLayout[layout_unknown_3d].row_major(
                IndexList[3](3, 2, 1),
            ),
        ),
        LayoutTensor[DType.index, layout_unknown_2d](
            computed_stack,
            RuntimeLayout[layout_unknown_2d].row_major(
                IndexList[2](3, 3),
            ),
        ),
    )

    for i in range(3):
        for j in range(3):
            assert_equal(computed_outputs[i, j], golden_outputs[i, j])


def main():
    test_where_size()
    test_where_size_bool()
    test_where()
    test_where_1d()
    test_where_bool()
