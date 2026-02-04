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

from itertools import product
from layout._layout import row_major
from layout._tile_tensor import TileTensor
from nn.arg_nonzero import arg_nonzero, arg_nonzero_shape
from testing import assert_equal

from utils import IndexList


# CHECK-LABEL: test_where_size
def test_where_size():
    print("== test_where_size")
    comptime rank = 3
    comptime values_shape = row_major[3, 2, 1]()
    var values_stack = InlineArray[Float32, values_shape.product()](
        uninitialized=True
    )
    var values = TileTensor(values_stack, values_shape)

    values[0, 0, 0] = 1.0
    values[0, 1, 0] = 2.0
    values[1, 0, 0] = 0.0
    values[1, 1, 0] = 0.0
    values[2, 0, 0] = 0.0
    values[2, 1, 0] = -3.0

    var output_shape = arg_nonzero_shape[DType.float32, True](
        values.make_dynamic[DType.int64]()
    )

    assert_equal(output_shape[0], 3)
    assert_equal(output_shape[1], 3)


# CHECK-LABEL: test_where_size_bool
def test_where_size_bool():
    print("== test_where_size_bool")
    comptime rank = 3
    comptime values_shape = row_major[3, 2, 1]()
    var values_stack = InlineArray[Scalar[DType.bool], values_shape.product()](
        uninitialized=True
    )
    var values = TileTensor(values_stack, values_shape)

    values[0, 0, 0] = True
    values[0, 1, 0] = True
    values[1, 0, 0] = False
    values[1, 1, 0] = False
    values[2, 0, 0] = Scalar[DType.bool](False)
    values[2, 1, 0] = Scalar[DType.bool](True)

    var output_shape = arg_nonzero_shape[DType.bool, True](
        values.make_dynamic[DType.int64]()
    )

    assert_equal(output_shape[0], 3)
    assert_equal(output_shape[1], 3)


# CHECK-LABEL: test_where
def test_where():
    print("== test_where")
    comptime rank = 3
    comptime values_shape = row_major[3, 2, 1]()
    var values_stack = InlineArray[Float32, values_shape.product()](
        uninitialized=True
    )
    var values = TileTensor(values_stack, values_shape)

    values[0, 0, 0] = 1.0
    values[0, 1, 0] = 2.0
    values[1, 0, 0] = 0.0
    values[1, 1, 0] = 0.0
    values[2, 0, 0] = 0.0
    values[2, 1, 0] = -3.0

    var computed_stack = InlineArray[Scalar[DType.int], 9](uninitialized=True)
    var computed_outputs = TileTensor[DType.int,](
        computed_stack,
        row_major[3, 3](),
    )

    var golden_stack = InlineArray[Scalar[DType.int], 9](uninitialized=True)
    var golden_outputs = TileTensor[DType.int,](
        golden_stack,
        row_major[3, 3](),
    )

    golden_outputs[0, 0] = 0
    golden_outputs[0, 1] = 0
    golden_outputs[0, 2] = 0
    golden_outputs[1, 0] = 0
    golden_outputs[1, 1] = 1
    golden_outputs[1, 2] = 0
    golden_outputs[2, 0] = 2
    golden_outputs[2, 1] = 1
    golden_outputs[2, 2] = 0

    arg_nonzero(
        values.make_dynamic[DType.int64](),
        computed_outputs.make_dynamic[DType.int64](),
    )

    for i, j in product(range(3), range(3)):
        assert_equal(computed_outputs[i, j], golden_outputs[i, j])


# CHECK-LABEL: test_where_1d
def test_where_1d():
    print("== test_where_1d")
    comptime num_elements = 12
    comptime num_indices = 6

    var values_stack = InlineArray[Float32, num_elements](uninitialized=True)
    var values = TileTensor(values_stack, row_major[num_elements]())

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

    var computed_stack = InlineArray[Scalar[DType.int], num_indices](
        uninitialized=True
    )
    var computed_outputs = TileTensor(
        computed_stack, row_major[num_indices, 1]()
    )

    var golden_stack = InlineArray[Scalar[DType.int], num_indices](
        uninitialized=True
    )
    var golden_outputs = TileTensor(golden_stack, row_major[num_indices]())

    golden_outputs[0] = 1
    golden_outputs[1] = 3
    golden_outputs[2] = 5
    golden_outputs[3] = 7
    golden_outputs[4] = 9
    golden_outputs[5] = 11

    arg_nonzero(
        values.make_dynamic[DType.int64](),
        computed_outputs.make_dynamic[DType.int64](),
    )

    for i in range(num_indices):
        assert_equal(computed_outputs[i, 0], golden_outputs[i])


# CHECK-LABEL: test_where_bool
def test_where_bool():
    print("== test_where_bool")
    comptime rank = 3
    comptime values_shape = row_major[3, 2, 1]()
    var values_stack = InlineArray[
        Scalar[DType.bool], Int(values_shape.product())
    ](uninitialized=True)
    var values = TileTensor(values_stack, values_shape)

    values[0, 0, 0] = True
    values[0, 1, 0] = True
    values[1, 0, 0] = False
    values[1, 1, 0] = False
    values[2, 0, 0] = False
    values[2, 1, 0] = True

    var computed_stack = InlineArray[Scalar[DType.int], 9](uninitialized=True)
    var computed_outputs = TileTensor(computed_stack, row_major[3, 3]())

    var golden_stack = InlineArray[Scalar[DType.int], 9](uninitialized=True)
    var golden_outputs = TileTensor(golden_stack, row_major[3, 3]())

    golden_outputs[0, 0] = 0
    golden_outputs[0, 1] = 0
    golden_outputs[0, 2] = 0
    golden_outputs[1, 0] = 0
    golden_outputs[1, 1] = 1
    golden_outputs[1, 2] = 0
    golden_outputs[2, 0] = 2
    golden_outputs[2, 1] = 1
    golden_outputs[2, 2] = 0

    arg_nonzero(
        values.make_dynamic[DType.int64](),
        computed_outputs.make_dynamic[DType.int64](),
    )

    for i, j in product(range(3), range(3)):
        assert_equal(computed_outputs[i, j], golden_outputs[i, j])


def main():
    test_where_size()
    test_where_size_bool()
    test_where()
    test_where_1d()
    test_where_bool()
