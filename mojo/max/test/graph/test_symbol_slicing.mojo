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
# NOTE: Takes ~10 minutes to run with asan
# UNSUPPORTED: asan
# RUN: mojo %s

from max.graph import Dim, Graph, TensorType, Type, _testing, ops
from max.tensor import Tensor, TensorShape
from testing import *


fn test_vector_index_int() raises:
    var g = Graph(TensorType(DType.int64, 3))
    var arg = g[0]
    g.output(arg[1])

    var x = Tensor[DType.int64](TensorShape(3), 1, 2, 3)
    var expected = Tensor[DType.int64](TensorShape(), 2)

    var actual = _testing.execute_unary[DType.int64, DType.int64](g, x)
    _testing.assert_tensors_equal[DType.int64](expected, actual)


fn test_vector_index_sym() raises:
    var g = Graph(
        List[Type](TensorType(DType.int64, 3), TensorType(DType.int64)),
    )
    var arg = g[0]
    var idx = g[1]
    g.output(arg[idx])

    var x = Tensor[DType.int64](TensorShape(3), 1, 2, 3)
    var i = Tensor[DType.int64](TensorShape(), 1)
    var expected = Tensor[DType.int64](TensorShape(), 2)

    var actual = _testing.execute_binary[DType.int64](g, x, i)
    _testing.assert_tensors_equal[DType.int64](expected, actual)


fn test_vector_index_neg_one() raises:
    var g = Graph(TensorType(DType.int64, 1, Dim.dynamic()))
    var arg = g[0]
    g.output(ops.shape_of(arg)[-1])

    var x = Tensor[DType.int64](TensorShape(1, 3), 1, 2, 3)
    var expected = Tensor[DType.int64](TensorShape(), 3)

    var actual = _testing.execute_unary[DType.int64, DType.int64](g, x)
    _testing.assert_tensors_equal[DType.int64](expected, actual)


fn test_vector_slice_int() raises:
    var g = Graph(TensorType(DType.int64, 3))
    var arg = g[0]
    g.output(arg[1:3])

    var x = Tensor[DType.int64](TensorShape(3), 1, 2, 3)
    var expected = Tensor[DType.int64](TensorShape(2), 2, 3)

    var actual = _testing.execute_unary[DType.int64, DType.int64](g, x)
    _testing.assert_tensors_equal[DType.int64](expected, actual)


fn test_vector_slice_sym() raises:
    var g = Graph(
        List[Type](TensorType(DType.int64, 3), TensorType(DType.int64)),
    )
    var arg = g[0]
    var idx = g[1]
    g.output(arg[idx : idx + 2, out_dims = List[Dim](2)])

    var x = Tensor[DType.int64](TensorShape(3), 1, 2, 3)
    var i = Tensor[DType.int64](TensorShape(), 1)
    var expected = Tensor[DType.int64](TensorShape(2), 2, 3)

    var actual = _testing.execute_binary[DType.int64](g, x, i)
    _testing.assert_tensors_equal[DType.int64](expected, actual)


fn test_vector_slice_new_sym() raises:
    var g = Graph(
        List[Type](TensorType(DType.int64, 3), TensorType(DType.int64)),
    )
    var arg = g[0]
    var len = g[1]
    g.output(arg[:len, out_dims = List[Dim]("len")])

    var x = Tensor[DType.int64](TensorShape(3), 1, 2, 3)
    var i = Tensor[DType.int64](TensorShape(), 2)
    var expected = Tensor[DType.int64](TensorShape(2), 1, 2)

    var actual = _testing.execute_binary[DType.int64](g, x, i)
    _testing.assert_tensors_equal[DType.int64](expected, actual)


fn test_vector_slice_sym_incorrect_size() raises:
    var g = Graph(
        List[Type](TensorType(DType.int64, 3), TensorType(DType.int64)),
    )
    var arg = g[0]
    var idx = g[1]
    g.output(arg[idx : idx + 2, out_dims = List[Dim](5)])

    var x = Tensor[DType.int64](TensorShape(3), 1, 2, 3)
    var i = Tensor[DType.int64](TensorShape(), 1)

    with assert_raises():
        _ = _testing.execute_binary[DType.int64](g, x, i)


fn test_vector_slice_known_size_restricts() raises:
    var g = Graph(TensorType(DType.int64, Dim.dynamic()))
    var arg = g[0]

    var sliced = arg[7:12:2]

    assert_equal(sliced.tensor_type().dims[0], 3)


fn test_vector_slice_unknown_size_raises() raises:
    var g = Graph(TensorType(DType.int64, Dim.dynamic()))
    var arg = g[0]

    with assert_raises(contains="Please set out_dims"):
        _ = arg[7:-1]


fn test_vector_slice_with_out_dims() raises:
    var g = Graph(TensorType(DType.int64, Dim.dynamic()))
    var arg = g[0]

    var sliced = arg[7:-1, out_dims = List[Dim](8)]

    assert_equal(sliced.tensor_type().dims[0], 8)


fn test_matrix_index_int() raises:
    var g = Graph(TensorType(DType.int64, 2, 3))
    var arg = g[0]
    g.output(arg[1])

    var x = Tensor[DType.int64](TensorShape(2, 3), 1, 2, 3, 4, 5, 6)
    var expected = Tensor[DType.int64](TensorShape(3), 4, 5, 6)

    var actual = _testing.execute_unary[DType.int64, DType.int64](g, x)
    _testing.assert_tensors_equal[DType.int64](expected, actual)


fn test_matrix_index_sym() raises:
    var g = Graph(
        List[Type](TensorType(DType.int64, 2, 3), TensorType(DType.int64)),
    )
    var arg = g[0]
    var idx = g[1]
    g.output(arg[idx])

    var x = Tensor[DType.int64](TensorShape(2, 3), 1, 2, 3, 4, 5, 6)
    var i = Tensor[DType.int64](TensorShape(), 1)
    var expected = Tensor[DType.int64](TensorShape(3), 4, 5, 6)

    var actual = _testing.execute_binary[DType.int64](g, x, i)
    _testing.assert_tensors_equal[DType.int64](expected, actual)


fn test_matrix_slice_int() raises:
    var g = Graph(TensorType(DType.int64, 3, 3))
    var arg = g[0]
    g.output(arg[1:3])

    var x = Tensor[DType.int64](TensorShape(3, 3), 1, 2, 3, 4, 5, 6, 7, 8, 9)
    var expected = Tensor[DType.int64](TensorShape(2, 3), 4, 5, 6, 7, 8, 9)

    var actual = _testing.execute_unary[DType.int64, DType.int64](g, x)
    _testing.assert_tensors_equal[DType.int64](expected, actual)


fn test_matrix_slice_sym() raises:
    var g = Graph(
        List[Type](TensorType(DType.int64, 3, 3), TensorType(DType.int64)),
    )
    var arg = g[0]
    var idx = g[1]
    g.output(arg[idx : idx + 2, out_dims = List[Dim](2)])

    var x = Tensor[DType.int64](TensorShape(3, 3), 1, 2, 3, 4, 5, 6, 7, 8, 9)
    var i = Tensor[DType.int64](TensorShape(), 1)
    var expected = Tensor[DType.int64](TensorShape(2, 3), 4, 5, 6, 7, 8, 9)

    var actual = _testing.execute_binary[DType.int64](g, x, i)
    _testing.assert_tensors_equal[DType.int64](expected, actual)


fn test_matrix_slice_double() raises:
    var g = Graph(
        List[Type](TensorType(DType.int64, 3, 3), TensorType(DType.int64)),
    )
    var arg = g[0]
    var idx = g[1]
    g.output(arg[idx : idx + 2, idx - 1 : idx + 1, out_dims = List[Dim](2, 2)])

    var x = Tensor[DType.int64](TensorShape(3, 3), 1, 2, 3, 4, 5, 6, 7, 8, 9)
    var i = Tensor[DType.int64](TensorShape(), 1)
    var expected = Tensor[DType.int64](TensorShape(2, 2), 4, 5, 7, 8)

    var actual = _testing.execute_binary[DType.int64](g, x, i)
    _testing.assert_tensors_equal[DType.int64](expected, actual)


def test_slice_multiple_static_slices():
    x = Tensor[DType.int64](TensorShape(2, 2, 2), 1, 2, 3, 4, 5, 6, 7, 8)

    g = Graph(TensorType(DType.int64, 2, 2, 2))
    slice_out = g[0][:, :1]
    assert_true(TensorType(DType.int64, 2, 1, 2) == slice_out.tensor_type())
    g.output(slice_out)
    _testing.assert_tensors_equal(
        Tensor[DType.int64](TensorShape(2, 1, 2), 1, 2, 5, 6),
        _testing.execute_unary[outtype = DType.int64](g, x),
    )

    g = Graph(TensorType(DType.int64, 2, 2, 2))
    slice_out = g[0][:, :1, :2]
    assert_true(TensorType(DType.int64, 2, 1, 2) == slice_out.tensor_type())
    g.output(slice_out)
    _testing.assert_tensors_equal(
        Tensor[DType.int64](TensorShape(2, 1, 2), 1, 2, 5, 6),
        _testing.execute_unary[outtype = DType.int64](g, x),
    )

    g = Graph(TensorType(DType.int64, 2, 2, 2))
    slice_out = g[0][1:]
    assert_true(TensorType(DType.int64, 1, 2, 2) == slice_out.tensor_type())
    g.output(slice_out)
    _testing.assert_tensors_equal(
        Tensor[DType.int64](TensorShape(1, 2, 2), 5, 6, 7, 8),
        _testing.execute_unary[outtype = DType.int64](g, x),
    )

    g = Graph(TensorType(DType.int64, 2, 2, 2))
    slice_out = g[0][:, :, 2:]
    assert_true(TensorType(DType.int64, 2, 2, 0) == slice_out.tensor_type())
    g.output(slice_out)
    _testing.assert_tensors_equal(
        Tensor[DType.int64](TensorShape(2, 2, 0)),
        _testing.execute_unary[outtype = DType.int64](g, x),
    )

    g = Graph(TensorType(DType.int64, "batch", Dim.dynamic(), "z"))
    with assert_raises(contains="Please set out_dims"):
        _ = g[0][1:, :1, :]

    g = Graph(TensorType(DType.int64, 2, 2, 2))
    with assert_raises(contains="unsupported"):
        g.output(g[0][:, :, ::-1])

    g = Graph(TensorType(DType.int64, 2, 2, 2))
    with assert_raises(contains="got 4 slices, tensor only has rank 3"):
        g.output(g[0][1:, 1:, 1:, 1:])


def main():
    test_vector_index_int()
    test_vector_index_neg_one()
    test_vector_index_sym()
    test_vector_slice_int()
    test_vector_slice_sym()
    test_vector_slice_new_sym()
    test_vector_slice_sym_incorrect_size()
    test_vector_slice_known_size_restricts()
    test_vector_slice_unknown_size_raises()
    test_vector_slice_with_out_dims()

    test_matrix_index_int()
    test_matrix_index_sym()
    test_matrix_slice_int()
    test_matrix_slice_sym()
    test_matrix_slice_double()

    test_slice_multiple_static_slices()
