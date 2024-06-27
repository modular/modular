# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo "%s"

from testing import assert_raises

from max.graph import _testing, Type, Graph, TensorType, Dim
from max.graph.ops.elementwise import equal, abs, round
from max.tensor import Tensor, TensorShape


fn test_basic() raises:
    var g = Graph(
        List[Type](
            TensorType(DType.float32, 1, 2, 3), TensorType(DType.float32, 2, 3)
        ),
    )
    g.output(g[0] + g[1])
    g.verify()

    var x = Tensor[DType.float32](
        TensorShape(1, 2, 3), -2.0, -1.0, 0.0, 1.0, 2.0, 3.0
    )
    var y = Tensor[DType.float32](
        TensorShape(2, 3), 1.0, 1.0, 0.0, 1.0, 0.0, 1.0
    )
    var expected = Tensor[DType.float32](
        TensorShape(1, 2, 3), -1.0, 0.0, 0.0, 2.0, 2.0, 4.0
    )

    var actual = _testing.execute_binary(g, x, y)
    _testing.assert_tensors_almost_equal(expected, actual, 1e-4)


fn test_literal() raises:
    var g = Graph(TensorType(DType.float32, 1, 2, 3))
    g.output(g[0] - 1.0)
    g.verify()

    var x = Tensor[DType.float32](
        TensorShape(1, 2, 3), -2.0, -1.0, 0.0, 1.0, 2.0, 3.0
    )
    var expected = Tensor[DType.float32](
        TensorShape(1, 2, 3), -3.0, -2.0, -1.0, 0.0, 1.0, 2.0
    )

    var actual = _testing.execute_unary(g, x)
    _testing.assert_tensors_almost_equal(expected, actual, 1e-4)


fn test_implicit_promotion() raises:
    var g = Graph(
        List[Type](TensorType(DType.int16, 3), TensorType(DType.float32, 3)),
    )
    g.output(g[0] * g[1])
    g.verify()

    var x = Tensor[DType.int16](TensorShape(3), -2, -1, 0)
    var y = Tensor[DType.float32](TensorShape(3), 1.0, 1.0, 3.0)
    var expected = Tensor[DType.float32](TensorShape(3), -2.0, -1.0, 0.0)

    var actual = _testing.execute_binary[
        DType.int16, DType.float32, DType.float32
    ](g, x, y)
    _testing.assert_tensors_almost_equal(expected, actual, 1e-4)


fn test_comparison_results_in_bool() raises:
    var g = Graph(
        List[Type](TensorType(DType.int16, 3), TensorType(DType.float32, 2, 3)),
    )
    g.output(equal(g[0], g[1]))
    g.verify()

    var x = Tensor[DType.int16](TensorShape(3), -2, -1, 0)
    var y = Tensor[DType.float32](TensorShape(2, 3), 1.0, -1.0, 3.0, 7, 2, 0)
    var expected = Tensor[DType.bool](
        TensorShape(2, 3), False, True, False, False, False, True
    )

    var actual = _testing.execute_binary[outtype = DType.bool](g, x, y)
    _testing.assert_tensors_equal(expected, actual)


fn test_bad_implicit_promotion() raises:
    var g = Graph(
        List[Type](TensorType(DType.int16, 3), TensorType(DType.float16, 3)),
    )
    with assert_raises(
        contains=(
            "Unsafe cast from si16 to f16. Insert an explicit cast op if this"
            " conversion is wanted."
        )
    ):
        _ = g[0] + g[1]


fn test_bad_broadcast_dynamic_one() raises:
    var g = Graph(
        List[Type](
            TensorType(DType.float32, Dim.dynamic()),
            TensorType(DType.float32, Dim.dynamic()),
        ),
    )
    g.output(g[0] + g[1])
    g.verify()

    var x = Tensor[DType.float32](TensorShape(3), -2, -1, 0)
    var y = Tensor[DType.float32](TensorShape(1), 1.0)
    var expected = Tensor[DType.float32](TensorShape(3), -1.0, 0.0, 1.0)

    with assert_raises(
        contains="RMO rebind shape check for broadcasting failed at: "
    ):
        _ = _testing.execute_binary(g, x, y)


fn test_matching_dynamic() raises:
    var g = Graph(
        List[Type](
            TensorType(DType.float32, Dim.dynamic()),
            TensorType(DType.float32, Dim.dynamic()),
        ),
    )
    g.output(g[0] + g[1])
    g.verify()

    var x = Tensor[DType.float32](TensorShape(3), -2, -1, 0)
    var y = Tensor[DType.float32](TensorShape(3), 1.0, 1.0, 1.0)
    var expected = Tensor[DType.float32](TensorShape(3), -1.0, 0.0, 1.0)

    var actual = _testing.execute_binary(g, x, y)
    _testing.assert_tensors_almost_equal(expected, actual, 1e-4)


fn test_unary() raises:
    var g = Graph(
        List[Type](
            TensorType(DType.int32, 1, 2, 3), TensorType(DType.int32, 1)
        ),
    )
    g.output(abs(g[0]))
    g.verify()

    var x = Tensor[DType.int32](TensorShape(1, 2, 3), -2, -1, 0, 1, 2, 3)
    var ignored = Tensor[DType.int32](TensorShape(1), 1)
    var expected = Tensor[DType.int32](TensorShape(1, 2, 3), 2, 1, 0, 1, 2, 3)

    var actual = _testing.execute_binary(g, x, ignored)
    _testing.assert_tensors_equal(expected, actual)


fn test_unary_float() raises:
    var g = Graph(
        List[Type](
            TensorType(DType.float32, 1, 2, 3), TensorType(DType.float32, 1)
        ),
    )
    g.output(round(g[0]))
    g.verify()

    var x = Tensor[DType.float32](
        TensorShape(1, 2, 3), -2.2, -1.7, 0.2, 1.2, 2.9, 3.6
    )
    var ignored = Tensor[DType.float32](TensorShape(1), -1.0)
    var expected = Tensor[DType.float32](
        TensorShape(1, 2, 3), -2.0, -2.0, 0.0, 1.0, 3.0, 4.0
    )

    var actual = _testing.execute_binary(g, x, ignored)
    _testing.assert_tensors_almost_equal(expected, actual, 1e-4)


fn test_unary_bad_dtype() raises:
    var g = Graph(
        List[Type](
            TensorType(DType.int32, 1, 2, 3), TensorType(DType.int32, 1)
        ),
    )
    with assert_raises(
        contains=(
            "mo.round only supports floating point inputs. Please explicitly"
            " cast to your desired float type first."
        )
    ):
        _ = round(g[0])


def main():
    test_basic()
    test_literal()
    test_implicit_promotion()
    test_matching_dynamic()
    test_comparison_results_in_bool()
    test_unary()
    test_unary_float()
    test_bad_implicit_promotion()
    test_bad_broadcast_dynamic_one()
    test_unary_bad_dtype()
