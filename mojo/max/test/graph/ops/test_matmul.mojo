# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo "%s"

from testing import assert_raises

from max.graph import ops, _testing, Type, Graph, TensorType
from max.tensor import Tensor, TensorShape


fn test_matmul() raises:
    var g = Graph(
        List[Type](
            TensorType(DType.float32, 2, 3), TensorType(DType.float32, 3, 2)
        ),
    )
    g.output(g[0] @ g[1])
    g.verify()

    var x = Tensor[DType.float32](
        TensorShape(2, 3), -2.0, -1.0, 0.0, 1.0, 2.0, 3.0
    )
    var y = Tensor[DType.float32](
        TensorShape(3, 2), 1.0, 1.0, 0.0, 1.0, 0.0, 1.0
    )
    var expected = Tensor[DType.float32](TensorShape(2, 2), -2, -3, 1, 6)

    var actual = _testing.execute_binary(g, x, y)
    _testing.assert_tensors_almost_equal(expected, actual, 1e-4)


fn test_matmul_type_promotion() raises:
    var g = Graph(
        List[Type](
            TensorType(DType.int8, 2, 3), TensorType(DType.float32, 3, 2)
        ),
    )
    g.output(g[0] @ g[1])
    g.verify()

    var x = Tensor[DType.int8](TensorShape(2, 3), -2, -1, 0, 1, 2, 3)
    var y = Tensor[DType.float32](
        TensorShape(3, 2), 1.0, 1.0, 0.0, 1.0, 0.0, 1.0
    )
    var expected = Tensor[DType.float32](TensorShape(2, 2), -2, -3, 1, 6)

    var actual = _testing.execute_binary[outtype = DType.float32](g, x, y)
    _testing.assert_tensors_almost_equal(expected, actual, 1e-4)


fn test_batch_matmul() raises:
    var g = Graph(
        List[Type](
            TensorType(DType.float32, 2, 1, 2, 3),
            TensorType(DType.float32, 1, 3, 2),
        ),
    )
    g.output(g[0] @ g[1])
    g.verify()

    var x = Tensor[DType.float32](
        TensorShape(2, 1, 2, 3),
        -2.0,
        -1.0,
        0.0,
        1.0,
        2.0,
        3.0,
        -2.0,
        -1.0,
        0.0,
        1.0,
        2.0,
        3.0,
    )
    var y = Tensor[DType.float32](
        TensorShape(1, 3, 2), 1.0, 1.0, 0.0, 1.0, 0.0, 1.0
    )
    var expected = Tensor[DType.float32](
        TensorShape(2, 1, 2, 2),
        -2,
        -3,
        1,
        6,
        -2,
        -3,
        1,
        6,
    )

    var actual = _testing.execute_binary(g, x, y)
    _testing.assert_tensors_almost_equal(expected, actual, 1e-4)


fn test_matmul_with_named_dims() raises:
    var g = Graph(
        List[Type](
            TensorType(DType.float32, "A", "B"),
            TensorType(DType.float32, "B", "C"),
        ),
    )
    g.output(g[0] @ g[1])
    g.verify()

    var x = Tensor[DType.float32](
        TensorShape(2, 3), -2.0, -1.0, 0.0, 1.0, 2.0, 3.0
    )
    var y = Tensor[DType.float32](
        TensorShape(3, 2), 1.0, 1.0, 0.0, 1.0, 0.0, 1.0
    )
    var expected = Tensor[DType.float32](TensorShape(2, 2), -2, -3, 1, 6)

    var actual = _testing.execute_binary(g, x, y)
    _testing.assert_tensors_almost_equal(expected, actual, 1e-4)


fn test_vector_lhs() raises:
    var g = Graph(
        List[Type](
            TensorType(DType.float32, 3), TensorType(DType.float32, 3, 2)
        ),
    )
    g.output(g[0] @ g[1])
    g.verify()

    var x = Tensor[DType.float32](TensorShape(3), -2.0, -1.0, 0.0)
    var y = Tensor[DType.float32](
        TensorShape(3, 2), 1.0, 1.0, 0.0, 1.0, 0.0, 1.0
    )
    var expected = Tensor[DType.float32](TensorShape(2), -2, -3)

    var actual = _testing.execute_binary(g, x, y)
    _testing.assert_tensors_almost_equal(expected, actual, 1e-4)


fn test_vector_rhs() raises:
    var g = Graph(
        List[Type](
            TensorType(DType.float32, 2, 3), TensorType(DType.float32, 3)
        ),
    )
    g.output(g[0] @ g[1])
    g.verify()

    var x = Tensor[DType.float32](
        TensorShape(2, 3), -2.0, -1.0, 0.0, 1.0, 2.0, 3.0
    )
    var y = Tensor[DType.float32](TensorShape(3), 1.0, 1.0, 0.0)
    var expected = Tensor[DType.float32](TensorShape(2), -3, 3)

    var actual = _testing.execute_binary(g, x, y)
    _testing.assert_tensors_almost_equal(expected, actual, 1e-4)


fn test_vector_both() raises:
    var g = Graph(
        List[Type](TensorType(DType.float32, 3), TensorType(DType.float32, 3)),
    )
    g.output(g[0] @ g[1])
    g.verify()

    var x = Tensor[DType.float32](TensorShape(3), -2.0, -1.0, 0.0)
    var y = Tensor[DType.float32](TensorShape(3, 2), 1.0, 1.0, 0.0)
    var expected = Tensor[DType.float32](TensorShape(), -3)

    var actual = _testing.execute_binary(g, x, y)
    _testing.assert_tensors_almost_equal(expected, actual, 1e-4)


def test_matmul_dimension_mismatch():
    var graph = Graph(TensorType(DType.float32, 2, "x"))

    var matmul_constant_value = Tensor[DType.float32](2, 6)
    matmul_constant_value._to_buffer().fill(0.15)
    var matmul_constant = graph.constant(matmul_constant_value)

    graph.output(ops.matmul(graph[0], matmul_constant))
    graph.verify()


def test_matmul_broadcast_fail():
    var graph = Graph(TensorType(DType.float32, "batch", 2, 6))

    var matmul_constant_value = Tensor[DType.float32](11, 6, 1)
    matmul_constant_value._to_buffer().fill(0.15)
    var matmul_constant = graph.constant(matmul_constant_value)

    graph.output(ops.matmul(graph[0], matmul_constant))
    graph.verify()


def test_matmul_scalar_unsupported():
    var graph = Graph(TensorType(DType.float32, 2, 6))

    var matmul_constant = graph.scalar[DType.float32](0.15)

    with assert_raises(contains="Scalar inputs are not supported"):
        _ = ops.matmul(graph[0], matmul_constant)


def main():
    test_matmul()
    test_matmul_type_promotion()
    test_batch_matmul()
    test_matmul_with_named_dims()
    test_vector_lhs()
    test_vector_rhs()
    test_vector_both()
    test_matmul_dimension_mismatch()
    test_matmul_broadcast_fail()
    test_matmul_scalar_unsupported()
