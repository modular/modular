# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo "%s"

from max.graph import _testing, Graph, TensorType
from max.graph.ops.linalg import layer_norm, range_fill, tile
from max.tensor import Tensor, TensorShape


fn test_layer_norm() raises:
    var g = Graph(TensorType(DType.float32, 4, 4))

    var gamma = g.constant(Tensor[DType.float32](TensorShape(4), 0.1))

    var beta = g.constant(Tensor[DType.float32](TensorShape(4), 0.2))

    g.output(
        layer_norm(
            g[0],
            gamma=gamma,
            beta=beta,
            epsilon=1e-05,
        )
    )
    g.verify()

    # fmt: off
    var x = Tensor[DType.float32](
        TensorShape(4, 4),
        -1.0, 0.0, 1.0, -2.0,
        -1.0, 0.0, 1.0, -2.0,
        -1.0, 0.0, 1.0, -2.0,
        -1.0, 0.0, 1.0, -2.0,
    )
    var expected = Tensor[DType.float32](
        TensorShape(4, 4),
        0.155278, 0.244721, 0.334163, 0.065836,
        0.155278, 0.244721, 0.334163, 0.065836,
        0.155278, 0.244721, 0.334163, 0.065836,
        0.155278, 0.244721, 0.334163, 0.065836,
    )
    # fmt: on

    var actual = _testing.execute_unary(g, x)
    _testing.assert_tensors_almost_equal(expected, actual, 1e-4)


fn test_range_fill() raises:
    var g = Graph(TensorType(DType.int32, 1, 4, 4, 1))

    var limit = g.scalar(12, DType.int32)

    var start = g.scalar(0, DType.int32)

    var step = g.scalar(1, DType.int32)

    g.output(
        range_fill(
            start,
            limit,
            step,
        )
    )
    g.verify()

    var x = Tensor[DType.int32](TensorShape(1, 4, 4, 1), 0)

    # fmt: off
    var expected = Tensor[DType.int32](
        TensorShape(12),
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    )
    # fmt: on

    var actual = _testing.execute_unary[DType.int32, DType.int32](g, x)
    _testing.assert_tensors_equal[DType.int32](expected, actual)


fn test_tile() raises:
    var g = Graph(TensorType(DType.float32, 2, 3))

    g.output(
        tile(
            g[0],
            List[Int64](3, 2),
        )
    )
    g.verify()

    # fmt: off
    var x = Tensor[DType.float32](
        TensorShape(2, 3),
        1.0, 0.5, 3.0,
        -1.0, 2.0, 4.0,
    )
    var expected = Tensor[DType.float32](
        TensorShape(6, 6),
        1.0, 0.5, 3.0, 1.0, 0.5, 3.0,
        -1.0, 2.0, 4.0, -1.0, 2.0, 4.0,
        1.0, 0.5, 3.0, 1.0, 0.5, 3.0,
        -1.0, 2.0, 4.0, -1.0, 2.0, 4.0,
        1.0, 0.5, 3.0, 1.0, 0.5, 3.0,
        -1.0, 2.0, 4.0, -1.0, 2.0, 4.0,
    )
    # fmt: on

    var actual = _testing.execute_unary(g, x)
    _testing.assert_tensors_almost_equal(expected, actual, 1e-4)


def main():
    test_range_fill()
    test_layer_norm()
    test_tile()
