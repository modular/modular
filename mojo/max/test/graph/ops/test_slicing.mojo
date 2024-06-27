# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo "%s"

from max.graph import _testing, Graph, TensorType
from max.graph.ops.slicing import select
from max.tensor import Tensor, TensorShape


fn test_select() raises:
    var g = Graph(TensorType(DType.bool, 1, 2, 2))

    var x = g.constant(
        Tensor[DType.float32](
            TensorShape(1, 2, 2),
            -1.5,
            2.5,
            -3.5,
            4.5,
        )
    )

    var y = g.constant(
        Tensor[DType.float32](
            TensorShape(1, 2, 2),
            -4.5,
            3.5,
            -2.5,
            1.5,
        )
    )

    g.output(
        select(
            g[0],
            x,
            y,
        )
    )
    g.verify()

    # fmt: off
    var input = Tensor[DType.bool](
        TensorShape(1, 2, 2),
        True, False,
        False, True,
    )
    var expected = Tensor[DType.float32](
        TensorShape(1, 2, 2),
        -1.5, 3.5,
        -2.5, 4.5,
    )
    # fmt: on

    var actual = _testing.execute_unary(g, input)
    _testing.assert_tensors_almost_equal(expected, actual, 1e-4)


def main():
    test_select()
