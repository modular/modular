# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo "%s"

from max.graph import Graph, TensorType, _testing, ops
from max.tensor import Tensor, TensorShape


fn test_squeeze() raises:
    var g = Graph(TensorType(DType.float32, "batch", 1, 3))
    g.output(ops.squeeze(g[0], 1))
    g.verify()

    var x = Tensor[DType.float32](TensorShape(1, 1, 3), 1, 2, 3)
    var expected = Tensor[DType.float32](TensorShape(1, 3), 1, 2, 3)

    var actual = _testing.execute_unary(g, x)
    _testing.assert_tensors_almost_equal(expected, actual)


def main():
    test_squeeze()
