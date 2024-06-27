# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo "%s"

from max.graph import _testing, Graph, TensorType, Type, Dim
from testing import assert_raises, assert_true
from max.tensor import Tensor, TensorShape


fn test_reshape() raises:
    var g = Graph(
        List[Type](
            TensorType(DType.int32, 3, 4),
            TensorType(DType.int32, 2, Dim("x"), 2),
        ),
    )

    # [3, 4] -> [6, 2]
    var reshape1 = g[0].reshape(6, 2)
    # TODO(GRA-552): Enable comparison of shape directly instead of printing full mlir
    assert_true(str(reshape1).endswith("!mo.tensor<[6, 2], si32>"))

    # [3, 4] -> [2, 3, -1]
    var reshape2 = g[0].reshape(2, 3, -1)
    # TODO(GRA-552): Enable comparison of shape directly instead of printing full mlir
    assert_true(str(reshape2).endswith("!mo.tensor<[2, 3, 2], si32>"))

    # [2, x, 2] -> [-1, x]
    var reshape3 = g[1].reshape(Dim(-1), Dim("x"))
    # TODO(GRA-552): Enable comparison of shape directly instead of printing full mlir
    assert_true(str(reshape3).endswith("!mo.tensor<[4, x], si32>"))


fn test_reshape_error() raises:
    var g = Graph(
        List[Type](
            TensorType(DType.int32, 3, 4),
        ),
    )

    # [3, 4] -> [6, 1]
    # NOTE error because `12 != 6`, user made a mistake here.
    with assert_raises(
        contains="[reshape] input and output number of elements must match"
    ):
        _ = g[0].reshape(6, 1)

    # [3, 4] -> [12, 0]
    # NOTE error because `12 != 0`, user made a mistake here.
    with assert_raises(
        contains="[reshape] input and output number of elements must match"
    ):
        _ = g[0].reshape(12, 0)

    # [3, 4] -> [-1, -1]
    # NOTE error because multiple `-1`, user made a mistake here.
    with assert_raises(
        contains="[reshape] multiple -1 detected in target shape"
    ):
        _ = g[0].reshape(-1, -1)

    # [3, 4] -> [0, -1]
    # NOTE error because `-1` can not be used with `0`, user made a mistake here.
    with assert_raises(
        contains=(
            "[reshape] cannot infer dimension when a specified output dimension"
            " is 0"
        )
    ):
        _ = g[0].reshape(0, -1)


fn test_reshape_runtime_zero() raises:
    var g = Graph(
        List[Type](
            TensorType(DType.int32, 2, Dim("x"), 2),
        ),
    )

    # [2, x, 2] -> [-1, x]
    var reshape = g[0].reshape(Dim(-1), Dim("x"))

    g.output(reshape)
    g.verify()

    var x = Tensor[DType.int16](TensorShape(2, 0, 2))

    with assert_raises(
        contains=(
            "[reshape] cannot infer dimension when a specified output dimension"
            " is 0"
        )
    ):
        _ = _testing.execute_unary(g, x)


def main():
    test_reshape()
    test_reshape_error()
    test_reshape_runtime_zero()
    # TODO(GRA-578): Once we have dim expression in, test a runtime negative number.
