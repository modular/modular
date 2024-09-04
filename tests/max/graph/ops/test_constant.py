# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
import numpy as np
from hypothesis import assume, given
from max import _graph
from max.dtype import DType
from max.graph import Graph, ops


def test_constant() -> None:
    with Graph("constants", input_types=()) as graph:
        const = np.array([0, 1, 2, 3, 4, 5]).astype(np.int64).reshape((2, 3))
        const = ops.constant(const, DType.from_numpy(const.dtype))

        graph.output(const)

        assert "0, 1, 2, 3, 4, 5" in str(graph._mlir_op)


def test_constant_transpose() -> None:
    with Graph("constants", input_types=()) as graph:
        const = np.array([0, 1, 2, 3, 4, 5]).astype(np.int64).reshape((2, 3)).T
        const = ops.constant(const, DType.from_numpy(const.dtype))

        graph.output(const)

        assert "0, 3, 1, 4, 2, 5" in str(graph._mlir_op)


@given(dtype=...)
def test_scalar(dtype: DType) -> None:
    # Can represent an integer value
    assume(dtype != DType.bool)
    # Not supported by numpy
    assume(dtype != DType.bfloat16)
    with Graph("scalar", input_types=()) as graph:
        const = 7.2
        const = ops.scalar(const, dtype)

        graph.output(const)

        mlir_dtype = _graph.dtype_type(graph._context, dtype._mlir)
        assert "7" in str(graph._mlir_op)
        assert str(mlir_dtype) in str(graph._mlir_op)
