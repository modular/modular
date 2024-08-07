# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
import numpy as np
import pytest
from max.graph import DType, Graph, ops


def test_constant() -> None:
    with Graph("constants", input_types=()) as graph:
        const = np.array([0, 1, 2, 3, 4, 5]).astype(np.int64).reshape((2, 3))
        const = ops.constant(const)

        graph.output(const)
        graph._mlir_op.verify()

        assert "0, 1, 2, 3, 4, 5" in str(graph._mlir_op)


def test_constant_transpose() -> None:
    with Graph("constants", input_types=()) as graph:
        const = np.array([0, 1, 2, 3, 4, 5]).astype(np.int64).reshape((2, 3)).T
        const = ops.constant(const)

        graph.output(const)
        graph._mlir_op.verify()

        assert "0, 3, 1, 4, 2, 5" in str(graph._mlir_op)
