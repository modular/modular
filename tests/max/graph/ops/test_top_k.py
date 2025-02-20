# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""ops.top_k tests."""

import numpy as np
import pytest
from max.dtype import DType
from max.graph import Graph, TensorType, ops


def test_top_k_output_tensor_types():
    input = np.array([0, 1, 2, 3, 4, 5]).astype(np.float32)
    k = 4

    expected_weight_tensor_type = TensorType(dtype=DType.float32, shape=[k])
    expected_idx_tensor_type = TensorType(dtype=DType.int64, shape=[k])

    with Graph(
        "top_k",
        input_types=[TensorType(dtype=DType.float32, shape=input.shape)],
    ) as graph:
        input_as_constant = ops.constant(input, DType.from_numpy(input.dtype))
        weight_tensor, idx_tensor = ops.top_k(input_as_constant, k)

        assert weight_tensor.type == expected_weight_tensor_type
        assert idx_tensor.type == expected_idx_tensor_type


def test_top_k_with_axis_greater_than_input_rank():
    input = np.array([0, 1, 2, 3, 4, 5]).astype(np.float32)
    k = 4

    axis_greater_than_input_rank = len(input.shape) + 1

    with Graph(
        "top_k",
        input_types=[TensorType(dtype=DType.float32, shape=input.shape)],
    ) as graph:
        input = np.array([0, 1, 2, 3, 4, 5]).astype(np.float32)
        input_as_constant = ops.constant(input, DType.from_numpy(input.dtype))
        with pytest.raises(IndexError):
            ops.top_k(input_as_constant, k, axis=axis_greater_than_input_rank)


def test_top_k_with_k_greater_than_input_length():
    input = np.array([0, 1, 2, 3, 4, 5]).astype(np.float32)
    k = 4

    k_greater_than_input_length = len(input) + 1

    with Graph(
        "top_k",
        input_types=[TensorType(dtype=DType.float32, shape=input.shape)],
    ) as graph:
        input_as_constant = ops.constant(input, DType.from_numpy(input.dtype))
        with pytest.raises(IndexError):
            ops.top_k(input_as_constant, k_greater_than_input_length)
