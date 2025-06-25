# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Tests for LayerNorm layer in max.nn."""

from __future__ import annotations

import pytest
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType
from max.nn.norm import LayerNorm

from .norm_test_utils import (
    COMMON_NORM_TEST_SHAPES,
    assert_op_output_shape,
    assert_single_op,
)


def test_layer_norm_dimension_mismatch():
    """Tests LayerNorm raises ValueError for dimension mismatches."""
    with pytest.raises(
        ValueError,
        match=r"Gamma size 32 does not match dimension of reduction 64",
    ):
        Graph(
            "test",
            forward=LayerNorm(
                dims=32, device=DeviceRef.CPU(), dtype=DType.float32
            ),
            input_types=[
                TensorType(DType.float32, (2, 10, 64), DeviceRef.CPU())
            ],
        )


def test_layer_norm_basic():
    """Tests basic LayerNorm functionality."""
    g = Graph(
        "test",
        forward=LayerNorm(dims=64, device=DeviceRef.CPU(), dtype=DType.float32),
        input_types=[TensorType(DType.float32, (2, 10, 64), DeviceRef.CPU())],
    )

    # Find the layer_norm op in the IR.
    layer_norm_op = assert_single_op(g, "mo.layer_norm")

    # Check the output type.
    assert_op_output_shape(layer_norm_op, "[2, 10, 64]")


def test_layer_norm_no_bias():
    """Tests LayerNorm without bias."""
    norm = LayerNorm(
        dims=64, device=DeviceRef.CPU(), dtype=DType.float32, use_bias=False
    )
    assert norm.bias is None

    g = Graph(
        "test",
        forward=norm,
        input_types=[TensorType(DType.float32, (2, 64), DeviceRef.CPU())],
    )

    # Check that the graph still works without bias.
    assert_single_op(g, "mo.layer_norm")


@pytest.mark.parametrize("shape, dim", COMMON_NORM_TEST_SHAPES)
def test_layer_norm_shapes(shape, dim):
    """Tests LayerNorm with various input shapes."""
    g = Graph(
        "test",
        forward=LayerNorm(
            dims=dim, device=DeviceRef.CPU(), dtype=DType.float32
        ),
        input_types=[TensorType(DType.float32, shape, DeviceRef.CPU())],
    )

    # Verify the graph contains a layer_norm op.
    assert_single_op(g, "mo.layer_norm")
