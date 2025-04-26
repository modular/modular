# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.graph Python bindings."""

import re

import numpy as np
import pytest
from max.dtype import DType
from max.graph import DeviceRef, Graph, Weight, ops


def test_add_weight() -> None:
    """Tests adding weights to the graph."""
    with Graph("graph_with_weights", input_types=()) as graph:
        w = Weight(
            "random_weight",
            dtype=DType.int64,
            shape=[5, 10],
            device=DeviceRef.CPU(),
        )

        w2 = Weight(
            "scalar_float",
            dtype=DType.float32,
            shape=[1],
            device=DeviceRef.CPU(),
        )

        graph.output(
            graph.add_weight(w),
            graph.add_weight(w2),
        )
        gen_mlir = str(graph._mlir_op).splitlines()
        # Most recent weight is at the top
        assert re.search(
            r'mo.constant.external.*name = "scalar_float".*!mo.tensor<\[1\], f32',
            gen_mlir[1],
        )
        assert re.search(
            r'mo.constant.external.*name = "random_weight".*!mo.tensor<\[5, 10\], si64',
            gen_mlir[2],
        )


def test_add_weights_with_sum() -> None:
    """Tests adding weights with a sum and ensuring that weights are added to the top of the graph."""
    with Graph("graph_with_weights", input_types=()) as graph:
        w1 = Weight(
            "random_weight1",
            dtype=DType.int64,
            shape=[5, 10],
            device=DeviceRef.CPU(),
        )

        w2 = Weight(
            "random_weight2",
            dtype=DType.int64,
            shape=[5, 10],
            device=DeviceRef.CPU(),
        )

        graph.output(
            graph.add_weight(w1) + graph.add_weight(w2),
        )
        gen_mlir = str(graph._mlir_op).splitlines()
        # Most recent weight is at the top
        assert re.search(
            r'mo.constant.external.*name = "random_weight2".*!mo.tensor<\[5, 10\], si64',
            gen_mlir[1],
        )
        assert re.search(
            r'mo.constant.external.*name = "random_weight1".*!mo.tensor<\[5, 10\], si64',
            gen_mlir[2],
        )


def test_add_same_weight() -> None:
    """Tests adding weights to the graph."""
    with Graph("graph_with_weights", input_types=()) as graph:
        w = Weight("w", dtype=DType.float32, shape=[], device=DeviceRef.CPU())
        value = graph.add_weight(w)

        # TODO(...): Make it return the exact same value
        # Adding the same Weight is fine, and should return a similar value
        value2 = graph.add_weight(w)
        assert value.type == value2.type

        # Test that adding a different Weight with the same name fails.
        w2 = Weight("w", dtype=DType.float32, shape=[], device=DeviceRef.CPU())

        with pytest.raises(ValueError, match="already exists"):
            graph.add_weight(w2)


def test_weight_is_value_like() -> None:
    with Graph("graph_with_weights", input_types=()) as graph:
        w = Weight("w", dtype=DType.float32, shape=[], device=DeviceRef.CPU())
        constant = ops.constant(np.array(1), DType.float32)
        graph.output(constant + w)
        gen_mlir = str(graph._mlir_op)
        assert re.search(
            r"mo.constant.external.*!mo.tensor<\[\], f32", gen_mlir
        )


def test_weight_outside_graph_error() -> None:
    w = Weight("w", dtype=DType.float32, shape=[], device=DeviceRef.CPU())
    with pytest.raises(ValueError, match="no parent graph"):
        _ = w * 5

    with pytest.raises(ValueError, match="no parent graph"):
        _ = ops.cast(w, DType.float64)
