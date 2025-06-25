# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Tests for RMSNorm layer in max.nn."""

from __future__ import annotations

import pytest
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType
from max.nn.norm import DistributedRMSNorm, RMSNorm

from .norm_test_utils import (
    COMMON_NORM_TEST_SHAPES,
    assert_op_output_shape,
    assert_single_op,
    find_ops_in_graph,
)


def test_rms_norm_dimension_mismatch() -> None:
    """Tests RMSNorm raises ValueError when weight dimension doesn't match input's last dimension."""
    with pytest.raises(
        ValueError, match=r"weight dimension.*must match.*last dimension"
    ):
        Graph(
            "test",
            forward=RMSNorm(dim=32, dtype=DType.float32),
            input_types=[
                TensorType(DType.float32, (2, 10, 64), DeviceRef.CPU())
            ],
        )


def test_rms_norm_basic() -> None:
    """Tests basic RMSNorm functionality."""
    norm = RMSNorm(dim=64, dtype=DType.float32)
    g = Graph(
        "test",
        forward=norm,
        input_types=[TensorType(DType.float32, (2, 10, 64), DeviceRef.CPU())],
    )

    # Find the rms_norm custom op in the IR.
    rms_norm_op = assert_single_op(g, "mo.custom", "rms_norm")

    # Check the output type matches input shape.
    assert_op_output_shape(rms_norm_op, "[2, 10, 64]")


@pytest.mark.parametrize("shape, dim", COMMON_NORM_TEST_SHAPES)
def test_rms_norm_shapes(shape, dim) -> None:
    """Tests RMSNorm with various input shapes."""
    g = Graph(
        "test",
        forward=RMSNorm(dim=dim, dtype=DType.float32),
        input_types=[TensorType(DType.float32, shape, DeviceRef.CPU())],
    )

    # Verify the graph contains an rms_norm op.
    assert_single_op(g, "mo.custom", "rms_norm")


def test_rms_norm_device_transfer() -> None:
    """Tests RMSNorm handles device transfers correctly."""
    device = DeviceRef.GPU(0)
    g = Graph(
        "test",
        forward=RMSNorm(dim=64, dtype=DType.float32),
        input_types=[TensorType(DType.float32, (2, 64), device)],
    )

    # Find transfer ops that move weight to GPU.
    transfer_ops = find_ops_in_graph(g, "rmo.mo.transfer")
    # Should have at least one transfer for the weight.
    assert len(transfer_ops) >= 1


def test_distributed_rms_norm() -> None:
    """Tests DistributedRMSNorm with multiple devices."""
    devices = [DeviceRef.GPU(0), DeviceRef.GPU(1)]
    norm = DistributedRMSNorm(dim=64, dtype=DType.float32, devices=devices)
    # DistributedRMSNorm expects a single input that will be split across devices.
    g = Graph(
        "test",
        forward=norm,
        input_types=[TensorType(DType.float32, (2, 64), DeviceRef.GPU(0))],
    )

    # Should have multiple rms_norm ops, one per device.
    rms_norm_ops = find_ops_in_graph(g, "mo.custom", "rms_norm")
    assert len(rms_norm_ops) == 2


def test_rms_norm_tensor_parallel_scenario() -> None:
    """Tests the specific tensor parallel scenario that causes issues."""
    # Simulate InternVL-38B device 1 with 12 heads (1536 dims) trying to use
    # full gamma (3200 dims).
    with pytest.raises(
        ValueError, match=r"RMSNorm weight dimension \(3200\).*input.*\(1536\)"
    ):
        Graph(
            "test",
            forward=RMSNorm(dim=3200, dtype=DType.bfloat16),
            input_types=[
                TensorType(DType.bfloat16, (1, 1025, 1536), DeviceRef.GPU(1))
            ],
        )
