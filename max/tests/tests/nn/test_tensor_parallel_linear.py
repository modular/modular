# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tests for tensor parallel linear layers in max.nn."""

from __future__ import annotations

import pytest
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, Weight
from max.nn.linear import ColumnParallelLinear


def test_column_parallel_linear_empty_devices() -> None:
    """Tests ColumnParallelLinear with an empty devices list."""
    with pytest.raises(
        ValueError,
        match="ColumnParallelLinear requires a non-empty devices argument",
    ):
        ColumnParallelLinear(
            in_dim=16, out_dim=32, dtype=DType.float32, devices=[]
        )


def test_column_parallel_linear_valid() -> None:
    """Tests ColumnParallelLinear with valid arguments."""
    gpu0 = DeviceRef.GPU(id=0)
    gpu1 = DeviceRef.GPU(id=1)
    linear = ColumnParallelLinear(
        in_dim=16, out_dim=32, dtype=DType.float32, devices=[gpu0, gpu1]
    )
    with Graph(
        "column_parallel_linear",
        input_types=[
            TensorType(DType.float32, shape=(1, 16), device=gpu0),
            TensorType(DType.float32, shape=(1, 16), device=gpu1),
        ],
    ) as graph:
        x0, x1 = linear([inp.tensor for inp in graph.inputs])
        assert x0.device == gpu0
        assert x1.device == gpu1
        graph.output(x0, x1)


def test_column_parallel_linear_tied_weight_valid() -> None:
    """Tests ColumnParallelLinear with a valid tied_weight."""
    gpu0 = DeviceRef.GPU(id=0)
    gpu1 = DeviceRef.GPU(id=1)

    tied_weight = Weight(
        name="tied_weight",
        dtype=DType.float32,
        shape=(32, 16),
        device=gpu0,
    )
    linear = ColumnParallelLinear(
        in_dim=16,
        out_dim=32,
        dtype=DType.float32,
        devices=[gpu0, gpu1],
        tied_weight=tied_weight,
    )
    assert linear.weight is tied_weight

    with Graph(
        "column_parallel_linear_tied",
        input_types=[
            TensorType(DType.float32, shape=(1, 16), device=gpu0),
            TensorType(DType.float32, shape=(1, 16), device=gpu1),
        ],
    ) as graph:
        x0, x1 = linear([inp.tensor for inp in graph.inputs])
        assert x0.device == gpu0
        assert x1.device == gpu1
        graph.output(x0, x1)


def test_column_parallel_linear_tied_weight_with_float8() -> None:
    """Tests ColumnParallelLinear with tied_weight and float8_config."""
    gpu0 = DeviceRef.GPU(id=0)
    tied_weight = Weight(
        name="tied_weight",
        dtype=DType.float32,
        shape=(16, 32),
        device=gpu0,
    )
    with pytest.raises(
        ValueError,
        match=(
            "float8 and bias are both unsupported by ColumnParallelLinear "
            "currently"
        ),
    ):
        ColumnParallelLinear(
            in_dim=16,
            out_dim=32,
            dtype=DType.float32,
            devices=[gpu0],
            tied_weight=tied_weight,
            # Dummy float8_config.
            float8_config={},
        )


def test_column_parallel_linear_tied_weight_with_bias() -> None:
    """Tests ColumnParallelLinear with tied_weight and has_bias=True."""
    gpu0 = DeviceRef.GPU(id=0)
    tied_weight = Weight(
        name="tied_weight",
        dtype=DType.float32,
        shape=(16, 32),
        device=gpu0,
    )
    with pytest.raises(
        ValueError,
        match=(
            "float8 and bias are both unsupported by ColumnParallelLinear "
            "currently"
        ),
    ):
        ColumnParallelLinear(
            in_dim=16,
            out_dim=32,
            dtype=DType.float32,
            devices=[gpu0],
            tied_weight=tied_weight,
            has_bias=True,
        )
