# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Tests for tensor parallel linear layers in max.nn."""

from __future__ import annotations

from typing import Any, cast

import pytest
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, Type, Weight
from max.nn import Signals
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
    devices = [gpu0, gpu1]
    signals = Signals(devices)
    linear = ColumnParallelLinear(
        in_dim=16, out_dim=32, dtype=DType.float32, devices=devices
    )

    with Graph(
        "column_parallel_linear",
        # https://github.com/python/mypy/issues/19413
        input_types=cast(
            list[Type[Any]],
            [
                TensorType(DType.float32, shape=(1, 16), device=gpu0),
                TensorType(DType.float32, shape=(1, 16), device=gpu1),
            ]
            + signals.input_types(),
        ),
    ) as graph:
        num_devices = len(devices)
        x0, x1 = linear(
            [inp.tensor for inp in graph.inputs[:num_devices]],
            [inp.buffer for inp in graph.inputs[num_devices:]],
        )
        assert x0.device == gpu0
        assert x1.device == gpu1
        graph.output(x0, x1)


def test_column_parallel_linear_tied_weight_valid() -> None:
    """Tests ColumnParallelLinear with a valid tied_weight."""
    gpu0 = DeviceRef.GPU(id=0)
    gpu1 = DeviceRef.GPU(id=1)
    devices = [gpu0, gpu1]
    signals = Signals(devices)

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
        devices=devices,
        tied_weight=tied_weight,
    )
    assert linear.weight is tied_weight

    with Graph(
        "column_parallel_linear_tied",
        # https://github.com/python/mypy/issues/19413
        input_types=cast(
            list[Type[Any]],
            [
                TensorType(DType.float32, shape=(1, 16), device=gpu0),
                TensorType(DType.float32, shape=(1, 16), device=gpu1),
            ]
            + signals.input_types(),
        ),
    ) as graph:
        num_devices = len(devices)
        x0, x1 = linear(
            [inp.tensor for inp in graph.inputs[:num_devices]],
            [inp.buffer for inp in graph.inputs[num_devices:]],
        )
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
