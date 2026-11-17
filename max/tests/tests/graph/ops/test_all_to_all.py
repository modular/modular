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
"""Test the ops.all_to_all operation (graph construction only)."""

from __future__ import annotations

import pytest
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.legacy import Signals


def _build(
    shapes: list[list[int | str]],
    split_axis: int,
    concat_axis: int,
    group_size: int | None = None,
    devices: list[DeviceRef] | None = None,
) -> list[list[str]]:
    """Stages an all-to-all and returns each output's dims as strings."""
    devices = devices or [DeviceRef.GPU(id=i) for i in range(len(shapes))]
    signals = Signals(devices)
    with Graph(
        "all_to_all",
        input_types=[
            *[
                TensorType(DType.bfloat16, shape, device=device)
                for shape, device in zip(shapes, devices, strict=True)
            ],
            *signals.input_types(),
        ],
    ) as graph:
        n = len(shapes)
        outputs = ops.all_to_all(
            [v.tensor for v in graph.inputs[:n]],
            [v.buffer for v in graph.inputs[n:]],
            split_axis=split_axis,
            concat_axis=concat_axis,
            group_size=group_size,
        )
        graph.output(*outputs)
        for output, device in zip(outputs, devices, strict=True):
            assert output.device == device
            assert output.dtype == DType.bfloat16
        assert "mo.distributed.all_to_all" in str(graph)
        return [[str(d) for d in output.shape] for output in outputs]


def test_same_axis_keeps_the_shape() -> None:
    """Splitting and concatenating one axis keeps every shape (DCP's merge)."""
    assert _build([["T", 64, 515]] * 8, 1, 1) == [["T", "64", "515"]] * 8


def test_different_axes_move_the_sharding() -> None:
    """The split axis shrinks by the group and the concat axis grows by it."""
    assert _build([[3, 8, 12]] * 4, 1, 0) == [["12", "2", "12"]] * 4


def test_groups_exchange_within_themselves() -> None:
    """Two groups of two may differ in their unsplit dims."""
    shapes: list[list[int | str]] = [[3, 8], [3, 8], [5, 8], [5, 8]]
    assert _build(shapes, 1, 0, group_size=2) == [
        ["6", "4"],
        ["6", "4"],
        ["10", "4"],
        ["10", "4"],
    ]


def test_an_indivisible_split_axis_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot split axis 1"):
        _build([[3, 6, 12]] * 4, 1, 1)


def test_mismatched_shapes_in_a_group_are_refused() -> None:
    with pytest.raises(ValueError, match="same shape"):
        _build([[3, 8], [4, 8]], 1, 1)


def test_repeated_devices_are_refused() -> None:
    with pytest.raises(ValueError, match="unique devices"):
        _build(
            [[2, 8]] * 2,
            1,
            1,
            devices=[DeviceRef.GPU(id=0), DeviceRef.GPU(id=0)],
        )
