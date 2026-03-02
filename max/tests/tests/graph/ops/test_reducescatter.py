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
"""Test the max.graph Python bindings for reducescatter."""

import pytest
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn import Signals


def test_reducescatter_rep_device() -> None:
    """Test unique device error for reducescatter."""
    devices = [
        DeviceRef.GPU(id=0),
        DeviceRef.GPU(id=0),
        DeviceRef.GPU(id=2),
        DeviceRef.GPU(id=3),
    ]
    signals = Signals(devices)

    with pytest.raises(
        ValueError,
        match=(
            r"reducescatter.sum operation must have unique devices across its"
            r" input tensors."
        ),
    ):
        with Graph(
            "reducescatter",
            input_types=[
                TensorType(
                    dtype=DType.float32, shape=[24, 5], device=devices[0]
                ),
                TensorType(
                    dtype=DType.float32, shape=[24, 5], device=devices[1]
                ),
                TensorType(
                    dtype=DType.float32, shape=[24, 5], device=devices[2]
                ),
                TensorType(
                    dtype=DType.float32, shape=[24, 5], device=devices[3]
                ),
                *signals.input_types(),
            ],
        ) as graph:
            reducescatter_outputs = ops.reducescatter.sum(
                inputs=(v.tensor for v in graph.inputs[: len(devices)]),
                signal_buffers=(v.buffer for v in graph.inputs[len(devices) :]),
            )
            graph.output(*reducescatter_outputs)


def test_reducescatter_wrong_shape() -> None:
    """Test wrong shape error for reducescatter."""
    devices = [
        DeviceRef.GPU(id=0),
        DeviceRef.GPU(id=1),
        DeviceRef.GPU(id=2),
        DeviceRef.GPU(id=3),
    ]
    signals = Signals(devices)

    with pytest.raises(
        ValueError,
        match=(
            r"reducescatter.sum operation must have the same shape across all"
            r" input tensors."
        ),
    ):
        with Graph(
            "reducescatter",
            input_types=[
                TensorType(
                    dtype=DType.float32, shape=[24, 5], device=devices[0]
                ),
                TensorType(
                    dtype=DType.float32, shape=[24, 2], device=devices[1]
                ),
                TensorType(
                    dtype=DType.float32, shape=[24, 5], device=devices[2]
                ),
                TensorType(
                    dtype=DType.float32, shape=[24, 5], device=devices[3]
                ),
                *signals.input_types(),
            ],
        ) as graph:
            reducescatter_outputs = ops.reducescatter.sum(
                inputs=(v.tensor for v in graph.inputs[: len(devices)]),
                signal_buffers=(v.buffer for v in graph.inputs[len(devices) :]),
            )
            graph.output(*reducescatter_outputs)


def test_reducescatter_basic() -> None:
    """Test basic reducescatter use case.

    With 4 devices and input shape [5, 24], output shape should be [5, 6]
    (24 / 4 = 6 along axis 1).
    """
    devices = [
        DeviceRef.GPU(id=0),
        DeviceRef.GPU(id=1),
        DeviceRef.GPU(id=2),
        DeviceRef.GPU(id=3),
    ]
    signals = Signals(devices)

    with Graph(
        "reducescatter",
        input_types=[
            TensorType(dtype=DType.float32, shape=[5, 24], device=devices[0]),
            TensorType(dtype=DType.float32, shape=[5, 24], device=devices[1]),
            TensorType(dtype=DType.float32, shape=[5, 24], device=devices[2]),
            TensorType(dtype=DType.float32, shape=[5, 24], device=devices[3]),
            *signals.input_types(),
        ],
    ) as graph:
        reducescatter_outputs = ops.reducescatter.sum(
            inputs=(v.tensor for v in graph.inputs[: len(devices)]),
            signal_buffers=(v.buffer for v in graph.inputs[len(devices) :]),
        )
        graph.output(*reducescatter_outputs)
        for output, device in zip(reducescatter_outputs, devices, strict=True):
            assert device == output.device
            assert output.shape[0] == 5
            assert output.shape[1] == 6  # 24 / 4


def test_reducescatter_axis0() -> None:
    """Test reducescatter with axis=0.

    With 2 devices and input shape [6, 10], output shape should be [3, 10]
    (6 / 2 = 3 along axis 0).
    """
    devices = [
        DeviceRef.GPU(id=0),
        DeviceRef.GPU(id=1),
    ]
    signals = Signals(devices)

    with Graph(
        "reducescatter",
        input_types=[
            TensorType(dtype=DType.float32, shape=[6, 10], device=devices[0]),
            TensorType(dtype=DType.float32, shape=[6, 10], device=devices[1]),
            *signals.input_types(),
        ],
    ) as graph:
        reducescatter_outputs = ops.reducescatter.sum(
            inputs=(v.tensor for v in graph.inputs[: len(devices)]),
            signal_buffers=(v.buffer for v in graph.inputs[len(devices) :]),
            axis=0,
        )
        graph.output(*reducescatter_outputs)
        for output, device in zip(reducescatter_outputs, devices, strict=True):
            assert device == output.device
            assert output.shape[0] == 3  # 6 / 2
            assert output.shape[1] == 10


def test_reducescatter_axis1() -> None:
    """Test reducescatter with explicit axis=1.

    With 2 devices and input shape [6, 10], output shape should be [6, 5]
    (10 / 2 = 5 along axis 1).
    """
    devices = [
        DeviceRef.GPU(id=0),
        DeviceRef.GPU(id=1),
    ]
    signals = Signals(devices)

    with Graph(
        "reducescatter",
        input_types=[
            TensorType(dtype=DType.float32, shape=[6, 10], device=devices[0]),
            TensorType(dtype=DType.float32, shape=[6, 10], device=devices[1]),
            *signals.input_types(),
        ],
    ) as graph:
        reducescatter_outputs = ops.reducescatter.sum(
            inputs=(v.tensor for v in graph.inputs[: len(devices)]),
            signal_buffers=(v.buffer for v in graph.inputs[len(devices) :]),
            axis=1,
        )
        graph.output(*reducescatter_outputs)
        for output, device in zip(reducescatter_outputs, devices, strict=True):
            assert device == output.device
            assert output.shape[0] == 6
            assert output.shape[1] == 5  # 10 / 2


def test_reducescatter_axis_out_of_bounds() -> None:
    """Test that out-of-bounds axis raises ValueError."""
    devices = [
        DeviceRef.GPU(id=0),
        DeviceRef.GPU(id=1),
    ]
    signals = Signals(devices)

    with pytest.raises(
        ValueError,
        match=r"axis .* is out of bounds",
    ):
        with Graph(
            "reducescatter",
            input_types=[
                TensorType(
                    dtype=DType.float32, shape=[6, 10], device=devices[0]
                ),
                TensorType(
                    dtype=DType.float32, shape=[6, 10], device=devices[1]
                ),
                *signals.input_types(),
            ],
        ) as graph:
            ops.reducescatter.sum(
                inputs=(v.tensor for v in graph.inputs[: len(devices)]),
                signal_buffers=(v.buffer for v in graph.inputs[len(devices) :]),
                axis=5,
            )


def test_reducescatter_ragged_axis0() -> None:
    """Test ragged partition where scatter dim is not evenly divisible.

    With 8 devices and input shape [9, 16], axis=0:
    GPU 0 gets (2, 16) (9 // 8 + 1 remainder) and GPUs 1-7 get (1, 16).
    """
    num_gpus = 8
    devices = [DeviceRef.GPU(id=i) for i in range(num_gpus)]
    signals = Signals(devices)

    with Graph(
        "reducescatter",
        input_types=[
            TensorType(dtype=DType.float32, shape=[9, 16], device=dev)
            for dev in devices
        ]
        + list(signals.input_types()),
    ) as graph:
        reducescatter_outputs = ops.reducescatter.sum(
            inputs=(v.tensor for v in graph.inputs[:num_gpus]),
            signal_buffers=(v.buffer for v in graph.inputs[num_gpus:]),
            axis=0,
        )
        graph.output(*reducescatter_outputs)
        for dev_idx, (output, device) in enumerate(
            zip(reducescatter_outputs, devices, strict=True)
        ):
            assert device == output.device
            expected_rows = 2 if dev_idx < 1 else 1  # 9 % 8 == 1
            assert output.shape[0] == expected_rows
            assert output.shape[1] == 16
