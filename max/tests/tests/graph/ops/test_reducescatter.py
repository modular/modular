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


def test_reducescatter_axis0_not_supported() -> None:
    """Test that axis != -1 raises NotImplementedError."""
    devices = [
        DeviceRef.GPU(id=0),
        DeviceRef.GPU(id=1),
    ]
    signals = Signals(devices)

    with pytest.raises(
        NotImplementedError,
        match=r"reducescatter.sum only supports axis=-1",
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
                axis=0,
            )
