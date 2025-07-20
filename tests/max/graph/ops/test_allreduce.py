# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""Test the max.graph Python bindings."""

import pytest
from conftest import tensor_types
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn import Signals

shared_types = st.shared(tensor_types())


def test_allreduce_rep_device() -> None:
    """Test unique device error for allreduce."""
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
            "allreduce.sum operation must have unique devices across its input"
            " tensors."
        ),
    ):
        with Graph(
            "allreduce",
            input_types=[
                TensorType(
                    dtype=DType.float32, shape=[6, 5], device=devices[0]
                ),
                TensorType(
                    dtype=DType.float32, shape=[6, 5], device=devices[1]
                ),
                TensorType(
                    dtype=DType.float32, shape=[6, 5], device=devices[2]
                ),
                TensorType(
                    dtype=DType.float32, shape=[6, 5], device=devices[3]
                ),
                *signals.input_types(),
            ],
        ) as graph:
            allreduce_outputs = ops.allreduce.sum(
                inputs=(v.tensor for v in graph.inputs[: len(devices)]),
                signal_buffers=(v.buffer for v in graph.inputs[len(devices) :]),
            )
            graph.output(
                allreduce_outputs[0],
                allreduce_outputs[1],
                allreduce_outputs[2],
                allreduce_outputs[3],
            )


def test_allreduce_wrong_shape() -> None:
    """Test wrong shape error for allreduce."""
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
            "allreduce.sum operation must have the same shape across all input"
            " tensors."
        ),
    ):
        with Graph(
            "allreduce",
            input_types=[
                TensorType(
                    dtype=DType.float32, shape=[6, 5], device=devices[0]
                ),
                TensorType(
                    dtype=DType.float32, shape=[6, 2], device=devices[1]
                ),
                TensorType(
                    dtype=DType.float32, shape=[6, 5], device=devices[2]
                ),
                TensorType(
                    dtype=DType.float32, shape=[6, 5], device=devices[3]
                ),
                *signals.input_types(),
            ],
        ) as graph:
            allreduce_outputs = ops.allreduce.sum(
                inputs=(v.tensor for v in graph.inputs[: len(devices)]),
                signal_buffers=(v.buffer for v in graph.inputs[len(devices) :]),
            )
            graph.output(
                allreduce_outputs[0],
                allreduce_outputs[1],
                allreduce_outputs[2],
                allreduce_outputs[3],
            )


def test_allreduce_basic() -> None:
    """Test basic allreduce use case."""
    devices = [
        DeviceRef.GPU(id=0),
        DeviceRef.GPU(id=1),
        DeviceRef.GPU(id=2),
        DeviceRef.GPU(id=3),
    ]
    signals = Signals(devices)

    with Graph(
        "allreduce",
        input_types=[
            TensorType(dtype=DType.float32, shape=[6, 5], device=devices[0]),
            TensorType(dtype=DType.float32, shape=[6, 5], device=devices[1]),
            TensorType(dtype=DType.float32, shape=[6, 5], device=devices[2]),
            TensorType(dtype=DType.float32, shape=[6, 5], device=devices[3]),
            *signals.input_types(),
        ],
    ) as graph:
        allreduce_outputs = ops.allreduce.sum(
            inputs=(v.tensor for v in graph.inputs[: len(devices)]),
            signal_buffers=(v.buffer for v in graph.inputs[len(devices) :]),
        )
        graph.output(
            allreduce_outputs[0],
            allreduce_outputs[1],
            allreduce_outputs[2],
            allreduce_outputs[3],
        )
        for output, device in zip(allreduce_outputs, devices):
            assert device == output.device
