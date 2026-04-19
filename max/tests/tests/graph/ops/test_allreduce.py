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
"""Test the max.graph Python bindings."""

import re

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
            r"allreduce.sum operation must have unique devices across its input"
            r" tensors."
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
            r"allreduce.sum operation must have the same shape across all input"
            r" tensors."
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
        for output, device in zip(allreduce_outputs, devices, strict=True):
            assert device == output.device


def test_allreduce_layer_stages_bundled_ir() -> None:
    """Test that ops.bundled_allreduce.sum generates bundled allreduce inside mo.parallel."""
    devices = [
        DeviceRef.GPU(id=0),
        DeviceRef.GPU(id=1),
    ]
    signals = Signals(devices)

    with Graph(
        "allreduce_bundled",
        input_types=[
            TensorType(dtype=DType.float32, shape=[6, 5], device=devices[0]),
            TensorType(dtype=DType.float32, shape=[6, 5], device=devices[1]),
            *signals.input_types(),
        ],
    ) as graph:
        tensor_inputs = [graph.inputs[i].tensor for i in range(len(devices))]
        signal_buffers = [
            graph.inputs[len(devices) + i].buffer for i in range(len(devices))
        ]

        allreduce_outputs = ops.bundled_allreduce.sum(
            tensor_inputs, signal_buffers
        )
        graph.output(*allreduce_outputs)

    ir = str(graph)

    assert re.search(r"mo\.parallel", ir), (
        "Expected mo.parallel region in IR but not found"
    )
    assert re.search(r"mo\.bundled\.allreduce\.sum", ir), (
        "Expected mo.bundled.allreduce.sum in IR but not found"
    )
    assert not re.search(r"mo\.distributed\.allreduce\.sum", ir), (
        "Did not expect mo.distributed.allreduce.sum in IR"
    )
