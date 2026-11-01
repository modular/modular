# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the MAX License:
# https://modular.com/legal/max-license
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""
Test for dataflow ordering of control flow operations with free values.

This test reproduces the issue fixed in GEX-2653 where chained control flow
operations with free values (values used in nested regions but not as
direct operands) could cause "unmapped free value" assertions during
compilation.

The bug pattern: Chained ops.cond operations where later operations reference
chain results from earlier operations inside their branches, and there are
operations in between that create ordering dependencies.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from max.driver import CPU, Accelerator, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    BufferType,
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    ops,
)


def fake_model_with_chained_cond(
    combined_inputs: TensorValue,
    data_parallel_splits: TensorValue,
    signal_buffers: Any,
    device_refs: list[DeviceRef],
) -> tuple[
    TensorValue, TensorValue, TensorValue, TensorValue, TensorValue, TensorValue
]:
    """
    Creates a graph with chained conditional operations that triggers the
    free value ordering issue.

    The pattern:
    1. Split input into two parts based on dynamic splits
    2. Apply conditional operations to each part
    3. Use allgather to combine results
    4. Split again and apply more conditional operations

    The key issue: Later conditional operations yield chains from earlier
    conditional operations inside their branches, creating free values
    that must be ordered correctly in the dataflow.
    """

    def apply_slice(
        xs: list[TensorValue], data_parallel_splits: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        a = ops.slice_tensor(
            xs[0],
            [
                (
                    slice(data_parallel_splits[0], data_parallel_splits[1]),
                    "combined_inputs1",
                ),
                ...,
            ],
        )
        b = ops.slice_tensor(
            xs[1],
            [
                (
                    slice(data_parallel_splits[1], data_parallel_splits[2]),
                    "combined_inputs2",
                ),
                ...,
            ],
        )
        return a, b

    def cond_op(x: TensorValue) -> TensorValue:
        """Apply conditional operation based on shape."""
        pred = ops.shape_to_tensor([x.shape[0]])[0] > 1

        def then_fn() -> TensorValue:
            return x * 2

        def else_fn() -> TensorValue:
            return x * 3

        return ops.cond(pred, [x.type], then_fn, else_fn)[0]

    def call_cond(x: TensorValue) -> TensorValue:
        """Split, apply cond to each part, and recombine."""
        x, y = ops.split(x, [1, 1], axis=1)
        x = cond_op(x)
        y = cond_op(y)
        return ops.concat([x, y], axis=1)

    # Transfer inputs to different devices
    combined_inputs_list = [
        combined_inputs.to(device_refs[0]),
        combined_inputs.to(device_refs[1]),
    ]

    # First slice and conditional operations
    a, b = apply_slice(combined_inputs_list, data_parallel_splits)
    a = call_cond(a)
    b = call_cond(b)

    # Allgather to combine results - this creates chain dependencies
    c = ops.allgather([a, b], signal_buffers)
    c = [c0.rebind(combined_inputs_list[0].shape) for c0 in c]

    # Second slice and conditional operations
    # These conditional operations will yield chains from the earlier
    # conditional operations, creating the free value pattern
    d, e = apply_slice(c, data_parallel_splits)
    d = call_cond(d)
    e = call_cond(e)

    return a, b, c[0], c[1], d, e


def create_and_compile_graph(
    inputs: list[Buffer],
    buffers: list[Buffer],
    devices: Sequence[Device],
) -> Any:
    """Build and compile the graph that triggers the free value issue."""

    def fn(
        *inputs: TensorValue,
    ) -> tuple[
        TensorValue,
        TensorValue,
        TensorValue,
        TensorValue,
        TensorValue,
        TensorValue,
    ]:
        inputs_0, inputs_1, *signal_buffers = inputs
        device_refs = [DeviceRef.from_device(device) for device in devices]

        return fake_model_with_chained_cond(
            inputs_0,
            inputs_1,
            signal_buffers,
            device_refs,
        )

    graph = Graph(
        "test_control_flow_subfree",
        fn,
        input_types=[
            TensorType(
                inputs[0].dtype,
                ["total_seq_len", 2],
                DeviceRef.from_device(inputs[0].device),
            ),
            TensorType(inputs[1].dtype, ["batch_size"], DeviceRef.CPU()),
        ]
        + [
            BufferType(
                buffer.dtype, buffer.shape, DeviceRef.from_device(buffer.device)
            )
            for buffer in buffers
        ],
    )

    session = InferenceSession(devices=devices)
    model = session.load(graph)
    return model


def test_control_flow_subfree_ordering() -> None:
    """
    Test that control flow operations with free values are correctly
    ordered in the dataflow.

    This test only validates the compilation succeeds - the assertion would
    occur during graph compilation, not execution.
    """
    # Use CPU and single GPU
    devices = [CPU(), Accelerator(0)]

    # Create dummy input tensors (values don't matter for compilation test)
    combined_inputs = Buffer.from_numpy(np.arange(6).reshape(3, 2)).to(CPU())
    data_parallel_splits = Buffer.from_numpy(
        np.array([0, 2, 3], dtype=np.int64)
    )

    # Create minimal signal buffers
    # Note: We use a smaller size to avoid memory issues since we're only
    # testing compilation, not execution
    signal_buffer_size = 1024  # Much smaller than Signals.NUM_BYTES
    signal_buffers = [
        Buffer.zeros(shape=(signal_buffer_size,), dtype=DType.uint8, device=dev)
        for dev in devices
    ]

    model = create_and_compile_graph(
        [combined_inputs, data_parallel_splits],
        signal_buffers,
        devices,
    )

    # If we reach here, the compilation succeeded, which is what we're testing
    assert model is not None
