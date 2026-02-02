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
"""GPU execution tests for ops.shard_and_stack.

These tests verify the H2D (Host-to-Device) transfer path of the shard_and_stack
operation. Each output must be mapped to a unique GPU device.
"""

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def reference_shard_and_stack(
    inputs: list[np.ndarray],
    tp_degree: int,
    axis: int = 0,
) -> list[np.ndarray]:
    """Reference implementation using numpy.

    Args:
        inputs: List of input arrays to shard.
        tp_degree: Number of splits to perform along the specified axis.
        axis: The axis along which to split each input. Defaults to 0.

    Returns:
        List of tp_degree output arrays, where output i contains the i-th chunk
        from all inputs stacked along dimension 0.
    """
    # Normalize negative axis
    if axis < 0:
        axis = inputs[0].ndim + axis

    # Get chunk size along specified axis
    chunk_size = inputs[0].shape[axis] // tp_degree

    # Split each input and collect chunks
    all_input_chunks = []
    for inp in inputs:
        split_indices = [chunk_size * (i + 1) for i in range(tp_degree - 1)]
        input_chunks = np.split(inp, split_indices, axis=axis)
        all_input_chunks.append(input_chunks)

    # Create tp_degree outputs by stacking the i-th chunk from each input
    outputs = []
    for tp_index in range(tp_degree):
        chunks_for_output = [chunks[tp_index] for chunks in all_input_chunks]
        outputs.append(np.stack(chunks_for_output, axis=0))

    return outputs


@pytest.mark.parametrize(
    "input_shape,num_inputs,axis",
    [
        ([128, 256], 2, 0),  # 2D, split axis 0
        ([64, 128, 256], 3, 1),  # 3D, split axis 1
        ([64, 128, 256], 2, 2),  # 3D, split axis 2
    ],
)
def test_shard_and_stack_to_multiple_gpus(
    input_shape: list[int],
    num_inputs: int,
    axis: int,
) -> None:
    """Test shard_and_stack distributing shards across multiple GPUs.

    This test verifies that the shard_and_stack operation correctly distributes
    different shards to different GPU devices, which is the primary use case
    for tensor parallelism. Each output is mapped to a unique GPU.
    """
    # Query available GPUs - tp_degree equals number of GPUs (one output per GPU)
    num_gpus = accelerator_count()
    assert num_gpus >= 2, "This test requires at least 2 GPUs"
    tp_degree = num_gpus

    # Ensure the split axis dimension is divisible by tp_degree
    # Adjust the shape if necessary by rounding up to the nearest multiple
    input_shape = list(input_shape)  # Make a mutable copy
    axis_dim = input_shape[axis]
    if axis_dim % tp_degree != 0:
        input_shape[axis] = axis_dim * tp_degree

    # Create session with CPU and all available GPU devices
    host = CPU()
    gpus = [Accelerator(i) for i in range(num_gpus)]
    session = InferenceSession(devices=[host, *gpus])

    # Create random input data
    np.random.seed(42)
    inputs = [
        np.random.uniform(-1.0, 1.0, size=input_shape).astype(np.float32)
        for _ in range(num_inputs)
    ]

    # Each output goes to a unique GPU
    devices = [DeviceRef.GPU(i) for i in range(num_gpus)]

    # Build graph: inputs on CPU, outputs distributed across all GPUs
    with Graph(
        f"shard_and_stack_multi_gpu_axis{axis}",
        input_types=[
            TensorType(DType.float32, input_shape, device=DeviceRef.CPU())
            for _ in range(num_inputs)
        ],
    ) as graph:
        outputs = ops.shard_and_stack(
            [inp.tensor for inp in graph.inputs], devices, axis
        )
        graph.output(*outputs)

    # Compile and execute
    model = session.load(graph)
    device_inputs = [
        Buffer.from_numpy(inp).to(model.input_devices[0]) for inp in inputs
    ]
    results = model.execute(*device_inputs)

    # Compute expected results
    expected_outputs = reference_shard_and_stack(inputs, tp_degree, axis)

    # Validate
    assert len(results) == tp_degree

    for tp_index, (result, expected) in enumerate(
        zip(results, expected_outputs, strict=True)
    ):
        assert isinstance(result, Buffer)
        np.testing.assert_array_almost_equal(
            result.to_numpy(),
            expected,
            decimal=5,
            err_msg=f"Mismatch for multi-GPU axis={axis}, tp_index={tp_index}",
        )
