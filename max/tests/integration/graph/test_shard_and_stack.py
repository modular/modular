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
"""Execution tests for ops.shard_and_stack."""

import numpy as np
import pytest
import torch
from max.driver import Buffer
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
    "dtype",
    [
        DType.int8,
        DType.int16,
        DType.int32,
        DType.int64,
    ],
)
def test_shard_and_stack_dtypes(
    session: InferenceSession,
    dtype: DType,
) -> None:
    """Test shard_and_stack with different data types."""
    input_shape = [8, 16]
    num_inputs = 2
    tp_degree = 2

    # Map DType to torch dtype
    dtype_map = {
        DType.int8: torch.int8,
        DType.int16: torch.int16,
        DType.int32: torch.int32,
        DType.int64: torch.int64,
    }
    torch_dtype = dtype_map[dtype]

    # Create torch tensors with the appropriate dtype
    torch_inputs = [
        torch.randint(-100, 100, input_shape, dtype=torch_dtype)
        for _ in range(num_inputs)
    ]

    # Convert to numpy for reference implementation (float32 for all)
    inputs = [inp.to(torch.float32).cpu().numpy() for inp in torch_inputs]

    # Create devices list for shard_and_stack
    devices = [DeviceRef.CPU()] * tp_degree

    # Build graph
    with Graph(
        f"shard_and_stack_{dtype}",
        input_types=[
            TensorType(dtype, input_shape, device=DeviceRef.CPU())
            for _ in range(num_inputs)
        ],
    ) as graph:
        outputs = ops.shard_and_stack(
            [inp.tensor for inp in graph.inputs], devices
        )
        graph.output(*outputs)

    # Compile and execute
    model = session.load(graph)
    device_inputs = [
        Buffer.from_dlpack(inp).to(model.input_devices[0])
        for inp in torch_inputs
    ]
    results = model.execute(*device_inputs)

    # Compute expected results
    expected_outputs = reference_shard_and_stack(inputs, tp_degree)

    # Validate
    assert len(results) == tp_degree

    for tp_index, (result, expected) in enumerate(
        zip(results, expected_outputs, strict=True)
    ):
        assert isinstance(result, Buffer)
        # Convert result to numpy (will be in the original dtype)
        # For comparison, convert to float32 if needed
        actual_np = result.to_numpy()
        if dtype == DType.float16:
            # Convert to float32 for comparison due to precision differences
            actual_np = torch.from_numpy(actual_np).to(torch.float32).numpy()

        np.testing.assert_array_equal(
            actual_np,
            expected,
            err_msg=f"Mismatch for dtype {dtype}, tp_index={tp_index}",
        )


@pytest.mark.parametrize(
    "input_shape,num_inputs,tp_degree,axis",
    [
        ([4, 8, 16], 2, 2, 0),  # 3D, split axis 0
        ([4, 8, 16], 2, 2, 1),  # 3D, split axis 1
        ([4, 8, 16], 2, 4, 2),  # 3D, split axis 2
    ],
)
def test_shard_and_stack_various_axes(
    session: InferenceSession,
    input_shape: list[int],
    num_inputs: int,
    tp_degree: int,
    axis: int,
) -> None:
    """Test shard_and_stack with various axis values."""
    # Create random input data
    np.random.seed(42)
    inputs = [
        np.random.uniform(-1.0, 1.0, size=input_shape).astype(np.float32)
        for _ in range(num_inputs)
    ]

    # Create devices list for shard_and_stack
    devices = [DeviceRef.CPU()] * tp_degree

    # Build graph
    with Graph(
        f"shard_and_stack_axis{axis}",
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
        np.testing.assert_array_equal(
            result.to_numpy(),
            expected,
            err_msg=f"Mismatch for axis={axis}, tp_index={tp_index}",
        )
