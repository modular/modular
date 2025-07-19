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
"""Op implementation for allgather."""

from __future__ import annotations

from collections.abc import Iterable

from max.mlir.dialects import mo

from ..graph import Graph
from ..type import TensorType, _ChainType
from ..value import BufferValue, TensorValue
from .concat import concat


def allgather(
    inputs: Iterable[TensorValue],
    signal_buffers: Iterable[BufferValue],
    axis: int = 0,
) -> list[TensorValue]:
    """Collective allgather operation.

    This op is a collective op which takes in tensors from different devices and
    outputs tensors on different devices.
    In particular, this operation will gather the inputs across different
    devices and concatenates them along the specified dimension.
    The result is then broadcasted back to the same devices that the inputs
    came from.

    Args:
        inputs: The input tensors to gather.
        signal_buffers: Device buffer values used for synchronization.
        axis: Dimension to concatenate the input tensors. Defaults to 0.

    Returns:
        An iterable outputs which all hold the gathered output. Each output
        tensor contains the concatenation of all inputs along the specified dimension.
    """
    # Convert `inputs` to list since we'll iterate over it multiple times.
    inputs = list(inputs)
    signal_buffers = list(signal_buffers)

    if len(inputs) != len(signal_buffers):
        raise ValueError(
            f"expected number of inputs ({len(inputs)}) and number of "
            f"signal buffers ({len(signal_buffers)}) to match"
        )

    if len(inputs) < 2:
        return inputs

    shape = inputs[0].shape
    dtype = inputs[0].dtype
    # Check that all inputs have the same rank and are compatible for concatenation
    if not all(input.shape.rank == shape.rank for input in inputs[1:]):
        raise ValueError(
            "allgather operation must have the same rank across all"
            " input tensors."
        )
    if not all(input.dtype == dtype for input in inputs[1:]):
        all_dtypes = ", ".join([str(x.dtype) for x in inputs])
        raise ValueError(
            "allgather operation must have the same dtype across all"
            f" input tensors. Got: {all_dtypes}"
        )
    if not all(input.device for input in inputs[1:]):
        raise ValueError(
            "allgather operation inputs must have an explicit device. "
            f"Got: {inputs}"
        )

    devices = []
    for input in inputs:
        if input.device in devices:
            all_devices = ", ".join([str(x.device) for x in inputs])
            raise ValueError(
                "allgather operation must have unique devices across its input "
                f"tensors. Got: {all_devices}"
            )
        devices.append(input.device)

    if not -shape.rank <= axis < shape.rank:
        raise IndexError(f"Dimension out of range {shape.rank}, {axis=}")
    if axis < 0:
        axis += shape.rank

    # Check that all dimensions except the concatenation dimension are the same.
    for i, input in enumerate(inputs[1:], 1):
        for d in range(shape.rank):
            if d != axis and input.shape[d] != shape[d]:
                raise ValueError(
                    f"allgather operation inputs must have the same shape in all "
                    f"dimensions except the concatenation dimension. Input 0 has "
                    f"shape {shape}, but input {i} has shape {input.shape}. "
                    f"Mismatch at dimension {d}."
                )

    num_devices = len(inputs)

    # Prepare output types - one per input per device.
    output_types = []
    for device_idx in range(num_devices):
        for input_idx in range(num_devices):
            output_types.append(
                TensorType(
                    dtype,
                    inputs[input_idx].shape,
                    device=inputs[device_idx].device,
                ).to_mlir()
            )

    # Get the current chain for synchronization
    in_chain = Graph.current._current_chain

    # Stage the allgather op with signal buffers and chain.
    *results, out_chain = Graph.current._add_op(
        mo.distributed_allgather,
        # Output types: tensors + chain
        output_types,
        _ChainType().to_mlir(),
        inputs,
        signal_buffers,
        in_chain,
    )

    # Update the chain
    Graph.current._update_chain(out_chain)

    # Convert results to TensorValues.
    all_outputs = [res.tensor for res in results]

    # Concatenate outputs for each device.
    concatenated_outputs = []
    for device_idx in range(num_devices):
        device_outputs = []
        for input_idx in range(num_devices):
            output_idx = device_idx * num_devices + input_idx
            device_outputs.append(all_outputs[output_idx])

        result = concat(device_outputs, axis=axis)
        concatenated_outputs.append(result)

    return concatenated_outputs
