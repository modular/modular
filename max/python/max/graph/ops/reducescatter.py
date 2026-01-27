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
"""Op implementation for reducescatter."""

from __future__ import annotations

from collections.abc import Iterable

from max._core.dialects import mo

from ..graph import Graph
from ..type import _ChainType
from ..value import BufferValueLike, TensorType, TensorValue, TensorValueLike
from .utils import _buffer_values, _tensor_values


def sum(
    inputs: Iterable[TensorValueLike],
    signal_buffers: Iterable[BufferValueLike],
    axis: int = 0,
) -> list[TensorValue]:
    """Collective reduce-scatter summation operation.

    This op is a collective op which takes in tensors from different devices
    and outputs tensors on different devices. Each device reduces (via
    summation) and stores a disjoint partition of the inputs from all devices.

    Args:
        inputs: The input tensors to reduce and scatter.
        signal_buffers: Device buffer values used for synchronization.
        axis: The axis along which to scatter the reduced result. Defaults to 0.

    Returns:
        An iterable of outputs where each device receives its portion of the
        scattered result. The output shape on each device is the input shape
        with dimension `axis` divided by the number of devices.
    """
    inputs = _tensor_values(inputs)
    signal_buffers = _buffer_values(signal_buffers)
    if len(inputs) != len(signal_buffers):
        raise ValueError(
            f"expected number of inputs ({len(inputs)}) and number of "
            f"signal buffers ({len(signal_buffers)}) to match"
        )

    devices = [input.device for input in inputs]
    num_devices = len(devices)

    if not all(input.shape == inputs[0].shape for input in inputs[1:]):
        raise ValueError(
            "reducescatter.sum operation must have the same shape across all"
            f" input tensors. Got: {inputs=}"
        )
    if len(set(devices)) < len(devices):
        raise ValueError(
            "reducescatter.sum operation must have unique devices across its "
            f"input tensors. Got: {devices=}"
        )

    # Compute output shape: input shape with scatter axis divided by num_devices
    input_shape = inputs[0].shape
    input_dtype = inputs[0].dtype
    if axis < 0:
        axis = input_shape.rank + axis
    if axis < 0 or axis >= input_shape.rank:
        raise ValueError(
            f"axis {axis} is out of bounds for tensor with rank {input_shape.rank}"
        )

    # TODO(KERN-2337): Support axis != 0 in the kernel.
    if axis != 0:
        raise NotImplementedError(
            f"reducescatter.sum only supports axis=0, got axis={axis}"
        )

    # Per-device execution model:
    # Create one reducescatter op per device, each threading the destination
    # device's chain independently.
    # Do not merge device chains.
    results = []
    graph = Graph.current
    for device in devices:
        in_chain = graph.device_chains[device]

        # Compute output shape for this device's portion
        output_shape_list = list(input_shape)
        scatter_dim = input_shape[axis]
        if scatter_dim is not None:
            # TODO(KERN-2337): Handle uneven division.
            output_shape_list[axis] = scatter_dim // num_devices
        else:
            output_shape_list[axis] = None
        output_type = TensorType(
            dtype=input_dtype,
            shape=output_shape_list,
            device=device,
        )

        # Each op takes all inputs but only produces output for its device.
        result, out_chain = Graph.current._add_op_generated(
            mo.DistributedReducescatterSumOp,
            # Single output tensor type.
            output_type,
            # Output chain type.
            _ChainType(),
            inputs,
            signal_buffers,
            in_chain,
            device,
        )

        results.append(result.tensor)
        # Advance only this device's chain.
        graph.device_chains[device] = out_chain

    return results
