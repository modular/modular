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
from max._core.dialects.builtin import IntegerAttr, IntegerType

from ..graph import Graph
from ..type import _ChainType
from ..value import BufferValueLike, TensorType, TensorValue, TensorValueLike
from .utils import _buffer_values, _tensor_values


def sum(
    inputs: Iterable[TensorValueLike],
    signal_buffers: Iterable[BufferValueLike],
    axis: int = -1,
) -> list[TensorValue]:
    """Collective reduce-scatter summation operation.

    This op is a collective op which takes in tensors from different devices
    and outputs tensors on different devices. Each device reduces (via
    summation) and stores a disjoint partition of the inputs from all devices.

    Args:
        inputs: The input tensors to reduce and scatter.
        signal_buffers: Device buffer values used for synchronization.
        axis: The axis along which to scatter the reduced result. Defaults to -1.

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

    # Resolve negative axis before passing to MLIR. The kernel treats axis=-1
    # as a distinct flat-1D codepath, so we must always pass a non-negative value.
    input_shape = inputs[0].shape
    input_dtype = inputs[0].dtype
    if axis < 0:
        axis = input_shape.rank + axis
    if axis < 0 or axis >= input_shape.rank:
        raise ValueError(
            f"axis {axis} is out of bounds for tensor with rank {input_shape.rank}"
        )

    graph = Graph.current

    # Compute output types for each device's portion.
    output_types = []
    for dev_idx, device in enumerate(devices):
        output_shape_list = list(input_shape)
        scatter_dim = input_shape[axis]
        if scatter_dim is not None:
            remainder = int(scatter_dim) % num_devices
            output_shape_list[axis] = (scatter_dim // num_devices) + (
                1 if dev_idx < remainder else 0
            )
        else:
            output_shape_list[axis] = None
        output_types.append(
            TensorType(
                dtype=input_dtype,
                shape=output_shape_list,
                device=device,
            )
        )

    # Merge all device chains into one input chain.
    in_chain = graph._merge_chains(
        [graph._current_chain, *(graph.device_chains[d] for d in devices)]
    )

    # Stage a single reducescatter op across all devices.
    axis_attr = IntegerAttr(IntegerType(64), axis)
    *results, out_chain = graph._add_op_generated(
        mo.DistributedReducescatterSumOp,
        output_types,
        _ChainType(),
        inputs,
        signal_buffers,
        in_chain,
        axis_attr,
    )

    # Update all chains.
    graph._update_chain(out_chain)
    for device in devices:
        graph.device_chains[device] = out_chain

    return [res.tensor for res in results]
