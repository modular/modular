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

import itertools
from collections.abc import Iterable
from typing import TypeVar

from max._core.dialects import mo

from ..graph import Graph
from ..type import TensorType, _ChainType
from ..value import (
    BufferValue,
    BufferValueLike,
    TensorValue,
    TensorValueLike,
)
from .concat import concat

T = TypeVar("T")


# Added to itertools in 3.12
def _batched(iterable: Iterable[T], n: int) -> Iterable[tuple[T, ...]]:
    """Batch data into lists of length n. The last batch may be shorter."""
    it = iter(iterable)
    while True:
        batch = tuple(itertools.islice(it, n))
        if not batch:
            return
        yield batch


def _buffer_values(values: Iterable[BufferValueLike]) -> list[BufferValue]:
    return [BufferValue(v) for v in values]


def _tensor_values(values: Iterable[TensorValueLike]) -> list[TensorValue]:
    return [TensorValue(v) for v in values]


def allgather(
    inputs: Iterable[TensorValueLike],
    signal_buffers: Iterable[BufferValueLike],
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
    inputs = _tensor_values(inputs)
    signal_buffers = _buffer_values(signal_buffers)

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
            f" input tensors. Got: {inputs=}"
        )
    if not all(input.dtype == dtype for input in inputs[1:]):
        raise ValueError(
            "allgather operation must have the same dtype across all"
            f" input tensors. Got: {inputs=}"
        )

    devices = [input.device for input in inputs]
    if len(set(devices)) < len(devices):
        raise ValueError(
            "allgather operation must have unique devices across its input "
            f"tensors. Got: {devices=}"
        )

    if not -shape.rank <= axis < shape.rank:
        raise IndexError(f"Dimension out of range {shape.rank}, {axis=}")
    if axis < 0:
        axis += shape.rank

    # Check that all dimensions except the concatenation dimension are the same.
    for i, dim in enumerate(inputs[0].shape):
        if i == axis:
            continue
        if not all(input.shape[i] == dim for input in inputs):
            raise ValueError(
                f"allgather operation inputs must have the same shape in all "
                f"dimensions except the concatenation dimension. {inputs=}"
            )

    # Prepare output types - one per input per device.
    output_types = [
        TensorType(dtype, input.shape, device)
        for device in devices
        for input in inputs
    ]

    # Get the current chain for synchronization.
    graph = Graph.current
    in_chain = graph._merge_chains(
        [graph._current_chain, *(graph.device_chains[d] for d in devices)]
    )

    # Stage the allgather op with signal buffers and chain.
    *results, out_chain = graph._add_op_generated(
        mo.DistributedAllgatherOp,
        # Output types: tensors + chain
        output_types,
        _ChainType(),
        inputs,
        signal_buffers,
        in_chain,
    )

    # Update the chain.
    graph._update_chain(out_chain)

    # Update device chains.
    for device in devices:
        graph.device_chains[device] = out_chain

    # Convert results to TensorValues.
    outputs = [res.tensor for res in results]
    return [
        concat(batch, axis=axis) for batch in _batched(outputs, len(inputs))
    ]
