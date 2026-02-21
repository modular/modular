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
"""Op implementation for allreduce."""

from __future__ import annotations

from collections.abc import Iterable

from max._core.dialects import builtin, mo

from ..graph import Graph
from ..type import _ChainType
from ..value import BufferValueLike, TensorType, TensorValue, TensorValueLike
from .utils import _buffer_values, _tensor_values


def sum(
    inputs: Iterable[TensorValueLike], signal_buffers: Iterable[BufferValueLike]
) -> list[TensorValue]:
    """Collective allreduce summation operation.

    This op is a collective op which takes in tensors from different devices and
    outputs tensors on different devices.
    In particular, this operation will gather the inputs across different
    devices and reduce them via a summation operation.
    The result is then broadcasted back to the same devices that the inputs
    came from.

    Args:
        inputs: The input tensors to reduce.
        signal_buffers: Device buffer values used for synchronization.

    Returns:
        An iterable outputs which all hold the reduction output.
    """
    inputs = _tensor_values(inputs)
    signal_buffers = _buffer_values(signal_buffers)
    if len(inputs) != len(signal_buffers):
        raise ValueError(
            f"expected number of inputs ({len(inputs)}) and number of "
            f"signal buffers ({len(signal_buffers)}) to match"
        )

    devices = [input.device for input in inputs]

    if not all(input.shape == inputs[0].shape for input in inputs[1:]):
        raise ValueError(
            "allreduce.sum operation must have the same shape across all"
            f" input tensors. Got: {inputs=}"
        )
    if len(set(devices)) < len(devices):
        raise ValueError(
            "allreduce.sum operation must have unique devices across its input "
            f"tensors. Got: {devices=}"
        )

    # Per-device execution model:
    # Create one allreduce op per device, each threading the destination
    # device's chain independently.
    # Do not merge device chains.
    results = []
    graph = Graph.current
    for input_tensor, device in zip(inputs, devices, strict=True):
        in_chain = graph.device_chains[device]
        # Each op takes all inputs but only produces output for its device.
        # hasDeviceBarrier indicates this op has internal device barriers,
        # so only same-device operand chains need to be waited on.
        result, out_chain = Graph.current._add_op_generated(
            mo.DistributedAllreduceSumOp,
            # Single output tensor type.
            input_tensor.type,
            # Output chain type.
            _ChainType(),
            inputs,
            signal_buffers,
            in_chain,
            device,
            has_device_barrier=builtin.UnitAttr(),
        )

        results.append(result.tensor)
        # Advance only this device's chain.
        graph.device_chains[device] = out_chain

    return results


def matmul_allreduce(
    inputs: Iterable[TensorValueLike],
    weights: Iterable[TensorValueLike],
    signal_buffers: Iterable[BufferValueLike],
) -> list[TensorValue]:
    """Performs batched matmul then all-reduce over the given inputs, weights, and signal buffers."""
    inputs = _tensor_values(inputs)
    weights = _tensor_values(weights)
    signal_buffers = _buffer_values(signal_buffers)

    def infer_out_type(a: TensorValue, b: TensorValue) -> TensorType:
        if a.rank != 2 or b.rank != 2:
            raise ValueError("matmul_allreduce inputs must be 2D")
        m = a.shape[-2]
        n = b.shape[-2]
        out_shape = a.shape[:-2] + [m, n]
        return TensorType(
            dtype=a.dtype,
            shape=out_shape,
            device=a.device,
        )

    in_chain = Graph.current._current_chain
    *results, out_chain = Graph.current._add_op_generated(
        mo.DistributedMatmulAllreduceOp,
        # Types for 2 outputs: chain, list of tensors
        [infer_out_type(a, b) for a, b in zip(inputs, weights, strict=True)],
        _ChainType(),
        inputs,
        weights,
        signal_buffers,
        in_chain,
    )

    Graph.current._update_chain(out_chain)
    return [res.tensor for res in results]
