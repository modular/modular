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
"""Op implementation for allreduce."""

from __future__ import annotations

from collections.abc import Iterable

from max._core.dialects import mo
from max.mlir.dialects import mo as _mo

from ..graph import Graph
from ..type import _ChainType
from ..value import BufferValue, TensorType, TensorValue


def sum(
    inputs: Iterable[TensorValue], signal_buffers: Iterable[BufferValue]
) -> list[TensorValue]:
    """Collective allreduce summation operation.

    This op is a collective op which takes in tensors from different devices and
    outputs tensors on different devices.
    In particular, this operation will gather the inputs across different
    devices and reduce them via a summation operation.
    The result is then broadcasted back to the same devices that the inputs
    came from.

    This version of the allreduce sum op uses device-to-device transfers and
    hence is expected to be much slower than the :obj:`ops.allreduce.sum` version.

    Args:
        inputs: The input tensors to reduce.
        signal_buffers: Device buffer values used for synchronization.

    Returns:
        An iterable outputs which all hold the reduction output.
    """
    # Convert `inputs` to list since we'll iterate over it twice.
    inputs = list(inputs)
    signal_buffers = list(signal_buffers)
    if len(inputs) != len(signal_buffers):
        raise ValueError(
            f"expected number of inputs ({len(inputs)}) and number of "
            f"signal buffers ({len(signal_buffers)}) to match"
        )

    shape = None
    devices = []

    for input in inputs:
        if not shape:
            shape = input.shape
        if input.shape != shape:
            raise ValueError(
                "allreduce.sum operation must have the same shape across all"
                " input tensors."
            )
        if not input.device:
            raise ValueError(
                f"allreduce.sum operation input = {input} needs to have an"
                " explicit device."
            )
        if input.device in devices:
            raise ValueError(
                "allreduce.sum operation must have unique devices across its"
                " input tensors."
            )
        devices.append(input.device)

    # Per-device execution model:
    # Create one allreduce op per device, each threading the destination
    # device's chain independently.
    # Do not merge device chains.
    results = []
    graph = Graph.current
    for input_tensor, device in zip(inputs, devices, strict=True):
        in_chain = graph.device_chains[device]
        # Each op takes all inputs but only produces output for its device.
        (result, out_chain), _ = Graph.current._add_op_get_op_with_results(
            _mo.distributed_allreduce_sum,
            # Single output tensor type.
            input_tensor.type.to_mlir(),
            # Output chain type.
            _ChainType().to_mlir(),
            inputs,
            signal_buffers,
            in_chain,
            device.to_mlir(),
        )

        results.append(result.tensor)
        # Advance only this device's chain.
        graph.device_chains[device] = out_chain

    return results


def matmul_allreduce(
    inputs: Iterable[TensorValue],
    weights: Iterable[TensorValue],
    signal_buffers: Iterable[BufferValue],
) -> list[TensorValue]:
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
        list(inputs),
        list(weights),
        signal_buffers,
        in_chain,
    )

    Graph.current._update_chain(out_chain)
    return [res.tensor for res in results]
