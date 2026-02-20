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
"""Graph builders for hidden state gather and gather-scatter operations."""

from __future__ import annotations

from collections.abc import Sequence

from max.dtype import DType
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops


def build_gather_scatter_graph(
    device_refs: Sequence[DeviceRef],
    dtype: DType,
    hidden_dim: int,
    max_storage_rows: int,
) -> Graph:
    """Builds a fused gather-scatter graph for hidden states.

    Each device has its own set of inputs with independent symbolic
    dimensions, allowing different batch sizes per device.

    Args:
        device_refs: Device references, one per device.
        dtype: Data type of the hidden states.
        hidden_dim: Hidden dimension size.
        max_storage_rows: Maximum number of rows in storage per device.

    Returns:
        A side-effect-only graph that mutates each device's ``hs_storage``
        in place.
    """
    input_types: list[TensorType | BufferType] = []
    for i, dev in enumerate(device_refs):
        input_types.extend(
            [
                TensorType(dtype, [f"src_rows_{i}", hidden_dim], dev),
                TensorType(DType.int64, [f"num_rows_{i}"], dev),
                BufferType(dtype, [max_storage_rows, hidden_dim], dev),
                TensorType(DType.int64, [f"num_rows_{i}", 1], dev),
            ]
        )

    with Graph(
        "gather_scatter_hidden_states", input_types=input_types
    ) as graph:
        for i in range(len(device_refs)):
            base = i * 4
            src = graph.inputs[base].tensor
            gather_idx = graph.inputs[base + 1].tensor
            hs_storage = graph.inputs[base + 2].buffer
            scatter_idx = graph.inputs[base + 3].tensor

            gathered = ops.gather(src, gather_idx, axis=0)
            current = ops.buffer_load(hs_storage)
            updated = ops.scatter_nd(current, gathered, scatter_idx)
            ops.buffer_store(hs_storage, updated)

        graph.output()
        return graph


def build_gather_graph(
    device_refs: Sequence[DeviceRef],
    dtype: DType,
    hidden_dim: int,
) -> Graph:
    """Builds a graph that gathers rows from a source tensor by index.

    Returns one output tensor per device.

    Args:
        device_refs: Device references, one per device.
        dtype: Data type of the hidden states.
        hidden_dim: Hidden dimension size.

    Returns:
        A graph that returns gathered rows, one output per device.
    """
    input_types: list[TensorType | BufferType] = []
    for i, dev in enumerate(device_refs):
        input_types.extend(
            [
                TensorType(dtype, [f"src_rows_{i}", hidden_dim], dev),
                TensorType(DType.int64, [f"num_gather_{i}"], dev),
            ]
        )

    with Graph("gather_hidden_states", input_types=input_types) as graph:
        outputs = []
        for i in range(len(device_refs)):
            base = i * 2
            src = graph.inputs[base].tensor
            indices = graph.inputs[base + 1].tensor
            gathered = ops.gather(src, indices, axis=0)
            outputs.append(gathered)

        graph.output(*outputs)
        return graph
