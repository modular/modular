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

from max.dtype import DType
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops


def build_gather_scatter_graph(
    device: DeviceRef,
    dtype: DType,
    hidden_dim: int,
    max_storage_rows: int,
) -> Graph:
    """Builds a fused gather-scatter graph for hidden state operations.

    Gathers rows from a source tensor and scatters them into storage in a
    single graph.

    Args:
        device: Device reference for the graph.
        dtype: Data type of the hidden states.
        hidden_dim: Hidden dimension size.
        max_storage_rows: Maximum number of rows in storage.

    Returns:
        A side-effect-only graph that mutates ``hs_storage`` in place.
    """
    input_types: list[TensorType | BufferType] = [
        # src: source tensor [src_rows, hidden_dim]
        TensorType(dtype, ["src_rows", hidden_dim], device),
        # gather_indices: which rows to gather from src [num_rows]
        TensorType(DType.int64, ["num_rows"], device),
        # hs_storage: mutable buffer [max_storage_rows, hidden_dim]
        BufferType(dtype, [max_storage_rows, hidden_dim], device),
        # scatter_indices: where to scatter into storage [num_rows, 1]
        TensorType(DType.int64, ["num_rows", 1], device),
    ]

    with Graph(
        "gather_scatter_hidden_states",
        input_types=input_types,
    ) as graph:
        src = graph.inputs[0].tensor
        gather_idx = graph.inputs[1].tensor
        hs_storage = graph.inputs[2].buffer
        scatter_idx = graph.inputs[3].tensor

        gathered = ops.gather(src, gather_idx, axis=0)
        current = ops.buffer_load(hs_storage)
        updated = ops.scatter_nd(current, gathered, scatter_idx)
        ops.buffer_store(hs_storage, updated)

        graph.output()

        return graph


def build_gather_graph(
    device: DeviceRef,
    dtype: DType,
    hidden_dim: int,
) -> Graph:
    """Builds a graph that gathers rows from a source tensor by index.

    Args:
        device: Device reference for the graph.
        dtype: Data type of the hidden states.
        hidden_dim: Hidden dimension size.

    Returns:
        A graph that returns gathered rows.
    """
    input_types: list[TensorType | BufferType] = [
        # src: source tensor [src_rows, hidden_dim]
        TensorType(dtype, ["src_rows", hidden_dim], device),
        # indices: row indices to gather [num_gather]
        TensorType(DType.int64, ["num_gather"], device),
    ]

    with Graph(
        "gather_hidden_states",
        input_types=input_types,
    ) as graph:
        src = graph.inputs[0].tensor
        indices = graph.inputs[1].tensor

        gathered = ops.gather(src, indices, axis=0)

        graph.output(gathered)

        return graph
