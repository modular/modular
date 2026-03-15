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
"""Graph builders for hidden state gather operations."""

from __future__ import annotations

from collections.abc import Sequence

from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kernels import extract_accepted_hs


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
    input_types: list[TensorType] = []
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


def build_extract_hs_graph(
    device_ref: DeviceRef,
    dtype: DType,
    hidden_dim: int,
) -> Graph:
    """Builds a graph wrapping the extract_accepted_hs custom op.

    Args:
        device_ref: Device reference for the graph.
        dtype: Data type of the hidden states.
        hidden_dim: Hidden dimension size.

    Returns:
        A graph that extracts accepted hidden states.
    """
    input_types = [
        TensorType(dtype, ["total_hs", hidden_dim], device=device_ref),
        TensorType(DType.uint32, ["offsets_len"], device=device_ref),
        TensorType(DType.int64, ["local_batch"], device=device_ref),
        TensorType(DType.int64, [1], device=DeviceRef.CPU()),
    ]
    with Graph("extract_accepted_hs", input_types=input_types) as graph:
        hs, hs_offsets, first_rejected, num_draft_tokens = graph.inputs
        accepted_hs, accepted_offsets = extract_accepted_hs(
            hs.tensor,
            hs_offsets.tensor,
            first_rejected.tensor,
            num_draft_tokens.tensor,
        )
        graph.output(accepted_hs, accepted_offsets)
        return graph
