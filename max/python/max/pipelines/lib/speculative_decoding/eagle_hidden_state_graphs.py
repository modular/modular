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
