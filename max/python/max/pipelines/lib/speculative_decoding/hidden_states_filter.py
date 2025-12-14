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

from typing import Any

import numpy as np
import numpy.typing as npt
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops


def filter_hidden_states(device: DeviceRef) -> Graph:
    """Create a graph to filter hidden states by keeping only specified indices.

    This is used in EAGLE speculative decoding to remove hidden states
    corresponding to sequences that have terminated (hit EOS or max length).

    Args:
        device: The device to run the graph on.

    Returns:
        A graph that takes hidden_states and gather_indices, and returns
        the filtered hidden states containing only the specified indices.
    """
    graph_inputs = [
        # Hidden states: [total_tokens, hidden_dim]
        TensorType(
            DType.bfloat16, ["total_tokens", "hidden_dim"], device=device
        ),
        # Indices to keep: [keep_count]
        TensorType(DType.int64, ["keep_count"], device=device),
    ]

    with Graph("filter_hidden_states", input_types=graph_inputs) as graph:
        hidden_states, gather_indices = graph.inputs

        filtered_hidden_states = ops.gather(
            hidden_states.tensor, gather_indices.tensor, axis=0
        )

        graph.output(filtered_hidden_states)

        return graph


def compute_filter_indices(
    first_rejected_tokens: npt.NDArray[np.integer[Any]],
    active_context_indices: list[int],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Compute indices to keep for the filter_hidden_states graph.

    Given the first_rejected_tokens array (which indicates how many draft tokens
    were accepted per batch element), compute the indices to keep and the offsets.

    Args:
        first_rejected_tokens: Array of shape [batch_size] where each element
            indicates the index of the first rejected token for that batch element.
        active_context_indices: List of indices of contexts that are still active.

    Returns:
        A tuple of (keep_indices, offsets):
        - keep_indices: Array of indices to keep from the hidden states
        - offsets: Cumulative offsets for each batch element [batch_size + 1]
    """
    offsets = np.concatenate(
        [[0], np.cumsum(first_rejected_tokens + 1)]
    ).astype(np.int64)
    keep_indices: list[int] = []
    for idx in active_context_indices:
        keep_indices.extend(range(int(offsets[idx]), int(offsets[idx + 1])))

    return np.array(keep_indices, dtype=np.int64), offsets
