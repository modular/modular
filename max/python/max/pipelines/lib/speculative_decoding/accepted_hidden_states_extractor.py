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

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops

if TYPE_CHECKING:
    from typing import Any


def accepted_hidden_states_extractor(device: DeviceRef) -> Graph:
    """Create a graph to extract accepted hidden states for each batch element.

    For EAGLE speculative decoding, after verification we need to extract
    the hidden states corresponding to:
    - The target sampled token (1 per batch element)
    - The accepted draft tokens (variable per batch element)

    Args:
        device: The device to run the graph on.

    Returns:
        A graph that takes hidden_states, logit_offsets, total_range, and
        output_offsets, and returns the concatenated accepted hidden states.

    Example:
        Given 3 batch elements with acceptance patterns:
        - Batch 0: 2 accepted (indices 0, 1)
        - Batch 1: 4 accepted (indices 4, 5, 6, 7)
        - Batch 2: 1 accepted (index 9)

        Inputs:
        - hidden_states: [12, hidden_dim]
        - logit_offsets: [0, 4, 9, 12]
        - total_range: [0, 1, 2, 3, 4, 5, 6]
        - output_offsets: [0, 2, 6]

        Output: hidden_states gathered at indices [0, 1, 4, 5, 6, 7, 9]
    """
    graph_inputs = [
        # Hidden states from target model: [total_tokens, hidden_dim]
        TensorType(
            DType.bfloat16, ["total_tokens", "hidden_dim"], device=device
        ),
        # Logit offsets (cumulative): [batch_size + 1]
        TensorType(DType.int64, ["offsets_len"], device=device),
        # Range [0, 1, 2, ..., total_accepted-1]: [total_accepted]
        TensorType(DType.int64, ["total_accepted"], device=device),
        # Output offsets (cumsum of num_accepted_per_batch): [batch_size]
        TensorType(DType.int64, ["batch_size"], device=device),
    ]

    with Graph(
        "extract_accepted_hidden_states", input_types=graph_inputs
    ) as graph:
        hidden_states, logit_offsets, total_range, output_offsets = graph.inputs

        input_start_offsets = logit_offsets.tensor[:-1]

        zeros = total_range.tensor * 0
        ones = ops.broadcast_to(
            ops.constant(1, DType.int64, device=device),
            output_offsets.tensor.shape,
        )
        markers = ops.scatter(zeros, ones, output_offsets.tensor, axis=0)
        batch_indices = ops.cumsum(markers) - 1

        batch_output_starts = ops.gather(
            output_offsets.tensor, batch_indices, axis=0
        )
        local_offsets = total_range.tensor - batch_output_starts

        batch_input_starts = ops.gather(
            input_start_offsets, batch_indices, axis=0
        )
        gather_indices = batch_input_starts + local_offsets
        accepted_hidden_states = ops.gather(
            hidden_states.tensor, gather_indices, axis=0
        )

        graph.output(accepted_hidden_states)

        return graph


def compute_extractor_inputs(
    first_rejected_tokens: npt.NDArray[np.integer[Any]],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Compute inputs for the accepted_hidden_states_extractor graph.

    Given the first_rejected_tokens array (which indicates how many draft tokens
    were accepted per batch element), compute the total_range and output_offsets
    arrays needed by the extractor graph.
    """
    num_accepted_per_batch = first_rejected_tokens + 1
    total_accepted = int(num_accepted_per_batch.sum())
    total_range = np.arange(total_accepted, dtype=np.int64)
    output_offsets = np.concatenate(
        [[0], np.cumsum(num_accepted_per_batch)[:-1]]
    ).astype(np.int64)
    return total_range, output_offsets
