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


def accepted_hidden_states_extractor(
    devices: DeviceRef | list[DeviceRef],
) -> Graph:
    """Create a graph to extract accepted hidden states for each batch element.

    For EAGLE speculative decoding, after verification we need to extract
    the hidden states corresponding to:
    - The target sampled token (1 per batch element)
    - The accepted draft tokens (variable per batch element)
    """
    if not isinstance(devices, list):
        devices = [devices]
    graph_inputs = []
    for i, dev in enumerate(devices):
        graph_inputs.extend(
            [
                # Hidden states from target model: [total_tokens, hidden_dim]
                TensorType(
                    DType.bfloat16,
                    [f"total_tokens_{i}", "hidden_dim"],
                    device=dev,
                ),
                # Logit offsets (cumulative): [batch_size + 1]
                TensorType(DType.int64, [f"offsets_len_{i}"], device=dev),
                # Range [0, 1, 2, ..., total_accepted-1]: [total_accepted]
                TensorType(DType.int64, [f"total_accepted_{i}"], device=dev),
                # Output offsets (cumsum of num_accepted_per_batch): [batch_size]
                TensorType(DType.int64, [f"batch_size_{i}"], device=dev),
            ]
        )

    with Graph(
        "extract_accepted_hidden_states", input_types=graph_inputs
    ) as graph:
        outputs = []
        for i, dev in enumerate(devices):
            base_idx = i * 4
            hidden_states = graph.inputs[base_idx]
            logit_offsets = graph.inputs[base_idx + 1]
            total_range = graph.inputs[base_idx + 2]
            output_offsets = graph.inputs[base_idx + 3]

            input_start_offsets = logit_offsets.tensor[:-1]

            zeros = total_range.tensor * 0
            ones = ops.broadcast_to(
                ops.constant(1, DType.int64, device=dev),
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
            outputs.append(accepted_hidden_states)

        graph.output(*outputs)

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


def compute_per_device_extractor_inputs(
    first_rejected_tokens: npt.NDArray[np.integer[Any]],
    data_parallel_splits: npt.NDArray[np.int64],
) -> list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]:
    """Compute extractor inputs per device for DP mode.

    Args:
        first_rejected_tokens: Array of shape [batch_size] where each element
            indicates the index of the first rejected token for that batch element.
        data_parallel_splits: Array of shape [num_devices + 1] containing the
            cumulative batch indices for each device.

    Returns:
        A list of tuples, one per device, each containing (total_range, output_offsets).
    """
    num_devices = len(data_parallel_splits) - 1
    per_device_inputs = []

    for dev_idx in range(num_devices):
        start_batch = int(data_parallel_splits[dev_idx])
        end_batch = int(data_parallel_splits[dev_idx + 1])
        device_first_rejected = first_rejected_tokens[start_batch:end_batch]
        total_range, output_offsets = compute_extractor_inputs(
            device_first_rejected
        )
        per_device_inputs.append((total_range, output_offsets))

    return per_device_inputs


def call_per_device_graph(
    graph_models: list[Any],
    per_device_inputs: list[list[Any]],
) -> list[Any]:
    """Call per-device graphs, skipping devices with empty inputs."""
    result_list: list[Any] = []

    for dev_idx, device_inputs in enumerate(per_device_inputs):
        hidden_states = device_inputs[0]
        if hidden_states.shape[0] == 0:
            # Skip empty hidden states, return original buffer
            result_list.append(hidden_states)
        else:
            # Call this device's graph
            (result,) = graph_models[dev_idx](*device_inputs)
            result_list.append(result)

    return result_list
