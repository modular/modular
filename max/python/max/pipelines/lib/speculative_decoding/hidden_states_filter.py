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

from typing import Any

import numpy as np
import numpy.typing as npt
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops


def filter_hidden_states(devices: DeviceRef | list[DeviceRef]) -> Graph:
    """Create a graph to filter hidden states by keeping only specified indices.

    This is used in EAGLE speculative decoding to remove hidden states
    corresponding to sequences that have terminated (hit EOS or max length).

    For data parallel models, this handles multiple devices where each device
    has its own slice of hidden states.

    Args:
        devices: A single device or list of devices.

    Returns:
        A graph with 2 inputs per device (hidden_states, gather_indices) and
        1 output per device (filtered hidden states).
    """
    if not isinstance(devices, list):
        devices = [devices]
    graph_inputs = []
    for i, dev in enumerate(devices):
        graph_inputs.extend(
            [
                # Hidden states: [total_tokens, hidden_dim]
                TensorType(
                    DType.bfloat16,
                    [f"total_tokens_{i}", "hidden_dim"],
                    device=dev,
                ),
                # Indices to keep: [keep_count]
                TensorType(DType.int64, [f"keep_count_{i}"], device=dev),
            ]
        )

    with Graph("filter_hidden_states", input_types=graph_inputs) as graph:
        outputs = []
        for i in range(len(devices)):
            base_idx = i * 2
            hidden_states = graph.inputs[base_idx]
            gather_indices = graph.inputs[base_idx + 1]

            filtered_hidden_states = ops.gather(
                hidden_states.tensor, gather_indices.tensor, axis=0
            )
            outputs.append(filtered_hidden_states)

        graph.output(*outputs)

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


def compute_per_device_filter_indices(
    first_rejected_tokens: npt.NDArray[np.integer[Any]],
    active_context_indices: list[int],
    data_parallel_splits: npt.NDArray[np.int64],
) -> list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]:
    """Compute filter indices per device for DP mode.

    Args:
        first_rejected_tokens: Array of shape [batch_size] where each element
            indicates the index of the first rejected token for that batch element.
        active_context_indices: List of global indices of contexts that are still active.
        data_parallel_splits: Array of shape [num_devices + 1] containing the
            cumulative batch indices for each device.

    Returns:
        A list of tuples, one per device, each containing (keep_indices, offsets).
    """
    num_devices = len(data_parallel_splits) - 1
    per_device_results = []

    for dev_idx in range(num_devices):
        start_batch = int(data_parallel_splits[dev_idx])
        end_batch = int(data_parallel_splits[dev_idx + 1])
        device_first_rejected = first_rejected_tokens[start_batch:end_batch]
        # Convert global indices to local indices for this device
        device_active_indices = [
            idx - start_batch
            for idx in active_context_indices
            if start_batch <= idx < end_batch
        ]
        keep_indices, offsets = compute_filter_indices(
            device_first_rejected, device_active_indices
        )
        per_device_results.append((keep_indices, offsets))

    return per_device_results


def compute_local_active_indices(
    active_context_indices: list[int],
    data_parallel_splits: npt.NDArray[np.int64],
) -> list[npt.NDArray[np.int64]]:
    """Compute local active indices per device for DP mode.

    Args:
        active_context_indices: List of global indices of contexts that are still active.
        data_parallel_splits: Array of shape [num_devices + 1] containing the
            cumulative batch indices for each device.

    Returns:
        A list of numpy arrays, one per device, each containing the local active
        indices for that device.
    """
    num_devices = len(data_parallel_splits) - 1
    per_device_indices = []

    for dev_idx in range(num_devices):
        start_batch = int(data_parallel_splits[dev_idx])
        end_batch = int(data_parallel_splits[dev_idx + 1])
        local_indices = [
            idx - start_batch
            for idx in active_context_indices
            if start_batch <= idx < end_batch
        ]
        per_device_indices.append(np.array(local_indices, dtype=np.int64))

    return per_device_indices
