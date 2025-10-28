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
from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops


def split_batch(
    devices: list[DeviceRef],
    input: TensorValue,
    input_row_offsets: TensorValue,
    data_parallel_splits: TensorValue,
) -> tuple[list[TensorValue], list[TensorValue]]:
    """Split a ragged input batch into data parallel batches.

    See `split_batch_replicated` for a version of this method that takes
    replicated inputs and input_row_offsets for each device.

    Example:
        devices = [device_1, device_2]
        input = [seq_1, seq_2, seq_3, seq_4]
        input_row_offsets = [0, offset_1, offset_2, offset_3, offset_4]
        data_parallel_splits = [0, 2, 4]

    Outputs:
        split_input = [seq_1, seq_2], [seq_3, seq_4]
        split_offsets = [0, offset_1, offset_2], [0, new_offset_3, new_offset_4]

    After being split, the outputs will be placed on the devices specified in
    `devices`.

    The size of data_parallel_splits must be equal to the number of devices + 1.

    Args:
        input: Input tensor of shape [total_seq_len, ...].
        input_row_offsets: Row offsets tensor indicating batch boundaries.
        data_parallel_splits: Tensor containing batch splits for each device.
            Must be located on CPU.

    Returns:
        Tuple of (split_input, split_offsets)
        where split_input and split_offsets are lists of tensors, one per device
    """
    cpu = DeviceRef.CPU()
    num_devices = len(devices)

    if num_devices == 0:
        raise ValueError("Expected at least one device")
    if num_devices == 1:
        # No splitting needed for single device
        return [input], [input_row_offsets]

    # Create copy of input_row_offsets that will be used as slice indices.
    input_row_offsets_int64 = input_row_offsets.cast(DType.int64).to(cpu)
    split_input = []
    split_offsets = []
    for i, device in enumerate(devices):
        start_offset = input_row_offsets[data_parallel_splits[i]]
        start_offset_i64 = input_row_offsets_int64[data_parallel_splits[i]]
        end_offset_i64 = input_row_offsets_int64[data_parallel_splits[i + 1]]
        token_slice = ops.slice_tensor(
            input,
            [
                (
                    slice(start_offset_i64, end_offset_i64),
                    f"input_split_{i}",
                ),
                ...,
            ],
        )
        if i + 1 >= num_devices:
            end_idx = None
        else:
            end_idx = data_parallel_splits[i + 1] + 1

        offsets_slice = (
            ops.slice_tensor(
                input_row_offsets,
                [
                    (
                        slice(data_parallel_splits[i], end_idx),
                        f"offset_split_{i}",
                    )
                ],
            )
            - start_offset
        )
        split_input.append(token_slice.to(device))
        split_offsets.append(offsets_slice.to(device))

    return split_input, split_offsets


def split_batch_replicated(
    devices: list[DeviceRef],
    input: list[TensorValue],
    input_row_offsets: list[TensorValue],
    input_row_offsets_int64: TensorValue,
    data_parallel_splits: TensorValue,
) -> tuple[list[TensorValue], list[TensorValue]]:
    """Split a ragged token batch into data parallel batches.

    This version takes a list of input and input_row_offsets replicated on
    each device. Also see `split_input` for a version of this method that takes
    a single ragged token batch.

    Example:
        devices = [device_1, device_2]
        input = [seq_1, seq_2, seq_3, seq_4] (replicated for each device)
        input_row_offsets = [0, offset_1, offset_2, offset_3, offset_4]
            (replicated for each device)
        data_parallel_splits = [0, 2, 4]

    Outputs:
        split_input = [seq_1, seq_2], [seq_3, seq_4]
        split_offsets = [0, offset_1, offset_2], [0, new_offset_3, new_offset_4]

    After being split, the outputs will be placed on the devices specified in
    `devices`.

    The size of data_parallel_splits must be equal to the number of devices + 1.

    Args:
        devices: List of devices to split the input on.
        input: List of input token tensors of shape [total_seq_len]. The
            list must be the same length as the number of devices.
        input_row_offsets: Row offsets tensor indicating batch boundaries.
            The list must be the same length as the number of devices.
        input_row_offsets_int64: Row offsets tensor indicating batch boundaries.
            Must be located on CPU.
        data_parallel_splits: Tensor containing batch splits for each device.
            Must be located on CPU.

    Returns:
        Tuple of (split_input, split_offsets)
        where split_input and split_offsets are lists of tensors, one per device
    """
    cpu = DeviceRef.CPU()
    num_devices = len(devices)

    if num_devices == 0:
        raise ValueError("Expected at least one device")
    if num_devices == 1:
        # No splitting needed for single device
        return list(input), list(input_row_offsets)

    # Check that input and input_row_offsets are replicated for each device.
    if len(input) != num_devices:
        raise ValueError(f"Expected {num_devices} input, got {len(input)}")
    if len(input_row_offsets) != num_devices:
        raise ValueError(
            f"Expected {num_devices} input_row_offsets, got {len(input_row_offsets)}"
        )

    for i, device in enumerate(devices):
        if input[i].device != device:
            raise ValueError(
                f"Expected token {i} to be located on device {device}, got {input[i].device}"
            )
        if input_row_offsets[i].device != device:
            raise ValueError(
                f"Expected input_row_offset {i} to be located on device {device}, got {input_row_offsets[i].device}"
            )

    split_input = []
    split_offsets = []
    for i in range(num_devices):
        # Offsets must be on CPU to be used as a slice index.
        start_offset = input_row_offsets[i][data_parallel_splits[i]]
        start_offset_i64 = input_row_offsets_int64[data_parallel_splits[i]]
        end_offset_i64 = input_row_offsets_int64[data_parallel_splits[i + 1]]
        token_slice = ops.slice_tensor(
            input[i],
            [
                (
                    slice(start_offset_i64.to(cpu), end_offset_i64.to(cpu)),
                    f"input_split_{i}",
                ),
                ...,
            ],
        )
        if i + 1 >= num_devices:
            end_idx = None
        else:
            end_idx = data_parallel_splits[i + 1] + 1

        offsets_slice = (
            ops.slice_tensor(
                input_row_offsets[i],
                [
                    (
                        slice(data_parallel_splits[i], end_idx),
                        f"offset_split_{i}",
                    )
                ],
            )
            - start_offset
        )
        split_input.append(token_slice)
        split_offsets.append(offsets_slice)

    return split_input, split_offsets
