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
"""Op implementation for shard_and_stack."""

from __future__ import annotations

from collections.abc import Sequence

from ..dim import StaticDim
from ..type import Device, DeviceRef, TensorType
from ..value import TensorValue, TensorValueLike
from .custom import custom
from .validation import (
    assert_on_host,
    assert_same_device,
    assert_same_dtype,
    assert_same_shape,
    assert_valid_axis,
)


def shard_and_stack(
    inputs: Sequence[TensorValueLike],
    devices: Sequence[Device | DeviceRef],
    axis: int = 0,
) -> list[TensorValue]:
    """Shards a list of input tensors along a specified axis, producing multiple outputs.

    This operation takes multiple input tensors, splits each along the specified axis
    into `len(devices)` chunks, and returns one output tensor per device. Each output
    contains the chunks at the corresponding index stacked from all inputs along
    a new dimension 0.

    This is useful for distributing model weights across multiple devices in
    tensor parallel configurations.

    For example, with 2 inputs A and B, axis=0, and 2 devices:
    - Input A shape [10, D], Input B shape [10, D]
    - Output 0: stack([A[0:5], B[0:5]]) -> shape [2, 5, D] on devices[0]
    - Output 1: stack([A[5:10], B[5:10]]) -> shape [2, 5, D] on devices[1]

    With axis=1 and 2 devices:
    - Input A shape [D, 10], Input B shape [D, 10]
    - Output 0: stack([A[:, 0:5], B[:, 0:5]]) -> shape [2, D, 5] on devices[0]
    - Output 1: stack([A[:, 5:10], B[:, 5:10]]) -> shape [2, D, 5] on devices[1]

    Args:
        inputs: A list of symbolic tensors to shard. All tensors must have
            the same shape, dtype, and device.
        devices: Target devices for each output tensor. The number of devices
            determines the number of splits. Each output tensor
            will be placed on the corresponding device. This enables direct
            host-to-device transfer without intermediate CPU storage.
        axis: The axis along which to split each input tensor. Defaults to 0.
            Supports negative indexing (e.g., -1 for last axis).

    Returns:
        A list of `len(devices)` tensors, each with shape
        [num_inputs, D0, ..., Daxis//len(devices), ..., Dn-1] where the input
        shape is [D0, ..., Daxis, ..., Dn-1]. Output i contains the stacked
        chunks at position i from all input tensors, placed on devices[i].

    Raises:
        ValueError: If inputs list is empty, if devices list is empty, if input
            tensors don't have matching shapes, if the dimension size at the
            axis is not evenly divisible by len(devices), or if axis is out of
            bounds.
    """
    if not inputs:
        raise ValueError("Must provide at least one input tensor.")

    if not devices:
        raise ValueError("Must provide at least one device.")

    num_splits = len(devices)

    # Convert inputs to TensorValue
    input_tensors = [TensorValue(x) for x in inputs]

    # Validate all inputs have the same shape
    first_shape = input_tensors[0].shape
    first_dtype = input_tensors[0].dtype
    input_rank = input_tensors[0].rank

    assert_same_device(*input_tensors)
    assert_same_dtype(*input_tensors)
    assert_same_shape(*input_tensors)

    # Validate axis in bounds (before normalization)
    assert_valid_axis(input_tensors[0], axis)

    # Validate inputs are on the host device
    assert_on_host(*input_tensors)

    # Normalize negative axis
    if axis < 0:
        axis = input_rank + axis

    # Validate divisibility (best-effort: only check if dimension is static)
    axis_dim = first_shape[axis]
    if isinstance(axis_dim, StaticDim):
        if int(axis_dim) % num_splits != 0:
            raise ValueError(
                f"Dimension size along axis {axis} ({axis_dim}) must be evenly "
                f"divisible by number of devices ({num_splits})"
            )

    # Calculate output shape
    # Each output has new dimension 0 for stacking, then original shape with
    # axis dimension divided
    chunk_size = axis_dim // num_splits

    # Build output shape: [num_inputs, D0, ..., Daxis//num_splits, ..., Dn-1]
    output_shape = [len(input_tensors), *first_shape]
    # Position axis+1 because we added new dimension 0
    output_shape[axis + 1] = chunk_size

    # Create one output type per device
    output_types = [
        TensorType(
            dtype=first_dtype,
            shape=output_shape,
            device=device,
        )
        for device in devices
    ]

    # Call the custom kernel operation
    results = custom(
        "mo.shard_and_stack",
        input_tensors[0].device,
        input_tensors,
        out_types=output_types,
        parameters={
            "axis": axis,
        },
    )

    return [r.tensor for r in results]
