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
"""Shared helpers for parsing device spec inputs."""

from __future__ import annotations

import logging
from typing import Any, Literal, cast

from max.driver import DeviceSpec, scan_available_devices

logger = logging.getLogger("max.pipelines")

DeviceHandleStr = Literal["cpu", "gpu", "default"]
DeviceHandleList = list[int]
DeviceHandle = DeviceHandleStr | DeviceHandleList


def normalize_device_specs_input(value: str | list[int]) -> DeviceHandle:
    """Normalize device input into either a string or list of ints.

    Args:
        value: The value provided as a string (e.g., "cpu", "gpu", "gpu:0,1,2")

    Returns:
        Either "cpu", "gpu", "default", or a non-empty list of GPU IDs as integers.

    Raises:
        ValueError: If the format is invalid
    """
    if isinstance(value, list):
        if not all(isinstance(part, int) for part in value):
            raise ValueError(
                "Device list must contain only integers, e.g. [0, 1]."
            )
        return value

    lowered = value.lower()
    if lowered in {"cpu", "gpu", "default"}:
        return cast(DeviceHandleStr, lowered)
    # By this point, we should only be left with a list of GPU IDs in a
    # gpu:<id1>,<id2> format.
    if not value.startswith("gpu:"):
        raise ValueError(f"Expected 'gpu:<id1>,<id2>' format, got '{value}'")
    # Remove the "gpu:" prefix and split the string by commas to get a list of GPU IDs.
    try:
        gpu_ids = value.removeprefix("gpu:").split(",")
        return [int(part) for part in gpu_ids]
    except ValueError:
        raise ValueError(  # noqa: B904
            f"{value!r} is not a valid device list. Use format 'cpu', 'gpu', or 'gpu:0,1'."
        )


def get_requested_gpu_ids(devices: DeviceHandle) -> list[int]:
    """Helper function to get requested GPU IDs from devices input.

    Args:
        devices: The devices input, either "gpu" or a list of GPU IDs

    Returns:
        List of requested GPU IDs
    """
    if devices == "gpu" or devices == "default":
        return [0]
    if isinstance(devices, list):
        return devices
    return []


def validate_gpu_ids(gpu_ids: list[int], available_gpu_ids: list[int]) -> None:
    """Helper function to validate requested GPU IDs against available ones.

    Args:
        gpu_ids: List of requested GPU IDs
        available_gpu_ids: List of available GPU IDs

    Raises:
        ValueError: If a requested GPU ID is not available
    """
    for gpu_id in gpu_ids:
        if gpu_id not in available_gpu_ids:
            if len(available_gpu_ids) == 0:
                raise ValueError(
                    f"GPU id {gpu_id} requested but no GPUs are available. "
                    f"Use valid device IDs or '--devices=cpu'."
                )
            else:
                raise ValueError(
                    f"GPU id {gpu_id} requested but only GPU IDs {available_gpu_ids} are "
                    f"available. Use valid device IDs or '--devices=cpu'."
                )


def device_specs_from_normalized_device_handle(
    devices: DeviceHandle,
) -> list[DeviceSpec]:
    """Convert normalized device input into validated DeviceSpec objects."""
    available_devices = scan_available_devices()
    available_gpu_ids = [
        device.id for device in available_devices if device.device_type == "gpu"
    ]
    if devices == "cpu":
        return [DeviceSpec.cpu()]
    if devices == "default" and len(available_gpu_ids) == 0:
        logger.info("No GPUs available, falling back to CPU")
        return [DeviceSpec.cpu()]

    requested_gpu_ids = get_requested_gpu_ids(devices=devices)
    validate_gpu_ids(
        gpu_ids=requested_gpu_ids,
        available_gpu_ids=available_gpu_ids,
    )

    return [DeviceSpec.accelerator(id=gpu_id) for gpu_id in requested_gpu_ids]


def _coerce_str(value: str) -> list[DeviceSpec]:
    """Coerce a single string into validated DeviceSpec objects."""
    normalized = normalize_device_specs_input(value)
    if normalized == "cpu":
        return [DeviceSpec.cpu()]
    if normalized == "gpu":
        return [DeviceSpec.accelerator(0)]
    if normalized == "default":
        return scan_available_devices()
    if isinstance(normalized, list):
        return [DeviceSpec.accelerator(id=gpu_id) for gpu_id in normalized]
    raise ValueError(f"Invalid device specs input: {value}")


def coerce_device_specs_input(
    value: Any,
) -> list[DeviceSpec]:
    """Coerce config device specs into validated DeviceSpec objects."""
    if isinstance(value, list):
        if all(isinstance(part, DeviceSpec) for part in value):
            return value
        if all(isinstance(part, int) for part in value):
            return [DeviceSpec.accelerator(id=gpu_id) for gpu_id in value]
        # Handle list of strings from CLI parsers (e.g. cyclopts) which may
        # produce ['gpu:0,1'], ['0', '1'], ['cpu'], etc.
        if all(isinstance(part, str) for part in value):
            # All numeric strings → treat as GPU IDs (e.g. ['0', '1']).
            try:
                gpu_ids = [int(part) for part in value]
                return [DeviceSpec.accelerator(id=gpu_id) for gpu_id in gpu_ids]
            except ValueError:
                pass
            # Single string element → delegate to full string parsing
            # (handles 'gpu:0,1', 'cpu', 'gpu', 'default', etc.).
            if len(value) == 1:
                return _coerce_str(value[0])
        raise ValueError(f"Invalid device specs input: {value!r}")
    if isinstance(value, str):
        return _coerce_str(value)
    raise ValueError(f"Invalid device specs input: {value!r}")
