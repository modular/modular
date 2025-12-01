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
"""MAX Driver APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# The Device class is not used in this file, but is used by others that import
# this file. Ruff will try to remove it, so ignore that Ruff check on this line.
from max._core.driver import (  # noqa: F401
    CPU,
    Accelerator,
    Device,
    DeviceStream,
    accelerator_count,
)


def accelerator_api() -> str:
    """Returns the API used to program the accelerator."""
    if accelerator_count() > 0:
        return Accelerator().api
    return CPU().api


def accelerator_architecture_name() -> str:
    """Returns the architecture name of the accelerator device."""
    if accelerator_count() > 0:
        return Accelerator().architecture_name
    return CPU().architecture_name


@dataclass(frozen=True)
class DeviceSpec:
    """Specification for a device, containing its ID and type.

    This class provides a way to specify device parameters like ID and type (CPU/GPU)
    for creating Device instances.
    """

    id: int
    """Provided id for this device."""

    device_type: Literal["cpu", "gpu"] = "cpu"
    """Type of specified device."""

    def __post_init__(self) -> None:
        if self.device_type == "gpu" and self.id < 0:
            raise ValueError(
                f"id provided {self.id} for accelerator must always be greater than 0"
            )

    @staticmethod
    def cpu(id: int = -1):  # noqa: ANN205
        """Creates a CPU device specification."""
        return DeviceSpec(id, "cpu")

    @staticmethod
    def accelerator(id: int = 0):  # noqa: ANN205
        """Creates an accelerator (GPU) device specification."""
        return DeviceSpec(id, "gpu")


def load_devices(device_specs: list[DeviceSpec]) -> list[Device]:
    """Initialize and return a list of devices, given a list of device specs."""
    num_devices_available = accelerator_count()
    devices: list[Device] = []
    for device_spec in device_specs:
        if device_spec.device_type == "cpu":
            devices.append(CPU(device_spec.id))
        else:
            if device_spec.id >= num_devices_available:
                if num_devices_available == 0:
                    reason = "no devices were found."
                else:
                    reason = f"only found {num_devices_available} devices."
                raise ValueError(
                    f"Device {device_spec.id} was requested but {reason}"
                )

            devices.append(Accelerator(device_spec.id))

    return devices


def scan_available_devices() -> list[DeviceSpec]:
    """Returns all accelerators if available, else return cpu."""
    accel_count = accelerator_count()
    if accel_count == 0:
        return [DeviceSpec.cpu()]
    else:
        return [DeviceSpec.accelerator(i) for i in range(accel_count)]


def devices_exist(devices: list[DeviceSpec]) -> bool:
    """Identify if devices exist."""
    available_devices = scan_available_devices()
    for device in devices:
        if device.device_type != "cpu" and device not in available_devices:
            return False

    return True


def calculate_virtual_device_count(*device_spec_lists: list[DeviceSpec]) -> int:
    """Calculate the minimum virtual device count needed for the given device specs.

    Args:
        *device_spec_lists: One or more lists of DeviceSpec objects (e.g., main devices
            and draft devices)

    Returns:
        The minimum number of virtual devices needed (max GPU ID + 1), or 1 if no GPUs
    """
    max_gpu_id = -1
    for device_specs in device_spec_lists:
        for device_spec in device_specs:
            if device_spec.device_type == "gpu":
                max_gpu_id = max(max_gpu_id, device_spec.id)

    return max(1, max_gpu_id + 1)


def calculate_virtual_device_count_from_cli(
    *device_inputs: str | list[int],
) -> int:
    """Calculate virtual device count from raw CLI inputs (before parsing).

    This helper works with the raw device input strings or lists before they're
    parsed into DeviceSpec objects. Used when virtual device mode needs to be
    enabled before device validation occurs.

    Args:
        *device_inputs: One or more raw device inputs - either strings like "gpu:0,1,2"
            or lists of integers like [0, 1, 2]

    Returns:
        The minimum number of virtual devices needed (max GPU ID + 1), or 1 if no GPUs
    """
    max_gpu_id = -1
    for device_input in device_inputs:
        if isinstance(device_input, list):
            # Handle list of GPU IDs like [0, 1, 2]
            if len(device_input) > 0:
                max_gpu_id = max(max_gpu_id, max(device_input))
        elif device_input in ("gpu", "default"):
            # Handle "gpu" or "default" which means GPU 0
            max_gpu_id = max(max_gpu_id, 0)
        # Other string formats (like "gpu:0,1,2") are handled by the DevicesOptionType parser

    return max(1, max_gpu_id + 1)
