# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
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
            msg = f"id provided {self.id} for accelerator must always be greater than 0"
            raise ValueError(msg)

    @staticmethod
    def cpu(id: int = -1):
        """Creates a CPU device specification."""
        return DeviceSpec(id, "cpu")

    @staticmethod
    def accelerator(id: int = 0):
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
                msg = f"Device {device_spec.id} was requested but "
                if num_devices_available == 0:
                    msg += "no devices were found."
                else:
                    msg += f"only found {num_devices_available} devices."
                raise ValueError(msg)

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
