# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""MAX Driver APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from max._core.driver import Device, accelerator_count


def CPU(id: int = -1) -> Device:
    """Creates a CPU device for the specified NUMA node.

    .. code-block:: python

        from max import driver

        # Create default CPU device
        device = driver.CPU()

        # Or specify NUMA node id if using NUMA architecture
        device = driver.CPU(id=0)  # First NUMA node
        device = driver.CPU(id=1)  # Second NUMA node

        # Get device id
        device_id = device.id
    """
    return Device.cpu(id)


def Accelerator(id: int = -1) -> Device:
    """Creates an accelerator device with the specified ID.

    Provides access to GPU or other hardware accelerators in the system.

    .. code-block:: python

        from max import driver

        # Create default accelerator (usually first available GPU)
        device = driver.Accelerator()

        # Or specify GPU id
        device = driver.Accelerator(id=0)  # First GPU
        device = driver.Accelerator(id=1)  # Second GPU

        # Get device id
        device_id = device.id
    """
    return Device.accelerator(id)


def accelerator_api() -> str:
    """Returns the API used to program the accelerator."""
    if accelerator_count() > 0:
        return Accelerator().api
    return CPU().api


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

    @staticmethod
    def cpu(id: int = -1):
        """Creates a CPU device specification."""
        return DeviceSpec(id, "cpu")

    @staticmethod
    def accelerator(id: int = -1):
        """Creates an accelerator (GPU) device specification."""
        return DeviceSpec(id, "gpu")
