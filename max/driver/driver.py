# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""MAX Driver APIs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from json import loads
from typing import Any, Literal, Mapping

from max._driver import Device as _Device
from max._driver import accelerator as _accelerator
from max._driver import accelerator_count as _accelerator_count
from max._driver import cpu_device as _cpu_device


@dataclass
class Device:
    """A hardware device abstraction for computation.

    Provides a unified interface for CPU and accelerator (e.g. GPU) devices.
    """

    # External users should never initialize these fields themselves.
    _device: _Device

    def __str__(self) -> str:
        return str(self._device)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Device) and str(self) == str(other)

    def __ne__(self, other: Any) -> bool:
        return not self == other

    @property
    def is_host(self):
        """Checks whether this device is the CPU (host) device.

        .. code-block:: python

            from max import driver

            device = driver.CPU()
            device.is_host
        """
        return self._device.is_host

    @property
    def stats(self) -> Mapping[str, Any]:
        """Provides real-time utilization data for the device.

        .. code-block:: python

            from max import driver

            device = driver.CPU()
            device.stats
        """
        stat = loads(self._device.stats)
        stat["timestamp"] = datetime.now().isoformat()
        return stat

    @property
    def label(self) -> str:
        """Provides the device type identifier.

        .. code-block:: python

            from max import driver

            device = driver.CPU()
            device.label
        """
        return self._device.label

    @property
    def api(self) -> str:
        """Provides the programming interface used by the device.

        Returns:
            The programming interface identifier used by the device.
            One of:

            - ``cpu`` for host devices.
            - ``cuda`` for NVIDIA GPUs.
            - ``hip`` for AMD GPUs.

        .. code-block:: python

            from max import driver

            device = driver.CPU()
            device.api
        """
        return self._device.api

    @property
    def id(self) -> int:
        """Provides the unique identifier for this device.

        For CPU devices, this represents the NUMA node ID.
        For GPU accelerators, this is the device ID relative to the host.
        Combined with :obj:`label`, this forms a unique device identifier
        (e.g., ``gpu:0`` or ``gpu:1``).

        .. code-block:: python

            from max import driver

            device = driver.CPU()
            device.id
        """
        return self._device.id

    @property
    def is_compatible(self) -> bool:
        """Returns whether this device is compatible with MAX."""
        return self._device.is_compatible

    @classmethod
    def cpu(cls, id: int = -1) -> Device:
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
        return cls(_cpu_device(id))

    @classmethod
    def accelerator(cls, id: int = -1) -> Device:
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
        return cls(_accelerator(id))


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


def accelerator_count() -> int:
    """Returns number of accelerator devices available."""
    return _accelerator_count()


def accelerator_api() -> str:
    """Returns the API used to program the accelerator."""
    if _accelerator_count() > 0:
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
