# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from json import loads
from typing import Any, Literal, Mapping

from max._driver import Device as _Device
from max._driver import cpu_device as _cpu_device
from max._driver import cuda_device as _cuda_device
from max._driver import accelerator_count as _accelerator_count


@dataclass
class Device:
    """Device object. Limited to CUDA and CPU devices for now."""

    # Note: External users should never initialize these fields themselves.
    _device: _Device

    def __str__(self) -> str:
        return str(self._device)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Device) and str(self) == str(other)

    def __ne__(self, other: Any) -> bool:
        return not self == other

    @property
    def is_host(self):
        """Returns whether the device is the CPU."""
        return self._device.is_host

    @property
    def stats(self) -> Mapping[str, Any]:
        """Returns utilization data for the device."""
        stat = loads(self._device.stats)
        stat["timestamp"] = datetime.now().isoformat()
        return stat

    @property
    def label(self) -> str:
        """Returns device label."""
        return self._device.label

    @property
    def id(self) -> int:
        """Returns device id."""
        return self._device.id

    @classmethod
    def cpu(cls, id: int = -1) -> Device:
        """Creates a CPU device with the provided numa id."""
        return cls(_cpu_device(id))

    @classmethod
    def cuda(cls, id: int = -1) -> Device:
        """Creates a CUDA device with the provided id."""
        return cls(_cuda_device(id))


def CPU(id: int = -1) -> Device:
    """Creates a CPU device with the provided numa id."""
    return Device.cpu(id)


def CUDA(id: int = -1) -> Device:
    """Creates a CUDA device with the provided id."""
    return Device.cuda(id)


def accelerator_count() -> int:
    """Returns number of GPU devices available."""
    return _accelerator_count()


@dataclass(frozen=True)
class DeviceSpec:
    id: int
    """Provided id for this device."""

    device_type: Literal["cpu", "gpu"] = "cpu"
    """Type of specified device."""

    @staticmethod
    def cpu(id: int = -1):
        return DeviceSpec(id, "cpu")

    @staticmethod
    def cuda(id: int = -1):
        return DeviceSpec(id, "gpu")
