# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from __future__ import annotations

from max._driver import Device as _Device
from max._driver import cpu_device as _cpu_device
from max._driver import cuda_device as _cuda_device


class Device:
    """Device object. Limited to CUDA and CPU devices for now."""

    _device: _Device

    def __init__(self, _impl: _Device) -> None:
        # Note: This initialization method should never be called by
        # external users.
        self._device = _impl

    def __str__(self) -> str:
        return str(self._device)

    @property
    def is_host(self):
        """Returns whether the device is the CPU."""
        return self._device.is_host

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
