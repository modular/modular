# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from abc import ABC, abstractmethod

from max._driver import Device as _Device
from max._driver import cpu_device as _cpu_device
from max._driver import cuda_device as _cuda_device


class Device(ABC):
    """Abstract device class. Base class for all device objects."""

    _device: _Device

    @abstractmethod
    def __init__(self, id: int = -1) -> None:
        ...

    def __str__(self) -> str:
        return str(self._device)

    @property
    def is_host(self) -> bool:
        return False


class CPU(Device):
    """This represents an instance of a logical cpu device."""

    _device: _Device

    def __init__(self, id: int = -1):
        # FIXME: Use device descriptors
        self._device = _cpu_device(id)

    @property
    def is_host(self) -> bool:
        return True


class CUDA(Device):
    """This represents a CUDA GPU device."""

    _device: _Device

    def __init__(self, id: int = -1):
        self._device = _cuda_device(id)
