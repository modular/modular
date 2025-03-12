# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from max._core.driver import __version__  # type: ignore

from .driver import (
    CPU,
    Accelerator,
    Device,
    DeviceSpec,
    accelerator_api,
    accelerator_count,
    devices_exist,
    load_devices,
    scan_available_devices,
)
from .tensor import DLPackArray, MemMapTensor, Tensor

del driver  # type: ignore
del tensor  # type: ignore

__all__ = [
    "CPU",
    "Accelerator",
    "Device",
    "DeviceSpec",
    "DLPackArray",
    "MemMapTensor",
    "Tensor",
    "accelerator_api",
    "load_devices",
    "scan_available_devices",
    "devices_exist",
]
