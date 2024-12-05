# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from max._driver.core import __version__

from .driver import CPU, CUDA, Device, DeviceSpec, accelerator_count
from .tensor import DLPackArray, MemMapTensor, Tensor

del driver  # type: ignore
del tensor  # type: ignore

__all__ = [
    "CPU",
    "CUDA",
    "Device",
    "DeviceSpec",
    "DLPackArray",
    "MemMapTensor",
    "Tensor",
]
