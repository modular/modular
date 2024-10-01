# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from max._driver.core import __version__

from .driver import CPU, CUDA, Device
from .tensor import DLPackArray, MemMapTensor, Tensor

del driver  # type: ignore
del tensor  # type: ignore
