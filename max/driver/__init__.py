# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from max._driver.core import __version__

from .driver import CPU, CUDA, Device
from .tensor import Tensor

del driver
del tensor
