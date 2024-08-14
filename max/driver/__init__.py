# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from max._driver.core import __version__

from .driver import CPU
from .dtype import DType
from .tensor import Tensor

del driver
del dtype
del tensor
