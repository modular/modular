# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .driver import CPU, __version__
from .dtype import DType
from .tensor import Tensor

del driver
del dtype
del tensor
