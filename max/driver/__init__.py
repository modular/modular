# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .driver import CPU, __version__
from .tensor import Tensor
from .types import dtype

del driver
del types
del tensor
