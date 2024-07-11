# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from typing import Tuple

from max.driver.driver_core import Tensor as _Tensor

from .driver import CPU
from .types import dtype


class Tensor:
    _impl: _Tensor

    def __init__(self, dt: dtype, shape: Tuple[int], device: CPU = CPU()):
        self._impl = _Tensor(int(dt), shape, device._device)

    @property
    def dtype(self) -> dtype:
        return dtype(self._impl.dtype)

    @property
    def shape(self) -> Tuple[int]:
        return self._impl.shape

    @property
    def rank(self) -> int:
        return self._impl.rank
