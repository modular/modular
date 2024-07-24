# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from __future__ import annotations

from typing import Any, Optional, Tuple

from max.driver.driver_core import Tensor as _Tensor

from .driver import CPU
from .types import dtype


class Tensor:
    """
    Device-resident tensor representation. Allocates contiguous memory on provided
    device to support a tensor of a given dtype and shape.
    :param dt: DType of tensor
    :param shape: Tuple of positive, non-zero integers denoting the tensor shape.
    :param device: Device to allocate tensor onto.
    """

    # We're storing the dtype explicitly to retain the int-name mapping
    # defined in `types.py`.
    dt: dtype
    _impl: _Tensor

    def __init__(
        self,
        dt: dtype,
        shape: Tuple[int, ...] = (),
        device: CPU = CPU(),
        **kwargs,
    ) -> None:
        self.dt = dt
        if "_impl" in kwargs:
            self._impl = kwargs["_impl"]
        else:
            self._impl = _Tensor(dt.id, shape, device._device)

    @property
    def dtype(self) -> dtype:
        """DType of constituent elements in tensor"""
        return self.dt

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of tensor"""
        return self._impl.shape

    @property
    def rank(self) -> int:
        """Tensor rank"""
        return self._impl.rank

    def __repr__(self) -> str:
        return f"max.driver.Tensor({self.dtype}, {self.shape})"

    def __setitem__(self, idx: Tuple[int, ...], value: Any) -> None:
        """Sets an item in the tensor"""
        self._impl.set(idx, value)

    def __getitem__(self, idx: Tuple[int, ...]) -> Tensor:
        """Gets an item from the tensor"""
        new_tensor = self._impl.get(idx)
        return Tensor(self.dt, _impl=new_tensor)

    def item(self) -> Any:
        """Returns the scalar value at a given location. Currently
        implemented only for zero-rank tensors. The return type is
        converted to a Python built-in type.
        """
        return self._impl.item()
