# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from __future__ import annotations

from typing import Any, Tuple

from max._driver import Tensor as _Tensor
from max.dtype import DType

from .driver import CPU


class Tensor:
    """
    Device-resident tensor representation. Allocates memory onto a given device
    with the provided shape and dtype. Tensors can be sliced to provide strided
    views of the underlying memory, but any tensors input into model execution
    must be contiguous. Does not currently support setting items across multiple
    indices, but does support numpy-style slicing.

    :param dt: DType of tensor
    :param shape: Tuple of positive, non-zero integers denoting the tensor shape.
    :param device: Device to allocate tensor onto.
    """

    # We're storing the dtype explicitly to retain the int-name mapping
    # defined in `types.py`.
    _impl: _Tensor

    def __init__(
        self,
        shape: Tuple[int, ...],
        dt: DType,
        device: CPU = CPU(),
        **kwargs,
    ) -> None:
        # Note that we ignore the dtype and shape arguments if we provide an
        # _impl. This is because the dtype and shape will be taken directly from
        # the preconstructed C++ representation.
        if "_impl" in kwargs:
            self._impl = kwargs["_impl"]
        else:
            self._impl = _Tensor(shape, dt._to(), device._device)

    @property
    def dtype(self) -> DType:
        """DType of constituent elements in tensor"""
        return DType._from(self._impl.dtype)

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
        """Gets a tensor slice. Supports full numpy-style slicing. Invocations
        using only integer-based indexes will return zero-rank tensors."""
        new_tensor = self._impl.get(idx)
        # The shape and dtype we pass into this constructor will be ignored.
        return Tensor((), DType.unknown, _impl=new_tensor)

    def item(self) -> Any:
        """Returns the scalar value at a given location. Currently
        implemented only for zero-rank tensors. The return type is
        converted to a Python built-in type.
        """
        return self._impl.item()
