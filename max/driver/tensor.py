# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from typing import Any, Tuple

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

    def __init__(self, dt: dtype, shape: Tuple[int], device: CPU = CPU()):
        self.dt = dt
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

    def __setitem__(self, idx: Tuple[int, ...], value: Any):
        """Sets an item in the tensor"""
        indices = self._canonicalize_indices(idx)
        self._impl.set_item(indices, value)

    def __getitem__(self, idx: Tuple[int, ...]) -> Any:
        """Gets an item from the tensor"""
        indices = self._canonicalize_indices(idx)
        return self._impl.get_item(indices)

    def _canonicalize_indices(
        self, indexes: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        """Transform arbitrary indexes into unsigned integers usable directly
        by the underlying C++ methods."""
        # Validate that indices are integers
        if not all([isinstance(elt, int) for elt in indexes]):
            raise TypeError("all indices should be integers")

        canonicalized = ()
        shape = self.shape
        for idx, elt in enumerate(indexes):
            # A negative index N corresponds to "N + 1" elements away from the
            # last element in the corresponding dimension.
            adjusted_idx = shape[idx] + elt if elt < 0 else elt
            if not (0 <= adjusted_idx < shape[idx]):
                raise IndexError(
                    "indices should not exceed the bounds of the tensor"
                )
            canonicalized = canonicalized + (adjusted_idx,)

        return canonicalized
