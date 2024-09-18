# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from __future__ import annotations

from itertools import product
from typing import Any, Generator, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
from max._driver import Tensor as _Tensor
from max.dtype import DType

from .driver import CPU, Device


@runtime_checkable
class DLPackArray(Protocol):
    def __dlpack__(self) -> Any:
        ...

    def __dlpack_device__(self) -> Any:
        ...


class Tensor(DLPackArray):
    """
    Device-resident tensor representation. Allocates memory onto a given device
    with the provided shape and dtype. Tensors can be sliced to provide strided
    views of the underlying memory, but any tensors input into model execution
    must be contiguous. Does not currently support setting items across multiple
    indices, but does support numpy-style slicing.

    :param dtype: DType of tensor
    :param shape: Tuple of positive, non-zero integers denoting the tensor shape.
    :param device: Device to allocate tensor onto.
    """

    # We're storing the dtype explicitly to retain the int-name mapping
    # defined in `types.py`.
    _impl: _Tensor

    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: DType,
        device: Device = CPU(),
        **kwargs,
    ) -> None:
        # Note that we ignore the dtype and shape arguments if we provide an
        # _impl. This is because the dtype and shape will be taken directly from
        # the preconstructed C++ representation.
        if "_impl" in kwargs:
            self._impl = kwargs["_impl"]
        else:
            self._impl = _Tensor(shape, dtype._to(), device._device)

    @classmethod
    def _from_impl(cls, impl: _Tensor) -> Tensor:
        # The error messages are confusing if accidentally passing an incorrect
        # type, so we assert.
        assert isinstance(impl, _Tensor)
        # The dtype and shape arguments are ignored.
        return cls((), DType.uint8, _impl=impl)

    @classmethod
    def zeros(
        cls, shape: Tuple[int, ...], dtype: DType, device: Device = CPU()
    ) -> Tensor:
        """Allocates an tensor with all elements initialized to zero."""
        tensor = cls(shape, dtype, device)
        tensor._impl.zeros()
        return tensor

    @property
    def dtype(self) -> DType:
        """DType of constituent elements in tensor."""
        return DType._from(self._impl.dtype)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of tensor."""
        return self._impl.shape

    @property
    def rank(self) -> int:
        """Tensor rank."""
        return self._impl.rank

    @property
    def device(self) -> Device:
        """Device on which tensor is resident."""
        return Device(self._impl.device)

    @property
    def is_contiguous(self) -> bool:
        """Whether or not tensor is contiguously allocated in memory. Returns
        false if the tensor is a non-contiguous slice.

        Currently, we consider certain situations that are contiguous as
        non-contiguous for the purposes of our engine. These situations include:
        * A tensor with negative steps."""
        return self._impl.is_contiguous

    def _iterate_indices(self) -> Generator[Tuple[int, ...]]:
        index_gen = [range(x) for x in self.shape]
        for idx in product(*index_gen):
            yield idx

    def contiguous(self) -> Tensor:
        """Creates a contiguous copy of the parent tensor."""
        tensor_copy = Tensor(self.shape, self.dtype)
        for idx in self._iterate_indices():
            tensor_copy[idx] = self[idx].item()
        return tensor_copy

    def __repr__(self) -> str:
        return f"max.driver.Tensor({self.dtype}, {self.shape})"

    def __setitem__(self, idx: Tuple[int, ...], value: Any) -> None:
        """Sets an item in the tensor."""
        self._impl.set(idx, value)

    def __getitem__(self, idx: Tuple[int, ...]) -> Tensor:
        """Gets a tensor slice. Supports full numpy-style slicing. Invocations
        using only integer-based indexes will return zero-rank tensors."""
        new_tensor = self._impl.get(idx)
        return Tensor._from_impl(new_tensor)

    def item(self) -> Any:
        """Returns the scalar value at a given location. Currently
        implemented only for zero-rank tensors. The return type is
        converted to a Python built-in type.
        """
        return self._impl.item()

    @property
    def is_host(self) -> bool:
        """Whether or not tensor is host-resident. Returns false for GPU tensors,
        true for CPU tensors."""
        return self._impl.is_host

    def copy_to(self, device: Device) -> Tensor:
        """Copies a tensor to the provided device."""
        copied = self._impl.copy_to(device._device)
        return Tensor(self.shape, self.dtype, device, _impl=copied)

    @classmethod
    def from_numpy(cls, arr: np.ndarray, device: Device = CPU()) -> Tensor:
        """Creates a tensor from a provided numpy array, allocated on the
        provided device.
        If the target device is a CPU, the underlying data will
        not be copied unless the array is noncontiguous. If it is, a contiguous
        copy will first be created.

        If the target device is a GPU, the device will be copied
        to the target."""
        input_arr = arr if arr.flags["C_CONTIGUOUS"] else np.ascontiguousarray(
            arr
        )
        tensor = cls._from_impl(_Tensor(input_arr, CPU()._device))

        if not device.is_host:
            return tensor.copy_to(device)
        return tensor

    def to_numpy(self) -> np.ndarray:
        """Converts the tensor to a numpy array."""
        return np.from_dlpack(self)

    def __dlpack_device__(self) -> Tuple[int, int]:
        """Implements part of the dlpack contract."""
        return self._impl.__dlpack_device__()

    def __dlpack__(self) -> Any:
        """Implements part of the dlpack contract."""
        return self._impl.__dlpack__()

    @classmethod
    def from_dlpack(cls, arr: Any, *, copy: Optional[bool] = None) -> Tensor:
        """Create a tensor from an object implementing the dlpack protocol.

        This usually does not result in a copy, and the producer of the object
        retains ownership of the underlying memory."""
        if isinstance(arr, np.ndarray):
            # TODO(MSDK-976): Older version of numpy don't support exporting
            # read-only arrays, so we copy if we can, and leave a hint if not.
            if copy is None and not arr.flags.writeable:
                copy = True
            if copy:
                arr = arr.copy()

            try:
                return cls._from_impl(_Tensor.from_dlpack(arr))
            except BufferError as e:
                msg = str(e)
                if msg.startswith("Cannot export readonly array"):
                    raise type(e)(
                        msg
                        + " Consider passing `copy = True` to"
                        " `Tensor.from_dlpack`."
                    )
                raise e

        if copy is not None:
            raise ValueError(
                "`Tensor.from_dlpack` support the `copy` flag only for numpy"
                " array inputs"
            )

        return cls._from_impl(_Tensor.from_dlpack(arr))
