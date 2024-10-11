# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from __future__ import annotations

import struct
from itertools import product
from mmap import mmap
from os import PathLike
from typing import (
    Any,
    Generator,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)

import numpy as np
from max._driver import Tensor as _Tensor
from max.dtype import DType

from .driver import CPU, Device

_IdxElType = Union[int, slice]
IndexType = Union[Sequence[_IdxElType], _IdxElType]
ShapeType = Sequence[int]


@runtime_checkable
class DLPackArray(Protocol):
    def __dlpack__(self) -> Any:
        ...

    def __dlpack_device__(self) -> Any:
        ...


_T = TypeVar("_T", bound="Tensor")


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
        shape: ShapeType,
        dtype: DType,
        device: Device = CPU(),
    ) -> None:
        self._impl = _Tensor(shape, dtype._to(), device._device)

    @classmethod
    def _from_impl(cls: Type[_T], impl: _Tensor) -> _T:
        # TODO: use typing.Self instead of TypeVar when we are on Python 3.11+.
        # The error messages are confusing if accidentally passing an incorrect
        # type, so we assert.
        assert isinstance(impl, _Tensor)
        # The dtype and shape arguments are ignored.
        tensor = cls.__new__(cls)
        tensor._impl = impl
        return tensor

    @classmethod
    def zeros(
        cls, shape: ShapeType, dtype: DType, device: Device = CPU()
    ) -> Tensor:
        """Allocates an tensor with all elements initialized to zero."""
        tensor = cls(shape, dtype, device)
        tensor._impl.zeros()
        return tensor

    @classmethod
    def scalar(cls, value: Any, dtype: DType, device: Device = CPU()) -> Tensor:
        """Create a scalar value of a given dtype and value."""
        tensor = cls((), dtype, CPU())
        tensor[0] = value  # type: ignore

        # We can't directly set GPU memory, so we just have to copy
        # the tensor over.
        return tensor.to(device)

    @property
    def dtype(self) -> DType:
        """DType of constituent elements in tensor."""
        return DType._from(self._impl.dtype)

    @property
    def shape(self) -> ShapeType:
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

    def _iterate_indices(self) -> Generator[ShapeType]:
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

    def __setitem__(self, idx: IndexType, value: Any) -> None:
        """Sets an item in the tensor."""
        self._impl.set(idx, value)

    def __getitem__(self, idx: IndexType) -> Tensor:
        """Gets a tensor slice. Supports full numpy-style slicing. Invocations
        using only integer-based indexes will return zero-rank tensors."""
        new_tensor = self._impl.get(idx)
        return self._from_impl(new_tensor)

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

    def copy(self, device: Optional[Device] = None) -> Tensor:
        """Create a deep copy on an optionally given device.

        If a device is None (default), a copy is created on the same device."""
        if device is None:
            device = self.device
        return self._from_impl(self._impl.copy_to(device._device))

    def to(self, device: Device) -> Tensor:
        """Return a tensor that's guaranteed to be on the given device.

        The tensor is only copied if the input device is different from the
        device upon which the tensor is already resident.
        """
        return self if self.device == device else self.copy(device)

    @property
    def num_elements(self) -> int:
        """Returns the number of elements in this tensor.

        Rank-0 tensors have 1 element by convention.
        """
        return self._impl.num_elements

    @property
    def element_size(self) -> int:
        """Return the size of the element type in bytes."""
        return self.dtype.size_in_bytes

    def view(self, dtype: DType, shape: Optional[ShapeType] = None) -> Tensor:
        """Return a new tensor with the given type and shape that shares the
        underlying memory.

        If the shape is not given, it will be deduced if possible, or a
        ValueError is raised.
        """
        if shape is None:
            last_axis_size = self.element_size * self.shape[-1]
            if last_axis_size % dtype.size_in_bytes:
                raise ValueError(
                    "When changing to a larger dtype, its size must be a"
                    " divisor of the total size in bytes of the last axis of"
                    " the array."
                )
            shape = (*self.shape[:-1], last_axis_size // dtype.size_in_bytes)

        return self._from_impl(self._impl.view(shape, dtype._to()))

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> Tensor:
        """Creates a tensor from a provided numpy array on the host device.

        The underlying data is not copied unless the array is noncontiguous. If
        it is, a contiguous copy will be returned."""

        # NOTE: np.ascontiguousarray only copies if needed.
        return cls.from_dlpack(np.ascontiguousarray(arr))

    def to_numpy(self) -> np.ndarray:
        """Converts the tensor to a numpy array.

        If the tensor is not on the host, an exception is raised.
        """
        try:
            return np.from_dlpack(self)  # type: ignore
        except RuntimeError as e:
            if str(e).startswith("Unsupported device in DLTensor"):
                raise RuntimeError(
                    f"Cannot convert tensor on {self.device} to numpy; move to"
                    " the host using `Tensor.to`"
                ) from e
            raise

    def __dlpack_device__(self) -> Tuple[int, int]:
        """Implements part of the dlpack contract."""
        return self._impl.__dlpack_device__()

    def __dlpack__(self, *, stream=None) -> Any:
        """Implements part of the dlpack contract."""
        return self._impl.__dlpack__(stream=stream)

    @classmethod
    def from_dlpack(
        cls: Type[_T], arr: Any, *, copy: Optional[bool] = None
    ) -> _T:
        """Create a tensor from an object implementing the dlpack protocol.

        This usually does not result in a copy, and the producer of the object
        retains ownership of the underlying memory."""
        if isinstance(arr, np.memmap):
            # TODO(MSDK-976): since `np.memmap`s are often read-only, we just
            # use our own memmap implementation here, but it would be better to
            # always delegate to from_dlpack.
            return MemMapTensor._from_numpy_memmap(arr)  # type: ignore
        if isinstance(arr, np.ndarray):
            # TODO(MSDK-976): Older version of numpy don't support exporting
            # read-only arrays, so we copy if we can, and leave a hint if not.
            if copy is None and not arr.flags.writeable:
                copy = True
            if copy:
                arr = arr.copy()

            # Numpy's dlpack implementation cannot handle its own bool types, so
            # we trick it into thinking it is uint8.
            is_bool = arr.dtype == bool
            if is_bool:
                arr = arr.view(np.uint8)

            try:
                tensor = cls._from_impl(_Tensor.from_dlpack(arr))
            except BufferError as e:
                msg = str(e)
                if msg.startswith("Cannot export readonly array"):
                    raise type(e)(
                        msg
                        + " Consider passing `copy = True` to"
                        " `Tensor.from_dlpack`."
                    )
                raise e

            return tensor.view(DType.bool) if is_bool else tensor  # type: ignore

        if copy is not None:
            raise ValueError(
                "`Tensor.from_dlpack` support the `copy` flag only for numpy"
                " array inputs"
            )

        return cls._from_impl(_Tensor.from_dlpack(arr))


class MemMapTensor(Tensor):
    """Create a memory-mapped tensor from a binary file on disk.

    The constructor argument semantics follow that of np.memmap.
    """

    _mmap: mmap
    _read_only: bool

    def __init__(
        self,
        filename: PathLike,
        dtype: DType,
        shape: ShapeType,
        mode="r+",
        offset=0,
    ):
        # Instead of implementing all the mmap-related logic, we just delegate
        # to numpy. By passing order="C", we ensure C-contiguous layout.
        arr = np.memmap(  # type: ignore
            filename, dtype.to_numpy(), mode, offset, shape, order="C"
        )
        assert arr.flags["C_CONTIGUOUS"]
        self._init_from_numpy_memmap(arr)

    def _init_from_numpy_memmap(self, arr: np.memmap):
        # TODO(MSDK-976): Ideally, we could just use DLPack to borrow the
        # underlying memory from numpy. But our numpy version doesn't allow
        # dlpack to be used on read-only arrays (common for memmaped weights).
        self._impl = _Tensor(arr, CPU()._device)

        # numpy does not attempt to free/close the mmap object it uses, so we
        # copy a reference to it, which should keep it alive as long as needed.
        self._mmap = arr._mmap  # type: ignore

        self._read_only = not arr.flags.writeable

    @classmethod
    def _from_numpy_memmap(cls, arr: np.memmap) -> MemMapTensor:
        tensor = cls.__new__(cls)
        tensor._init_from_numpy_memmap(arr)
        return tensor

    def __dlpack__(self, *, stream=None) -> Any:
        """Implements part of the dlpack contract."""
        # We must ensure that the underlying mmap doesn't get closed.
        return self._impl.__dlpack__(stream=stream, _mmap=self._mmap)

    @property
    def read_only(self) -> bool:
        return self._read_only

    def __setitem__(self, idx: IndexType, value: Any) -> None:
        """Sets an item in the tensor."""
        if self.read_only:
            raise ValueError("Cannot modify read-only MemMapTensor")
        super().__setitem__(idx, value)


def load_max_tensor(path: PathLike) -> Tensor:
    """Experimental method for loading serialized MAX tensors.

    Max tensors can be exported by creating a graph and calling `Value.print()`
    with the `BINARY_MAX_CHECKPOINT` option.

    Args:
        path: Path to tensor (should end with .max)

    Returns:
        A `Tensor` created from the path. The shape and dtype are read
        from the file.

    Raises:
        ValueError if the file format is not the MAX checkpoint format.
    """

    with open(path, "rb") as f:
        header = f.read(8)
        if header != b"\x93\xf0\x9f\x94\xa5\x2b\x2b\x93":
            raise ValueError(
                f"{path} is not a max checkpoint. If this file was saved "
                'from the "BINARY" debug print option (and not '
                '"BINARY_MAX_CHECKPOINT"), please initialize `MemmapTensor` '
                "directly."
            )
        major_version, _minor_version = struct.unpack("2i", f.read(8))

        # Hardcoded but we should move to a robust versioning system if this
        # method is ever used outside of debugging.
        if major_version > 0:
            raise ValueError("Unable to read from version > 0.")

        metadata_size = struct.unpack("Q", f.read(8))[0]

        key_size = struct.unpack("i", f.read(4))[0]
        unused_key = f.read(key_size).decode("utf-8")
        dtype, rank = struct.unpack("2b", f.read(2))
        dtype = DType(dtype)
        shape = tuple(struct.unpack("i", f.read(4))[0] for _ in range(rank))
        offset = struct.unpack("Q", f.read(8))[0]

        bytes_read = 4 + key_size + 2 + 4 * rank + 8
        if bytes_read != metadata_size:
            raise ValueError(
                "Multiple tensors found in .max file. This is currently not"
                " supported."
            )

        if dtype == DType.bfloat16:
            dtype = DType.uint8
            shape = tuple(dim * 2 for dim in shape)
            tensor = MemMapTensor(path, dtype, shape, mode="r", offset=offset)
            return tensor.view(DType.bfloat16)
        else:
            return MemMapTensor(path, dtype, shape, mode="r", offset=offset)
