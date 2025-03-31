# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from __future__ import annotations

import struct
from collections.abc import Generator, Sequence
from itertools import product
from mmap import mmap
from os import PathLike
from typing import Any, Optional, Protocol, Union, runtime_checkable

import numpy as np
from max._core.driver import Tensor as Tensor
from max.dtype import DType

from .driver import CPU

_IdxElType = Union[int, slice]
IndexType = Union[Sequence[_IdxElType], _IdxElType]
ShapeType = Sequence[int]


@runtime_checkable
class DLPackArray(Protocol):
    def __dlpack__(self) -> Any: ...

    def __dlpack_device__(self) -> Any: ...


def _iterate_indices(self) -> Generator[ShapeType]:
    yield from product(*map(range, self.shape))


def _contiguous(self) -> Tensor:
    """Creates a contiguous copy of the parent tensor."""
    tensor_copy = Tensor(self.shape, self.dtype)
    for idx in self._iterate_indices():
        tensor_copy[idx] = self[idx].item()
    return tensor_copy


def _aligned(self, alignment: int | None = None) -> bool:
    """Returns whether the tensor is aligned to the desired alignment."""
    return self.__aligned(alignment or self.dtype.align)


def _repr(self) -> str:
    return f"max.driver.Tensor({self.dtype}, {self.shape}, {self.device.api}[{self.device.id}])"


def _view(self, dtype: DType, shape: Optional[ShapeType] = None) -> Tensor:
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

    return self._view(dtype, shape)


def inplace_copy_from(self, src: Tensor) -> None:
    """Copy the contents of another tensor into this one. These tensors may
    be on different devices.

    Requires that both tensors are contiguous and have same size."""
    # check that both tensors are contiguous
    if not self.is_contiguous:
        raise ValueError("Cannot copy from non-contiguous tensor")
    if not src.is_contiguous:
        raise ValueError("Cannot copy to non-contiguous tensor")

    # check that both tensors have same size
    if self.num_elements != src.num_elements:
        raise ValueError("Cannot copy tensors of different sizes")

    # check that both tensors have the same dtype
    if self.dtype != src.dtype:
        raise ValueError("Cannot copy tensors of different dtypes")

    self._inplace_copy_from(src)


def _from_numpy(arr: np.ndarray) -> Tensor:
    """Creates a tensor from a provided numpy array on the host device.

    The underlying data is not copied unless the array is noncontiguous. If
    it is, a contiguous copy will be returned."""

    # NOTE: np.ascontiguousarray only copies if needed.
    # Skip np.contiguousarray for scalars since it converts them to rank-1.
    return Tensor.from_dlpack(np.ascontiguousarray(arr) if arr.shape else arr)


def _to_numpy(self) -> np.ndarray:
    """Converts the tensor to a numpy array.

    If the tensor is not on the host, an exception is raised.
    """
    try:
        return np.from_dlpack(self.to(CPU()))
    except RuntimeError as e:
        if str(e).startswith("Unsupported device in DLTensor"):
            raise RuntimeError(
                f"Cannot convert tensor on {self.device} to numpy; move to"
                " the host using `Tensor.to`"
            ) from e
        raise


def _from_dlpack(array: Any, *, copy: Optional[bool] = None) -> Tensor:
    """Create a tensor from an object implementing the dlpack protocol.

    This usually does not result in a copy, and the producer of the object
    retains ownership of the underlying memory."""
    if isinstance(array, np.memmap):
        # TODO(MSDK-976): since `np.memmap`s are often read-only, we just
        # use our own memmap implementation here, but it would be better to
        # always delegate to from_dlpack.
        return MemMapTensor._from_numpy_memmap(array)
    if isinstance(array, np.ndarray):
        if not array.flags.c_contiguous:
            raise ValueError(
                "driver tensor's from_dlpack only accepts contiguous arrays. "
                "First call np.ascontiguousarray(array)"
            )

        # TODO(MSDK-976): Older version of numpy don't support exporting
        # read-only arrays, so we copy if we can, and leave a hint if not.
        if copy is None and not array.flags.writeable:
            copy = True
        if copy:
            array = array.copy()

        # Numpy's dlpack implementation cannot handle its own bool types, so
        # we trick it into thinking it is uint8.
        is_bool = array.dtype == bool
        if is_bool:
            array = array.view(np.uint8)

        try:
            tensor = Tensor._from_dlpack(array)
        except BufferError as e:
            msg = str(e)
            if msg.startswith("Cannot export readonly array"):
                raise type(e)(
                    msg
                    + " Consider passing `copy = True` to `Tensor.from_dlpack`."
                )
            raise e

        return tensor.view(DType.bool) if is_bool else tensor

    # Short circuit if it's our type.
    if isinstance(array, Tensor):
        return array.copy() if copy else array

    if copy is not None:
        raise ValueError(
            "`Tensor.from_dlpack` supports the `copy` flag only for numpy"
            " array and `Tensor` inputs"
        )

    return Tensor._from_dlpack(array)


Tensor._iterate_indices = _iterate_indices  # type: ignore[method-assign]
Tensor.contiguous = _contiguous  # type: ignore[method-assign]
Tensor._aligned = _aligned  # type: ignore[method-assign]
Tensor.__repr__ = _repr  # type: ignore[method-assign]
Tensor.view = _view  # type: ignore[method-assign]
Tensor.inplace_copy_from = inplace_copy_from  # type: ignore[method-assign]
Tensor.from_numpy = _from_numpy  # type: ignore[method-assign]
Tensor.to_numpy = _to_numpy  # type: ignore[method-assign]
Tensor.from_dlpack = _from_dlpack  # type: ignore[method-assign]


class MemMapTensor(Tensor):
    """Create a memory-mapped tensor from a binary file on disk.

    The constructor argument semantics follow that of np.memmap.
    """

    _mmap: mmap
    _read_only: bool

    def __init__(
        self,
        filename: PathLike | str,
        dtype: DType,
        shape: ShapeType | int,
        mode: np._MemMapModeKind = "r+",
        offset=0,
    ) -> None:
        # Instead of implementing all the mmap-related logic, we just delegate
        # to numpy. By passing order="C", we ensure C-contiguous layout.
        arr: np.memmap = np.memmap(
            filename,
            dtype.to_numpy(),
            mode,
            offset,
            # NOTE: prior to NumPy 2.0, `shape` must be `tuple` or `int`.
            shape if isinstance(shape, int) else tuple(shape),
            order="C",
        )
        assert arr.flags["C_CONTIGUOUS"]
        self._init_from_numpy_memmap(arr)

    def _init_from_numpy_memmap(self, arr: np.memmap) -> None:
        # TODO(MSDK-976): Ideally, we could just use DLPack to borrow the
        # underlying memory from numpy. But our numpy version doesn't allow
        # dlpack to be used on read-only arrays (common for memmaped weights).
        super().__init__(arr, CPU())

        # numpy does not attempt to free/close the mmap object it uses, so we
        # copy a reference to it, which should keep it alive as long as needed.
        self._mmap = arr._mmap  # type: ignore

        self._read_only = not arr.flags.writeable

    def _init_from_tensor(self, tensor: Tensor, mm: mmap, ro: bool) -> None:
        super().__init__(tensor)
        self._mmap = mm
        self._read_only = ro

    @classmethod
    def _from_numpy_memmap(cls, arr: np.memmap) -> MemMapTensor:
        tensor = cls.__new__(cls)
        tensor._init_from_numpy_memmap(arr)
        return tensor

    def __dlpack__(self, *, stream=None) -> Any:  # type: ignore[override]
        """Implements part of the dlpack contract."""
        # We must ensure that the underlying mmap doesn't get closed.
        return super().__dlpack__(stream=stream, _mmap=self._mmap)

    @property
    def read_only(self) -> bool:
        return self._read_only

    def view(
        self, dtype: DType, shape: ShapeType | None = None
    ) -> MemMapTensor:
        """Creates a view of the memory-mapped tensor with new type and shape.

        Preserves the original memory mapping properties (read-only status and
        file reference) while reinterpreting the bytes.
        Shares underlying storage -- modifications affect all views of the same
        memory region.

        Args:
            dtype: New data type for interpreting the bytes.
                Size must divide evenly into the original tensor's last
                dimension byte size.
            shape: Optional new shape.
                Inferred from dtype size if not provided.
                Product of dimensions must match original element count
                adjusted for dtype size.

        Returns:
            New :obj:`MemMapTensor` instance with specified dtype/shape, sharing the
            original memory-mapped file properties.

        Raises:
            ValueError: For invalid shape/dtype combinations or boundary overflows.
            TypeError: For undefined byte interpretations.
        """
        viewed = super().view(dtype, shape)
        result = MemMapTensor.__new__(MemMapTensor)
        result._init_from_tensor(viewed, self._mmap, self._read_only)
        return result

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

        # '2I' = 2 4-byte integers (major_version, minor_version).
        major_version, _minor_version = struct.unpack("2I", f.read(8))

        # Hardcoded but we should move to a robust versioning system if this
        # method is ever used outside of debugging.
        if major_version > 0:
            raise ValueError("Unable to read from version > 0.")

        # 'Q' = 8-byte unsigned long long (metadata_size).
        metadata_size = struct.unpack("Q", f.read(8))[0]

        # 'I' = 4-byte unsigned integer (key_size).
        key_size = struct.unpack("I", f.read(4))[0]

        unused_key = f.read(key_size).decode("utf-8")

        # '2B' = 2 unsigned bytes (dtype, rank).
        dtype, rank = struct.unpack("2B", f.read(2))

        dtype = DType(dtype)

        # 'I' = 4-byte unsigned integer (each dimension in shape tuple).
        shape = tuple(struct.unpack("I", f.read(4))[0] for _ in range(rank))

        # 'Q' = 8-byte unsigned long long (offset).
        offset = struct.unpack("Q", f.read(8))[0]

        bytes_read = 4 + key_size + 2 + 4 * rank + 8
        if bytes_read != metadata_size:
            raise ValueError(
                "Multiple tensors found in .max file. This is currently not supported."
            )

        if dtype == DType.bfloat16:
            # Only modify last dimension for byte expansion.
            new_shape = list(shape)
            if len(new_shape) == 0:
                # Handle scalar case.
                new_shape = [2]
            else:
                # Expand last dimension for uint8 bytes.
                new_shape[-1] *= 2

            tensor = MemMapTensor(
                path,
                DType.uint8,
                new_shape,
                mode="r",
                offset=offset,
            )
            return tensor.view(DType.bfloat16)
        else:
            return MemMapTensor(path, dtype, shape, mode="r", offset=offset)
