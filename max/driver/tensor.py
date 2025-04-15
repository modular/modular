# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from __future__ import annotations

import struct
from collections.abc import Generator, Sequence
from functools import wraps
from itertools import product
from os import PathLike
from typing import Any, Optional, Union

import numpy as np
from max._core.driver import Tensor as Tensor
from max.dtype import DType

from .driver import CPU

_IdxElType = Union[int, slice]
IndexType = Union[Sequence[_IdxElType], _IdxElType]
ShapeType = Sequence[int]


def _iterate_indices(self) -> Generator[ShapeType]:
    yield from product(*map(range, self.shape))


def _contiguous(self) -> Tensor:
    """Creates a contiguous copy of the parent tensor."""
    tensor_copy = Tensor(self.dtype, self.shape)
    for idx in self._iterate_indices():
        tensor_copy[idx] = self[idx].item()
    return tensor_copy


def _repr(self) -> str:
    return f"max.driver.Tensor({self.dtype}, {self.shape}, {self.stream})"


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


@wraps(Tensor.mmap)
def _mmap(
    filename: PathLike | str,
    dtype: DType,
    shape: ShapeType | int,
    mode: np._MemMapModeKind = "copyonwrite",
    offset=0,
):
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
    return Tensor.from_dlpack(arr)


Tensor._iterate_indices = _iterate_indices  # type: ignore[method-assign]
Tensor.contiguous = _contiguous  # type: ignore[method-assign]
Tensor.__repr__ = _repr  # type: ignore[method-assign]
Tensor.view = _view  # type: ignore[method-assign]
Tensor.inplace_copy_from = inplace_copy_from  # type: ignore[method-assign]
Tensor.from_numpy = _from_numpy  # type: ignore[method-assign]
Tensor.to_numpy = _to_numpy  # type: ignore[method-assign]
Tensor.from_dlpack = _from_dlpack  # type: ignore[method-assign]
Tensor.mmap = _mmap  # type: ignore[method-assign]


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
                '"BINARY_MAX_CHECKPOINT"), please initialize `Tensor.mmap` '
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

            tensor = Tensor.mmap(
                path,
                DType.uint8,
                new_shape,
                mode="r",
                offset=offset,
            )
            return tensor.view(DType.bfloat16)
        else:
            return Tensor.mmap(path, dtype, shape, mode="r", offset=offset)
