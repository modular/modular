# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Data types for tensors in MAX Engine."""

from __future__ import annotations

import numpy as np
from max._core.dtype import DType as DType


def _missing(value) -> DType | None:
    if isinstance(value, str):
        return _MLIR_TO_DTYPE[value]
    return None


def _repr(self) -> str:
    return self.name


def _mlir(self) -> str:
    return _DTYPE_TO_MLIR[self]


_DTYPE_TO_NUMPY = {
    DType.bool: np.bool_,
    DType.int8: np.int8,
    DType.int16: np.int16,
    DType.int32: np.int32,
    DType.int64: np.int64,
    DType.uint8: np.uint8,
    DType.uint16: np.uint16,
    DType.uint32: np.uint32,
    DType.uint64: np.uint64,
    DType.float8_e4m3: np.uint8,
    DType.float8_e4m3fn: np.uint8,
    DType.float8_e4m3fnuz: np.uint8,
    DType.float8_e5m2: np.uint8,
    DType.float8_e5m2fnuz: np.uint8,
    DType.float16: np.float16,
    DType.float32: np.float32,
    DType.float64: np.float64,
}

_NUMPY_TO_DTYPE = {
    np.bool_: DType.bool,
    np.int8: DType.int8,
    np.int16: DType.int16,
    np.int32: DType.int32,
    np.int64: DType.int64,
    np.uint8: DType.uint8,
    np.uint16: DType.uint16,
    np.uint32: DType.uint32,
    np.uint64: DType.uint64,
    np.float16: DType.float16,
    np.float32: DType.float32,
    np.float64: DType.float64,
}


def _to_numpy(self) -> np.dtype:
    """Converts this ``DType`` to the corresponding NumPy dtype.

    Returns:
        DType: The corresponding NumPy dtype object.

    Raises:
        ValueError: If the dtype is not supported.
    """

    if numpy_dtype := _DTYPE_TO_NUMPY.get(self):
        return np.dtype(numpy_dtype)
    raise ValueError(f"unsupported DType to convert to NumPy: {self}")


def _from_numpy(dtype: np.dtype) -> DType:
    """Converts a NumPy dtype to the corresponding DType.

    Args:
        dtype (np.dtype): The NumPy dtype to convert.

    Returns:
        DType: The corresponding DType enum value.

    Raises:
        ValueError: If the input dtype is not supported.
    """

    # Handle both np.dtype objects and numpy type objects.
    np_type = dtype.type if isinstance(dtype, np.dtype) else dtype

    if max_dtype := _NUMPY_TO_DTYPE.get(np_type):
        return max_dtype
    raise ValueError(f"unsupported NumPy dtype: {dtype}")


def _align(self) -> int:
    """Returns the alignment of the dtype."""
    if self is DType.bfloat16:
        # Use float16 alignment since np.bfloat16 doesn't exist.
        return np.dtype(np.float16).alignment
    return np.dtype(self.to_numpy()).alignment


DType._missing_ = _missing  # type: ignore[method-assign]
DType.__repr__ = _repr  # type: ignore[method-assign]
DType._mlir = property(_mlir)  # type: ignore[assignment]
DType.align = property(_align)  # type: ignore[assignment]
DType.to_numpy = _to_numpy  # type: ignore[method-assign]
DType.from_numpy = _from_numpy  # type: ignore[method-assign]


_DTYPE_TO_MLIR = {
    DType.bool: "bool",
    DType.int8: "si8",
    DType.int16: "si16",
    DType.int32: "si32",
    DType.int64: "si64",
    DType.uint8: "ui8",
    DType.uint16: "ui16",
    DType.uint32: "ui32",
    DType.uint64: "ui64",
    DType.float8_e4m3: "f8e4m3",
    DType.float8_e4m3fn: "f8e4m3fn",
    DType.float8_e4m3fnuz: "f8e4m3fnuz",
    DType.float8_e5m2: "f8e5m2",
    DType.float8_e5m2fnuz: "f8e5m2fnuz",
    DType.float16: "f16",
    DType.float32: "f32",
    DType.float64: "f64",
    DType.bfloat16: "bf16",
    DType._unknown: "invalid",
}

_MLIR_TO_DTYPE = {v: k for k, v in _DTYPE_TO_MLIR.items()}

# Load these methods only if torch is available.
try:
    import torch  # type: ignore

    _DTYPE_TO_TORCH = {
        DType.bool: torch.bool,
        DType.int8: torch.int8,
        DType.int16: torch.int16,
        DType.int32: torch.int32,
        DType.int64: torch.int64,
        DType.uint8: torch.uint8,
        DType.uint16: torch.uint16,
        DType.uint32: torch.uint32,
        DType.uint64: torch.uint64,
        DType.float16: torch.float16,
        DType.float32: torch.float32,
        DType.float64: torch.float64,
        DType.bfloat16: torch.bfloat16,
        DType.float8_e4m3fn: torch.float8_e4m3fn,
        DType.float8_e4m3fnuz: torch.float8_e4m3fnuz,
        DType.float8_e5m2: torch.float8_e5m2,
        DType.float8_e5m2fnuz: torch.float8_e5m2fnuz,
    }

    _TORCH_TO_DTYPE = {v: k for k, v in _DTYPE_TO_TORCH.items()}

    def _max_to_torch_type(dtype: DType) -> torch.dtype:
        return _DTYPE_TO_TORCH[dtype]

    def _torch_to_max_type(dtype: torch.dtype) -> DType:
        return _TORCH_TO_DTYPE[dtype]

except ImportError as e:

    def _max_to_torch_type(dtype: DType) -> torch.dtype:
        raise e

    def _torch_to_max_type(dtype: torch.dtype) -> DType:
        raise e


def max_to_torch_type(dtype: DType) -> torch.dtype:
    """Converts a MAX dtype.DType to a torch.dtype.

    Args:
        dtype (dtype.DType): MAX dtype.DType.

    Returns:
        torch.dtype: The corresponding torch.dtype.
    """
    return _max_to_torch_type(dtype)


def torch_to_max_type(dtype: torch.dtype) -> DType:
    """Converts a torch.dtype to a MAX dtype.DType.

    Args:
        dtype (torch.dtype): PyTorch torch.dtype.

    Returns:
        dtype.DType: The corresponding dtype.DType.
    """
    return _torch_to_max_type(dtype)
