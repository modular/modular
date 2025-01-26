# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from enum import Enum

import numpy as np
from max._dtype import DType as _DType

# DType UInt8 values from Support/include/Support/ML/DType.h
mIsInteger = 1 << 7
mIsFloat = 1 << 6
mIsComplex = 1 << 5
mIsSigned = 1
kIntWidthShift = 1


class DType(Enum):
    """The tensor data type."""

    # The unknown DType is used for passing python objects to the MAX Engine.
    _unknown = 0  # Original name: invalid
    # si1 = (0 << kIntWidthShift) | mIsInteger | mIsSigned
    # ui1 = (0 << kIntWidthShift) | mIsInteger
    # si2 = (1 << kIntWidthShift) | mIsInteger | mIsSigned
    # ui2 = (1 << kIntWidthShift) | mIsInteger
    # si4 = (2 << kIntWidthShift) | mIsInteger | mIsSigned
    # ui4 = (2 << kIntWidthShift) | mIsInteger

    int8 = (3 << kIntWidthShift) | mIsInteger | mIsSigned  # Original name: si8
    uint8 = (3 << kIntWidthShift) | mIsInteger  # Original name: ui8
    # Original name: si16
    int16 = (4 << kIntWidthShift) | mIsInteger | mIsSigned
    uint16 = (4 << kIntWidthShift) | mIsInteger  # Original name: ui16

    # Original name: si32
    int32 = (5 << kIntWidthShift) | mIsInteger | mIsSigned
    uint32 = (5 << kIntWidthShift) | mIsInteger  # Original name: ui32
    # Original name: si64
    int64 = (6 << kIntWidthShift) | mIsInteger | mIsSigned
    uint64 = (6 << kIntWidthShift) | mIsInteger  # Original name: ui64
    # si128 = (7 << kIntWidthShift) | mIsInteger | mIsSigned
    # ui128 = (7 << kIntWidthShift) | mIsInteger

    f8e5m2 = 0 | mIsFloat
    f8e4m3 = 1 | mIsFloat
    # f8e3m4 = 2 | mIsFloat
    f8e5m2fnuz = 3 | mIsFloat
    f8e4m3fnuz = 4 | mIsFloat
    float16 = 5 | mIsFloat  # Original name: f16
    bfloat16 = 6 | mIsFloat  # Original name: bf16
    float32 = 7 | mIsFloat  # Original name: f32
    float64 = 8 | mIsFloat  # Original name: f64
    # f128 = 9 | mIsFloat

    # f24 = 10 | mIsFloat
    # f80 = 11 | mIsFloat
    # tf32 = 12 | mIsFloat

    bool = 1  # Original name: kBool

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            return _MLIR_TO_DTYPE[value]
        return None

    @classmethod
    def _from(cls, dtype: _DType):
        try:
            return cls.__dict__[dtype.name]
        except KeyError as e:
            if dtype.name == "unknown":
                return DType._unknown
            raise e

    def _to(self):
        try:
            return _DType.__dict__[self.name]
        except KeyError as e:
            if self.name == "_unknown":
                return _DType.unknown
            raise e

    def __repr__(self) -> str:
        return self.name

    @property
    def _mlir(self):
        return _DTYPE_TO_MLIR[self]

    def to_numpy(self) -> np.dtype:
        """Converts a NumPy dtype to the corresponding DType.

        Args:
            dtype (np.dtype): The NumPy dtype to convert.

        Returns:
            DType: The corresponding DType enum value.

        Raises:
            ValueError: If the input dtype is not supported.
        """
        dtype_to_numpy: dict[DType, np.dtype] = {
            DType.bool: np.dtype(np.bool_),
            DType.int8: np.dtype(np.int8),
            DType.int16: np.dtype(np.int16),
            DType.int32: np.dtype(np.int32),
            DType.int64: np.dtype(np.int64),
            DType.uint8: np.dtype(np.uint8),
            DType.uint16: np.dtype(np.uint16),
            DType.uint32: np.dtype(np.uint32),
            DType.uint64: np.dtype(np.uint64),
            DType.f8e4m3: np.dtype(np.uint8),
            DType.f8e4m3fnuz: np.dtype(np.uint8),
            DType.f8e5m2: np.dtype(np.uint8),
            DType.f8e5m2fnuz: np.dtype(np.uint8),
            DType.float16: np.dtype(np.float16),
            DType.float32: np.dtype(np.float32),
            DType.float64: np.dtype(np.float64),
        }

        if self in dtype_to_numpy:
            return dtype_to_numpy[self]
        else:
            raise ValueError(f"unsupported DType to convert to NumPy: {self}")

    @classmethod
    def from_numpy(cls, dtype: np.dtype) -> DType:
        """Converts a NumPy dtype to the corresponding DType.

        Args:
            dtype (np.dtype): The NumPy dtype to convert.

        Returns:
            DType: The corresponding DType enum value.

        Raises:
            ValueError: If the input dtype is not supported.
        """
        numpy_to_dtype = {
            np.bool_: cls.bool,
            np.int8: cls.int8,
            np.int16: cls.int16,
            np.int32: cls.int32,
            np.int64: cls.int64,
            np.uint8: cls.uint8,
            np.uint16: cls.uint16,
            np.uint32: cls.uint32,
            np.uint64: cls.uint64,
            np.float16: cls.float16,
            np.float32: cls.float32,
            np.float64: cls.float64,
        }

        # Handle both np.dtype objects and numpy type objects.
        np_type = dtype.type if isinstance(dtype, np.dtype) else dtype

        if np_type in numpy_to_dtype:
            return numpy_to_dtype[np_type]
        else:
            raise ValueError(f"unsupported NumPy dtype: {dtype}")

    @property
    def align(self) -> int:
        """Returns the alignment of the dtype."""
        if self is DType.bfloat16:
            # Use float16 alignment since np.bfloat16 doesn't exist.
            return np.dtype(np.float16).alignment

        return np.dtype(self.to_numpy()).alignment

    def is_integral(self) -> __builtins__.bool:
        """Returns true if the dtype is an integer."""
        return self.value & mIsInteger != 0

    def is_float(self) -> __builtins__.bool:
        """Returns true if the dtype is floating point."""
        if self.is_integral():
            return False
        return self.value & mIsFloat != 0

    @property
    def size_in_bytes(self) -> int:
        """Returns the size of the dtype in bytes."""
        return self._to().size_in_bytes


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
    DType.f8e4m3: "f8e4m3",
    DType.f8e4m3fnuz: "f8e4m3fnuz",
    DType.f8e5m2: "f8e5m2",
    DType.f8e5m2fnuz: "f8e5m2fnuz",
    DType.float16: "f16",
    DType.float32: "f32",
    DType.float64: "f64",
    DType.bfloat16: "bf16",
    DType._unknown: "invalid",
}


_MLIR_TO_DTYPE = {
    "bool": DType.bool,
    "si8": DType.int8,
    "si16": DType.int16,
    "si32": DType.int32,
    "si64": DType.int64,
    "ui8": DType.uint8,
    "ui16": DType.uint16,
    "ui32": DType.uint32,
    "ui64": DType.uint64,
    "f16": DType.float16,
    "f32": DType.float32,
    "f64": DType.float64,
    "bf16": DType.bfloat16,
    "f8e4m3": DType.f8e4m3,
    "f8e4m3fnuz": DType.f8e4m3fnuz,
    "f8e5m2": DType.f8e5m2,
    "f8e5m2fnuz": DType.f8e5m2fnuz,
    "invalid": DType._unknown,
}
