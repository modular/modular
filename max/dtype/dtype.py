# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from enum import Enum

import numpy as np
from max._dtype import DType as _DType


class DType(Enum):
    """The tensor data type."""

    bool = 0
    int8 = 1
    int16 = 2
    int32 = 3
    int64 = 4
    uint8 = 5
    uint16 = 6
    uint32 = 7
    uint64 = 8
    float16 = 9
    float32 = 10
    float64 = 11
    bfloat16 = 12
    unknown = 13

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            return _MLIR_TO_DTYPE[value]
        return None

    @classmethod
    def _from(cls, dtype: _DType):
        obj = cls.__dict__[dtype.name]
        return obj

    def _to(self):
        return _DType.__dict__[self.name]

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
        dtype_to_numpy = {
            DType.bool: np.bool_,
            DType.int8: np.int8,
            DType.int16: np.int16,
            DType.int32: np.int32,
            DType.int64: np.int64,
            DType.uint8: np.uint8,
            DType.uint16: np.uint16,
            DType.uint32: np.uint32,
            DType.uint64: np.uint64,
            DType.float16: np.float16,
            DType.float32: np.float32,
            DType.float64: np.float64,
        }

        if self in dtype_to_numpy:
            return dtype_to_numpy[self]
        else:
            raise ValueError(f"unsupported DType to convert to NumPy: {self}")

    @classmethod
    def from_numpy(cls, dtype: np.dtype) -> "DType":
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
    DType.float16: "f16",
    DType.float32: "f32",
    DType.float64: "f64",
    DType.bfloat16: "bf16",
    DType.unknown: "invalid",
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
    "invalid": DType.unknown,
}
