# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""The DType, which mirrors Mojo stdlib dtype.

TODO(MSDK-597): move this to a common location shared by
`max.{driver,engine,graph}`.
"""

from __future__ import annotations

from enum import Enum

import numpy as np


class DType(Enum):
    """The tensor data type."""

    bool = 0, "bool"
    int8 = 1, "si8"
    int16 = 2, "si16"
    int32 = 3, "si32"
    int64 = 4, "si64"
    uint8 = 5, "ui8"
    uint16 = 6, "ui16"
    uint32 = 7, "ui32"
    uint64 = 8, "ui64"
    float16 = 9, "f16"
    float32 = 10, "f32"
    float64 = 11, "f64"

    def __repr__(self) -> str:
        return self.name

    @property
    def _mlir(self):
        return self.value[1]

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
