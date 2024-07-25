# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""The DType, which mirrors Mojo stdlib dtype.

TODO(MSDK-597): move this to a common location shared by
`max.{driver,engine,graph}`.
"""

from enum import Enum


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
