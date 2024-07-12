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
    unknown = 12

    def __repr__(self) -> str:
        return self.name
