# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# Need this import to silence error from circular type dependency in ._from
# method
from __future__ import annotations

from enum import Enum

from max.driver.driver_core import DType as _DType


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
    def _from(cls, dtype: _DType) -> DType:
        obj = cls.__dict__[dtype.name]
        return obj

    def _to(self) -> _DType:
        return _DType.__dict__[self.name]

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other: DType) -> bool:
        return self._to() == other._to()
