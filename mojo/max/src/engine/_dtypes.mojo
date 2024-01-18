# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@value
@register_passable("trivial")
struct EngineDType:
    """Mojo representation of engine's dtypes."""

    var dtype: Int32

    alias unknown: Int32 = 0

    alias int8: Int32 = 1
    alias int16: Int32 = 2
    alias int32: Int32 = 3
    alias int64: Int32 = 4

    alias uint8: Int32 = 10
    alias uint16: Int32 = 11
    alias uint32: Int32 = 12
    alias uint64: Int32 = 13

    alias float16: Int32 = 20
    alias float32: Int32 = 21
    alias float64: Int32 = 22

    alias bool: Int32 = 30

    fn __init__(dtype: DType) -> Self:
        if dtype == DType.int8:
            return Self {dtype: EngineDType.int8}
        if dtype == DType.int16:
            return Self {dtype: EngineDType.int16}
        if dtype == DType.int32:
            return Self {dtype: EngineDType.int32}
        if dtype == DType.int64:
            return Self {dtype: EngineDType.int64}

        if dtype == DType.uint8:
            return Self {dtype: EngineDType.uint8}
        if dtype == DType.uint16:
            return Self {dtype: EngineDType.uint16}
        if dtype == DType.uint32:
            return Self {dtype: EngineDType.uint32}
        if dtype == DType.uint64:
            return Self {dtype: EngineDType.uint64}

        if dtype == DType.float16:
            return Self {dtype: EngineDType.float16}
        if dtype == DType.float32:
            return Self {dtype: EngineDType.float32}
        if dtype == DType.float64:
            return Self {dtype: EngineDType.float64}

        if dtype == DType.bool:
            return Self {dtype: EngineDType.bool}

        return EngineDType.unknown

    fn to_dtype(self) -> DType:
        if self == EngineDType.int8:
            return DType.int8
        if self == EngineDType.int16:
            return DType.int16
        if self == EngineDType.int32:
            return DType.int32
        if self == EngineDType.int64:
            return DType.int64

        if self == EngineDType.uint8:
            return DType.uint8
        if self == EngineDType.uint16:
            return DType.uint16
        if self == EngineDType.uint32:
            return DType.uint32
        if self == EngineDType.uint64:
            return DType.uint64

        if self == EngineDType.float16:
            return DType.float16
        if self == EngineDType.float32:
            return DType.float32
        if self == EngineDType.float64:
            return DType.float64

        if self == EngineDType.bool:
            return DType.bool

        return DType.invalid

    fn __eq__(self, other: EngineDType) -> Bool:
        return self.dtype == other.dtype
