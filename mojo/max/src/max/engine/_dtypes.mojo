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
    alias bfloat16: Int32 = 23

    alias bool: Int32 = 30

    @implicit
    fn __init__(out self, dtype: Int32):
        self.dtype = dtype

    @implicit
    fn __init__(out self, dtype: DType):
        if dtype is DType.int8:
            self.dtype = EngineDType.int8
        elif dtype is DType.int16:
            self.dtype = EngineDType.int16
        elif dtype is DType.int32:
            self.dtype = EngineDType.int32
        elif dtype is DType.int64:
            self.dtype = EngineDType.int64

        elif dtype is DType.uint8:
            self.dtype = EngineDType.uint8
        elif dtype is DType.uint16:
            self.dtype = EngineDType.uint16
        elif dtype is DType.uint32:
            self.dtype = EngineDType.uint32
        elif dtype is DType.uint64:
            self.dtype = EngineDType.uint64

        elif dtype is DType.float16:
            self.dtype = EngineDType.float16
        elif dtype is DType.float32:
            self.dtype = EngineDType.float32
        elif dtype is DType.float64:
            self.dtype = EngineDType.float64
        elif dtype is DType.bfloat16:
            self.dtype = EngineDType.bfloat16

        elif dtype is DType.bool:
            self.dtype = EngineDType.bool
        else:
            self.dtype = EngineDType.unknown

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

        if self == EngineDType.bfloat16:
            return DType.bfloat16

        return DType.invalid

    fn __eq__(self, other: EngineDType) -> Bool:
        return self.dtype == other.dtype
