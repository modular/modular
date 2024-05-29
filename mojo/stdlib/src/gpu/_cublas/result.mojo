# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@value
@register_passable("trivial")
struct Result:
    var _value: Int32
    alias SUCCESS = Result(0)
    alias NOT_INITIALIZED = Result(1)
    alias ALLOC_FAILED = Result(3)
    alias INVALID_VALUE = Result(7)
    alias ARCH_MISMATCH = Result(8)
    alias MAPPING_ERROR = Result(11)
    alias EXECUTION_FAILED = Result(13)
    alias INTERNAL_ERROR = Result(14)
    alias NOT_SUPPORTED = Result(15)
    alias LICENSE_ERROR = Result(16)

    fn __init__(inout self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __str__(self) -> String:
        if self == Self.SUCCESS:
            return "SUCCESS"
        if self == Self.NOT_INITIALIZED:
            return "NOT_INITIALIZED"
        if self == Self.ALLOC_FAILED:
            return "ALLOC_FAILED"
        if self == Self.INVALID_VALUE:
            return "INVALID_VALUE"
        if self == Self.ARCH_MISMATCH:
            return "ARCH_MISMATCH"
        if self == Self.MAPPING_ERROR:
            return "MAPPING_ERROR"
        if self == Self.EXECUTION_FAILED:
            return "EXECUTION_FAILED"
        if self == Self.INTERNAL_ERROR:
            return "INTERNAL_ERROR"
        if self == Self.NOT_SUPPORTED:
            return "NOT_SUPPORTED"
        if self == Self.LICENSE_ERROR:
            return "LICENSE_ERROR"
        return abort[String]("invalid Result entry")

    fn __int__(self) -> Int:
        return int(self._value)
