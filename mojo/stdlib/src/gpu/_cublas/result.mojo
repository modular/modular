# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@value
@register_passable("trivial")
struct Result:
    var _value: Int32
    alias SUCCESS = Self(0)
    alias NOT_INITIALIZED = Self(1)
    alias ALLOC_FAILED = Self(3)
    alias INVALID_VALUE = Self(7)
    alias ARCH_MISMATCH = Self(8)
    alias MAPPING_ERROR = Self(11)
    alias EXECUTION_FAILED = Self(13)
    alias INTERNAL_ERROR = Self(14)
    alias NOT_SUPPORTED = Self(15)
    alias LICENSE_ERROR = Self(16)

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
