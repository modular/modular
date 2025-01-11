# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from os import abort


@value
@register_passable("trivial")
struct Result(Writable):
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

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        if self == Self.SUCCESS:
            return writer.write("SUCCESS")
        if self == Self.NOT_INITIALIZED:
            return writer.write("NOT_INITIALIZED")
        if self == Self.ALLOC_FAILED:
            return writer.write("ALLOC_FAILED")
        if self == Self.INVALID_VALUE:
            return writer.write("INVALID_VALUE")
        if self == Self.ARCH_MISMATCH:
            return writer.write("ARCH_MISMATCH")
        if self == Self.MAPPING_ERROR:
            return writer.write("MAPPING_ERROR")
        if self == Self.EXECUTION_FAILED:
            return writer.write("EXECUTION_FAILED")
        if self == Self.INTERNAL_ERROR:
            return writer.write("INTERNAL_ERROR")
        if self == Self.NOT_SUPPORTED:
            return writer.write("NOT_SUPPORTED")
        if self == Self.LICENSE_ERROR:
            return writer.write("LICENSE_ERROR")

        return abort("unreachable: invalid Result entry")

    fn __int__(self) -> Int:
        return Int(self._value)
