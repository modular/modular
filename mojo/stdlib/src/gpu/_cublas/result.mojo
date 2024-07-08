# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@value
@register_passable("trivial")
struct Result(Formattable):
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

    @no_inline
    fn __str__(self) -> String:
        return String.format_sequence(self)

    @no_inline
    fn format_to(self, inout writer: Formatter):
        if self == Self.SUCCESS:
            writer.write("SUCCESS")
        if self == Self.NOT_INITIALIZED:
            writer.write("NOT_INITIALIZED")
        if self == Self.ALLOC_FAILED:
            writer.write("ALLOC_FAILED")
        if self == Self.INVALID_VALUE:
            writer.write("INVALID_VALUE")
        if self == Self.ARCH_MISMATCH:
            writer.write("ARCH_MISMATCH")
        if self == Self.MAPPING_ERROR:
            writer.write("MAPPING_ERROR")
        if self == Self.EXECUTION_FAILED:
            writer.write("EXECUTION_FAILED")
        if self == Self.INTERNAL_ERROR:
            writer.write("INTERNAL_ERROR")
        if self == Self.NOT_SUPPORTED:
            writer.write("NOT_SUPPORTED")
        if self == Self.LICENSE_ERROR:
            writer.write("LICENSE_ERROR")

        return abort("unreachable: invalid Result entry")

    fn __int__(self) -> Int:
        return int(self._value)
