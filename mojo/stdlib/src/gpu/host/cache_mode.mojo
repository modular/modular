# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@value
@register_passable("trivial")
struct CacheMode:
    """Caching modes for dlcm."""

    var _value: Int

    alias NONE = Self(0)
    """Compile with no -dlcm flag specified."""

    alias L1_CACHE_DISABLED = Self(1)
    """Compile with L1 cache disabled."""

    alias L1_CACHE_ENABLED = Self(2)
    """Compile with L1 cache enabled."""

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    fn __init__(out self, *, other: Self):
        """Explicitly construct a deep copy of the provided value.

        Args:
            other: The value to copy.
        """
        self = other

    @always_inline
    fn __int__(self) -> Int:
        return self._value

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn write_to[W: Writer](self, inout writer: W):
        return writer.write(self._value)
