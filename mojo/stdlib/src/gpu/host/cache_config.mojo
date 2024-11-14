# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@value
@register_passable("trivial")
struct CacheConfig(CollectionElement, EqualityComparable):
    var code: Int32

    alias PREFER_NONE = Self(0)
    """No preference for shared memory or L1 (default)."""

    alias PREFER_SHARED = Self(1)
    """Prefer larger shared memory and smaller L1 cache."""

    alias PREFER_L1 = Self(2)
    """Prefer larger L1 cache and smaller shared memory."""

    alias PREFER_EQUAL = Self(3)
    """Prefer equal sized L1 cache and shared memory."""

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        return self.code == other.code

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __init__(out self, *, other: Self):
        """Explicitly construct a deep copy of the provided value.

        Args:
            other: The value to copy.
        """
        self = other
