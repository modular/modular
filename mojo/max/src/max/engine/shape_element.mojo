# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""ShapeElement API.  See ShapeElement struct definition for details."""

from collections.string import StringSlice
from os import abort


@value
@register_passable
struct _ShapeElementType(EqualityComparable):
    """Type of a shape element."""

    var _value: UInt8

    alias STATIC = _ShapeElementType(0)
    alias UNNAMED_DYNAMIC = _ShapeElementType(1)
    alias NAMED_DYNAMIC = _ShapeElementType(2)

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)


# N.B.: We do not use @value since that would expose a constructor that takes
# _type, _static, and _name directly, while we want to keep this representation
# hidden.
struct ShapeElement(Copyable, Movable, EqualityComparable):
    """A single dimension of a possibly-dynamic shape.

    A shape element can be static or dynamic.  If dynamic, it can be named or
    unnamed.  Named dynamic shape elements must match across all inputs of a
    model.
    """

    var _type: _ShapeElementType
    var _static: Int64
    var _name: String

    @implicit
    fn __init__(out self, static: Int):
        """Create a static shape element with the given static dimension value.

        Args:
            static: The static size of the dimension.
        """
        self = Self(Int64(static))

    @implicit
    fn __init__(out self, static: Int64):
        """Create a static shape element with the given static dimension value.

        Args:
            static: The static size of the dimension.
        """
        self._type = _ShapeElementType.STATIC
        self._static = static
        self._name = String()

    # TODO(MSTDL-715):
    #   This initializer should not be necessary, we should need
    #   only the initilaizer from a `NoneType`.
    @implicit
    fn __init__(out self, unnamed: NoneType._mlir_type):
        """Create an unnamed dynamic shape element.

        Args:
            unnamed: None.
        """
        self = Self(NoneType(unnamed))

    @implicit
    fn __init__(out self, unnamed: NoneType):
        """Create an unnamed dynamic shape element.

        Args:
            unnamed: None.
        """
        self._type = _ShapeElementType.UNNAMED_DYNAMIC
        self._static = 0
        self._name = String()

    @implicit
    fn __init__(out self, owned name: String):
        """Create a dynamic shape element with the given name.

        Args:
            name:
                The name of the dimension, which must be non-empty, consist
                only of alphanumeric characters and underscores, and must not
                begin with a digit.
        """
        self._type = _ShapeElementType.NAMED_DYNAMIC
        self._static = 0
        self._name = name^

    @implicit
    fn __init__(out self, name: StringSlice):
        """Create a dynamic shape element with the given name.

        Args:
            name:
                The name of the dimension, which must be non-empty, consist
                only of alphanumeric characters and underscores, and must not
                begin with a digit.
        """
        self = Self(String(name))

    @implicit
    fn __init__(out self, name: StringLiteral):
        """Create a dynamic shape element with the given name.

        Args:
            name:
                The name of the dimension, which must be non-empty, consist
                only of alphanumeric characters and underscores, and must not
                begin with a digit.
        """
        self = Self(String(name))

    fn __moveinit__(out self, owned other: Self):
        """Initialize from another owned ShapeElement.

        Args:
            other: The ShapeElement to copy from.
        """
        self._type = other._type^
        self._static = other._static
        self._name = other._name^

    fn __copyinit__(out self, other: Self):
        """Create a copy of another ShapeElement.

        Args:
            other: The ShapeElement to copy from.
        """
        self._type = other._type
        self._static = other._static
        self._name = other._name

    @always_inline
    fn copy(self) -> Self:
        """Explicitly construct a copy of self.

        Returns:
            A copy of this value.
        """
        return self

    fn __eq__(self, other: Self) -> Bool:
        """Determine if this shape element is equal to another.

        Note that this is structural, not necessarily semantic -- two unnamed
        dynamic shape elements compare as equal, even though in a model such
        shape elements are not necessarily required to be equal.

        Args:
            other: ShapeElement to compare to.

        Returns: True if shape elements are equal; False otherwise.
        """
        if self._type != other._type:
            return False
        if self.is_static():
            return self._static == other._static
        if self.is_unnamed_dynamic():
            return True
        if self.is_named_dynamic():
            return self._name == other._name
        return abort[Bool]("unhandled type case in ShapeElement.__eq__")

    fn __ne__(self, other: Self) -> Bool:
        """Determine if this shape element is unequal to another.

        Note that this is structural, not necessarily semantic -- two unnamed
        dynamic shape elements compare as equal, even though in a model such
        shape elements are not necessarily required to be equal.

        Args:
            other: ShapeElement to compare to.

        Returns: True if shape elements are unequal; False otherwise.
        """
        return not (self == other)

    # TODO: These should be '@property' once Mojo supports that.
    fn is_static(self) -> Bool:
        """Whether this shape element is static.

        Returns: True if this shape element is static; False otherwise.
        """
        return self._type == _ShapeElementType.STATIC

    fn is_dynamic(self) -> Bool:
        """Whether this shape element is a dynamic dimension.

        Returns:
            True if this shape element is a dynamic dimension; False otherwise.
        """
        return self.is_unnamed_dynamic() or self.is_named_dynamic()

    fn is_unnamed_dynamic(self) -> Bool:
        """Whether this shape element is an unnamed dynamic dimension.

        Returns:
            True if this shape element is an unnamed dynamic dimension; False
            otherwise.
        """
        return self._type == _ShapeElementType.UNNAMED_DYNAMIC

    fn is_named_dynamic(self) -> Bool:
        """Whether this shape element is a named dynamic dimension.

        Returns:
            True if this shape element is a named dynamic dimension; False
            otherwise.
        """
        return self._type == _ShapeElementType.NAMED_DYNAMIC

    fn static_value(self) -> Int64:
        """Return size of this static shape element.

        Returns:
            Size of this static shape element, or 0 if not a static shape
            element.
        """
        if not self.is_static():
            return 0
        return self._static

    # N.B.: Returns a String rather than a StringRef for safety.  When Mojo
    # supports richer origins, we could return a StringRef instead.
    fn name(self) -> String:
        """Return name of this named dynamic shape element.

        Returns:
            Name of this named dynamic shape element, or empty string if not a
            named dynamic shape element.
        """
        if not self.is_named_dynamic():
            return ""
        return self._name
