# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Provides utilities for working with static and variadic lists.

You can import these APIs from the `buffer` package. For example:

```mojo
from buffer import Dim
```
"""

from utils import IndexList, StaticTuple, unroll

# ===-----------------------------------------------------------------------===#
# Dim
# ===-----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct Dim(Intable, Stringable, Writable, ImplicitlyBoolable):
    """A static or dynamic dimension modeled with an optional integer.

    This class is meant to represent an optional static dimension. When a value
    is present, the dimension has that static value. When a value is not
    present, the dimension is dynamic.
    """

    alias _sentinel = -31337
    """The sentinel value to use if the dimension is dynamic.  This value was
    chosen to be a visible-in-the-debugger sentinel.  We can't use Int.MIN
    because that value is target-dependent and won't fold in parameters."""

    var _value_or_missing: Int
    """The dimension value to use or `_sentinel` if the dimension is dynamic."""

    @always_inline("nodebug")
    @implicit
    fn __init__[I: Intable](mut self, value: I):
        """Creates a statically-known dimension.

        Parameters:
            I: The Intable type.

        Args:
            value: The static dimension value.
        """
        self._value_or_missing = Int(value)

    @always_inline("nodebug")
    @implicit
    fn __init__[I: Indexer](mut self, value: I):
        """Creates a statically-known dimension.

        Parameters:
            I: A type that can be used as an index.

        Args:
            value: The static dimension value.
        """
        self = Dim(index(value))

    @always_inline("builtin")
    @implicit
    fn __init__(out self, value: __mlir_type.index):
        """Creates a statically-known dimension.

        Args:
            value: The static dimension value.
        """
        self._value_or_missing = Int(value)

    @always_inline("builtin")
    @implicit
    fn __init__(out self, value: Int):
        """Creates a statically-known dimension.

        Args:
            value: The static dimension value.
        """
        self._value_or_missing = value

    @always_inline("builtin")
    fn __init__(out self):
        """Creates a dynamic dimension with no static value."""
        self._value_or_missing = Self._sentinel

    @always_inline("builtin")
    fn __bool__(self) -> Bool:
        """Returns True if the dimension has a static value.

        Returns:
            Whether the dimension has a static value.
        """
        return self._value_or_missing != Self._sentinel

    @always_inline("builtin")
    fn __as_bool__(self) -> Bool:
        """Returns True if the dimension has a static value.

        Returns:
            Whether the dimension has a static value.
        """
        return self.__bool__()

    @always_inline("nodebug")
    fn has_value(self) -> Bool:
        """Returns True if the dimension has a static value.

        Returns:
            Whether the dimension has a static value.
        """
        return self.__bool__()

    @always_inline("nodebug")
    fn is_dynamic(self) -> Bool:
        """Returns True if the dimension has a dynamic value.

        Returns:
            Whether the dimension is dynamic.
        """
        return not self.has_value()

    @always_inline("nodebug")
    fn get(self) -> Int:
        """Gets the static dimension value.

        Returns:
            The static dimension value.
        """
        # TODO: Shouldn't this assert the value is present?
        return self._value_or_missing

    @always_inline
    fn is_multiple[alignment: Int](self) -> Bool:
        """Checks if the dimension is aligned.

        Parameters:
            alignment: The alignment requirement.

        Returns:
            Whether the dimension is aligned.
        """
        if self.is_dynamic():
            return False
        return self.get() % alignment == 0

    @always_inline("nodebug")
    fn __index__(self) -> __mlir_type.index:
        """Convert to index.

        Returns:
            The corresponding __mlir_type.index value.
        """
        return self.get().value

    @always_inline("nodebug")
    fn __mul__(self, rhs: Dim) -> Dim:
        """Multiplies two dimensions.

        If either are unknown, the result is unknown as well.

        Args:
            rhs: The other dimension.

        Returns:
            The product of the two dimensions.
        """
        if not self or not rhs:
            return Dim()
        return Dim(self.get() * rhs.get())

    @always_inline("nodebug")
    fn __imul__(mut self, rhs: Dim):
        """Inplace multiplies two dimensions.

        If either are unknown, the result is unknown as well.

        Args:
            rhs: The other dimension.
        """
        self = self * rhs

    @always_inline
    fn __floordiv__(self, rhs: Dim) -> Dim:
        """Divide by the given dimension and round towards negative infinity.

        If either are unknown, the result is unknown as well.

        Args:
            rhs: The divisor dimension.

        Returns:
            The floor division of the two dimensions.
        """
        if not self or not rhs:
            return Dim()
        return Dim(self.get() // rhs.get())

    @always_inline
    fn __rfloordiv__(self, rhs: Dim) -> Dim:
        """Divide the given argument by self and round towards negative
        infinity.

        If either are unknown, the result is unknown as well.

        Args:
            rhs: The dimension to divide by this Dim.

        Returns:
            The floor of the argument divided by self.
        """
        return rhs // self

    @always_inline("nodebug")
    fn __int__(self) -> Int:
        """Gets the static dimension value.

        Returns:
            The static dimension value.
        """
        return self.get()

    @always_inline("nodebug")
    fn __eq__(self, rhs: Dim) -> Bool:
        """Compares two dimensions for equality.

        Args:
            rhs: The other dimension.

        Returns:
            True if the dimensions are the same.
        """
        if self and rhs:
            return self.get() == rhs.get()
        return (not self) == (not rhs)

    @always_inline("nodebug")
    fn __ne__(self, rhs: Dim) -> Bool:
        """Compare two dimensions for inequality.

        Args:
            rhs: The dimension to compare.

        Returns:
            True if they are not equal.
        """
        return not self == rhs

    @no_inline
    fn __str__(self) -> String:
        """Converts the Dim to a String. If the value is unknown, then the
        string "?" is returned.

        Returns:
            The string representation of the type.
        """
        return String.write(self)

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        """
        Formats this DimList to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """

        if self.is_dynamic():
            return writer.write("?")
        else:
            return writer.write(Int(self))

    fn or_else(self, default: Int) -> Int:
        """Return the underlying value contained in the Optional or a default
        value if the Optional's underlying value is not present.

        Args:
            default: The new value to use if no value was present.

        Returns:
            The underlying value contained in the Optional or a default value.
        """
        if self:
            return self.get()
        return default


# ===-----------------------------------------------------------------------===#
# DimList
# ===-----------------------------------------------------------------------===#


@register_passable("trivial")
struct DimList(
    Sized,
    Stringable,
    Representable,
    Writable,
):
    """This type represents a list of dimensions. Each dimension may have a
    static value or not have a value, which represents a dynamic dimension."""

    var value: VariadicList[Dim]
    """The underlying storage for the list of dimensions."""

    @always_inline("nodebug")
    @implicit
    fn __init__[Intable: Intable](mut self, value: Intable):
        """Creates a dimension list from the given list of values.

        Parameters:
            Intable: A type able to be converted to an `Int`.

        Args:
            value: The initial dim values list.
        """
        self.value = VariadicList[Dim](Int(value))

    @always_inline("nodebug")
    @implicit
    fn __init__[I: Indexer](mut self, values: (I,)):
        """Creates a dimension list from the given list of values.

        Parameters:
            I: A type that can be used as an index.

        Args:
            values: The initial dim values list.
        """
        self.value = VariadicList[Dim](Int(values[0]))

    @always_inline("nodebug")
    @implicit
    fn __init__[I0: Indexer, I1: Indexer](mut self, values: (I0, I1)):
        """Creates a dimension list from the given list of values.

        Parameters:
            I0: A type that can be used as an Index.
            I1: A type that can be used as an Index.

        Args:
            values: The initial dim values list.
        """
        self.value = VariadicList[Dim](Int(values[0]), Int(values[1]))

    @always_inline("nodebug")
    @implicit
    fn __init__[
        I0: Indexer, I1: Indexer, I2: Indexer
    ](mut self, values: (I0, I1, I2)):
        """Creates a dimension list from the given list of values.

        Parameters:
            I0: A type that can be used as an Index.
            I1: A type that can be used as an Index.
            I2: A type that can be used as an Index.

        Args:
            values: The initial dim values list.
        """
        self.value = VariadicList[Dim](
            Int(values[0]), Int(values[1]), Int(values[2])
        )

    @always_inline("nodebug")
    fn __init__[I0: Indexer, I1: Indexer](mut self, val0: I0, val1: I1):
        """Creates a dimension list from the given list of values.

        Parameters:
            I0: A type that can be used as an Index.
            I1: A type that can be used as an Index.

        Args:
            val0: The initial dim value.
            val1: The initial dim value.
        """
        self.value = VariadicList[Dim](Int(val0), Int(val1))

    @always_inline("nodebug")
    fn __init__[
        I0: Indexer, I1: Indexer, I2: Indexer
    ](mut self, val0: I0, val1: I1, val2: I2):
        """Creates a dimension list from the given list of values.

        Parameters:
            I0: A type that can be used as an Index.
            I1: A type that can be used as an Index.
            I2: A type that can be used as an Index.

        Args:
            val0: The initial dim value.
            val1: The initial dim value.
            val2: The initial dim value.
        """
        self.value = VariadicList[Dim](Int(val0), Int(val1), Int(val2))

    @always_inline("nodebug")
    fn __init__[
        I0: Indexer, I1: Indexer, I2: Indexer, I3: Indexer
    ](mut self, val0: I0, val1: I1, val2: I2, val3: I3):
        """Creates a statically-known dimension.

        Parameters:
            I0: A type that can be used as an Index.
            I1: A type that can be used as an Index.
            I2: A type that can be used as an Index.
            I3: A type that can be used as an Index.

        Args:
            val0: The initial dim value.
            val1: The initial dim value.
            val2: The initial dim value.
            val3: The initial dim value.
        """
        self = VariadicList[Dim](
            Int(val0),
            Int(val1),
            Int(val2),
            Int(val3),
        )

    @always_inline("nodebug")
    @implicit
    fn __init__(out self, values: VariadicList[Dim]):
        """Creates a dimension list from the given list of values.

        Args:
            values: The initial dim values list.
        """
        self.value = values

    @always_inline("nodebug")
    @implicit
    fn __init__(out self, *values: Dim):
        """Creates a dimension list from the given Dim values.

        Args:
            values: The initial dim values.
        """
        self.value = values

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        """Gets the size of the DimList.

        Returns:
            The number of elements in the DimList.
        """
        return len(self.value)

    @always_inline("nodebug")
    fn get[i: Int](self) -> Int:
        """Gets the static dimension value at a specified index.

        Parameters:
            i: The dimension index.

        Returns:
            The static dimension value at the specified index.
        """
        constrained[i >= 0, "index must be positive"]()
        return self.value[i].get()

    @always_inline("nodebug")
    fn at[i: Int](self) -> Dim:
        """Gets the dimension at a specified index.

        Parameters:
            i: The dimension index.

        Returns:
            The dimension at the specified index.
        """
        constrained[i >= 0, "index must be positive"]()
        return self.value[i]

    @always_inline("nodebug")
    fn has_value[i: Int](self) -> Bool:
        """Returns True if the dimension at the given index has a static value.

        Parameters:
            i: The dimension index.

        Returns:
            Whether the specified dimension has a static value.
        """
        constrained[i >= 0, "index must be positive"]()
        return self.value[i].__bool__()

    @always_inline
    fn product[length: Int](self) -> Dim:
        """Computes the product of the first `length` dimensions in the list.

        If any are dynamic, the result is a dynamic dimension value.

        Parameters:
            length: The number of elements in the list.

        Returns:
            The product of the first `length` dimensions.
        """
        return self.product[0, length]()

    @always_inline
    fn product[start: Int, end: Int](self) -> Dim:
        """Computes the product of a range of the dimensions in the list.

        If any in the range are dynamic, the result is a dynamic dimension
        value.

        Parameters:
            start: The starting index.
            end: The end index.

        Returns:
            The product of all the dimensions.
        """

        if not self.all_known[start, end]():
            return Dim()

        var res = 1

        @parameter
        for i in range(start, end):
            res *= self.value[i].get()
        return res

    @always_inline
    fn product(self) -> Dim:
        """Computes the product of all the dimensions in the list.

        If any are dynamic, the result is a dynamic dimension value.

        Returns:
            The product of all the dimensions.
        """
        var res = 1
        for i in range(len(self)):
            if not self.value[i]:
                return Dim()
            var val = self.value[i].get()
            if val:
                res *= val
        return res

    @always_inline
    fn _contains_impl[i: Int, length: Int](self, value: Dim) -> Bool:
        @parameter
        if i >= length:
            return False
        else:
            return self.at[i]() == value or self._contains_impl[i + 1, length](
                value
            )

    @always_inline
    fn contains[length: Int](self, value: Dim) -> Bool:
        """Determines whether the dimension list contains a specified dimension
        value.

        Parameters:
            length: The number of elements in the list.

        Args:
            value: The value to find.

        Returns:
            True if the list contains a dimension of the specified value.
        """
        return self._contains_impl[0, length](value)

    @always_inline
    fn all_known[length: Int](self) -> Bool:
        """Determines whether all dimensions are statically known.

        Parameters:
            length: The number of elements in the list.

        Returns:
            True if all dimensions have a static value.
        """
        return not self.contains[length](Dim())

    @always_inline
    fn all_known[start: Int, end: Int](self) -> Bool:
        """Determines whether all dimensions within [start, end) are statically
        known.

        Parameters:
            start: The first queried dimension.
            end: The last queried dimension.

        Returns:
            True if all queried dimensions have a static value.
        """
        return not self._contains_impl[start, end](Dim())

    @always_inline
    fn into_index_list[rank: Int](self) -> IndexList[rank]:
        """Copy the DimList values into an `IndexList`, providing the rank.

        Parameters:
            rank: The rank of the output IndexList.

        Returns:
            An IndexList with the same dimensions as the DimList.

        ```mojo
        from buffer import DimList

        var dim_list = DimList(2, 4)
        var index_list = dim_list.into_index_list[rank=2]()
        ```
        .
        """
        var num_elements = len(self)
        debug_assert(
            rank == num_elements,
            "[DimList] mismatch in the number of elements",
        )
        var index_list = IndexList[rank]()

        @parameter
        for idx in range(rank):
            index_list[idx] = Int(self.at[idx]())

        return index_list

    @always_inline
    @staticmethod
    fn create_unknown[length: Int]() -> Self:
        """Creates a dimension list of all dynamic dimension values.

        Parameters:
            length: The number of elements in the list.

        Returns:
            A list of all dynamic dimension values.
        """
        constrained[length > 0, "length must be positive"]()

        return VariadicList[Dim](
            __mlir_op.`pop.variadic.splat`[
                numElements = length.value,
                _type = __mlir_type[`!kgen.variadic<`, Dim, `>`],
            ](Dim())
        )

    fn __str__(self) -> String:
        """Converts the DimList to a String. The String is a comma separated
        list of the string representation of Dim.

        Returns:
            The string representation of the type.
        """
        return String.write(self)

    fn __repr__(self) -> String:
        """Converts the DimList to a readable String representation.

        Returns:
            The string representation of the type.
        """
        return "DimList(" + String(self) + ")"

    @always_inline("nodebug")
    fn __eq__(self, rhs: DimList) -> Bool:
        """Compares two DimLists for equality.

        DimLists are considered equal if all non-dynamic Dims have similar
        values and all dynamic Dims in self are also dynamic in rhs.

        Args:
            rhs: The other DimList.

        Returns:
            True if the DimLists are the same.
        """
        if len(self) != len(rhs):
            return False

        for i in range(len(self)):
            if self.value[i] != rhs.value[i]:
                return False

        return True

    fn write_to[W: Writer](self, mut writer: W):
        """
        Formats this DimList to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """

        writer.write("[")

        for i in range(len(self)):
            if i:
                writer.write(", ")
            writer.write(self.value[i])

        writer.write("]")


@always_inline
fn _make_tuple[
    size: Int, *, unsigned: Bool = False
](values: DimList, out result: IndexList[size, unsigned=unsigned]):
    """Creates a tuple constant using the specified values.

    Args:
        values: The list of values.

    Returns:
        A tuple with the values filled in.
    """
    var array = __mlir_op.`pop.array.repeat`[
        _type = __mlir_type[
            `!pop.array<`, size.value, `, `, result._int_type, `>`
        ]
    ](result._int_type(0))

    @parameter
    for idx in range(size):
        array = __mlir_op.`pop.array.replace`[
            _type = __mlir_type[
                `!pop.array<`, size.value, `, `, result._int_type, `>`
            ],
            index = idx.value,
        ](result._int_type(values.at[idx]().get()), array)

    return __type_of(result)(array)


@always_inline
fn _make_partially_static_index_list[
    size: Int, static_list: DimList, *, unsigned: Bool = False
](dynamic_list: IndexList, out result: IndexList[size, unsigned=unsigned]):
    """Creates a tuple constant using the specified values.

    Args:
        dynamic_list: The dynamic list of values.

    Returns:
        A tuple with the values filled in.
    """
    var array = __mlir_op.`pop.array.repeat`[
        _type = __mlir_type[
            `!pop.array<`, size.value, `, `, result._int_type, `>`
        ]
    ](result._int_type(0))

    @parameter
    for idx in range(size):

        @parameter
        if static_list.at[idx]().is_dynamic():
            array = __mlir_op.`pop.array.replace`[
                _type = __mlir_type[
                    `!pop.array<`, size.value, `, `, result._int_type, `>`
                ],
                index = idx.value,
            ](result._int_type(dynamic_list[idx]), array)
        else:
            array = __mlir_op.`pop.array.replace`[
                _type = __mlir_type[
                    `!pop.array<`, size.value, `, `, result._int_type, `>`
                ],
                index = idx.value,
            ](result._int_type(static_list.at[idx]().get()), array)

    return __type_of(result)(array)
