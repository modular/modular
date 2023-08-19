# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Provides utilities for working with static and variadic lists."""

from Assert import assert_param
from TypeUtilities import rebind
from memory.unsafe import Pointer

# ===----------------------------------------------------------------------===#
# Dim
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct Dim:
    """A static or dynamic dimension modeled with an optional integer.

    This class is meant to represent an optional static dimension. When a value
    is present, the dimension has that static value. When a value is not
    present, the dimension is dynamic.
    """

    alias type = __mlir_type[`!pop.variant<i1, `, Int, `>`]
    var value: Self.type
    """Either a boolean indicating that the dimension is dynamic, or the static
    value of the dimension."""

    @always_inline
    fn __init__(value: Int) -> Dim:
        """Creates a statically-known dimension.

        Args:
            value: The static dimension value.

        Returns:
            A dimension with a static value.
        """
        return __mlir_op.`pop.variant.create`[_type : Self.type](value)

    @always_inline
    fn __init__(value: __mlir_type.index) -> Dim:
        """Creates a statically-known dimension.

        Args:
            value: The static dimension value.

        Returns:
            A dimension with a static value.
        """
        return Int(value)

    @always_inline
    fn __init__() -> Dim:
        """Creates a dynamic dimension.

        Returns:
            A dimension value with no static value.
        """
        return __mlir_op.`pop.variant.create`[_type : Self.type](
            __mlir_attr.`0 : i1`
        )

    @always_inline
    fn __bool__(self) -> Bool:
        """Returns True if the dimension has a static value.

        Returns:
            Whether the dimension has a static value.
        """
        return __mlir_op.`pop.variant.is`[testType : __mlir_attr[Int]](
            self.value
        )

    @always_inline
    fn has_value(self) -> Bool:
        """Returns True if the dimension has a static value.

        Returns:
            Whether the dimension has a static value.
        """
        return self.__bool__()

    @always_inline
    fn is_dynamic(self) -> Bool:
        """Returns True if the dimension has a dynamic value.

        Returns:
            Whether the dimension is dynamic.
        """
        return not self.has_value()

    @always_inline
    fn get(self) -> Int:
        """Gets the static dimension value.

        Returns:
            The static dimension value.
        """
        return __mlir_op.`pop.variant.get`[_type:Int](self.value)

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

    @always_inline
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

    @always_inline
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


# ===----------------------------------------------------------------------===#
# DimList
# ===----------------------------------------------------------------------===#


@register_passable("trivial")
struct DimList:
    """This type represents a list of dimensions. Each dimension may have a
    static value or not have a value, which represents a dynamic dimension."""

    var value: VariadicList[Dim]
    """The underlying storage for the list of dimensions."""

    @always_inline("nodebug")
    fn __init__(values: VariadicList[Dim]) -> Self:
        """Creates a dimension list from the given list of values.

        Args:
            values: The initial dim values list.

        Returns:
            A dimension list.
        """
        return Self {value: values}

    @always_inline("nodebug")
    fn __init__(*values: Dim) -> Self:
        """Creates a dimension list from the given Dim values.

        Args:
            values: The initial dim values.

        Returns:
            A dimension list.
        """
        return VariadicList[Dim](values)

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        """Gets the size of the DimList.

        Returns:
            The number of elements in the DimList.
        """
        return self.value.__len__()

    @always_inline("nodebug")
    fn at[i: Int](self) -> Dim:
        """Gets the dimension at a specified index.

        Parameters:
            i: The dimension index.

        Returns:
            The dimension at the specified index.
        """
        assert_param[i >= 0, "negative index"]()
        return self.value[i]

    @always_inline
    fn _product_impl[i: Int, end: Int](self) -> Dim:
        @parameter
        if i >= end:
            return 1
        else:
            return self.at[i]() * self._product_impl[i + 1, end]()

    @always_inline
    fn product[length: Int](self) -> Dim:
        """Computes the product of all the dimensions in the list.

        If any are dynamic, the result is a dynamic dimension value.

        Parameters:
            length: The number of elements in the list.

        Returns:
            The product of all the dimensions.
        """
        return self._product_impl[0, length]()

    @always_inline
    fn product_range[start: Int, end: Int](self) -> Dim:
        """Computes the product of a range of the dimensions in the list.

        If any in the range are dynamic, the result is a dynamic dimension
        value.

        Parameters:
            start: The starting index.
            end: The end index.

        Returns:
            The product of all the dimensions.
        """
        return self._product_impl[start, end]()

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
    @staticmethod
    fn create_unknown[length: Int]() -> Self:
        """Creates a dimension list of all dynamic dimension values.

        Parameters:
            length: The number of elements in the list.

        Returns:
            A list of all dynamic dimension values.
        """
        assert_param[length > 0, "length must be positive"]()
        alias u = Dim()

        @parameter
        if length == 1:
            return rebind[Self](DimList(u))
        elif length == 2:
            return rebind[Self](DimList(u, u))
        elif length == 3:
            return rebind[Self](DimList(u, u, u))
        elif length == 4:
            return rebind[Self](DimList(u, u, u, u))
        elif length == 5:
            return rebind[Self](DimList(u, u, u, u, u))
        elif length == 6:
            return rebind[Self](DimList(u, u, u, u, u, u))
        elif length == 7:
            return rebind[Self](DimList(u, u, u, u, u, u, u))
        elif length == 8:
            return rebind[Self](DimList(u, u, u, u, u, u, u, u))
        elif length == 9:
            return rebind[Self](DimList(u, u, u, u, u, u, u, u, u))
        elif length == 10:
            return rebind[Self](DimList(u, u, u, u, u, u, u, u, u, u))
        elif length == 11:
            return rebind[Self](DimList(u, u, u, u, u, u, u, u, u, u, u))
        elif length == 12:
            return rebind[Self](DimList(u, u, u, u, u, u, u, u, u, u, u, u))
        elif length == 13:
            return rebind[Self](DimList(u, u, u, u, u, u, u, u, u, u, u, u, u))
        elif length == 14:
            return rebind[Self](
                DimList(u, u, u, u, u, u, u, u, u, u, u, u, u, u)
            )
        elif length == 15:
            return rebind[Self](
                DimList(u, u, u, u, u, u, u, u, u, u, u, u, u, u, u)
            )
        else:
            return rebind[Self](
                DimList(u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u)
            )


# ===----------------------------------------------------------------------===#
# VariadicList / VariadicListMem
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn _convert_int_to_index_variadic_list[
    sz: Int
](l: VariadicList[Int]) -> VariadicList[__mlir_type.`index`]:
    """Converts VariadicList[Int] to VariadicList[__mlir_type.`index`].

    Constraints:
        Maximum supported list length is 15.

    Parameters:
        sz: The length of the list.

    Args:
        l: The list to convert.

    Returns:
        The produced index-typed list.
    """

    @parameter
    if sz == 0:
        return VariadicList[__mlir_type.`index`]()
    elif sz == 1:
        return VariadicList[__mlir_type.`index`](l[0].value)
    elif sz == 2:
        return VariadicList[__mlir_type.`index`](l[0].value, l[1].value)
    elif sz == 3:
        return VariadicList[__mlir_type.`index`](
            l[0].value, l[1].value, l[2].value
        )
    elif sz == 4:
        return VariadicList[__mlir_type.`index`](
            l[0].value, l[1].value, l[2].value, l[3].value
        )
    elif sz == 5:
        return VariadicList[__mlir_type.`index`](
            l[0].value, l[1].value, l[2].value, l[3].value, l[4].value
        )
    elif sz == 6:
        return VariadicList[__mlir_type.`index`](
            l[0].value,
            l[1].value,
            l[2].value,
            l[3].value,
            l[4].value,
            l[5].value,
        )
    elif sz == 7:
        return VariadicList[__mlir_type.`index`](
            l[0].value,
            l[1].value,
            l[2].value,
            l[3].value,
            l[4].value,
            l[5].value,
            l[6].value,
        )
    elif sz == 8:
        return VariadicList[__mlir_type.`index`](
            l[0].value,
            l[1].value,
            l[2].value,
            l[3].value,
            l[4].value,
            l[5].value,
            l[6].value,
            l[7].value,
        )
    elif sz == 9:
        return VariadicList[__mlir_type.`index`](
            l[0].value,
            l[1].value,
            l[2].value,
            l[3].value,
            l[4].value,
            l[5].value,
            l[6].value,
            l[7].value,
            l[8].value,
        )
    elif sz == 10:
        return VariadicList[__mlir_type.`index`](
            l[0].value,
            l[1].value,
            l[2].value,
            l[3].value,
            l[4].value,
            l[5].value,
            l[6].value,
            l[7].value,
            l[8].value,
            l[9].value,
        )
    elif sz == 11:
        return VariadicList[__mlir_type.`index`](
            l[0].value,
            l[1].value,
            l[2].value,
            l[3].value,
            l[4].value,
            l[5].value,
            l[6].value,
            l[7].value,
            l[8].value,
            l[9].value,
            l[10].value,
        )
    elif sz == 12:
        return VariadicList[__mlir_type.`index`](
            l[0].value,
            l[1].value,
            l[2].value,
            l[3].value,
            l[4].value,
            l[5].value,
            l[6].value,
            l[7].value,
            l[8].value,
            l[9].value,
            l[10].value,
            l[11].value,
        )
    elif sz == 13:
        return VariadicList[__mlir_type.`index`](
            l[0].value,
            l[1].value,
            l[2].value,
            l[3].value,
            l[4].value,
            l[5].value,
            l[6].value,
            l[7].value,
            l[8].value,
            l[9].value,
            l[10].value,
            l[11].value,
            l[12].value,
        )
    elif sz == 14:
        return VariadicList[__mlir_type.`index`](
            l[0].value,
            l[1].value,
            l[2].value,
            l[3].value,
            l[4].value,
            l[5].value,
            l[6].value,
            l[7].value,
            l[8].value,
            l[9].value,
            l[10].value,
            l[11].value,
            l[12].value,
            l[13].value,
        )
    elif sz == 15:
        return VariadicList[__mlir_type.`index`](
            l[0].value,
            l[1].value,
            l[2].value,
            l[3].value,
            l[4].value,
            l[5].value,
            l[6].value,
            l[7].value,
            l[8].value,
            l[9].value,
            l[10].value,
            l[11].value,
            l[12].value,
            l[13].value,
            l[14].value,
        )
    assert_param[sz == 16]()
    return VariadicList[__mlir_type.`index`](
        l[0].value,
        l[1].value,
        l[2].value,
        l[3].value,
        l[4].value,
        l[5].value,
        l[6].value,
        l[7].value,
        l[8].value,
        l[9].value,
        l[10].value,
        l[11].value,
        l[12].value,
        l[13].value,
        l[14].value,
        l[15].value,
    )


@register_passable("trivial")
struct VariadicList[type: AnyType]:
    """A utility class to access variadic function arguments. Provides a "list"
    view of the function argument so that the size of the argument list and each
    individual argument can be accessed.

    Parameters:
        type: The type of the elements in the list.
    """

    alias StorageType = __mlir_type[`!kgen.variadic<`, type, `>`]
    var value: Self.StorageType
    """The underlying storage for the variadic list."""

    @always_inline
    fn __init__(*value: type) -> Self:
        """Constructs a VariadicList from a variadic list of arguments.

        Args:
            value: The variadic argument list to construct the variadic list
              with.

        Returns:
            The VariadicList constructed.
        """
        return Self(value)

    @always_inline
    fn __init__(value: Self.StorageType) -> Self:
        """Constructs a VariadicList from a variadic argument type.

        Args:
            value: The variadic argument to construct the list with.

        Returns:
            The VariadicList constructed.
        """
        return Self {value: value}

    @always_inline
    fn __len__(self) -> Int:
        """Gets the size of the list.

        Returns:
            The number of elements on the variadic list.
        """

        return __mlir_op.`pop.variadic.size`(self.value)

    @always_inline
    fn __getitem__(self, index: Int) -> type:
        """Gets a single element on the variadic list.

        Args:
            index: The index of the element to access on the list.

        Returns:
            The element on the list corresponding to the given index.
        """
        return __mlir_op.`pop.variadic.get`(self.value, index.value)


@register_passable("trivial")
struct VariadicListMem[type: AnyType]:
    """A utility class to access variadic function arguments of memory-only
    types that may have ownership. It exposes pointers to the elements in a way
    that can be enumerated.  Each element may be accessed with
    `__get_address_as_lvalue`.

    Parameters:
        type: The type of the elements in the list.
    """

    alias StorageType = __mlir_type[`!kgen.variadic<!pop.pointer<`, type, `>>`]
    var value: Self.StorageType
    """The underlying storage, a variadic list of pointers to elements of the
    given type."""

    @always_inline
    fn __init__(value: Self.StorageType) -> Self:
        """Constructs a VariadicList from a variadic argument type.

        Args:
            value: The variadic argument to construct the list with.

        Returns:
            The VariadicList constructed.
        """
        return Self {value: value}

    @always_inline
    fn __len__(self) -> Int:
        """Gets the size of the list.

        Returns:
            The number of elements on the variadic list.
        """

        return __mlir_op.`pop.variadic.size`(self.value)

    @always_inline
    fn __getitem__(self, index: Int) -> __mlir_type[`!pop.pointer<`, type, `>`]:
        """Gets a single element on the variadic list.

        Args:
            index: The index of the element to access on the list.

        Returns:
            A low-level pointer to the element on the list corresponding to the
            given index.
        """
        return __mlir_op.`pop.variadic.get`(self.value, index.value)
