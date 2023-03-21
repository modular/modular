# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Int import Int

from Assert import assert_param, assert_param_bool, debug_assert
from Functional import unroll
from TypeUtilities import rebind
from Assert import assert_param_bool_msg


# ===----------------------------------------------------------------------===#
# Dim
# ===----------------------------------------------------------------------===#


@register_passable
struct Dim:
    """A static or dynamic dimension modeled with an optional integer. This
    class is meant to represent an optional static dimension. When a value is
    present, the dimension has that static value. When a value is not present,
    the dimension is dynamic."""

    alias type = __mlir_type[`!pop.variant<i1, `, Int, `>`]
    var value: type

    @always_inline
    fn __init__(value: type) -> Dim:
        """Create a dimension from its underlying value type.

        Args:
            value (type): The underlying value.
        Returns:
            Dim: A dimension value.
        """
        return Dim {value: value}

    @always_inline
    fn __init__(value: Int) -> Dim:
        """Create a statically-known dimension.

        Args:
            value (Int): The static dimension value.
        Returns:
            Dim: A dimension with a static value.
        """
        return __mlir_op.`pop.variant.create`[_type:type](value)

    @always_inline
    fn __init__(value: __mlir_type.index) -> Dim:
        """Create a statically-known dimension.

        Args:
            value (__mlir_type.index): The static dimension value.
        Returns:
            Dim: A dimension with a static value.
        """
        return Int(value)

    @always_inline
    fn __init__() -> Dim:
        """Create a dynamic dimension.

        Returns:
            Dim: A dimension value with no static value.
        """
        return __mlir_op.`pop.variant.create`[_type:type](__mlir_attr.`0 : i1`)

    @always_inline
    fn __copy__(self) -> Dim:
        """Clone the dimension.

        Args:
            self (Self): The value to clone.
        Returns:
            Dim: A copy of the dimension.
        """
        return self.value

    @always_inline
    fn __bool__(self) -> Bool:
        """Return true if the dimension has a static value.

        Args:
            self (Self): The dimension.
        Returns:
            Bool: Whether the dimension has a value.
        """
        return __mlir_op.`pop.variant.is`[testType : __mlir_attr[Int]](
            self.value
        )

    @always_inline
    fn has_value(self) -> Bool:
        """Return true if the dimension has a static value.

        Args:
            self (Self): The dimension.
        Returns:
            Bool: Whether the dimension has a value.
        """
        return self.__bool__()

    @always_inline
    fn is_dynamic(self) -> Bool:
        """Return true if the dimension has a dynamic value.

        Args:
            self (Self): The dimension.
        Returns:
            Bool: Whether the dimension is dynamic.
        """
        return not self.has_value()

    @always_inline
    fn get(self) -> Int:
        """Get the static dimension value.

        Args:
            self (Self): The dimension.
        Returns:
            Bool: Whether the dimension is dynamic.
        """
        return __mlir_op.`pop.variant.get`[_type:Int](self.value)

    @always_inline
    fn __mul__(self, rhs: Dim) -> Dim:
        """Multiply two dimensions. If either are unknown, the result is unknown
        as well.

        Args:
            self (Self): The dimension.
            rhs (Dim): The other dimension.
        Returns:
            Dim: The product of the two dimensions.
        """
        if not self or not rhs:
            return Dim()
        return Dim(self.get() * rhs.get())

    @always_inline
    fn __eq__(self, rhs: Dim) -> Bool:
        """Compare two dimensions for equality.

        Args:
            self (Self): The dimension.
            rhs (Dim): The other dimension.
        Returns:
            Bool: True if the dimensions are the same.
        """
        if self and rhs:
            return self.get() == rhs.get()
        return (not self) == (not rhs)

    @staticmethod
    @always_inline
    fn from_index[value: __mlir_type.index]() -> Dim:
        """Create a dimension from an index value.

        ParamArgs:
            value (__mlir_type.index): The index dimension value.
        Returns:
            Dim: A dimension value.
        """

        @parameter
        if value == __mlir_attr.`#kgen.unknown : index`:
            return Dim()
        else:
            return Dim(Int(value))


# ===----------------------------------------------------------------------===#
# DimList
# ===----------------------------------------------------------------------===#


@register_passable
struct DimList[length: Int]:
    """This type represents a list of dimensions. Each dimension may have a
    static value or not have a value, which represents a dynamic dimension."""

    alias list_type = __mlir_type[
        `!kgen.list<`, Dim, `[`, length.__as_mlir_index(), `]>`
    ]
    var value: list_type

    @always_inline("nodebug")
    fn __init__(value: list_type) -> Self:
        """Create a dimension list from the underlying value type.

        Args:
            value (list_type): The underlying value.
        Returns:
            Self: A dimension list.
        """
        return Self {value: value}

    @always_inline("nodebug")
    fn __clone__(self&) -> Self:
        """Copy a dimension list.

        Args:
            self (Self): The list to copy.
        Returns:
            Self: A dimension list.
        """
        return Self {value: self.value}

    @always_inline("nodebug")
    fn at[i: Int](self) -> Dim:
        """Get the dimension at a specified index.

        ParamArgs:
            i (Int): The dimension index.
        Args:
            self (Self): The list to index.
        Returns:
            Dim: The dimension at the specified index.
        """
        assert_param_bool_msg[i >= 0, "negative index"]()
        assert_param_bool_msg[i < length, "index exceeds length"]()
        return __mlir_op.`pop.list.get`[index : i.__as_mlir_index()](self.value)

    @always_inline
    fn _product_impl[i: Int, end: Int](self) -> Dim:
        @parameter
        if i >= end:
            return Int(1)
        else:
            return self.at[i]() * self._product_impl[i + 1, end]()

    @always_inline
    fn product(self) -> Dim:
        """Compute the product of all the dimensions in the list. If any are
        dynamic, the result is a dynamic dimension value.

        Args:
            self (Self): The list whose product to find.
        Returns:
            Dim: The product of all the dimensions.
        """
        return self._product_impl[0, length]()

    @always_inline
    fn product_range[start: Int, end: Int](self) -> Dim:
        """Compute the product of a range of the dimensions in the list. If any
        in the range are dynamic, the result is a dynamic dimension value.

        ParamArgs:
            start (Int): The starting index.
            end (Int): The end index.
        Args:
            self (Self): The list whose product to find.
        Returns:
            Dim: The product of all the dimensions.
        """
        return self._product_impl[start, end]()

    @always_inline
    fn _contains_impl[i: Int](self, value: Dim) -> Bool:
        @parameter
        if i >= length:
            return False
        else:
            return self.at[i]() == value or self._contains_impl[i + 1](value)

    @always_inline
    fn contains(self, value: Dim) -> Bool:
        """Determine whether the dimension list contains a specified dimension
        value.

        Args:
            self (Self): The list to search.
            value (Dim): The value to find.
        Returns:
            Bool: True if the list contains a dimension of the specified value.
        """
        return self._contains_impl[0](value)

    @always_inline
    fn all_known(self) -> Bool:
        """Determine whether all dimensions are statically known.

        Args:
            self (Self): The list to check.
        Returns:
            Bool: True if all dimensions have a static value.
        """
        return not self.contains(Dim())

    @always_inline
    @staticmethod
    fn create_unknown() -> Self:
        """Create a dimension list of all dynamic dimension values.

        Returns:
            Self: A list of all dynamic dimension values.
        """
        assert_param_bool_msg[length > 0, "length must be positive"]()
        alias u = Dim()

        @parameter
        if length == 1:
            return rebind[Self](create_dim_list(u))
        elif length == 2:
            return rebind[Self](create_dim_list(u, u))
        elif length == 3:
            return rebind[Self](create_dim_list(u, u, u))
        elif length == 4:
            return rebind[Self](create_dim_list(u, u, u, u))
        elif length == 5:
            return rebind[Self](create_dim_list(u, u, u, u, u))
        elif length == 6:
            return rebind[Self](create_dim_list(u, u, u, u, u, u))
        elif length == 7:
            return rebind[Self](create_dim_list(u, u, u, u, u, u, u))
        elif length == 8:
            return rebind[Self](create_dim_list(u, u, u, u, u, u, u, u))
        elif length == 9:
            return rebind[Self](create_dim_list(u, u, u, u, u, u, u, u, u))
        elif length == 10:
            return rebind[Self](create_dim_list(u, u, u, u, u, u, u, u, u, u))
        elif length == 11:
            return rebind[Self](
                create_dim_list(u, u, u, u, u, u, u, u, u, u, u)
            )
        elif length == 12:
            return rebind[Self](
                create_dim_list(u, u, u, u, u, u, u, u, u, u, u, u)
            )
        elif length == 13:
            return rebind[Self](
                create_dim_list(u, u, u, u, u, u, u, u, u, u, u, u, u)
            )
        elif length == 14:
            return rebind[Self](
                create_dim_list(u, u, u, u, u, u, u, u, u, u, u, u, u, u)
            )
        elif length == 15:
            return rebind[Self](
                create_dim_list(u, u, u, u, u, u, u, u, u, u, u, u, u, u, u)
            )
        else:
            return rebind[Self](
                create_dim_list(u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u)
            )


# ===----------------------------------------------------------------------===#
# create_dim_list
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn create_dim_list(e0: Dim) -> DimList[1]:
    """Creates a list given a type and elements.

    Args:
        e0 (Dim): The 1st element of the returned list.

    Returns:
        DimList[1]: The list containing the elements.
    """
    return __mlir_op.`pop.list.create`[
        _type : __mlir_type[`!kgen.list<`, Dim, `[1]>`]
    ](e0)


@always_inline("nodebug")
fn create_dim_list(e0: Dim, e1: Dim) -> DimList[2]:
    """Creates a list given a type and elements.

    Args:
        e0 (Dim): The 1st element of the returned list.
        e1 (Dim): The 2nd element of the returned list.

    Returns:
        DimList[2]: The list containing the elements.
    """
    return __mlir_op.`pop.list.create`[
        _type : __mlir_type[`!kgen.list<`, Dim, `[2]>`]
    ](e0, e1)


@always_inline("nodebug")
fn create_dim_list(e0: Dim, e1: Dim, e2: Dim) -> DimList[3]:
    """Creates a list given a type and elements.

    Args:
        e0 (Dim): The 1st element of the returned list.
        e1 (Dim): The 2nd element of the returned list.
        e2 (Dim): The 3rd element of the returned list.

    Returns:
        DimList[3]: The list containing the elements.
    """
    return __mlir_op.`pop.list.create`[
        _type : __mlir_type[`!kgen.list<`, Dim, `[3]>`]
    ](e0, e1, e2)


@always_inline("nodebug")
fn create_dim_list(e0: Dim, e1: Dim, e2: Dim, e3: Dim) -> DimList[4]:
    """Creates a list given a type and elements.

    Args:
        e0 (Dim): The 1st element of the returned list.
        e1 (Dim): The 2nd element of the returned list.
        e2 (Dim): The 3rd element of the returned list.
        e3 (Dim): The 4th element of the returned list.

    Returns:
        DimList[4]: The list containing the elements.
    """
    return __mlir_op.`pop.list.create`[
        _type : __mlir_type[`!kgen.list<`, Dim, `[4]>`]
    ](e0, e1, e2, e3)


@always_inline("nodebug")
fn create_dim_list(e0: Dim, e1: Dim, e2: Dim, e3: Dim, e4: Dim) -> DimList[5]:
    """Creates a list given a type and elements.

    Args:
        e0 (Dim): The 1st element of the returned list.
        e1 (Dim): The 2nd element of the returned list.
        e2 (Dim): The 3rd element of the returned list.
        e3 (Dim): The 4th element of the returned list.
        e4 (Dim): The 5th element of the returned list.

    Returns:
        DimList[5]: The list containing the elements.
    """
    return __mlir_op.`pop.list.create`[
        _type : __mlir_type[`!kgen.list<`, Dim, `[5]>`]
    ](e0, e1, e2, e3, e4)


@always_inline("nodebug")
fn create_dim_list(
    e0: Dim, e1: Dim, e2: Dim, e3: Dim, e4: Dim, e5: Dim
) -> DimList[6]:
    """Creates a list given a type and elements.

    Args:
        e0 (Dim): The 1st element of the returned list.
        e1 (Dim): The 2nd element of the returned list.
        e2 (Dim): The 3rd element of the returned list.
        e3 (Dim): The 4th element of the returned list.
        e4 (Dim): The 5th element of the returned list.
        e5 (Dim): The 6th element of the returned list.

    Returns:
        DimList[6]: The list containing the elements.
    """
    return __mlir_op.`pop.list.create`[
        _type : __mlir_type[`!kgen.list<`, Dim, `[6]>`]
    ](e0, e1, e2, e3, e4, e5)


@always_inline("nodebug")
fn create_dim_list(
    e0: Dim,
    e1: Dim,
    e2: Dim,
    e3: Dim,
    e4: Dim,
    e5: Dim,
    e6: Dim,
) -> DimList[7]:
    """Creates a list given a type and elements.

    Args:
        e0 (Dim): The 1st element of the returned list.
        e1 (Dim): The 2nd element of the returned list.
        e2 (Dim): The 3rd element of the returned list.
        e3 (Dim): The 4th element of the returned list.
        e4 (Dim): The 5th element of the returned list.
        e5 (Dim): The 6th element of the returned list.
        e6 (Dim): The 7th element of the returned list.

    Returns:
        DimList[7]: The list containing the elements.
    """
    return __mlir_op.`pop.list.create`[
        _type : __mlir_type[`!kgen.list<`, Dim, `[7]>`]
    ](e0, e1, e2, e3, e4, e5, e6)


@always_inline("nodebug")
fn create_dim_list(
    e0: Dim,
    e1: Dim,
    e2: Dim,
    e3: Dim,
    e4: Dim,
    e5: Dim,
    e6: Dim,
    e7: Dim,
) -> DimList[8]:
    """Creates a list given a type and elements.

    Args:
        e0 (Dim): The 1st element of the returned list.
        e1 (Dim): The 2nd element of the returned list.
        e2 (Dim): The 3rd element of the returned list.
        e3 (Dim): The 4th element of the returned list.
        e4 (Dim): The 5th element of the returned list.
        e5 (Dim): The 6th element of the returned list.
        e6 (Dim): The 7th element of the returned list.
        e7 (Dim): The 8th element of the returned list.

    Returns:
        DimList[8]: The list containing the elements.
    """
    return __mlir_op.`pop.list.create`[
        _type : __mlir_type[`!kgen.list<`, Dim, `[8]>`]
    ](e0, e1, e2, e3, e4, e5, e6, e7)


@always_inline("nodebug")
fn create_dim_list(
    e0: Dim,
    e1: Dim,
    e2: Dim,
    e3: Dim,
    e4: Dim,
    e5: Dim,
    e6: Dim,
    e7: Dim,
    e8: Dim,
) -> DimList[9]:
    """Creates a list given a type and elements.

    Args:
        e0 (Dim): The 1st element of the returned list.
        e1 (Dim): The 2nd element of the returned list.
        e2 (Dim): The 3rd element of the returned list.
        e3 (Dim): The 4th element of the returned list.
        e4 (Dim): The 5th element of the returned list.
        e5 (Dim): The 6th element of the returned list.
        e6 (Dim): The 7th element of the returned list.
        e7 (Dim): The 8th element of the returned list.
        e8 (Dim): The 9th element of the returned list.

    Returns:
        DimList[9]: The list containing the elements.
    """
    return __mlir_op.`pop.list.create`[
        _type : __mlir_type[`!kgen.list<`, Dim, `[9]>`]
    ](e0, e1, e2, e3, e4, e5, e6, e7, e8)


@always_inline("nodebug")
fn create_dim_list(
    e0: Dim,
    e1: Dim,
    e2: Dim,
    e3: Dim,
    e4: Dim,
    e5: Dim,
    e6: Dim,
    e7: Dim,
    e8: Dim,
    e9: Dim,
) -> DimList[10]:
    """Creates a list given a type and elements.

    Args:
        e0 (Dim): The 1st element of the returned list.
        e1 (Dim): The 2nd element of the returned list.
        e2 (Dim): The 3rd element of the returned list.
        e3 (Dim): The 4th element of the returned list.
        e4 (Dim): The 5th element of the returned list.
        e5 (Dim): The 6th element of the returned list.
        e6 (Dim): The 7th element of the returned list.
        e7 (Dim): The 8th element of the returned list.
        e8 (Dim): The 9th element of the returned list.
        e9 (Dim): The 10th element of the returned list.

    Returns:
        DimList[10]: The list containing the elements.
    """
    return __mlir_op.`pop.list.create`[
        _type : __mlir_type[`!kgen.list<`, Dim, `[10]>`]
    ](e0, e1, e2, e3, e4, e5, e6, e7, e8, e9)


@always_inline("nodebug")
fn create_dim_list(
    e0: Dim,
    e1: Dim,
    e2: Dim,
    e3: Dim,
    e4: Dim,
    e5: Dim,
    e6: Dim,
    e7: Dim,
    e8: Dim,
    e9: Dim,
    e10: Dim,
) -> DimList[11]:
    """Creates a list given a type and elements.

    Args:
        e0 (Dim): The 1st element of the returned list.
        e1 (Dim): The 2nd element of the returned list.
        e2 (Dim): The 3rd element of the returned list.
        e3 (Dim): The 4th element of the returned list.
        e4 (Dim): The 5th element of the returned list.
        e5 (Dim): The 6th element of the returned list.
        e6 (Dim): The 7th element of the returned list.
        e7 (Dim): The 8th element of the returned list.
        e8 (Dim): The 9th element of the returned list.
        e9 (Dim): The 10th element of the returned list.
        e10 (Dim): The 11th element of the returned list.

    Returns:
        DimList[11]: The list containing the elements.
    """
    return __mlir_op.`pop.list.create`[
        _type : __mlir_type[`!kgen.list<`, Dim, `[11]>`]
    ](e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10)


@always_inline("nodebug")
fn create_dim_list(
    e0: Dim,
    e1: Dim,
    e2: Dim,
    e3: Dim,
    e4: Dim,
    e5: Dim,
    e6: Dim,
    e7: Dim,
    e8: Dim,
    e9: Dim,
    e10: Dim,
    e11: Dim,
) -> DimList[12]:
    """Creates a list given a type and elements.

    Args:
        e0 (Dim): The 1st element of the returned list.
        e1 (Dim): The 2nd element of the returned list.
        e2 (Dim): The 3rd element of the returned list.
        e3 (Dim): The 4th element of the returned list.
        e4 (Dim): The 5th element of the returned list.
        e5 (Dim): The 6th element of the returned list.
        e6 (Dim): The 7th element of the returned list.
        e7 (Dim): The 8th element of the returned list.
        e8 (Dim): The 9th element of the returned list.
        e9 (Dim): The 10th element of the returned list.
        e10 (Dim): The 11th element of the returned list.
        e11 (Dim): The 12th element of the returned list.

    Returns:
        DimList[12]: The list containing the elements.
    """
    return __mlir_op.`pop.list.create`[
        _type : __mlir_type[`!kgen.list<`, Dim, `[12]>`]
    ](e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11)


@always_inline("nodebug")
fn create_dim_list(
    e0: Dim,
    e1: Dim,
    e2: Dim,
    e3: Dim,
    e4: Dim,
    e5: Dim,
    e6: Dim,
    e7: Dim,
    e8: Dim,
    e9: Dim,
    e10: Dim,
    e11: Dim,
    e12: Dim,
) -> DimList[13]:
    """Creates a list given a type and elements.

    Args:
        e0 (Dim): The 1st element of the returned list.
        e1 (Dim): The 2nd element of the returned list.
        e2 (Dim): The 3rd element of the returned list.
        e3 (Dim): The 4th element of the returned list.
        e4 (Dim): The 5th element of the returned list.
        e5 (Dim): The 6th element of the returned list.
        e6 (Dim): The 7th element of the returned list.
        e7 (Dim): The 8th element of the returned list.
        e8 (Dim): The 9th element of the returned list.
        e9 (Dim): The 10th element of the returned list.
        e10 (Dim): The 11th element of the returned list.
        e11 (Dim): The 12th element of the returned list.
        e12 (Dim): The 13th element of the returned list.

    Returns:
        DimList[13]: The list containing the elements.
    """
    return __mlir_op.`pop.list.create`[
        _type : __mlir_type[`!kgen.list<`, Dim, `[13]>`]
    ](e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12)


@always_inline("nodebug")
fn create_dim_list(
    e0: Dim,
    e1: Dim,
    e2: Dim,
    e3: Dim,
    e4: Dim,
    e5: Dim,
    e6: Dim,
    e7: Dim,
    e8: Dim,
    e9: Dim,
    e10: Dim,
    e11: Dim,
    e12: Dim,
    e13: Dim,
) -> DimList[14]:
    """Creates a list given a type and elements.

    Args:
        e0 (Dim): The 1st element of the returned list.
        e1 (Dim): The 2nd element of the returned list.
        e2 (Dim): The 3rd element of the returned list.
        e3 (Dim): The 4th element of the returned list.
        e4 (Dim): The 5th element of the returned list.
        e5 (Dim): The 6th element of the returned list.
        e6 (Dim): The 7th element of the returned list.
        e7 (Dim): The 8th element of the returned list.
        e8 (Dim): The 9th element of the returned list.
        e9 (Dim): The 10th element of the returned list.
        e10 (Dim): The 11th element of the returned list.
        e11 (Dim): The 12th element of the returned list.
        e12 (Dim): The 13th element of the returned list.
        e13 (Dim): The 14th element of the returned list.

    Returns:
        DimList[14]: The list containing the elements.
    """
    return __mlir_op.`pop.list.create`[
        _type : __mlir_type[`!kgen.list<`, Dim, `[14]>`]
    ](e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13)


@always_inline("nodebug")
fn create_dim_list(
    e0: Dim,
    e1: Dim,
    e2: Dim,
    e3: Dim,
    e4: Dim,
    e5: Dim,
    e6: Dim,
    e7: Dim,
    e8: Dim,
    e9: Dim,
    e10: Dim,
    e11: Dim,
    e12: Dim,
    e13: Dim,
    e14: Dim,
) -> DimList[15]:
    """Creates a list given a type and elements.

    Args:
        e0 (Dim): The 1st element of the returned list.
        e1 (Dim): The 2nd element of the returned list.
        e2 (Dim): The 3rd element of the returned list.
        e3 (Dim): The 4th element of the returned list.
        e4 (Dim): The 5th element of the returned list.
        e5 (Dim): The 6th element of the returned list.
        e6 (Dim): The 7th element of the returned list.
        e7 (Dim): The 8th element of the returned list.
        e8 (Dim): The 9th element of the returned list.
        e9 (Dim): The 10th element of the returned list.
        e10 (Dim): The 11th element of the returned list.
        e11 (Dim): The 12th element of the returned list.
        e12 (Dim): The 13th element of the returned list.
        e13 (Dim): The 14th element of the returned list.
        e14 (Dim): The 15th element of the returned list.

    Returns:
        DimList[15]: The list containing the elements.
    """
    return __mlir_op.`pop.list.create`[
        _type : __mlir_type[`!kgen.list<`, Dim, `[15]>`]
    ](e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14)


@always_inline("nodebug")
fn create_dim_list(
    e0: Dim,
    e1: Dim,
    e2: Dim,
    e3: Dim,
    e4: Dim,
    e5: Dim,
    e6: Dim,
    e7: Dim,
    e8: Dim,
    e9: Dim,
    e10: Dim,
    e11: Dim,
    e12: Dim,
    e13: Dim,
    e14: Dim,
    e15: Dim,
) -> DimList[16]:
    """Creates a list given a type and elements.

    Args:
        e0 (Dim): The 1st element of the returned list.
        e1 (Dim): The 2nd element of the returned list.
        e2 (Dim): The 3rd element of the returned list.
        e3 (Dim): The 4th element of the returned list.
        e4 (Dim): The 5th element of the returned list.
        e5 (Dim): The 6th element of the returned list.
        e6 (Dim): The 7th element of the returned list.
        e7 (Dim): The 8th element of the returned list.
        e8 (Dim): The 9th element of the returned list.
        e9 (Dim): The 10th element of the returned list.
        e10 (Dim): The 11th element of the returned list.
        e11 (Dim): The 12th element of the returned list.
        e12 (Dim): The 13th element of the returned list.
        e13 (Dim): The 14th element of the returned list.
        e14 (Dim): The 15th element of the returned list.
        e15 (Dim): The 16th element of the returned list.

    Returns:
        DimList[16]: The list containing the elements.
    """
    return __mlir_op.`pop.list.create`[
        _type : __mlir_type[`!kgen.list<`, Dim, `[16]>`]
    ](e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15)


# ===----------------------------------------------------------------------===#
# create_kgen_list
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn create_kgen_list[
    type: __mlir_type.`!kgen.mlirtype`
](e0: type) -> __mlir_type[`!kgen.list<`, type, `[1]>`]:
    """Creates a list given a type and elements.

    Args:
        type (!kgen.mlirtype): The list type.
        e0 (type): The 1st element of the returned list.

    Returns:
        !kgen.list<type[1]>: The list containing the elements.
    """
    return __mlir_op.`pop.list.create`[
        _type : __mlir_type[`!kgen.list<`, type, `[1]>`]
    ](e0)


@always_inline("nodebug")
fn create_kgen_list[
    type: __mlir_type.`!kgen.mlirtype`
](e0: type, e1: type) -> __mlir_type[`!kgen.list<`, type, `[2]>`]:
    """Creates a list given a type and elements.

    Args:
        type (!kgen.mlirtype): The list type.
        e0 (type): The 1st element of the returned list.
        e1 (type): The 2nd element of the returned list.

    Returns:
        !kgen.list<type[2]>: The list containing the elements.
    """
    return __mlir_op.`pop.list.create`[
        _type : __mlir_type[`!kgen.list<`, type, `[2]>`]
    ](e0, e1)


@always_inline("nodebug")
fn create_kgen_list[
    type: __mlir_type.`!kgen.mlirtype`
](e0: type, e1: type, e2: type) -> __mlir_type[`!kgen.list<`, type, `[3]>`]:
    """Creates a list given a type and elements.

    Args:
        type (!kgen.mlirtype): The list type.
        e0 (type): The 1st element of the returned list.
        e1 (type): The 2nd element of the returned list.
        e2 (type): The 3rd element of the returned list.

    Returns:
        !kgen.list<type[3]>: The list containing the elements.
    """
    return __mlir_op.`pop.list.create`[
        _type : __mlir_type[`!kgen.list<`, type, `[3]>`]
    ](e0, e1, e2)


@always_inline("nodebug")
fn create_kgen_list[
    type: __mlir_type.`!kgen.mlirtype`
](e0: type, e1: type, e2: type, e3: type) -> __mlir_type[
    `!kgen.list<`, type, `[4]>`
]:
    """Creates a list given a type and elements.

    Args:
        type (!kgen.mlirtype): The list type.
        e0 (type): The 1st element of the returned list.
        e1 (type): The 2nd element of the returned list.
        e2 (type): The 3rd element of the returned list.
        e3 (type): The 4th element of the returned list.

    Returns:
        !kgen.list<type[4]>: The list containing the elements.
    """
    return __mlir_op.`pop.list.create`[
        _type : __mlir_type[`!kgen.list<`, type, `[4]>`]
    ](e0, e1, e2, e3)


@always_inline("nodebug")
fn create_kgen_list[
    type: __mlir_type.`!kgen.mlirtype`
](e0: type, e1: type, e2: type, e3: type, e4: type,) -> __mlir_type[
    `!kgen.list<`, type, `[5]>`
]:
    """Creates a list given a type and elements.

    Args:
        type (!kgen.mlirtype): The list type.
        e0 (type): The 1st element of the returned list.
        e1 (type): The 2nd element of the returned list.
        e2 (type): The 3rd element of the returned list.
        e3 (type): The 4th element of the returned list.
        e4 (type): The 5th element of the returned list.

    Returns:
        !kgen.list<type[5]>: The list containing the elements.
    """
    return __mlir_op.`pop.list.create`[
        _type : __mlir_type[`!kgen.list<`, type, `[5]>`]
    ](e0, e1, e2, e3, e4)


@always_inline("nodebug")
fn create_kgen_list[
    type: __mlir_type.`!kgen.mlirtype`
](e0: type, e1: type, e2: type, e3: type, e4: type, e5: type) -> __mlir_type[
    `!kgen.list<`, type, `[6]>`
]:
    """Creates a list given a type and elements.

    Args:
        type (!kgen.mlirtype): The list type.
        e0 (type): The 1st element of the returned list.
        e1 (type): The 2nd element of the returned list.
        e2 (type): The 3rd element of the returned list.
        e3 (type): The 4th element of the returned list.
        e4 (type): The 5th element of the returned list.
        e5 (type): The 6th element of the returned list.

    Returns:
        !kgen.list<type[6]>: The list containing the elements.
    """
    return __mlir_op.`pop.list.create`[
        _type : __mlir_type[`!kgen.list<`, type, `[6]>`]
    ](e0, e1, e2, e3, e4, e5)


@always_inline("nodebug")
fn create_kgen_list[
    type: __mlir_type.`!kgen.mlirtype`
](
    e0: type,
    e1: type,
    e2: type,
    e3: type,
    e4: type,
    e5: type,
    e6: type,
) -> __mlir_type[`!kgen.list<`, type, `[7]>`]:
    """Creates a list given a type and elements.

    Args:
        type (!kgen.mlirtype): The list type.
        e0 (type): The 1st element of the returned list.
        e1 (type): The 2nd element of the returned list.
        e2 (type): The 3rd element of the returned list.
        e3 (type): The 4th element of the returned list.
        e4 (type): The 5th element of the returned list.
        e5 (type): The 6th element of the returned list.
        e6 (type): The 7th element of the returned list.

    Returns:
        !kgen.list<type[7]>: The list containing the elements.
    """
    return __mlir_op.`pop.list.create`[
        _type : __mlir_type[`!kgen.list<`, type, `[7]>`]
    ](e0, e1, e2, e3, e4, e5, e6)


@always_inline("nodebug")
fn create_kgen_list[
    type: __mlir_type.`!kgen.mlirtype`
](
    e0: type,
    e1: type,
    e2: type,
    e3: type,
    e4: type,
    e5: type,
    e6: type,
    e7: type,
) -> __mlir_type[`!kgen.list<`, type, `[8]>`]:
    """Creates a list given a type and elements.

    Args:
        type (!kgen.mlirtype): The list type.
        e0 (type): The 1st element of the returned list.
        e1 (type): The 2nd element of the returned list.
        e2 (type): The 3rd element of the returned list.
        e3 (type): The 4th element of the returned list.
        e4 (type): The 5th element of the returned list.
        e5 (type): The 6th element of the returned list.
        e6 (type): The 7th element of the returned list.
        e7 (type): The 8th element of the returned list.

    Returns:
        !kgen.list<type[8]>: The list containing the elements.
    """
    return __mlir_op.`pop.list.create`[
        _type : __mlir_type[`!kgen.list<`, type, `[8]>`]
    ](e0, e1, e2, e3, e4, e5, e6, e7)


@always_inline("nodebug")
fn create_kgen_list[
    type: __mlir_type.`!kgen.mlirtype`
](
    e0: type,
    e1: type,
    e2: type,
    e3: type,
    e4: type,
    e5: type,
    e6: type,
    e7: type,
    e8: type,
) -> __mlir_type[`!kgen.list<`, type, `[9]>`]:
    """Creates a list given a type and elements.

    Args:
        type (!kgen.mlirtype): The list type.
        e0 (type): The 1st element of the returned list.
        e1 (type): The 2nd element of the returned list.
        e2 (type): The 3rd element of the returned list.
        e3 (type): The 4th element of the returned list.
        e4 (type): The 5th element of the returned list.
        e5 (type): The 6th element of the returned list.
        e6 (type): The 7th element of the returned list.
        e7 (type): The 8th element of the returned list.
        e8 (type): The 9th element of the returned list.

    Returns:
        !kgen.list<type[9]>: The list containing the elements.
    """
    return __mlir_op.`pop.list.create`[
        _type : __mlir_type[`!kgen.list<`, type, `[9]>`]
    ](e0, e1, e2, e3, e4, e5, e6, e7, e8)


@always_inline("nodebug")
fn create_kgen_list[
    type: __mlir_type.`!kgen.mlirtype`
](
    e0: type,
    e1: type,
    e2: type,
    e3: type,
    e4: type,
    e5: type,
    e6: type,
    e7: type,
    e8: type,
    e9: type,
) -> __mlir_type[`!kgen.list<`, type, `[10]>`]:
    """Creates a list given a type and elements.

    Args:
        type (!kgen.mlirtype): The list type.
        e0 (type): The 1st element of the returned list.
        e1 (type): The 2nd element of the returned list.
        e2 (type): The 3rd element of the returned list.
        e3 (type): The 4th element of the returned list.
        e4 (type): The 5th element of the returned list.
        e5 (type): The 6th element of the returned list.
        e6 (type): The 7th element of the returned list.
        e7 (type): The 8th element of the returned list.
        e8 (type): The 9th element of the returned list.
        e9 (type): The 10th element of the returned list.

    Returns:
        !kgen.list<type[10]>: The list containing the elements.
    """
    return __mlir_op.`pop.list.create`[
        _type : __mlir_type[`!kgen.list<`, type, `[10]>`]
    ](e0, e1, e2, e3, e4, e5, e6, e7, e8, e9)


@always_inline("nodebug")
fn create_kgen_list[
    type: __mlir_type.`!kgen.mlirtype`
](
    e0: type,
    e1: type,
    e2: type,
    e3: type,
    e4: type,
    e5: type,
    e6: type,
    e7: type,
    e8: type,
    e9: type,
    e10: type,
) -> __mlir_type[`!kgen.list<`, type, `[11]>`]:
    """Creates a list given a type and elements.

    Args:
        type (!kgen.mlirtype): The list type.
        e0 (type): The 1st element of the returned list.
        e1 (type): The 2nd element of the returned list.
        e2 (type): The 3rd element of the returned list.
        e3 (type): The 4th element of the returned list.
        e4 (type): The 5th element of the returned list.
        e5 (type): The 6th element of the returned list.
        e6 (type): The 7th element of the returned list.
        e7 (type): The 8th element of the returned list.
        e8 (type): The 9th element of the returned list.
        e9 (type): The 10th element of the returned list.
        e10 (type): The 11th element of the returned list.

    Returns:
        !kgen.list<type[11]>: The list containing the elements.
    """
    return __mlir_op.`pop.list.create`[
        _type : __mlir_type[`!kgen.list<`, type, `[11]>`]
    ](e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10)


@always_inline("nodebug")
fn create_kgen_list[
    type: __mlir_type.`!kgen.mlirtype`
](
    e0: type,
    e1: type,
    e2: type,
    e3: type,
    e4: type,
    e5: type,
    e6: type,
    e7: type,
    e8: type,
    e9: type,
    e10: type,
    e11: type,
) -> __mlir_type[`!kgen.list<`, type, `[12]>`]:
    """Creates a list given a type and elements.

    Args:
        type (!kgen.mlirtype): The list type.
        e0 (type): The 1st element of the returned list.
        e1 (type): The 2nd element of the returned list.
        e2 (type): The 3rd element of the returned list.
        e3 (type): The 4th element of the returned list.
        e4 (type): The 5th element of the returned list.
        e5 (type): The 6th element of the returned list.
        e6 (type): The 7th element of the returned list.
        e7 (type): The 8th element of the returned list.
        e8 (type): The 9th element of the returned list.
        e9 (type): The 10th element of the returned list.
        e10 (type): The 11th element of the returned list.
        e11 (type): The 12th element of the returned list.

    Returns:
        !kgen.list<type[12]>: The list containing the elements.
    """
    return __mlir_op.`pop.list.create`[
        _type : __mlir_type[`!kgen.list<`, type, `[12]>`]
    ](e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11)


@always_inline("nodebug")
fn create_kgen_list[
    type: __mlir_type.`!kgen.mlirtype`
](
    e0: type,
    e1: type,
    e2: type,
    e3: type,
    e4: type,
    e5: type,
    e6: type,
    e7: type,
    e8: type,
    e9: type,
    e10: type,
    e11: type,
    e12: type,
) -> __mlir_type[`!kgen.list<`, type, `[13]>`]:
    """Creates a list given a type and elements.

    Args:
        type (!kgen.mlirtype): The list type.
        e0 (type): The 1st element of the returned list.
        e1 (type): The 2nd element of the returned list.
        e2 (type): The 3rd element of the returned list.
        e3 (type): The 4th element of the returned list.
        e4 (type): The 5th element of the returned list.
        e5 (type): The 6th element of the returned list.
        e6 (type): The 7th element of the returned list.
        e7 (type): The 8th element of the returned list.
        e8 (type): The 9th element of the returned list.
        e9 (type): The 10th element of the returned list.
        e10 (type): The 11th element of the returned list.
        e11 (type): The 12th element of the returned list.
        e12 (type): The 13th element of the returned list.

    Returns:
        !kgen.list<type[13]>: The list containing the elements.
    """
    return __mlir_op.`pop.list.create`[
        _type : __mlir_type[`!kgen.list<`, type, `[13]>`]
    ](e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12)


@always_inline("nodebug")
fn create_kgen_list[
    type: __mlir_type.`!kgen.mlirtype`
](
    e0: type,
    e1: type,
    e2: type,
    e3: type,
    e4: type,
    e5: type,
    e6: type,
    e7: type,
    e8: type,
    e9: type,
    e10: type,
    e11: type,
    e12: type,
    e13: type,
) -> __mlir_type[`!kgen.list<`, type, `[14]>`]:
    """Creates a list given a type and elements.

    Args:
        type (!kgen.mlirtype): The list type.
        e0 (type): The 1st element of the returned list.
        e1 (type): The 2nd element of the returned list.
        e2 (type): The 3rd element of the returned list.
        e3 (type): The 4th element of the returned list.
        e4 (type): The 5th element of the returned list.
        e5 (type): The 6th element of the returned list.
        e6 (type): The 7th element of the returned list.
        e7 (type): The 8th element of the returned list.
        e8 (type): The 9th element of the returned list.
        e9 (type): The 10th element of the returned list.
        e10 (type): The 11th element of the returned list.
        e11 (type): The 12th element of the returned list.
        e12 (type): The 13th element of the returned list.
        e13 (type): The 14th element of the returned list.

    Returns:
        !kgen.list<type[14]>: The list containing the elements.
    """
    return __mlir_op.`pop.list.create`[
        _type : __mlir_type[`!kgen.list<`, type, `[14]>`]
    ](e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13)


@always_inline("nodebug")
fn create_kgen_list[
    type: __mlir_type.`!kgen.mlirtype`
](
    e0: type,
    e1: type,
    e2: type,
    e3: type,
    e4: type,
    e5: type,
    e6: type,
    e7: type,
    e8: type,
    e9: type,
    e10: type,
    e11: type,
    e12: type,
    e13: type,
    e14: type,
) -> __mlir_type[`!kgen.list<`, type, `[15]>`]:
    """Creates a list given a type and elements.

    Args:
        type (!kgen.mlirtype): The list type.
        e0 (type): The 1st element of the returned list.
        e1 (type): The 2nd element of the returned list.
        e2 (type): The 3rd element of the returned list.
        e3 (type): The 4th element of the returned list.
        e4 (type): The 5th element of the returned list.
        e5 (type): The 6th element of the returned list.
        e6 (type): The 7th element of the returned list.
        e7 (type): The 8th element of the returned list.
        e8 (type): The 9th element of the returned list.
        e9 (type): The 10th element of the returned list.
        e10 (type): The 11th element of the returned list.
        e11 (type): The 12th element of the returned list.
        e12 (type): The 13th element of the returned list.
        e13 (type): The 14th element of the returned list.
        e14 (type): The 15th element of the returned list.

    Returns:
        !kgen.list<type[15]>: The list containing the elements.
    """
    return __mlir_op.`pop.list.create`[
        _type : __mlir_type[`!kgen.list<`, type, `[15]>`]
    ](e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14)


@always_inline("nodebug")
fn create_kgen_list[
    type: __mlir_type.`!kgen.mlirtype`
](
    e0: type,
    e1: type,
    e2: type,
    e3: type,
    e4: type,
    e5: type,
    e6: type,
    e7: type,
    e8: type,
    e9: type,
    e10: type,
    e11: type,
    e12: type,
    e13: type,
    e14: type,
    e15: type,
) -> __mlir_type[`!kgen.list<`, type, `[16]>`]:
    """Creates a list given a type and elements.

    Args:
        type (!kgen.mlirtype): The list type.
        e0 (type): The 1st element of the returned list.
        e1 (type): The 2nd element of the returned list.
        e2 (type): The 3rd element of the returned list.
        e3 (type): The 4th element of the returned list.
        e4 (type): The 5th element of the returned list.
        e5 (type): The 6th element of the returned list.
        e6 (type): The 7th element of the returned list.
        e7 (type): The 8th element of the returned list.
        e8 (type): The 9th element of the returned list.
        e9 (type): The 10th element of the returned list.
        e10 (type): The 11th element of the returned list.
        e11 (type): The 12th element of the returned list.
        e12 (type): The 13th element of the returned list.
        e13 (type): The 14th element of the returned list.
        e14 (type): The 15th element of the returned list.
        e15 (type): The 16th element of the returned list.

    Returns:
        !kgen.list<type[16]>: The list containing the elements.
    """
    return __mlir_op.`pop.list.create`[
        _type : __mlir_type[`!kgen.list<`, type, `[16]>`]
    ](e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15)


# ===----------------------------------------------------------------------===#
# _get_kgen_list_item
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn _get_kgen_list_item[
    index: Int,
    size: __mlir_type.index,
    type: __mlir_type.`!kgen.mlirtype`,
](lst: __mlir_type[`!kgen.list<`, type, `[`, size, `]>`]) -> type:
    """Gets the list element of an input list at position `index`.

    Args:
        index (index): the position to get the value from.
        size (index): the size of the list.
        type (!kgen.mlirtype): the element type of the list.
        lst (!pop.list<type[size]>): the list to get the values from.

    Returns:
        type: The value at position `index` in the list.
    """
    assert_param_bool[index <= size]()
    return __mlir_op.`pop.list.get`[
        index : index.__as_mlir_index(), _type:type
    ](lst)


# ===----------------------------------------------------------------------===#
# contains
# ===----------------------------------------------------------------------===#


fn contains[
    size: __mlir_type.index
](
    elem: __mlir_type.index, lst: __mlir_type[`!kgen.list<index[`, size, `]>`]
) -> Bool:
    return _contains_impl[0, size](elem, lst)


fn _contains_impl[
    idx: __mlir_type.index, size: __mlir_type.index
](
    elem: __mlir_type.index, lst: __mlir_type[`!kgen.list<index[`, size, `]>`]
) -> Bool:
    @parameter
    if idx == size:
        return False
    let ok = __mlir_op.`index.cmp`[
        pred : __mlir_attr.`#index<cmp_predicate eq>`
    ](_get_kgen_list_item[idx, size, __mlir_type.index](lst), elem)
    return Bool(ok) or _contains_impl[idx + 1, size](elem, lst)


# ===----------------------------------------------------------------------===#
# product
# ===----------------------------------------------------------------------===#


fn product_range[
    start_idx: Int,
    end_idx: Int,
    size: __mlir_type.index,
](lst: __mlir_type[`!kgen.list<index[`, size, `]>`]) -> Int:
    return _product_range_impl[start_idx, start_idx, end_idx, size](lst)


fn _product_range_impl[
    idx: Int,
    start_idx: Int,
    end_idx: Int,
    size: __mlir_type.index,
](lst: __mlir_type[`!kgen.list<index[`, size, `]>`]) -> Int:
    assert_param_bool[idx <= end_idx]()

    @parameter
    if idx == end_idx:
        return 1
    else:
        return _get_kgen_list_item[idx, size, __mlir_type.index](
            lst
        ) * _product_range_impl[idx + 1, start_idx, end_idx, size](lst)


fn product[
    size: __mlir_type.index
](lst: __mlir_type[`!kgen.list<index[`, size, `]>`]) -> Int:
    return product_range[0, size, size](lst)


fn product_or_unknown[
    rank: __mlir_type.index,
    shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
    start_idx: Int,
    end_idx: Int,
]() -> __mlir_type.index:
    @parameter
    if is_all_known_range[start_idx, end_idx, rank, shape]():
        return product_range[start_idx, end_idx, rank](shape).__as_mlir_index()
    return __mlir_attr.`#kgen.unknown : index`


# ===----------------------------------------------------------------------===#
# is_all_known
# ===----------------------------------------------------------------------===#


fn is_all_known[
    rank: __mlir_type.index,
    shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
]() -> Bool:
    return is_all_known_range[0, rank, rank, shape]()


fn is_all_known_range[
    start_idx: Int,
    end_idx: Int,
    rank: __mlir_type.index,
    shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
]() -> Bool:
    return is_all_known_range_impl[start_idx, start_idx, end_idx, rank, shape]()


fn is_all_known_range_impl[
    index: Int,
    start_idx: Int,
    end_idx: Int,
    rank: __mlir_type.index,
    shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
]() -> Bool:
    assert_param_bool[index <= end_idx]()

    @parameter
    if index == end_idx:
        return True
    else:
        alias static_dim_value = _get_kgen_list_item[
            index.__as_mlir_index(), rank, __mlir_type.index
        ](shape)
        return (
            Bool(static_dim_value != __mlir_attr.`#kgen.unknown : index`)
            and is_all_known_range_impl[
                index + 1, start_idx, end_idx, rank, shape
            ]()
        )


# ===----------------------------------------------------------------------===#
# VariadicList
# ===----------------------------------------------------------------------===#


@register_passable
struct VariadicList[type: __mlir_type.`!kgen.mlirtype`]:
    """A utility class to access variadic function arguments. Provides a "list"
    view of the function argument so that the size of the argument list and each
    individual argument can be accessed.
    """

    alias StorageType = __mlir_type[`!kgen.variadic<`, type, `>`]
    var value: StorageType

    fn __copy__(self) -> Self:
        return Self {value: self.value}

    fn __init__(*value: type) -> Self:
        """Constructs a VariadicList from a variadic list of arguments.

        Args:
            *value (type):
                The variadic argument list to construct the variadic list with.

        Returns (VariadicList):
            The VariadicList constructed.
        """
        return Self(value)

    fn __init__(value: StorageType) -> VariadicList[type]:
        """Constructs a VariadicList from a variadic argument type.

        Args:
            value (StorageType):
                The variadic argument to construct the list with.

        Returns (VariadicList):
            The VariadicList constructed.
        """
        return Self {value: value}

    fn __len__(self) -> Int:
        """Gets the size of the list.

        Returns:
            The number of elements on the variadic list.
        """

        return __mlir_op.`pop.variadic.size`(self.value)

    fn __getitem__(self, index: Int) -> type:
        """Accessor to a single element on the variadic list.
        Args:
            index (Int):
                The index of the element to access on the list.
        Returns (type):
            The element on the list corresponding to the given index.
        """
        return __mlir_op.`pop.variadic.get`(self.value, index.__as_mlir_index())

    fn __getitem__(self, index: __mlir_type.index) -> type:
        """Accessor to a single element on the variadic list.
        Args:
            index (Int):
                The index of the element to access on the list.
        Returns (type):
            The element on the list corresponding to the given index.
        """
        return __mlir_op.`pop.variadic.get`(self.value, index)
