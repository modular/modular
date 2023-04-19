# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module provides utilities for working with static and variadic lists."""

from Assert import assert_param, assert_param_msg
from TypeUtilities import rebind
from Pointer import Pointer


# ===----------------------------------------------------------------------===#
# Dim
# ===----------------------------------------------------------------------===#


@register_passable("trivial")
struct Dim:
    """A static or dynamic dimension modeled with an optional integer.

    This class is meant to represent an optional static dimension. When a value
    is present, the dimension has that static value. When a value is not
    present, the dimension is dynamic.
    """

    alias type = __mlir_type[`!pop.variant<i1, `, Int, `>`]
    var value: type

    @always_inline
    fn __init__(value: type) -> Dim:
        """Create a dimension from its underlying value type.

        Args:
            value: The underlying value.

        Returns:
            A dimension value.
        """
        return Dim {value: value}

    @always_inline
    fn __init__(value: Int) -> Dim:
        """Create a statically-known dimension.

        Args:
            value: The static dimension value.

        Returns:
            A dimension with a static value.
        """
        return __mlir_op.`pop.variant.create`[_type:type](value)

    @always_inline
    fn __init__(value: __mlir_type.index) -> Dim:
        """Create a statically-known dimension.

        Args:
            value: The static dimension value.

        Returns:
            A dimension with a static value.
        """
        return Int(value)

    @always_inline
    fn __init__() -> Dim:
        """Create a dynamic dimension.

        Returns:
            A dimension value with no static value.
        """
        return __mlir_op.`pop.variant.create`[_type:type](__mlir_attr.`0 : i1`)

    @always_inline
    fn __bool__(self) -> Bool:
        """Return True if the dimension has a static value.

        Returns:
            Whether the dimension has a static value.
        """
        return __mlir_op.`pop.variant.is`[testType : __mlir_attr[Int]](
            self.value
        )

    @always_inline
    fn has_value(self) -> Bool:
        """Return True if the dimension has a static value.

        Returns:
            Whether the dimension has a static value.
        """
        return self.__bool__()

    @always_inline
    fn is_dynamic(self) -> Bool:
        """Return True if the dimension has a dynamic value.

        Returns:
            Whether the dimension is dynamic.
        """
        return not self.has_value()

    @always_inline
    fn get(self) -> Int:
        """Get the static dimension value.

        Returns:
            The static dimension value.
        """
        return __mlir_op.`pop.variant.get`[_type:Int](self.value)

    @always_inline
    fn is_multiple[alignment: Int](self) -> Bool:
        """Check if the dimension is aligned.

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
        """Multiply two dimensions.

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
        """Compare two dimensions for equality.

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
struct DimList[length: Int]:
    """This type represents a list of dimensions. Each dimension may have a
    static value or not have a value, which represents a dynamic dimension."""

    alias list_type = __mlir_type[`!kgen.variadic<`, Dim, `>`]
    var value: list_type

    @always_inline("nodebug")
    fn __init__(value: list_type) -> Self:
        """Create a dimension list from the underlying value type.

        Args:
            value: The underlying value.

        Returns:
            A dimension list.
        """
        return Self {value: value}

    @always_inline("nodebug")
    fn __clone__(self&) -> Self:
        """Copy a dimension list.

        Returns:
            A dimension list.
        """
        return Self {value: self.value}

    @always_inline("nodebug")
    fn at[i: Int](self) -> Dim:
        """Get the dimension at a specified index.

        ParamArgs:
            i: The dimension index.

        Returns:
            The dimension at the specified index.
        """
        assert_param_msg[i >= 0, "negative index"]()
        assert_param_msg[i < length, "index exceeds length"]()
        return __mlir_op.`pop.variadic.get`(self.value, i.__as_mlir_index())

    @always_inline
    fn _product_impl[i: Int, end: Int](self) -> Dim:
        @parameter
        if i >= end:
            return 1
        else:
            return self.at[i]() * self._product_impl[i + 1, end]()

    @always_inline
    fn product(self) -> Dim:
        """Compute the product of all the dimensions in the list.

        If any are dynamic, the result is a dynamic dimension value.

        Returns:
            The product of all the dimensions.
        """
        return self._product_impl[0, length]()

    @always_inline
    fn product_range[start: Int, end: Int](self) -> Dim:
        """Compute the product of a range of the dimensions in the list.

        If any in the range are dynamic, the result is a dynamic dimension
        value.

        ParamArgs:
            start: The starting index.
            end: The end index.

        Returns:
            The product of all the dimensions.
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
            value: The value to find.

        Returns:
            True if the list contains a dimension of the specified value.
        """
        return self._contains_impl[0](value)

    @always_inline
    fn all_known(self) -> Bool:
        """Determine whether all dimensions are statically known.

        Returns:
            True if all dimensions have a static value.
        """
        return not self.contains(Dim())

    @always_inline
    @staticmethod
    fn create_unknown() -> Self:
        """Create a dimension list of all dynamic dimension values.

        Returns:
            A list of all dynamic dimension values.
        """
        assert_param_msg[length > 0, "length must be positive"]()
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
        e0: The 1st element of the returned list.

    Returns:
        The list containing the elements.
    """
    return __mlir_op.`pop.variadic.create`[
        _type : __mlir_type[`!kgen.variadic<`, Dim, `>`]
    ](e0)


@always_inline("nodebug")
fn create_dim_list(e0: Dim, e1: Dim) -> DimList[2]:
    """Creates a list given a type and elements.

    Args:
        e0: The 1st element of the returned list.
        e1: The 2nd element of the returned list.

    Returns:
        The list containing the elements.
    """
    return __mlir_op.`pop.variadic.create`[
        _type : __mlir_type[`!kgen.variadic<`, Dim, `>`]
    ](e0, e1)


@always_inline("nodebug")
fn create_dim_list(e0: Dim, e1: Dim, e2: Dim) -> DimList[3]:
    """Creates a list given a type and elements.

    Args:
        e0: The 1st element of the returned list.
        e1: The 2nd element of the returned list.
        e2: The 3rd element of the returned list.

    Returns:
        The list containing the elements.
    """
    return __mlir_op.`pop.variadic.create`[
        _type : __mlir_type[`!kgen.variadic<`, Dim, `>`]
    ](e0, e1, e2)


@always_inline("nodebug")
fn create_dim_list(e0: Dim, e1: Dim, e2: Dim, e3: Dim) -> DimList[4]:
    """Creates a list given a type and elements.

    Args:
        e0: The 1st element of the returned list.
        e1: The 2nd element of the returned list.
        e2: The 3rd element of the returned list.
        e3: The 4th element of the returned list.

    Returns:
        The list containing the elements.
    """
    return __mlir_op.`pop.variadic.create`[
        _type : __mlir_type[`!kgen.variadic<`, Dim, `>`]
    ](e0, e1, e2, e3)


@always_inline("nodebug")
fn create_dim_list(e0: Dim, e1: Dim, e2: Dim, e3: Dim, e4: Dim) -> DimList[5]:
    """Creates a list given a type and elements.

    Args:
        e0: The 1st element of the returned list.
        e1: The 2nd element of the returned list.
        e2: The 3rd element of the returned list.
        e3: The 4th element of the returned list.
        e4: The 5th element of the returned list.

    Returns:
        The list containing the elements.
    """
    return __mlir_op.`pop.variadic.create`[
        _type : __mlir_type[`!kgen.variadic<`, Dim, `>`]
    ](e0, e1, e2, e3, e4)


@always_inline("nodebug")
fn create_dim_list(
    e0: Dim, e1: Dim, e2: Dim, e3: Dim, e4: Dim, e5: Dim
) -> DimList[6]:
    """Creates a list given a type and elements.

    Args:
        e0: The 1st element of the returned list.
        e1: The 2nd element of the returned list.
        e2: The 3rd element of the returned list.
        e3: The 4th element of the returned list.
        e4: The 5th element of the returned list.
        e5: The 6th element of the returned list.

    Returns:
        The list containing the elements.
    """
    return __mlir_op.`pop.variadic.create`[
        _type : __mlir_type[`!kgen.variadic<`, Dim, `>`]
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
        e0: The 1st element of the returned list.
        e1: The 2nd element of the returned list.
        e2: The 3rd element of the returned list.
        e3: The 4th element of the returned list.
        e4: The 5th element of the returned list.
        e5: The 6th element of the returned list.
        e6: The 7th element of the returned list.

    Returns:
        The list containing the elements.
    """
    return __mlir_op.`pop.variadic.create`[
        _type : __mlir_type[`!kgen.variadic<`, Dim, `>`]
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
        e0: The 1st element of the returned list.
        e1: The 2nd element of the returned list.
        e2: The 3rd element of the returned list.
        e3: The 4th element of the returned list.
        e4: The 5th element of the returned list.
        e5: The 6th element of the returned list.
        e6: The 7th element of the returned list.
        e7: The 8th element of the returned list.

    Returns:
        The list containing the elements.
    """
    return __mlir_op.`pop.variadic.create`[
        _type : __mlir_type[`!kgen.variadic<`, Dim, `>`]
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
        e0: The 1st element of the returned list.
        e1: The 2nd element of the returned list.
        e2: The 3rd element of the returned list.
        e3: The 4th element of the returned list.
        e4: The 5th element of the returned list.
        e5: The 6th element of the returned list.
        e6: The 7th element of the returned list.
        e7: The 8th element of the returned list.
        e8: The 9th element of the returned list.

    Returns:
        The list containing the elements.
    """
    return __mlir_op.`pop.variadic.create`[
        _type : __mlir_type[`!kgen.variadic<`, Dim, `>`]
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
        e0: The 1st element of the returned list.
        e1: The 2nd element of the returned list.
        e2: The 3rd element of the returned list.
        e3: The 4th element of the returned list.
        e4: The 5th element of the returned list.
        e5: The 6th element of the returned list.
        e6: The 7th element of the returned list.
        e7: The 8th element of the returned list.
        e8: The 9th element of the returned list.
        e9: The 10th element of the returned list.

    Returns:
        The list containing the elements.
    """
    return __mlir_op.`pop.variadic.create`[
        _type : __mlir_type[`!kgen.variadic<`, Dim, `>`]
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
        e0: The 1st element of the returned list.
        e1: The 2nd element of the returned list.
        e2: The 3rd element of the returned list.
        e3: The 4th element of the returned list.
        e4: The 5th element of the returned list.
        e5: The 6th element of the returned list.
        e6: The 7th element of the returned list.
        e7: The 8th element of the returned list.
        e8: The 9th element of the returned list.
        e9: The 10th element of the returned list.
        e10: The 11th element of the returned list.

    Returns:
        The list containing the elements.
    """
    return __mlir_op.`pop.variadic.create`[
        _type : __mlir_type[`!kgen.variadic<`, Dim, `>`]
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
        e0: The 1st element of the returned list.
        e1: The 2nd element of the returned list.
        e2: The 3rd element of the returned list.
        e3: The 4th element of the returned list.
        e4: The 5th element of the returned list.
        e5: The 6th element of the returned list.
        e6: The 7th element of the returned list.
        e7: The 8th element of the returned list.
        e8: The 9th element of the returned list.
        e9: The 10th element of the returned list.
        e10: The 11th element of the returned list.
        e11: The 12th element of the returned list.

    Returns:
        The list containing the elements.
    """
    return __mlir_op.`pop.variadic.create`[
        _type : __mlir_type[`!kgen.variadic<`, Dim, `>`]
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
        e0: The 1st element of the returned list.
        e1: The 2nd element of the returned list.
        e2: The 3rd element of the returned list.
        e3: The 4th element of the returned list.
        e4: The 5th element of the returned list.
        e5: The 6th element of the returned list.
        e6: The 7th element of the returned list.
        e7: The 8th element of the returned list.
        e8: The 9th element of the returned list.
        e9: The 10th element of the returned list.
        e10: The 11th element of the returned list.
        e11: The 12th element of the returned list.
        e12: The 13th element of the returned list.

    Returns:
        The list containing the elements.
    """
    return __mlir_op.`pop.variadic.create`[
        _type : __mlir_type[`!kgen.variadic<`, Dim, `>`]
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
        e0: The 1st element of the returned list.
        e1: The 2nd element of the returned list.
        e2: The 3rd element of the returned list.
        e3: The 4th element of the returned list.
        e4: The 5th element of the returned list.
        e5: The 6th element of the returned list.
        e6: The 7th element of the returned list.
        e7: The 8th element of the returned list.
        e8: The 9th element of the returned list.
        e9: The 10th element of the returned list.
        e10: The 11th element of the returned list.
        e11: The 12th element of the returned list.
        e12: The 13th element of the returned list.
        e13: The 14th element of the returned list.

    Returns:
        The list containing the elements.
    """
    return __mlir_op.`pop.variadic.create`[
        _type : __mlir_type[`!kgen.variadic<`, Dim, `>`]
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
        e0: The 1st element of the returned list.
        e1: The 2nd element of the returned list.
        e2: The 3rd element of the returned list.
        e3: The 4th element of the returned list.
        e4: The 5th element of the returned list.
        e5: The 6th element of the returned list.
        e6: The 7th element of the returned list.
        e7: The 8th element of the returned list.
        e8: The 9th element of the returned list.
        e9: The 10th element of the returned list.
        e10: The 11th element of the returned list.
        e11: The 12th element of the returned list.
        e12: The 13th element of the returned list.
        e13: The 14th element of the returned list.
        e14: The 15th element of the returned list.

    Returns:
        The list containing the elements.
    """
    return __mlir_op.`pop.variadic.create`[
        _type : __mlir_type[`!kgen.variadic<`, Dim, `>`]
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
        e0: The 1st element of the returned list.
        e1: The 2nd element of the returned list.
        e2: The 3rd element of the returned list.
        e3: The 4th element of the returned list.
        e4: The 5th element of the returned list.
        e5: The 6th element of the returned list.
        e6: The 7th element of the returned list.
        e7: The 8th element of the returned list.
        e8: The 9th element of the returned list.
        e9: The 10th element of the returned list.
        e10: The 11th element of the returned list.
        e11: The 12th element of the returned list.
        e12: The 13th element of the returned list.
        e13: The 14th element of the returned list.
        e14: The 15th element of the returned list.
        e15: The 16th element of the returned list.

    Returns:
        The list containing the elements.
    """
    return __mlir_op.`pop.variadic.create`[
        _type : __mlir_type[`!kgen.variadic<`, Dim, `>`]
    ](e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15)


# ===----------------------------------------------------------------------===#
# create_kgen_list
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn create_kgen_list[
    type: AnyType
](e0: type) -> __mlir_type[`!kgen.variadic<`, type, `>`]:
    """Creates a list given a type and elements.

    Parameters:
        type: The list type.

    Args:
        e0: The 1st element of the returned list.

    Returns:
        The list containing the elements.
    """
    return __mlir_op.`pop.variadic.create`[
        _type : __mlir_type[`!kgen.variadic<`, type, `>`]
    ](e0)


@always_inline("nodebug")
fn create_kgen_list[
    type: AnyType
](e0: type, e1: type) -> __mlir_type[`!kgen.variadic<`, type, `>`]:
    """Creates a list given a type and elements.

    Parameters:
        type: The list type.

    Args:
        e0: The 1st element of the returned list.
        e1: The 2nd element of the returned list.

    Returns:
        The list containing the elements.
    """
    return __mlir_op.`pop.variadic.create`[
        _type : __mlir_type[`!kgen.variadic<`, type, `>`]
    ](e0, e1)


@always_inline("nodebug")
fn create_kgen_list[
    type: AnyType
](e0: type, e1: type, e2: type) -> __mlir_type[`!kgen.variadic<`, type, `>`]:
    """Creates a list given a type and elements.

    Parameters:
        type: The list type.

    Args:
        e0: The 1st element of the returned list.
        e1: The 2nd element of the returned list.
        e2: The 3rd element of the returned list.

    Returns:
        The list containing the elements.
    """
    return __mlir_op.`pop.variadic.create`[
        _type : __mlir_type[`!kgen.variadic<`, type, `>`]
    ](e0, e1, e2)


@always_inline("nodebug")
fn create_kgen_list[
    type: AnyType
](e0: type, e1: type, e2: type, e3: type) -> __mlir_type[
    `!kgen.variadic<`, type, `>`
]:
    """Creates a list given a type and elements.

    Parameters:
        type: The list type.

    Args:
        e0: The 1st element of the returned list.
        e1: The 2nd element of the returned list.
        e2: The 3rd element of the returned list.
        e3: The 4th element of the returned list.

    Returns:
        The list containing the elements.
    """
    return __mlir_op.`pop.variadic.create`[
        _type : __mlir_type[`!kgen.variadic<`, type, `>`]
    ](e0, e1, e2, e3)


@always_inline("nodebug")
fn create_kgen_list[
    type: AnyType
](e0: type, e1: type, e2: type, e3: type, e4: type,) -> __mlir_type[
    `!kgen.variadic<`, type, `>`
]:
    """Creates a list given a type and elements.

    Parameters:
        type: The list type.

    Args:
        e0: The 1st element of the returned list.
        e1: The 2nd element of the returned list.
        e2: The 3rd element of the returned list.
        e3: The 4th element of the returned list.
        e4: The 5th element of the returned list.

    Returns:
        The list containing the elements.
    """
    return __mlir_op.`pop.variadic.create`[
        _type : __mlir_type[`!kgen.variadic<`, type, `>`]
    ](e0, e1, e2, e3, e4)


@always_inline("nodebug")
fn create_kgen_list[
    type: AnyType
](e0: type, e1: type, e2: type, e3: type, e4: type, e5: type) -> __mlir_type[
    `!kgen.variadic<`, type, `>`
]:
    """Creates a list given a type and elements.

    Parameters:
        type: The list type.

    Args:
        e0: The 1st element of the returned list.
        e1: The 2nd element of the returned list.
        e2: The 3rd element of the returned list.
        e3: The 4th element of the returned list.
        e4: The 5th element of the returned list.
        e5: The 6th element of the returned list.

    Returns:
        The list containing the elements.
    """
    return __mlir_op.`pop.variadic.create`[
        _type : __mlir_type[`!kgen.variadic<`, type, `>`]
    ](e0, e1, e2, e3, e4, e5)


@always_inline("nodebug")
fn create_kgen_list[
    type: AnyType
](
    e0: type,
    e1: type,
    e2: type,
    e3: type,
    e4: type,
    e5: type,
    e6: type,
) -> __mlir_type[`!kgen.variadic<`, type, `>`]:
    """Creates a list given a type and elements.

    Parameters:
        type: The list type.

    Args:
        e0: The 1st element of the returned list.
        e1: The 2nd element of the returned list.
        e2: The 3rd element of the returned list.
        e3: The 4th element of the returned list.
        e4: The 5th element of the returned list.
        e5: The 6th element of the returned list.
        e6: The 7th element of the returned list.

    Returns:
        The list containing the elements.
    """
    return __mlir_op.`pop.variadic.create`[
        _type : __mlir_type[`!kgen.variadic<`, type, `>`]
    ](e0, e1, e2, e3, e4, e5, e6)


@always_inline("nodebug")
fn create_kgen_list[
    type: AnyType
](
    e0: type,
    e1: type,
    e2: type,
    e3: type,
    e4: type,
    e5: type,
    e6: type,
    e7: type,
) -> __mlir_type[`!kgen.variadic<`, type, `>`]:
    """Creates a list given a type and elements.

    Parameters:
        type: The list type.

    Args:
        e0: The 1st element of the returned list.
        e1: The 2nd element of the returned list.
        e2: The 3rd element of the returned list.
        e3: The 4th element of the returned list.
        e4: The 5th element of the returned list.
        e5: The 6th element of the returned list.
        e6: The 7th element of the returned list.
        e7: The 8th element of the returned list.

    Returns:
        The list containing the elements.
    """
    return __mlir_op.`pop.variadic.create`[
        _type : __mlir_type[`!kgen.variadic<`, type, `>`]
    ](e0, e1, e2, e3, e4, e5, e6, e7)


@always_inline("nodebug")
fn create_kgen_list[
    type: AnyType
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
) -> __mlir_type[`!kgen.variadic<`, type, `>`]:
    """Creates a list given a type and elements.

    Parameters:
        type: The list type.

    Args:
        e0: The 1st element of the returned list.
        e1: The 2nd element of the returned list.
        e2: The 3rd element of the returned list.
        e3: The 4th element of the returned list.
        e4: The 5th element of the returned list.
        e5: The 6th element of the returned list.
        e6: The 7th element of the returned list.
        e7: The 8th element of the returned list.
        e8: The 9th element of the returned list.

    Returns:
        The list containing the elements.
    """
    return __mlir_op.`pop.variadic.create`[
        _type : __mlir_type[`!kgen.variadic<`, type, `>`]
    ](e0, e1, e2, e3, e4, e5, e6, e7, e8)


@always_inline("nodebug")
fn create_kgen_list[
    type: AnyType
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
) -> __mlir_type[`!kgen.variadic<`, type, `>`]:
    """Creates a list given a type and elements.

    Parameters:
        type: The list type.

    Args:
        e0: The 1st element of the returned list.
        e1: The 2nd element of the returned list.
        e2: The 3rd element of the returned list.
        e3: The 4th element of the returned list.
        e4: The 5th element of the returned list.
        e5: The 6th element of the returned list.
        e6: The 7th element of the returned list.
        e7: The 8th element of the returned list.
        e8: The 9th element of the returned list.
        e9: The 10th element of the returned list.

    Returns:
        The list containing the elements.
    """
    return __mlir_op.`pop.variadic.create`[
        _type : __mlir_type[`!kgen.variadic<`, type, `>`]
    ](e0, e1, e2, e3, e4, e5, e6, e7, e8, e9)


@always_inline("nodebug")
fn create_kgen_list[
    type: AnyType
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
) -> __mlir_type[`!kgen.variadic<`, type, `>`]:
    """Creates a list given a type and elements.

    Parameters:
        type: The list type.

    Args:
        e0: The 1st element of the returned list.
        e1: The 2nd element of the returned list.
        e2: The 3rd element of the returned list.
        e3: The 4th element of the returned list.
        e4: The 5th element of the returned list.
        e5: The 6th element of the returned list.
        e6: The 7th element of the returned list.
        e7: The 8th element of the returned list.
        e8: The 9th element of the returned list.
        e9: The 10th element of the returned list.
        e10: The 11th element of the returned list.

    Returns:
        The list containing the elements.
    """
    return __mlir_op.`pop.variadic.create`[
        _type : __mlir_type[`!kgen.variadic<`, type, `>`]
    ](e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10)


@always_inline("nodebug")
fn create_kgen_list[
    type: AnyType
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
) -> __mlir_type[`!kgen.variadic<`, type, `>`]:
    """Creates a list given a type and elements.

    Parameters:
        type: The list type.

    Args:
        e0: The 1st element of the returned list.
        e1: The 2nd element of the returned list.
        e2: The 3rd element of the returned list.
        e3: The 4th element of the returned list.
        e4: The 5th element of the returned list.
        e5: The 6th element of the returned list.
        e6: The 7th element of the returned list.
        e7: The 8th element of the returned list.
        e8: The 9th element of the returned list.
        e9: The 10th element of the returned list.
        e10: The 11th element of the returned list.
        e11: The 12th element of the returned list.

    Returns:
        The list containing the elements.
    """
    return __mlir_op.`pop.variadic.create`[
        _type : __mlir_type[`!kgen.variadic<`, type, `>`]
    ](e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11)


@always_inline("nodebug")
fn create_kgen_list[
    type: AnyType
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
) -> __mlir_type[`!kgen.variadic<`, type, `>`]:
    """Creates a list given a type and elements.

    Parameters:
        type: The list type.

    Args:
        e0: The 1st element of the returned list.
        e1: The 2nd element of the returned list.
        e2: The 3rd element of the returned list.
        e3: The 4th element of the returned list.
        e4: The 5th element of the returned list.
        e5: The 6th element of the returned list.
        e6: The 7th element of the returned list.
        e7: The 8th element of the returned list.
        e8: The 9th element of the returned list.
        e9: The 10th element of the returned list.
        e10: The 11th element of the returned list.
        e11: The 12th element of the returned list.
        e12: The 13th element of the returned list.

    Returns:
        The list containing the elements.
    """
    return __mlir_op.`pop.variadic.create`[
        _type : __mlir_type[`!kgen.variadic<`, type, `>`]
    ](e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12)


@always_inline("nodebug")
fn create_kgen_list[
    type: AnyType
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
) -> __mlir_type[`!kgen.variadic<`, type, `>`]:
    """Creates a list given a type and elements.

    Parameters:
        type: The list type.

    Args:
        e0: The 1st element of the returned list.
        e1: The 2nd element of the returned list.
        e2: The 3rd element of the returned list.
        e3: The 4th element of the returned list.
        e4: The 5th element of the returned list.
        e5: The 6th element of the returned list.
        e6: The 7th element of the returned list.
        e7: The 8th element of the returned list.
        e8: The 9th element of the returned list.
        e9: The 10th element of the returned list.
        e10: The 11th element of the returned list.
        e11: The 12th element of the returned list.
        e12: The 13th element of the returned list.
        e13: The 14th element of the returned list.

    Returns:
        The list containing the elements.
    """
    return __mlir_op.`pop.variadic.create`[
        _type : __mlir_type[`!kgen.variadic<`, type, `>`]
    ](e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13)


@always_inline("nodebug")
fn create_kgen_list[
    type: AnyType
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
) -> __mlir_type[`!kgen.variadic<`, type, `>`]:
    """Creates a list given a type and elements.

    Parameters:
        type: The list type.

    Args:
        e0: The 1st element of the returned list.
        e1: The 2nd element of the returned list.
        e2: The 3rd element of the returned list.
        e3: The 4th element of the returned list.
        e4: The 5th element of the returned list.
        e5: The 6th element of the returned list.
        e6: The 7th element of the returned list.
        e7: The 8th element of the returned list.
        e8: The 9th element of the returned list.
        e9: The 10th element of the returned list.
        e10: The 11th element of the returned list.
        e11: The 12th element of the returned list.
        e12: The 13th element of the returned list.
        e13: The 14th element of the returned list.
        e14: The 15th element of the returned list.

    Returns:
        The list containing the elements.
    """
    return __mlir_op.`pop.variadic.create`[
        _type : __mlir_type[`!kgen.variadic<`, type, `>`]
    ](e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14)


@always_inline("nodebug")
fn create_kgen_list[
    type: AnyType
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
) -> __mlir_type[`!kgen.variadic<`, type, `>`]:
    """Creates a list given a type and elements.

    Parameters:
        type: The list type.

    Args:
        e0: The 1st element of the returned list.
        e1: The 2nd element of the returned list.
        e2: The 3rd element of the returned list.
        e3: The 4th element of the returned list.
        e4: The 5th element of the returned list.
        e5: The 6th element of the returned list.
        e6: The 7th element of the returned list.
        e7: The 8th element of the returned list.
        e8: The 9th element of the returned list.
        e9: The 10th element of the returned list.
        e10: The 11th element of the returned list.
        e11: The 12th element of the returned list.
        e12: The 13th element of the returned list.
        e13: The 14th element of the returned list.
        e14: The 15th element of the returned list.
        e15: The 16th element of the returned list.

    Returns:
        The list containing the elements.
    """
    return __mlir_op.`pop.variadic.create`[
        _type : __mlir_type[`!kgen.variadic<`, type, `>`]
    ](e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15)


# ===----------------------------------------------------------------------===#
# _get_kgen_list_item
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn _get_kgen_list_item[
    index: Int,
    size: Int,
    type: AnyType,
](lst: __mlir_type[`!kgen.variadic<`, type, `>`]) -> type:
    """Gets the list element of an input list at position `index`.

    Parameters:
        index: the position to get the value from.
        size: the size of the list.
        type: the element type of the list.

    Args:
        lst: the list to get the values from.

    Returns:
        The value at position `index` in the list.
    """
    assert_param[index <= size]()
    return __mlir_op.`pop.variadic.get`[_type:type](
        lst, index.__as_mlir_index()
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
        return VariadicList[__mlir_type.`index`](l[0].__as_mlir_index())
    elif sz == 2:
        return VariadicList[__mlir_type.`index`](
            l[0].__as_mlir_index(), l[1].__as_mlir_index()
        )
    elif sz == 3:
        return VariadicList[__mlir_type.`index`](
            l[0].__as_mlir_index(),
            l[1].__as_mlir_index(),
            l[2].__as_mlir_index(),
        )
    elif sz == 4:
        return VariadicList[__mlir_type.`index`](
            l[0].__as_mlir_index(),
            l[1].__as_mlir_index(),
            l[2].__as_mlir_index(),
            l[3].__as_mlir_index(),
        )
    elif sz == 5:
        return VariadicList[__mlir_type.`index`](
            l[0].__as_mlir_index(),
            l[1].__as_mlir_index(),
            l[2].__as_mlir_index(),
            l[3].__as_mlir_index(),
            l[4].__as_mlir_index(),
        )
    elif sz == 6:
        return VariadicList[__mlir_type.`index`](
            l[0].__as_mlir_index(),
            l[1].__as_mlir_index(),
            l[2].__as_mlir_index(),
            l[3].__as_mlir_index(),
            l[4].__as_mlir_index(),
            l[5].__as_mlir_index(),
        )
    elif sz == 7:
        return VariadicList[__mlir_type.`index`](
            l[0].__as_mlir_index(),
            l[1].__as_mlir_index(),
            l[2].__as_mlir_index(),
            l[3].__as_mlir_index(),
            l[4].__as_mlir_index(),
            l[5].__as_mlir_index(),
            l[6].__as_mlir_index(),
        )
    elif sz == 8:
        return VariadicList[__mlir_type.`index`](
            l[0].__as_mlir_index(),
            l[1].__as_mlir_index(),
            l[2].__as_mlir_index(),
            l[3].__as_mlir_index(),
            l[4].__as_mlir_index(),
            l[5].__as_mlir_index(),
            l[6].__as_mlir_index(),
            l[7].__as_mlir_index(),
        )
    elif sz == 9:
        return VariadicList[__mlir_type.`index`](
            l[0].__as_mlir_index(),
            l[1].__as_mlir_index(),
            l[2].__as_mlir_index(),
            l[3].__as_mlir_index(),
            l[4].__as_mlir_index(),
            l[5].__as_mlir_index(),
            l[6].__as_mlir_index(),
            l[7].__as_mlir_index(),
            l[8].__as_mlir_index(),
        )
    elif sz == 10:
        return VariadicList[__mlir_type.`index`](
            l[0].__as_mlir_index(),
            l[1].__as_mlir_index(),
            l[2].__as_mlir_index(),
            l[3].__as_mlir_index(),
            l[4].__as_mlir_index(),
            l[5].__as_mlir_index(),
            l[6].__as_mlir_index(),
            l[7].__as_mlir_index(),
            l[8].__as_mlir_index(),
            l[9].__as_mlir_index(),
        )
    elif sz == 11:
        return VariadicList[__mlir_type.`index`](
            l[0].__as_mlir_index(),
            l[1].__as_mlir_index(),
            l[2].__as_mlir_index(),
            l[3].__as_mlir_index(),
            l[4].__as_mlir_index(),
            l[5].__as_mlir_index(),
            l[6].__as_mlir_index(),
            l[7].__as_mlir_index(),
            l[8].__as_mlir_index(),
            l[9].__as_mlir_index(),
            l[10].__as_mlir_index(),
        )
    elif sz == 12:
        return VariadicList[__mlir_type.`index`](
            l[0].__as_mlir_index(),
            l[1].__as_mlir_index(),
            l[2].__as_mlir_index(),
            l[3].__as_mlir_index(),
            l[4].__as_mlir_index(),
            l[5].__as_mlir_index(),
            l[6].__as_mlir_index(),
            l[7].__as_mlir_index(),
            l[8].__as_mlir_index(),
            l[9].__as_mlir_index(),
            l[10].__as_mlir_index(),
            l[11].__as_mlir_index(),
        )
    elif sz == 13:
        return VariadicList[__mlir_type.`index`](
            l[0].__as_mlir_index(),
            l[1].__as_mlir_index(),
            l[2].__as_mlir_index(),
            l[3].__as_mlir_index(),
            l[4].__as_mlir_index(),
            l[5].__as_mlir_index(),
            l[6].__as_mlir_index(),
            l[7].__as_mlir_index(),
            l[8].__as_mlir_index(),
            l[9].__as_mlir_index(),
            l[10].__as_mlir_index(),
            l[11].__as_mlir_index(),
            l[12].__as_mlir_index(),
        )
    elif sz == 14:
        return VariadicList[__mlir_type.`index`](
            l[0].__as_mlir_index(),
            l[1].__as_mlir_index(),
            l[2].__as_mlir_index(),
            l[3].__as_mlir_index(),
            l[4].__as_mlir_index(),
            l[5].__as_mlir_index(),
            l[6].__as_mlir_index(),
            l[7].__as_mlir_index(),
            l[8].__as_mlir_index(),
            l[9].__as_mlir_index(),
            l[10].__as_mlir_index(),
            l[11].__as_mlir_index(),
            l[12].__as_mlir_index(),
            l[13].__as_mlir_index(),
        )
    elif sz == 15:
        return VariadicList[__mlir_type.`index`](
            l[0].__as_mlir_index(),
            l[1].__as_mlir_index(),
            l[2].__as_mlir_index(),
            l[3].__as_mlir_index(),
            l[4].__as_mlir_index(),
            l[5].__as_mlir_index(),
            l[6].__as_mlir_index(),
            l[7].__as_mlir_index(),
            l[8].__as_mlir_index(),
            l[9].__as_mlir_index(),
            l[10].__as_mlir_index(),
            l[11].__as_mlir_index(),
            l[12].__as_mlir_index(),
            l[13].__as_mlir_index(),
            l[14].__as_mlir_index(),
        )
    assert_param[sz == 16]()
    return VariadicList[__mlir_type.`index`](
        l[0].__as_mlir_index(),
        l[1].__as_mlir_index(),
        l[2].__as_mlir_index(),
        l[3].__as_mlir_index(),
        l[4].__as_mlir_index(),
        l[5].__as_mlir_index(),
        l[6].__as_mlir_index(),
        l[7].__as_mlir_index(),
        l[8].__as_mlir_index(),
        l[9].__as_mlir_index(),
        l[10].__as_mlir_index(),
        l[11].__as_mlir_index(),
        l[12].__as_mlir_index(),
        l[13].__as_mlir_index(),
        l[14].__as_mlir_index(),
        l[15].__as_mlir_index(),
    )


@register_passable("trivial")
struct VariadicList[type: AnyType]:
    """A utility class to access variadic function arguments. Provides a "list"
    view of the function argument so that the size of the argument list and each
    individual argument can be accessed.
    """

    alias StorageType = __mlir_type[`!kgen.variadic<`, type, `>`]
    var value: StorageType

    fn __init__(*value: type) -> Self:
        """Constructs a VariadicList from a variadic list of arguments.

        Args:
            value: The variadic argument list to construct the variadic list
              with.

        Returns:
            The VariadicList constructed.
        """
        return Self(value)

    fn __init__(value: StorageType) -> Self:
        """Constructs a VariadicList from a variadic argument type.

        Args:
            value: The variadic argument to construct the list with.

        Returns:
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
            index: The index of the element to access on the list.

        Returns:
            The element on the list corresponding to the given index.
        """
        return __mlir_op.`pop.variadic.get`(self.value, index.__as_mlir_index())


@register_passable("trivial")
struct VariadicListMem[type: AnyType]:
    """A utility class to access variadic function arguments of memory-only
    types that may have ownership. It exposes pointers to the elements in a way
    that can be enumerated.  Each element may be accessed with
    `__get_address_as_lvalue`.
    """

    alias StorageType = __mlir_type[`!kgen.variadic<!pop.pointer<`, type, `>>`]
    var value: StorageType

    fn __init__(value: StorageType) -> Self:
        """Constructs a VariadicList from a variadic argument type.

        Args:
            value: The variadic argument to construct the list with.

        Returns:
            The VariadicList constructed.
        """
        return Self {value: value}

    fn __len__(self) -> Int:
        """Gets the size of the list.

        Returns:
            The number of elements on the variadic list.
        """

        return __mlir_op.`pop.variadic.size`(self.value)

    fn __getitem__(self, index: Int) -> __mlir_type[`!pop.pointer<`, type, `>`]:
        """Accessor to a single element on the variadic list.

        Args:
            index: The index of the element to access on the list.

        Returns:
            A low-level pointer to the element on the list corresponding to the
            given index.
        """
        return __mlir_op.`pop.variadic.get`(self.value, index.__as_mlir_index())
