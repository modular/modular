# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Int import Int

from Assert import assert_param, debug_assert
from Functional import unroll
from TypeUtilities import rebind


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
    index: __mlir_type.index,
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
    assert_param[index <= size]()
    return __mlir_op.`pop.list.get`[index:index, _type:type](lst)


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
    start_idx: __mlir_type.index,
    end_idx: __mlir_type.index,
    size: __mlir_type.index,
](lst: __mlir_type[`!kgen.list<index[`, size, `]>`]) -> Int:
    return _product_range_impl[start_idx, start_idx, end_idx, size](lst)


fn _product_range_impl[
    idx: __mlir_type.index,
    start_idx: __mlir_type.index,
    end_idx: __mlir_type.index,
    size: __mlir_type.index,
](lst: __mlir_type[`!kgen.list<index[`, size, `]>`]) -> Int:
    assert_param[idx <= end_idx]()

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
    start_idx: __mlir_type.index,
    end_idx: __mlir_type.index,
]() -> __mlir_type.index:
    @parameter
    if is_all_known_range[start_idx, end_idx, rank, shape]():
        return product_range[start_idx, end_idx, rank](shape).__as_mlir_index()
    return __mlir_attr.`#kgen.unknown : index`


# ===----------------------------------------------------------------------===#
# is_all_known
# ===----------------------------------------------------------------------===#


fn is_all_known[
    rank: __mlir_type.index, shape: __mlir_type[`!kgen.list<index[`, rank, `]>`]
]() -> Bool:
    return is_all_known_range[0, rank, rank, shape]()


fn is_all_known_range[
    start_idx: __mlir_type.index,
    end_idx: __mlir_type.index,
    rank: __mlir_type.index,
    shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
]() -> Bool:
    return is_all_known_range_impl[start_idx, start_idx, end_idx, rank, shape]()


fn is_all_known_range_impl[
    index: __mlir_type.index,
    start_idx: __mlir_type.index,
    end_idx: __mlir_type.index,
    rank: __mlir_type.index,
    shape: __mlir_type[`!kgen.list<index[`, rank, `]>`],
]() -> Bool:
    assert_param[index <= end_idx]()

    @parameter
    if index == end_idx:
        return True
    else:
        alias static_dim_value = _get_kgen_list_item[
            index, rank, __mlir_type.index
        ](shape)
        return (
            Bool(static_dim_value != __mlir_attr.`#kgen.unknown : index`)
            and is_all_known_range_impl[
                index + 1, start_idx, end_idx, rank, shape
            ]()
        )


# ===----------------------------------------------------------------------===#
# create_kgen_list_unknown
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn create_kgen_list_unknown[
    len: __mlir_type.index
]() -> __mlir_type[`!kgen.list<`, __mlir_type.index, `[`, len, `]>`]:
    """Creates a list of LEN kgen.unknown elements."""
    alias unknown = __mlir_attr.`#kgen.unknown : index`

    @parameter
    if len == 1:
        return rebind[
            __mlir_type[`!kgen.list<`, __mlir_type.index, `[`, len, `]>`]
        ](create_kgen_list(unknown))

    @parameter
    if len == 2:
        return rebind[
            __mlir_type[`!kgen.list<`, __mlir_type.index, `[`, len, `]>`]
        ](create_kgen_list(unknown, unknown))

    @parameter
    if len == 3:
        return rebind[
            __mlir_type[`!kgen.list<`, __mlir_type.index, `[`, len, `]>`]
        ](create_kgen_list(unknown, unknown, unknown))

    @parameter
    if len == 4:
        return rebind[
            __mlir_type[`!kgen.list<`, __mlir_type.index, `[`, len, `]>`]
        ](create_kgen_list(unknown, unknown, unknown, unknown))

    @parameter
    if len == 5:
        return rebind[
            __mlir_type[`!kgen.list<`, __mlir_type.index, `[`, len, `]>`]
        ](create_kgen_list(unknown, unknown, unknown, unknown, unknown))

    debug_assert(False, "unreachable")
    return rebind[
        __mlir_type[`!kgen.list<`, __mlir_type.index, `[`, len, `]>`]
    ](create_kgen_list(unknown))


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

    fn __clone__(self&) -> Self:
        return Self {value: self.value}

    fn __new__(*value: type) -> Self:
        """Constructs a VariadicList from a variadic list of arguments.

        Args:
            *value (type):
                The variadic argument list to construct the variadic list with.

        Returns (VariadicList):
            The VariadicList constructed.
        """
        return Self(value)

    fn __new__(value: StorageType) -> VariadicList[type]:
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
