# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Int import Int
from Bool import Bool
from Assert import assert_param
from Functional import repeat


# ===----------------------------------------------------------------------===#
# create_kgen_list
# ===----------------------------------------------------------------------===#


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


# ===----------------------------------------------------------------------===#
# _get_kgen_list_item
# ===----------------------------------------------------------------------===#


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
) -> __mlir_type.i1:
    return _contains_impl[0, size](elem, lst)


@interface
fn _contains_impl[
    idx: __mlir_type.index, size: __mlir_type.index
](
    elem: __mlir_type.index, lst: __mlir_type[`!kgen.list<index[`, size, `]>`]
) -> __mlir_type.i1:
    ...


@implements(_contains_impl)
fn _contains_impl_base[
    idx: __mlir_type.index, size: __mlir_type.index
](
    elem: __mlir_type.index, lst: __mlir_type[`!kgen.list<index[`, size, `]>`]
) -> __mlir_type.i1:
    assert_param[idx == size]()
    return __mlir_attr.`0:i1`


@implements(_contains_impl)
fn _contains_impl_iter[
    idx: __mlir_type.index, size: __mlir_type.index
](
    elem: __mlir_type.index, lst: __mlir_type[`!kgen.list<index[`, size, `]>`]
) -> __mlir_type.i1:
    assert_param[idx < size]()
    let ok = __mlir_op.`index.cmp`[
        pred : __mlir_attr.`#index<cmp_predicate eq>`
    ](_get_kgen_list_item[idx, size, __mlir_type.index](lst), elem)
    return ok


# ===----------------------------------------------------------------------===#
# product
# ===----------------------------------------------------------------------===#


fn product[
    size: __mlir_type.index
](lst: __mlir_type[`!kgen.list<index[`, size, `]>`]) -> Int:
    return _product_impl[0, size](lst)


@interface
fn _product_impl[
    idx: __mlir_type.index, size: __mlir_type.index
](lst: __mlir_type[`!kgen.list<index[`, size, `]>`]) -> Int:
    ...


@implements(_product_impl)
fn _product_impl_base[
    idx: __mlir_type.index, size: __mlir_type.index
](lst: __mlir_type[`!kgen.list<index[`, size, `]>`]) -> Int:
    assert_param[idx == size]()
    return 1


@implements(_product_impl)
fn _product_impl_iter[
    idx: __mlir_type.index, size: __mlir_type.index
](lst: __mlir_type[`!kgen.list<index[`, size, `]>`]) -> Int:
    assert_param[idx < size]()
    return _get_kgen_list_item[idx, size, __mlir_type.index](
        lst
    ) * _product_impl[idx + 1, size](lst)


# ===----------------------------------------------------------------------===#
# create_kgen_list_unknown
# ===----------------------------------------------------------------------===#


@interface
fn create_kgen_list_unknown[
    len: __mlir_type.index
]() -> __mlir_type[`!kgen.list<`, __mlir_type.index, `[len]>`]:
    """Creates a list of LEN kgen.unknown elements."""
    ...


@implements(create_kgen_list_unknown)
fn create_kgen_list_unknown_1[
    len: __mlir_type.index
]() -> __mlir_type[`!kgen.list<`, __mlir_type.index, `[len]>`]:
    """Creates a list of 1 kgen.unknown element."""
    assert_param[len == 1]()
    let u = __mlir_attr.`#kgen.unknown : index`
    let l = __mlir_op.`pop.list.create`[
        _type : __mlir_type[`!kgen.list<`, __mlir_type.index, `[1]>`]
    ](u)
    return __mlir_op.`kgen.rebind`[
        _type : __mlir_type[`!kgen.list<`, __mlir_type.index, `[len]>`]
    ](l)


@implements(create_kgen_list_unknown)
fn create_kgen_list_unknown_2[
    len: __mlir_type.index
]() -> __mlir_type[`!kgen.list<`, __mlir_type.index, `[len]>`]:
    """Creates a list of 2 kgen.unknown elements."""
    assert_param[len == 2]()
    let u = __mlir_attr.`#kgen.unknown : index`
    let l = __mlir_op.`pop.list.create`[
        _type : __mlir_type[`!kgen.list<`, __mlir_type.index, `[2]>`]
    ](u, u)

    return __mlir_op.`kgen.rebind`[
        _type : __mlir_type[`!kgen.list<`, __mlir_type.index, `[len]>`]
    ](l)


@implements(create_kgen_list_unknown)
fn create_kgen_list_unknown_3[
    len: __mlir_type.index
]() -> __mlir_type[`!kgen.list<`, __mlir_type.index, `[len]>`]:
    """Creates a list of 3 kgen.unknown elements."""
    assert_param[len == 3]()
    let u = __mlir_attr.`#kgen.unknown : index`
    let l = __mlir_op.`pop.list.create`[
        _type : __mlir_type[`!kgen.list<`, __mlir_type.index, `[3]>`]
    ](u, u, u)

    return __mlir_op.`kgen.rebind`[
        _type : __mlir_type[`!kgen.list<`, __mlir_type.index, `[len]>`]
    ](l)


@implements(create_kgen_list_unknown)
fn create_kgen_list_unknown_4[
    len: __mlir_type.index
]() -> __mlir_type[`!kgen.list<`, __mlir_type.index, `[len]>`]:
    """Creates a list of 4 kgen.unknown elements."""
    assert_param[len == 4]()
    let u = __mlir_attr.`#kgen.unknown : index`
    let l = __mlir_op.`pop.list.create`[
        _type : __mlir_type[`!kgen.list<`, __mlir_type.index, `[4]>`]
    ](u, u, u, u)

    return __mlir_op.`kgen.rebind`[
        _type : __mlir_type[`!kgen.list<`, __mlir_type.index, `[len]>`]
    ](l)


@implements(create_kgen_list_unknown)
fn create_kgen_list_unknown_5[
    len: __mlir_type.index
]() -> __mlir_type[`!kgen.list<`, __mlir_type.index, `[len]>`]:
    """Creates a list of 5 kgen.unknown elements."""
    assert_param[len == 5]()
    let u = __mlir_attr.`#kgen.unknown : index`
    let l = __mlir_op.`pop.list.create`[
        _type : __mlir_type[`!kgen.list<`, __mlir_type.index, `[5]>`]
    ](u, u, u, u, u)

    return __mlir_op.`kgen.rebind`[
        _type : __mlir_type[`!kgen.list<`, __mlir_type.index, `[len]>`]
    ](l)
