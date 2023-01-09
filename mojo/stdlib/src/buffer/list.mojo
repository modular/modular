# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


fn create_kgen_list_1[
    type: __mlir_type.`!kgen.mlirtype`, e0: type
]() -> __mlir_type[`!kgen.list<`, type, `[1]>`]:
    """Creates a list given a type and elements.

    Args:
        type (!kgen.mlirtype): The list type.
        e0 (type): The 1st element of the returned list.

    Returns:
        !kgen.list<type[1]>: The list containing the elements.
    """
    return __mlir_attr[`#kgen.list<`, e0, `> : !kgen.list<`, type, `[1]>`]


fn create_kgen_list_2[
    type: __mlir_type.`!kgen.mlirtype`, e0: type, e1: type
]() -> __mlir_type[`!kgen.list<`, type, `[2]>`]:
    """Creates a list given a type and elements.

    Args:
        type (!kgen.mlirtype): The list type.
        e0 (type): The 1st element of the returned list.
        e1 (type): The 2nd element of the returned list.

    Returns:
        !kgen.list<type[2]>: The list containing the elements.
    """
    return __mlir_attr[
        `#kgen.list<`, e0, `,`, e1, `> : !kgen.list<`, type, `[2]>`
    ]


fn create_kgen_list_3[
    type: __mlir_type.`!kgen.mlirtype`, e0: type, e1: type, e2: type
]() -> __mlir_type[`!kgen.list<`, type, `[3]>`]:
    """Creates a list given a type and elements.

    Args:
        type (!kgen.mlirtype): The list type.
        e0 (type): The 1st element of the returned list.
        e1 (type): The 2nd element of the returned list.
        e2 (type): The 3rd element of the returned list.

    Returns:
        !kgen.list<type[3]>: The list containing the elements.
    """
    return __mlir_attr[
        `#kgen.list<`, e0, `,`, e1, `,`, e2, `> : !kgen.list<`, type, `[3]>`
    ]


fn create_kgen_list_4[
    type: __mlir_type.`!kgen.mlirtype`, e0: type, e1: type, e2: type, e3: type
]() -> __mlir_type[`!kgen.list<`, type, `[4]>`]:
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
    return __mlir_attr[
        `#kgen.list<`,
        e0,
        `,`,
        e1,
        `,`,
        e2,
        `,`,
        e3,
        `> : !kgen.list<`,
        type,
        `[4]>`,
    ]


fn create_kgen_list_5[
    type: __mlir_type.`!kgen.mlirtype`,
    e0: type,
    e1: type,
    e2: type,
    e3: type,
    e4: type,
]() -> __mlir_type[`!kgen.list<`, type, `[5]>`]:
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
    return __mlir_attr[
        `#kgen.list<`,
        e0,
        `,`,
        e1,
        `,`,
        e2,
        `,`,
        e3,
        `,`,
        e4,
        `> : !kgen.list<`,
        type,
        `[5]>`,
    ]


fn create_kgen_list_6[
    type: __mlir_type.`!kgen.mlirtype`,
    e0: type,
    e1: type,
    e2: type,
    e3: type,
    e4: type,
    e5: type,
]() -> __mlir_type[`!kgen.list<`, type, `[6]>`]:
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
    return __mlir_attr[
        `#kgen.list<`,
        e0,
        `,`,
        e1,
        `,`,
        e2,
        `,`,
        e3,
        `,`,
        e4,
        `,`,
        e5,
        `> : !kgen.list<`,
        type,
        `[6]>`,
    ]


fn create_kgen_list_7[
    type: __mlir_type.`!kgen.mlirtype`,
    e0: type,
    e1: type,
    e2: type,
    e3: type,
    e4: type,
    e5: type,
    e6: type,
]() -> __mlir_type[`!kgen.list<`, type, `[7]>`]:
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
    return __mlir_attr[
        `#kgen.list<`,
        e0,
        `,`,
        e1,
        `,`,
        e2,
        `,`,
        e3,
        `,`,
        e4,
        `,`,
        e5,
        `,`,
        e6,
        `> : !kgen.list<`,
        type,
        `[7]>`,
    ]


fn create_kgen_list_8[
    type: __mlir_type.`!kgen.mlirtype`,
    e0: type,
    e1: type,
    e2: type,
    e3: type,
    e4: type,
    e5: type,
    e6: type,
    e7: type,
]() -> __mlir_type[`!kgen.list<`, type, `[8]>`]:
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
    return __mlir_attr[
        `#kgen.list<`,
        e0,
        `,`,
        e1,
        `,`,
        e2,
        `,`,
        e3,
        `,`,
        e4,
        `,`,
        e5,
        `,`,
        e6,
        `,`,
        e7,
        `> : !kgen.list<`,
        type,
        `[8]>`,
    ]


fn create_kgen_list_9[
    type: __mlir_type.`!kgen.mlirtype`,
    e0: type,
    e1: type,
    e2: type,
    e3: type,
    e4: type,
    e5: type,
    e6: type,
    e7: type,
    e8: type,
]() -> __mlir_type[`!kgen.list<`, type, `[9]>`]:
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
    return __mlir_attr[
        `#kgen.list<`,
        e0,
        `,`,
        e1,
        `,`,
        e2,
        `,`,
        e3,
        `,`,
        e4,
        `,`,
        e5,
        `,`,
        e6,
        `,`,
        e7,
        `,`,
        e8,
        `> : !kgen.list<`,
        type,
        `[9]>`,
    ]


fn create_kgen_list_10[
    type: __mlir_type.`!kgen.mlirtype`,
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
]() -> __mlir_type[`!kgen.list<`, type, `[10]>`]:
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
    return __mlir_attr[
        `#kgen.list<`,
        e0,
        `,`,
        e1,
        `,`,
        e2,
        `,`,
        e3,
        `,`,
        e4,
        `,`,
        e5,
        `,`,
        e6,
        `,`,
        e7,
        `,`,
        e8,
        `,`,
        e9,
        `> : !kgen.list<`,
        type,
        `[10]>`,
    ]
