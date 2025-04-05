# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from sys.info import _current_target
from collections.string import StaticString


fn get_linkage_name[
    func_type: AnyTrivialRegType, //,
    target: __mlir_type.`!kgen.target`,
    func: func_type,
]() -> StaticString:
    """Returns `func` symbol name.

    Parameters:
        func_type: Type of func.
        target: The compilation target.
        func: A mojo function.

    Returns:
        Symbol name.
    """
    return __mlir_attr[
        `#kgen.param.expr<get_linkage_name,`,
        target,
        `,`,
        func,
        `> : !kgen.string`,
    ]


fn get_linkage_name[
    func_type: AnyTrivialRegType, //,
    func: func_type,
]() -> StaticString:
    """Returns `func` symbol name.

    Parameters:
        func_type: Type of func.
        func: A mojo function.

    Returns:
        Symbol name.
    """
    return get_linkage_name[_current_target(), func]()
