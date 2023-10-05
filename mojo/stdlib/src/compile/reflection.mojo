# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


fn _get_linkage_name[func_type: AnyType, func: func_type->asm: StringLiteral]():
    param_return[
        __mlir_attr[
            `#kgen.param.expr<get_linkage_name,`,
            func,
            `> : !kgen.string`,
        ]
    ]


fn get_linkage_name[func_type: AnyType, func: func_type]() -> StringLiteral:
    """Returns `func` symbol name.

    Parameters:
        func_type: Type of func.
        func: A mojo function.

    Returns:
        Symbol name.
    """
    alias fn_name: StringLiteral
    _get_linkage_name[func_type, func -> fn_name]()
    return fn_name
