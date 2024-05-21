# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


fn get_linkage_name[
    inferred func_type: AnyRegType,
    target: __mlir_type.`!kgen.target`,
    func: func_type,
]() -> StringLiteral:
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
