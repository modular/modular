# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

alias Int = __mlir_type.index
alias AnyRegType = __mlir_type.`!kgen.type`

alias `1` = __mlir_attr.`1 : index`
alias `2` = __mlir_attr.`2 : index`
alias `3` = __mlir_attr.`3 : index`
alias `4` = __mlir_attr.`4 : index`
alias `5` = __mlir_attr.`5 : index`
alias `6` = __mlir_attr.`6 : index`
alias `7` = __mlir_attr.`7 : index`
alias `8` = __mlir_attr.`8 : index`
alias `9` = __mlir_attr.`9 : index`


@register_passable
struct Error:
    pass


struct object:
    pass


# ===----------------------------------------------------------------------=== #
# Value Stubs
# ===----------------------------------------------------------------------=== #


trait Copyable:
    fn __copyinit__(inout self, existing: Self, /):
        pass


trait Movable:
    fn __moveinit__(inout self, owned existing: Self, /):
        pass


# ===----------------------------------------------------------------------=== #
# Builtin Collection Stubs
# ===----------------------------------------------------------------------=== #


@register_passable
struct VariadicList[type: AnyRegType]:
    alias storage_type = __mlir_type[`!kgen.variadic<`, type, `>`]

    fn __init__(value: Self.storage_type) -> Self:
        return Self {}
