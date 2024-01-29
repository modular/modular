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


@register_passable
struct Error:
    pass


trait Copyable:
    fn __copyinit__(inout self, existing: Self, /):
        pass


trait Movable:
    fn __moveinit__(inout self, owned existing: Self, /):
        pass
