# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

alias Int = __mlir_type.index
alias AnyRegType = __mlir_type.`!kgen.type`
alias StringLiteral = __mlir_type.`!kgen.string`
alias Float = __mlir_type.`!pop.scalar<f64>`

alias `0` = __mlir_attr.`0 : index`
alias `1` = __mlir_attr.`1 : index`
alias `2` = __mlir_attr.`2 : index`
alias `3` = __mlir_attr.`3 : index`
alias `4` = __mlir_attr.`4 : index`
alias `5` = __mlir_attr.`5 : index`
alias `6` = __mlir_attr.`6 : index`
alias `7` = __mlir_attr.`7 : index`
alias `8` = __mlir_attr.`8 : index`
alias `9` = __mlir_attr.`9 : index`
alias `10` = __mlir_attr.`10 : index`
alias `42` = __mlir_attr.`42 : index`
alias `123` = __mlir_attr.`123 : index`


@register_passable
struct Error:
    pass


struct object:
    pass


@register_passable("trivial")
struct Bool(AnyType):
    var x: __mlir_type.i1

    fn __mlir_i1__(self) -> __mlir_type.i1:
        return self.x


@register_passable("trivial")
struct Slice:
    fn __init__(end: Int) -> Self:
        return Self {}

    fn __init__(start: Int, end: Int) -> Self:
        return Self {}

    fn __init__[
        T0: AnyRegType, T1: AnyRegType, T2: AnyRegType
    ](start: T0, end: T1, step: T2) -> Self:
        return Self {}


# ===----------------------------------------------------------------------=== #
# Value Stubs
# ===----------------------------------------------------------------------=== #


trait Copyable:
    fn __copyinit__(inout self, existing: Self, /):
        pass


trait Movable:
    fn __moveinit__(inout self, owned existing: Self, /):
        pass


trait AnyType:
    fn __del__(owned self, /):
        ...


# ===----------------------------------------------------------------------=== #
# Builtin Collection Stubs
# ===----------------------------------------------------------------------=== #


@register_passable
struct VariadicList[type: AnyRegType]:
    alias storage_type = __mlir_type[`!kgen.variadic<`, type, `>`]

    fn __init__(value: Self.storage_type) -> Self:
        return Self {}
