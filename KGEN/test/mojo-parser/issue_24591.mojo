# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate %s -import-mojo --mojo-disable-builtins | kgen-opt -verify-parameters -mlir-print-op-generic | FileCheck %s

alias AnyType = __mlir_type.`!kgen.mlirtype`
alias NoneType = __mlir_type.`!kgen.none`


@register_passable
struct Optional[T: AnyType]:
    fn __init__(none: NoneType) -> Self:
        pass


alias int = __mlir_type.index


@register_passable
struct Param[x: int]:
    pass


@register_passable("trivial")
struct IntLiteral:
    fn __init__(y: __mlir_type.`!kgen.int_literal`) -> Self:
        pass


# COM: Check the TypeSignatureType attribute. This is the only memory-only
# COM: struct so we can match with 0.
# CHECK: "lit.struct.decl"() <{convention = 0 :
# CHECK-SAME: signature = !lit.type_signature<"x": index, "y": [[OPT:.*:@Optional]]<:type !lit.signature<<"y": index>() ownedresult -> !kgen.declref<[[P:.*@Param]]<*(1,0)>,
struct Thing[x: int, y: Optional[fn[y: int] () -> Param[x]] = None]:
    alias z = 1
