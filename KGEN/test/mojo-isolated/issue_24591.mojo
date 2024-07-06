# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | kgen-opt -verify-parameters -mlir-print-op-generic | FileCheck %s

alias AnyTrivialRegType = __mlir_type.`!kgen.type`
alias NoneType = __mlir_type.`!kgen.none`


@register_passable
struct Optional[T: AnyTrivialRegType]:
    fn __init__(inout self, none: NoneType):
        pass


alias int = __mlir_type.index


@register_passable
struct Param[x: int]:
    pass


@register_passable("trivial")
struct IntLiteral:
    fn __init__(inout self, y: __mlir_type.`!kgen.int_literal`):
        pass


# COM: Check the TypeSignatureType attribute. This is the only memory-only
# COM: struct so we can match with 0.
# CHECK: "lit.struct.decl"() <{convention = 0 :
# CHECK-SAME: signature = !lit.type_signature<"x": index, "y": [[OPT:.*:@Optional]]<:type !lit.signature<<"y": index>() -> !lit.struct<#Param <*(1,0)>>
struct Thing[x: int, y: Optional[fn[y: int] () -> Param[x]] = None]:
    alias z = 1
