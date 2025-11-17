# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | kgen-opt -mlir-print-op-generic | FileCheck %s

comptime AnyTrivialRegType = __mlir_type.`!kgen.type`
comptime NoneType = __mlir_type.`!kgen.none`


@register_passable
struct Optional[T: AnyTrivialRegType]:
    @implicit
    fn __init__(out self, none: NoneType):
        pass


comptime Index = __mlir_type.index


@register_passable
struct Param[x: Index]:
    pass

# Check the TypeSignatureType attribute. This is the only memory-only
# struct so we can match with 0.
# CHECK: "lit.struct.decl"() {{.*}} convention = 0 :
# CHECK-SAME: signature = !lit.type_signature<"x": index, "y": [[OPT:.*:@Optional]]<:type !lit.generator<<"y": index>() -> !lit.struct<#Param <*(1,0)>>
struct Thing[x: Index, y: Optional[fn[y: Index] () -> Param[x]] = None]:
    comptime z = 1
