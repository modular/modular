# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | kgen-opt -mlir-print-op-generic | FileCheck %s

alias NoneType = __mlir_type.`!kgen.none`

@register_passable
struct Optional[T: AnyTrivialRegType]:
    @implicit
    fn __init__(out self, none: NoneType):
        pass


@register_passable
struct Param[x: Int]:
    pass

# Check the TypeSignatureType attribute. This is the only memory-only
# struct so we can match with 0.
# CHECK: "lit.struct.decl"() {{.*}} convention = 0 :
# CHECK-SAME: signature = !lit.type_signature<"x": !Int, "y": !lit.struct<#Optional{{.*}}!lit.generator<<"y": !Int>() -> !lit.struct<#Param <:!Int *(1,0)>>
struct Thing[x: Int, y: Optional[fn[y: Int] () -> Param[x]] = None]:
    alias z = 1
