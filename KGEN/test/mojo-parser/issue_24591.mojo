# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | kgen-opt -mlir-print-op-generic | FileCheck %s

comptime NoneType = __mlir_type.`!kgen.none`


struct Optional[T: __mlir_type.`!kgen.type`](RegisterPassable):
    @implicit
    def __init__(out self, none: NoneType):
        pass


struct Param[x: Int](RegisterPassable):
    pass


# Check the TypeSignatureType attribute. This is the only memory-only
# struct so we can match with 0.
# CHECK: "lit.struct.decl"() {{.*}} convention = 0 :
# CHECK-SAME: signature = !lit.type_signature<"x": !Int, "y": !lit.struct<#Optional{{.*}}!lit.generator<<"y": !Int>() -> !lit.struct<#Param <:!Int *(1,0)>>
struct Thing[x: Int, y: Optional[def[y: Int]() thin -> Param[x]] = None]:
    comptime z = 1
