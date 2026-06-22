# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s

# Test @align decorator with parametric alignment values.


# Test basic parametric alignment using struct parameter
# CHECK-LABEL: lit.struct.decl @AlignedBuffer
# CHECK-SAME: minAlignment = #lit.struct.extract<:!Int alignment, "_mlir_value"> : index
@align(alignment)
struct AlignedBuffer[alignment: Int]:
    var data: Int


# Test parametric alignment with TrivialRegisterPassable
# CHECK-LABEL: lit.struct.decl @AlignedTrivialParam
# CHECK-SAME: register_passable_trivial
# CHECK-SAME: minAlignment = #lit.struct.extract<:!Int n, "_mlir_value"> : index
@align(n)
struct AlignedTrivialParam[n: Int](TrivialRegisterPassable):
    var value: __mlir_type.index


# Test parametric alignment combined with multiple parameters
# CHECK-LABEL: lit.struct.decl @MultiParam
# CHECK-SAME: minAlignment = #lit.struct.extract<:!Int align_val, "_mlir_value"> : index
@align(align_val)
struct MultiParam[T: __mlir_type.`!kgen.type`, align_val: Int]:
    var value: Self.T

    def __del__(deinit self):
        pass


# Test parametric alignment with an expression (n * 2)
# CHECK-LABEL: lit.struct.decl @AlignedExpr
# CHECK-SAME: minAlignment = #kgen.cast_to_builtin<#kgen.param.expr<mul, #kgen.cast_from_builtin<#lit.struct.extract<:!Int n, "_mlir_value"> : index> : !kgen.scalar<index>, #kgen<simd 2> : !kgen.scalar<index>> : !kgen.scalar<index>> : index
@align(n * 2)
struct AlignedExpr[n: Int]:
    var data: Int
