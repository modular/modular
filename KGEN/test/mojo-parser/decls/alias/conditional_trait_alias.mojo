# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values | FileCheck %s


trait RequiresIntAlias:
    comptime Value: Int


struct ConditionallyConforms[value: Int](RequiresIntAlias) where value > 0:
    # Ensure the generator constraint has been discharged in the witness.
    comptime Value: Int where Self.value > 0 = Self.value
    # CHECK: kgen.conformance @"{{.*}}RequiresIntAlias"
    # CHECK-NEXT: kgen.witness "Value" : !Int = value


trait RequiresParametricIntAlias:
    comptime Value[offset: Int]: Int


struct ConditionallyConformsParametric[value: Int](
    RequiresParametricIntAlias
) where (value > 0):
    # Ensure the generator constraint has been discharged in the witness.
    comptime Value[offset: Int]: Int where Self.value > 0 = (
        Self.value + offset
    )
    # CHECK: kgen.conformance @"{{.*}}RequiresParametricIntAlias"
    # CHECK-NEXT: kgen.witness "Value" : !lit.generator<<"offset": !Int>!Int> = #kgen.gen<
    # CHECK-SAME: @"__add__(::Int,::Int)", value, *(0,0)
    # CHECK-SAME: add(from_builtin(#lit.struct.extract<:!Int value, "_mlir_value">), from_builtin(#lit.struct.extract<:!Int *(0,0), "_mlir_value">))
