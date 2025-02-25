// Tests we properly annotate kernels with scalar arguments under the new
// extensibility api.

// RUN: kgen-opt %s --mogg-annotate-kernels | FileCheck %s

!Int = !lit.struct<@stdlib::@builtin::@int::@Int>
!DType = !lit.struct<@stdlib::@builtin::@dtype::@DType>
#SIMD = #lit<symbol@stdlib::@builtin::@simd::@SIMD>
!MyScalar = !lit.struct<#SIMD <:!DType {:dtype si8}, :!Int {1}>>
!MyScalarWithUnbound = !lit.struct<#SIMD <:!DType *"type`2x2", :!Int {1}>>

// Hard coded registration function, has special `mogg.intrinsic_register`
lit.fn @"register(::StringLiteral)"(%name: !lit.struct<@stdlib::@builtin::@string_literal::@StringLiteral>, %num_dps_outputs: !lit.struct<@stdlib::@builtin::@int::@Int> = {1}) -> !kgen.none attributes {mogg.intrinsic_register, sourceName = "register", specialFnKind = 0 : i8} {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// A basic case with an "execute" and function that takes scalar arguments
// CHECK-LABEL: lit.struct.decl @just_execute_with_scalar
lit.struct.decl @just_execute_with_scalar(trait<@stdlib::@builtin::@anytype::@AnyType>)
  decorators <:none apply(:!lit.generator<("name": !lit.struct<@stdlib::@builtin::@string_literal::@StringLiteral>, "num_dps_outputs": !lit.struct<@stdlib::@builtin::@int::@Int> = {1}) -> !kgen.none> @"register(::StringLiteral)", {:string "imposter_add"}, {1})> {

  // CHECK: lit.fn export @execute
  // CHECK-SAME: mogg.arg_type_names = ["test1::test1", "stdlib::SIMD", "stdlib::SIMD"]
  // CHECK-SAME: mogg.value_params = [unit, {size = #lit.struct<{value = 1}> : !Int, type = #lit.struct<{value: dtype = si8}> : !DType}, {size = #lit.struct<{value = 1}> : !Int, type = #kgen.param.decl.ref<"type`2x2"> : !DType}]
  lit.fn export @"execute"(%z: !lit.struct<@test1>, %y : !MyScalar, %x : !MyScalarWithUnbound) -> !kgen.none
    attributes {
        isStatic,
        sourceName = "execute",
        specialFnKind = 0 : i8} {
    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
  }
}
