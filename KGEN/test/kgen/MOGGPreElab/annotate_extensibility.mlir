// Tests we properly annotate kernels under the new extensibility api.
// These consist of a struct with a special `register` decorator and methods
// such as "execute" (running the op), and "shape" (runtime shape function).

// RUN: kgen-opt %s --mogg-annotate-kernels | FileCheck %s

// Hard coded registration function, has special `mogg.intrinsic_register`
lit.fn @"register(::StringLiteral)"(%name: !lit.struct<@stdlib::@builtin::@string_literal::@StringLiteral>, %num_dps_outputs: !lit.struct<@stdlib::@builtin::@int::@Int> = {1}) -> !kgen.none attributes {mogg.intrinsic_register, sourceName = "register", specialFnKind = 0 : i8} {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// A basic case with an "execute" and "shape" function that take only tensors
// CHECK-LABEL: lit.struct.decl @test_execute_and_shape
lit.struct.decl @test_execute_and_shape(trait<@stdlib::@builtin::@anytype::@AnyType>)
  decorators <:none apply(:!lit.generator<("name": !lit.struct<@stdlib::@builtin::@string_literal::@StringLiteral>, "num_dps_outputs": !lit.struct<@stdlib::@builtin::@int::@Int> = {1}) -> !kgen.none> @"register(::StringLiteral)", {:string "imposter_add"}, {1})> {

  // CHECK: lit.fn export @execute
  // CHECK: mogg.arg_params = [unit, [#kgen.param.decl.ref<"dtype"> : !lit.struct<@DType>], unit]
  // CHECK-SAME: mogg.arg_src_names = ["z", "x", "y"]
  // CHECK-SAME: mogg.arg_type_names = ["test1::test1", "test2::test2", "test3::test3"]
  // CHECK-SAME: mogg.execute = "imposter_add"
  lit.fn export @"execute"(%z: !lit.struct<@test1>, %x: !lit.struct<@test2<:@DType *"dtype">>, %y: !lit.struct<@test3>) -> !kgen.none
    attributes {
        isStatic,
        sourceName = "execute",
        specialFnKind = 0 : i8} {
    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
  }

  // CHECK: lit.fn export @shape
  // CHECK: mogg.arg_params = [{{\[}}#kgen.param.decl.ref<"dtype"> : !lit.struct<@DType>], unit]
  // CHECK-SAME: mogg.arg_src_names = ["x", "y"]
  // CHECK-SAME: mogg.arg_type_names = ["test2::test2", "test3::test3"]
  // CHECK-SAME: mogg.shape = "imposter_add"
  lit.fn export @"shape"(%x: !lit.struct<@test2<:@DType *"dtype">>, %y: !lit.struct<@test3>) -> !kgen.none
    attributes {
        isStatic,
        sourceName = "shape", specialFnKind = 0 : i8} {
    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
  }
}
