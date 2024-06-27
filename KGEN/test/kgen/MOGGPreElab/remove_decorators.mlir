// RUN: kgen-opt %s --mogg-annotate | FileCheck %s

lit.func @kernel() capturing -> !kgen.none
  decorators <:none apply(:!lit.signature<("name": !lit.declref<@stdlib::@builtin::@string_literal::@StringLiteral> borrow, "priority": !lit.declref<@stdlib::@builtin::@int::@Int> borrow) -> !kgen.none> @register::@register::@"mogg_register_override(stdlib::builtin::string_literal::StringLiteral,stdlib::builtin::int::Int)", {:string "kernel_reg_test"}, {4242})> attributes {linkageName = "kernel", sourceName = "kernel", specialFnKind = 0 : i8} {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}
// CHECK: lit.func export @kernel() {{.*}} mogg.kernel = [#lit.struct<{value: string = "kernel_reg_test"}> : !StringLiteral, #lit.struct<{value = 4242}> : !Int]


lit.func @copy_construct() capturing -> !kgen.none no_inline
  decorators <:none apply(:!lit.signature<() -> !kgen.none> @register::@register::@"mogg_tensor_copy_constructor()")> attributes {sourceName = "__copyinit__", specialFnKind = 3 : i8} {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}
// CHECK: lit.func @copy_construct() {{.*}} attributes {mogg.tensor_copy_construct


lit.func export @elementwise_user() -> !kgen.none always_inline
decorators <
  :!lit.signature<() -> !kgen.none> @register::@register::@"mogg_elementwise()"
  > attributes {linkageName = "abs_wrapped", sourceName = "abs_wrapped", specialFnKind = 0 : i8} {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}
// CHECK: lit.func export @elementwise_user() {{.*}}attributes{{.*}}mogg.elementwise

lit.func @extensibility_kernel[imm *"x`", mut *"__result__`1"]<type: @stdlib::@builtin::@dtype::@DType, rank: @stdlib::@builtin::@int::@Int>(%x: !lit.ref<@extensibility::@tensor::@Tensor<:@stdlib::@builtin::@dtype::@DType type, :@stdlib::@builtin::@int::@Int rank>, imm *"x`"> borrow_in_mem, ?, %__result__: !lit.ref<@extensibility::@tensor::@Tensor<:@stdlib::@builtin::@dtype::@DType type, :@stdlib::@builtin::@int::@Int rank>, mut *"__result__`1"> byref_result) -> !kgen.none
decorators <:none apply(:!lit.signature<("name": !lit.declref<@stdlib::@builtin::@string_literal::@StringLiteral> borrow) -> !kgen.none> @register::@register::@"mogg_register(stdlib::builtin::string_literal::StringLiteral)", {:string "my_kernel"})> attributes {sourceName = "foo", specialFnKind = 0 : i8} {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}
// CHECK: lit.func export @extensibility_kernel{{.*}} attributes {mogg.allocs = [1 : index],{{.*}} mogg.kernel = [#lit.struct<{value: string = "my_kernel"}> : !StringLiteral, -1]


lit.func @argument_names_and_params[imm *"x`", mut *"__result__`1"]<type: @stdlib::@builtin::@dtype::@DType, rank: @stdlib::@builtin::@int::@Int>(%x: !lit.ref<@MOGGTensor::@Tensor<:@stdlib::@builtin::@dtype::@DType type, :@stdlib::@builtin::@int::@Int rank>, imm *"x`"> borrow_in_mem, ?, %__result__: !lit.ref<@MOGGTensor::@Tensor<:@stdlib::@builtin::@dtype::@DType type, :@stdlib::@builtin::@int::@Int rank>, mut *"__result__`1"> byref_result) -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}
// CHECK: lit.func @argument_names_and_params{{.*}} attributes {mogg.arg_params = {{\[\[}}#kgen.param.decl.ref<"type"> : !DType, #kgen.param.decl.ref<"rank"> : !Int], [#kgen.param.decl.ref<"type"> : !DType, #kgen.param.decl.ref<"rank"> : !Int{{\]\]}}, mogg.arg_src_names = ["x", "__result__"], mogg.arg_type_names = ["MOGGTensor::Tensor", "MOGGTensor::Tensor"]}
