// RUN: kgen-opt %s --mogg-autoparameterize | FileCheck %s

kgen.generator @CREATE_NONE_SPEC(%arg0: !kgen.pointer<struct<(variadic<index>, variadic<index>) memoryOnly>> init_self, %arg1: !kgen.variadic<index> borrow, %arg2: !kgen.variadic<index> borrow) -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

kgen.generator  export @__static_tensor_spec_for_arg(%arg0: !kgen.string borrow, %arg1: !kgen.pointer<struct<(variadic<index>, variadic<index>) memoryOnly>> byref_result) -> !kgen.none no_inline attributes {mogg.tensor_spec_hook} {
  kgen.param.declare *"__result__`": struct<()> = <{  }>
  kgen.param.declare *"TENSOR_SPEC_NONE`1": struct<(variadic<index>, variadic<index>) memoryOnly> = <apply_result_slot(:(!kgen.pointer<struct<(variadic<index>, variadic<index>) memoryOnly>> init_self) -> !kgen.none @CREATE_NONE_SPEC)>
  %0 = kgen.param.materialize: struct<(variadic<index>, variadic<index>) memoryOnly> = <*"TENSOR_SPEC_NONE`1">
  pop.store %0, %arg1 : !kgen.pointer<struct<(variadic<index>, variadic<index>) memoryOnly>>
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

kgen.generator  @function_with_attr<type: dtype, rank>(%arg0: !kgen.pointer<struct<(pointer<none>, array<rank, index>, array<rank, index>) memoryOnly>> borrow_in_mem) -> !kgen.none attributes {mogg.arg_params = [[#kgen.param.decl.ref<"type"> : !kgen.dtype, #kgen.param.decl.ref<"rank"> : index]], mogg.arg_src_names = ["x"], mogg.arg_type_names = ["max::UnsafeTensorSlice"]} {
  kgen.param.declare *"spec`1": struct<(variadic<index>, variadic<index>) memoryOnly> = <apply_result_slot(:(!kgen.string borrow, !kgen.pointer<struct<(variadic<index>, variadic<index>) memoryOnly>> byref_result) -> !kgen.none @__static_tensor_spec_for_arg, "x")>
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}


kgen.generator  @function_calling_func_with_attr<type: dtype, rank>(%arg0: !kgen.pointer<struct<(pointer<none>, array<rank, index>, array<rank, index>) memoryOnly>> borrow_in_mem, %arg1: !kgen.pointer<struct<(pointer<none>, array<rank, index>, array<rank, index>) memoryOnly>> borrow_in_mem) -> !kgen.none attributes {mogg.arg_params = [[#kgen.param.decl.ref<"type"> : !kgen.dtype, #kgen.param.decl.ref<"rank"> : index]], mogg.arg_src_names = ["x", "y"], mogg.arg_type_names = ["max::UnsafeTensorSlice", "max::UnsafeTensorSlice"]} {
  %5 = kgen.call @function_with_attr<:dtype type, rank>(%arg1) : (!kgen.pointer<struct<(pointer<none>, array<rank, index>, array<rank, index>) memoryOnly>> borrow_in_mem) -> !kgen.none
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

kgen.generator  export @some_kernel<type: dtype, rank>(%arg0: !kgen.pointer<struct<(pointer<none>, array<rank, index>, array<rank, index>) memoryOnly>> inout, %arg1: !kgen.pointer<struct<(pointer<none>, array<rank, index>, array<rank, index>) memoryOnly>> borrow_in_mem, %arg2: !kgen.pointer<struct<(pointer<none>, array<rank, index>, array<rank, index>) memoryOnly>> borrow_in_mem) attributes {mogg.arg_params = [[#kgen.param.decl.ref<"type"> : !kgen.dtype, #kgen.param.decl.ref<"rank"> : index], [#kgen.param.decl.ref<"type"> : !kgen.dtype, #kgen.param.decl.ref<"rank"> : index], [#kgen.param.decl.ref<"type"> : !kgen.dtype, #kgen.param.decl.ref<"rank"> : index]], mogg.arg_src_names = ["out", "x", "y"], mogg.arg_type_names = ["max::UnsafeTensorSlice", "max::UnsafeTensorSlice", "max::UnsafeTensorSlice"], mogg.execute = "test_func" : !kgen.string} {
  %1 = kgen.call @function_with_attr<:dtype type, rank>(%arg1) : (!kgen.pointer<struct<(pointer<none>, array<rank, index>, array<rank, index>) memoryOnly>> borrow_in_mem) -> !kgen.none
  %2 = kgen.call @function_calling_func_with_attr<:dtype type, rank>(%arg1, %arg2) : (!kgen.pointer<struct<(pointer<none>, array<rank, index>, array<rank, index>) memoryOnly>> borrow_in_mem, !kgen.pointer<struct<(pointer<none>, array<rank, index>, array<rank, index>) memoryOnly>> borrow_in_mem) -> !kgen.none


  %3 = pop.stack_allocation 1 x struct<(pointer<none>, array<rank, index>, array<rank, index>) memoryOnly>

  // COM: Check we propagate null for tensors we don't know about.
  %4 = kgen.call @function_with_attr<:dtype type, rank>(%3) : (!kgen.pointer<struct<(pointer<none>, array<rank, index>, array<rank, index>) memoryOnly>> borrow_in_mem) -> !kgen.none

  kgen.return
}

// COM: Old kernel should have the mogg attrs stripped.
// CHECK: kgen.generator export @some_kernel

// CHECK: kgen.generator  @function_with_attr_{{[[0-9]]*}}<type: dtype, rank, __MOGG_SPEC0: struct<(variadic<index>, variadic<index>) memoryOnly>>
// CHECK-NEXT: kgen.param.declare *"spec`1": struct<(variadic<index>, variadic<index>) memoryOnly> = <__MOGG_SPEC0>

// CHECK: kgen.generator  @function_calling_func_with_attr_1<type: dtype, rank, __MOGG_SPEC1: struct<(variadic<index>, variadic<index>) memoryOnly>>
// CHECK-NEXT: kgen.call @function_with_attr_0<:dtype type, rank, :struct<(variadic<index>, variadic<index>) memoryOnly> __MOGG_SPEC1>

// CHECK: kgen.generator export @some_kernel_{{[[0-9]]*}}<type: dtype, rank, __MOGG_SPEC0: struct<({{.*}}) memoryOnly>, __MOGG_SPEC1: struct<(variadic<index>, variadic<index>) memoryOnly>, __MOGG_SPEC2: struct<(variadic<index>, variadic<index>) memoryOnly>>
// CHECK-NEXT: kgen.call @function_with_attr_{{[[0-9]]*}}<:dtype type, rank, :struct<(variadic<index>, variadic<index>) memoryOnly> __MOGG_SPEC1>
// CHECK-NEXT: kgen.call @function_calling_func_with_attr_{{[[0-9]]*}}<:dtype type, rank, :struct<(variadic<index>, variadic<index>) memoryOnly> __MOGG_SPEC2>
// CHECK-NEXT: stack_alloc
// CHECK-NEXT: function_with_attr_0<:dtype type, rank, :struct<(variadic<index>, variadic<index>) memoryOnly> apply_result_slot({{.*}} @CREATE_NONE_SPEC)
