// RUN: kgen-opt %s --mogg-autoparameterize | FileCheck %s

#type_value = #kgen.type<struct<() memoryOnly>, {"_get_dtype" : () -> !kgen.dtype = @"tensor_utils::unsafe_tensor_slice::UnsafeTensorSlice::_get_dtype()"<:dtype type, rank>, "_get_static_rank" : () -> index = @"tensor_utils::unsafe_tensor_slice::UnsafeTensorSlice::_get_static_rank()"<:dtype type, rank>, "__del__" : (!kgen.pointer<struct<() memoryOnly>> owned_in_mem) -> !kgen.none = @"tensor_utils::unsafe_tensor_slice::UnsafeTensorSlice::__del__(tensor_utils::unsafe_tensor_slice::UnsafeTensorSlice[$0, $1])"<:dtype type, rank>}> : !kgen.type
module {

kgen.generator @CREATE_NONE_SPEC<type: dtype, rank>(%arg0: !kgen.pointer<struct<(variadic<index>, variadic<index>, variant<<index>(!pop.array<rank, index>) capturing -> !pop.simd<*(0,0), type>, i1>, variant<<index>(!pop.array<rank, index>, !pop.simd<*(0,0), type>) capturing -> !kgen.none, i1>) memoryOnly>> byref_result) -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

kgen.generator export @specsof<T: type>(%arg0: !kgen.string, %arg1: !kgen.pointer<struct<(variadic<index>, variadic<index>, variant<<index>(!pop.array<apply(:() -> index get_type_method(T, "_get_static_rank")), index>) capturing -> !pop.simd<*(0,0), apply(:() -> !kgen.dtype get_type_method(T, "_get_dtype"))>, i1>, variant<<index>(!pop.array<apply(:() -> index get_type_method(T, "_get_static_rank")), index>, !pop.simd<*(0,0), apply(:() -> !kgen.dtype get_type_method(T, "_get_dtype"))>) capturing -> !kgen.none, i1>) memoryOnly>> byref_result) -> !kgen.none attributes {mogg.intrinsic_tensor_spec_hook} {
  kgen.param.declare *"TENSOR_SPEC_NONE`1": struct<(variadic<index>, variadic<index>, variant<<index>(!pop.array<apply(:() -> index get_type_method(T, "_get_static_rank")), index>) capturing -> !pop.simd<*(0,0), apply(:() -> !kgen.dtype get_type_method(T, "_get_dtype"))>, i1>, variant<<index>(!pop.array<apply(:() -> index get_type_method(T, "_get_static_rank")), index>, !pop.simd<*(0,0), apply(:() -> !kgen.dtype get_type_method(T, "_get_dtype"))>) capturing -> !kgen.none, i1>) memoryOnly> = <apply_result_slot(:(!kgen.pointer<struct<(variadic<index>, variadic<index>, variant<<index>(!pop.array<apply(:() -> index get_type_method(T, "_get_static_rank")), index>) capturing -> !pop.simd<*(0,0), apply(:() -> !kgen.dtype get_type_method(T, "_get_dtype"))>, i1>, variant<<index>(!pop.array<apply(:() -> index get_type_method(T, "_get_static_rank")), index>, !pop.simd<*(0,0), apply(:() -> !kgen.dtype get_type_method(T, "_get_dtype"))>) capturing -> !kgen.none, i1>) memoryOnly>> byref_result) -> !kgen.none @CREATE_NONE_SPEC<:dtype apply(:() -> !kgen.dtype get_type_method(T, "_get_dtype")), apply(:() -> index get_type_method(T, "_get_static_rank"))>)>
  %0 = kgen.param.materialize: struct<(variadic<index>, variadic<index>, variant<<index>(!pop.array<apply(:() -> index get_type_method(T, "_get_static_rank")), index>) capturing -> !pop.simd<*(0,0), apply(:() -> !kgen.dtype get_type_method(T, "_get_dtype"))>, i1>, variant<<index>(!pop.array<apply(:() -> index get_type_method(T, "_get_static_rank")), index>, !pop.simd<*(0,0), apply(:() -> !kgen.dtype get_type_method(T, "_get_dtype"))>) capturing -> !kgen.none, i1>) memoryOnly> = <*"TENSOR_SPEC_NONE`1">
  pop.store %0, %arg1 : !kgen.pointer<struct<(variadic<index>, variadic<index>, variant<<index>(!pop.array<apply(:() -> index get_type_method(T, "_get_static_rank")), index>) capturing -> !pop.simd<*(0,0), apply(:() -> !kgen.dtype get_type_method(T, "_get_dtype"))>, i1>, variant<<index>(!pop.array<apply(:() -> index get_type_method(T, "_get_static_rank")), index>, !pop.simd<*(0,0), apply(:() -> !kgen.dtype get_type_method(T, "_get_dtype"))>) capturing -> !kgen.none, i1>) memoryOnly>>
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

kgen.generator @function_with_attr<type: dtype, rank>(%arg0: !kgen.pointer<struct<() memoryOnly>> borrow_in_mem) -> !kgen.none attributes {mogg.arg_params = [[#kgen.param.decl.ref<"type"> : !kgen.dtype, #kgen.param.decl.ref<"rank"> : index]], mogg.arg_src_names = ["a"], mogg.arg_type_names = ["tensor_utils::UnsafeTensorSlice"]} {
  kgen.param.declare *"spec`1": struct<(variadic<index>, variadic<index>, variant<<index>(!pop.array<apply(:() -> index @"tensor_utils::unsafe_tensor_slice::UnsafeTensorSlice::_get_static_rank()"<:dtype type, rank>), index>) capturing -> !pop.simd<*(0,0), apply(:() -> !kgen.dtype @"tensor_utils::unsafe_tensor_slice::UnsafeTensorSlice::_get_dtype()"<:dtype type, rank>)>, i1>, variant<<index>(!pop.array<apply(:() -> index @"tensor_utils::unsafe_tensor_slice::UnsafeTensorSlice::_get_static_rank()"<:dtype type, rank>), index>, !pop.simd<*(0,0), apply(:() -> !kgen.dtype @"tensor_utils::unsafe_tensor_slice::UnsafeTensorSlice::_get_dtype()"<:dtype type, rank>)>) capturing -> !kgen.none, i1>) memoryOnly> = <apply_result_slot(:(!kgen.string, !kgen.pointer<struct<(variadic<index>, variadic<index>, variant<<index>(!pop.array<apply(:() -> index @"tensor_utils::unsafe_tensor_slice::UnsafeTensorSlice::_get_static_rank()"<:dtype type, rank>), index>) capturing -> !pop.simd<*(0,0), apply(:() -> !kgen.dtype @"tensor_utils::unsafe_tensor_slice::UnsafeTensorSlice::_get_dtype()"<:dtype type, rank>)>, i1>, variant<<index>(!pop.array<apply(:() -> index @"tensor_utils::unsafe_tensor_slice::UnsafeTensorSlice::_get_static_rank()"<:dtype type, rank>), index>, !pop.simd<*(0,0), apply(:() -> !kgen.dtype @"tensor_utils::unsafe_tensor_slice::UnsafeTensorSlice::_get_dtype()"<:dtype type, rank>)>) capturing -> !kgen.none, i1>) memoryOnly>> byref_result) -> !kgen.none @specsof<:type #type_value>, "a")>
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

kgen.generator @function_calling_func_with_attr<type: dtype, rank>(%arg0: !kgen.pointer<struct<() memoryOnly>> borrow_in_mem, %arg1: !kgen.pointer<struct<() memoryOnly>> borrow_in_mem) -> !kgen.none attributes {mogg.arg_params = [[#kgen.param.decl.ref<"type"> : !kgen.dtype, #kgen.param.decl.ref<"rank"> : index], [#kgen.param.decl.ref<"type"> : !kgen.dtype, #kgen.param.decl.ref<"rank"> : index]], mogg.arg_src_names = ["a", "b"], mogg.arg_type_names = ["tensor_utils::UnsafeTensorSlice", "tensor_utils::UnsafeTensorSlice"]} {
  %0 = kgen.call @function_with_attr<:dtype type, rank>(%arg1) : (!kgen.pointer<struct<() memoryOnly>> borrow_in_mem) -> !kgen.none
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

kgen.generator export @some_kernel<spec: struct<() memoryOnly>, type: dtype, rank>(%arg0: !kgen.pointer<struct<() memoryOnly>> inout, %arg1: !kgen.pointer<struct<() memoryOnly>> borrow_in_mem, %arg2: !kgen.pointer<struct<() memoryOnly>> borrow_in_mem) attributes {mogg.arg_params = [[#kgen.param.decl.ref<"type"> : !kgen.dtype, #kgen.param.decl.ref<"rank"> : index], [#kgen.param.decl.ref<"type"> : !kgen.dtype, #kgen.param.decl.ref<"rank"> : index], [#kgen.param.decl.ref<"type"> : !kgen.dtype, #kgen.param.decl.ref<"rank"> : index]], mogg.arg_src_names = ["c", "a", "b"], mogg.arg_type_names = ["tensor_utils::UnsafeTensorSlice", "tensor_utils::UnsafeTensorSlice", "tensor_utils::UnsafeTensorSlice"], mogg.execute = "some_kernel" : !kgen.string, mogg.spec = "spec"} {
  %0 = kgen.call @function_with_attr<:dtype type, rank>(%arg1) : (!kgen.pointer<struct<() memoryOnly>> borrow_in_mem) -> !kgen.none
  %1 = kgen.call @function_calling_func_with_attr<:dtype type, rank>(%arg1, %arg2) : (!kgen.pointer<struct<() memoryOnly>> borrow_in_mem, !kgen.pointer<struct<() memoryOnly>> borrow_in_mem) -> !kgen.none

  kgen.return
}

}

// COM: Old kernel should have the mogg attrs stripped.
// CHECK: kgen.generator export @some_kernel

// CHECK: kgen.generator  @function_with_attr_{{[[0-9]]*}}<type: dtype, rank, __MOGG_SPEC0: struct<({{.*}}) memoryOnly>>
// CHECK-NEXT: kgen.param.declare *"spec`1": struct<({{.*}}) memoryOnly> = <__MOGG_SPEC0>

// CHECK: kgen.generator  @function_calling_func_with_attr_{{[[0-9]]*}}<type: dtype, rank, __MOGG_SPEC1: struct<({{.*}}) memoryOnly>>
// CHECK-NEXT: kgen.call @function_with_attr_{{[[0-9]]*}}<:dtype type, rank, :struct<({{.*}}) memoryOnly> __MOGG_SPEC1>

// CHECK: kgen.generator export @some_kernel_{{[[0-9]]*}}<spec: struct<() memoryOnly>, type: dtype, rank, __MOGG_SPEC1: struct<({{.*}}) memoryOnly>, __MOGG_SPEC2: struct<({{.*}}) memoryOnly>>
// CHECK-NEXT: kgen.call @function_with_attr_{{[[0-9]]*}}<:dtype type, rank, :struct<({{.*}}) memoryOnly> __MOGG_SPEC1>
// CHECK-NEXT: kgen.call @function_calling_func_with_attr_{{[[0-9]]*}}<:dtype type, rank, :struct<({{.*}}) memoryOnly> __MOGG_SPEC2>
