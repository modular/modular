// RUN: kgen-opt %s --mogg-autospecialize | FileCheck %s

// Note, the structs (e.g. `ManagedTensorSlice`) are refer to stubbed classes to
// be used for testing.
#type_value = #kgen.type<struct<() memoryOnly>, {"_get_dtype" : () -> !kgen.dtype = @"tensor_utils::managed_tensor_slice::ManagedTensorSlice::_get_dtype()"<:dtype type, rank>, "_get_static_rank" : () -> index = @"tensor_utils::managed_tensor_slice::ManagedTensorSlice::_get_static_rank()"<:dtype type, rank>, "__del__" : (!kgen.pointer<struct<() memoryOnly>> owned_in_mem) -> !kgen.none = @"tensor_utils::managed_tensor_slice::ManagedTensorSlice::__del__(tensor_utils::managed_tensor_slice::ManagedTensorSlice[$0, $1])"<:dtype type, rank>}> : !kgen.type
module {

// Stub we look for to know the type signature of the tensor spec.
kgen.generator @CREATE_NONE_SPEC<type: dtype, rank>() -> !kgen.struct<(variadic<index>, variadic<index>, variant<<index>(!pop.array<rank, index>) capturing -> !pop.simd<*(0,0), type>, i1>, variant<<index>(!pop.array<rank, index>, !pop.simd<*(0,0), type>) capturing -> !kgen.none, i1>)> {
  kgen.unreachable
}

// Stub function we look for that fetches the static tensor spec. Calls to this
// will be replaced with the correct parameterization.
//
// e.g.
//
// fn foo[type: DType, rank: Int](input: ManagedTensorSlice[type, rank]):
//    alias static_shape = specsof[type, rank]("input").shape
//    ...
//
// Will become something analogous to:
//
// fn foo_1[static_spec: StaticTensorSpec, type: DType, rank: Int](input: ManagedTensorSlice[type, rank]):
//    # Refers to the construct populated by "input"
//    alias static_shape = static_spec.shape
//    ...
kgen.generator export @specsof<T: type>(%arg0: !kgen.string) -> !kgen.struct<(variadic<index>, variadic<index>, variant<<index>(!pop.array<apply(:() -> index get_vtable_entry(T, "_get_static_rank")), index>) capturing -> !pop.simd<*(0,0), apply(:() -> !kgen.dtype get_vtable_entry(T, "_get_dtype"))>, i1>, variant<<index>(!pop.array<apply(:() -> index get_vtable_entry(T, "_get_static_rank")), index>, !pop.simd<*(0,0), apply(:() -> !kgen.dtype get_vtable_entry(T, "_get_dtype"))>) capturing -> !kgen.none, i1>)> attributes {mogg.intrinsic_tensor_spec_hook} {
  kgen.param.declare *"TENSOR_SPEC_NONE`1": struct<(variadic<index>, variadic<index>, variant<<index>(!pop.array<apply(:() -> index get_vtable_entry(T, "_get_static_rank")), index>) capturing -> !pop.simd<*(0,0), apply(:() -> !kgen.dtype get_vtable_entry(T, "_get_dtype"))>, i1>, variant<<index>(!pop.array<apply(:() -> index get_vtable_entry(T, "_get_static_rank")), index>, !pop.simd<*(0,0), apply(:() -> !kgen.dtype get_vtable_entry(T, "_get_dtype"))>) capturing -> !kgen.none, i1>)> = <apply(:() -> !kgen.struct<(variadic<index>, variadic<index>, variant<<index>(!pop.array<apply(:() -> index get_vtable_entry(T, "_get_static_rank")), index>) capturing -> !pop.simd<*(0,0), apply(:() -> !kgen.dtype get_vtable_entry(T, "_get_dtype"))>, i1>, variant<<index>(!pop.array<apply(:() -> index get_vtable_entry(T, "_get_static_rank")), index>, !pop.simd<*(0,0), apply(:() -> !kgen.dtype get_vtable_entry(T, "_get_dtype"))>) capturing -> !kgen.none, i1>)> @CREATE_NONE_SPEC<:dtype apply(:() -> !kgen.dtype get_vtable_entry(T, "_get_dtype")), apply(:() -> index get_vtable_entry(T, "_get_static_rank"))>)>
  kgen.unreachable
}


// Analogous to something like:
//
// fn function_with_attr[type: DType, rank: Int](a: ManagedTensorSlice[type, rank]):
//    alias spec1 = specsof[type, rank]('a')
kgen.generator @function_with_attr<type: dtype, rank>(%arg0: !kgen.pointer<struct<() memoryOnly>> read_mem) -> !kgen.none attributes {mogg.arg_params = [[#kgen.param.decl.ref<"type"> : !kgen.dtype, #kgen.param.decl.ref<"rank"> : index]], mogg.arg_src_names = ["a"], mogg.arg_type_names = ["tensor_utils::ManagedTensorSlice"]} {
  kgen.param.declare *"spec`1": struct<(variadic<index>, variadic<index>, variant<<index>(!pop.array<apply(:() -> index @"tensor_utils::managed_tensor_slice::ManagedTensorSlice::_get_static_rank()"<:dtype type, rank>), index>) capturing -> !pop.simd<*(0,0), apply(:() -> !kgen.dtype @"tensor_utils::managed_tensor_slice::ManagedTensorSlice::_get_dtype()"<:dtype type, rank>)>, i1>, variant<<index>(!pop.array<apply(:() -> index @"tensor_utils::managed_tensor_slice::ManagedTensorSlice::_get_static_rank()"<:dtype type, rank>), index>, !pop.simd<*(0,0), apply(:() -> !kgen.dtype @"tensor_utils::managed_tensor_slice::ManagedTensorSlice::_get_dtype()"<:dtype type, rank>)>) capturing -> !kgen.none, i1>)> = <apply(:(!kgen.string) -> !kgen.struct<(variadic<index>, variadic<index>, variant<<index>(!pop.array<apply(:() -> index @"tensor_utils::managed_tensor_slice::ManagedTensorSlice::_get_static_rank()"<:dtype type, rank>), index>) capturing -> !pop.simd<*(0,0), apply(:() -> !kgen.dtype @"tensor_utils::managed_tensor_slice::ManagedTensorSlice::_get_dtype()"<:dtype type, rank>)>, i1>, variant<<index>(!pop.array<apply(:() -> index @"tensor_utils::managed_tensor_slice::ManagedTensorSlice::_get_static_rank()"<:dtype type, rank>), index>, !pop.simd<*(0,0), apply(:() -> !kgen.dtype @"tensor_utils::managed_tensor_slice::ManagedTensorSlice::_get_dtype()"<:dtype type, rank>)>) capturing -> !kgen.none, i1>)> @specsof<:type #type_value>, "a")>
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// Analogous to something like:
// fn function_with_attr[type: DType, rank: Int](a: ManagedTensorSlice[type, rank], b: ManagedTensorSlice[type, rank]):
//    function_with_attr(b)
kgen.generator @function_calling_func_with_attr<type: dtype, rank>(%arg0: !kgen.pointer<struct<() memoryOnly>> read_mem, %arg1: !kgen.pointer<struct<() memoryOnly>> read_mem) -> !kgen.none attributes {mogg.arg_params = [[#kgen.param.decl.ref<"type"> : !kgen.dtype, #kgen.param.decl.ref<"rank"> : index], [#kgen.param.decl.ref<"type"> : !kgen.dtype, #kgen.param.decl.ref<"rank"> : index]], mogg.arg_src_names = ["a", "b"], mogg.arg_type_names = ["tensor_utils::ManagedTensorSlice", "tensor_utils::ManagedTensorSlice"]} {
  %0 = kgen.call @function_with_attr<:dtype type, rank>(%arg1) : (!kgen.pointer<struct<() memoryOnly>> read_mem) -> !kgen.none
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// Analogous to something like:
// fn function_with_attr[type: DType, rank: Int](c: ManagedTensorSlice[type, rank], a: ManagedTensorSlice[type, rank], b: ManagedTensorSlice[type, rank]):
//    function_with_attr(b)
//    funciton_with_attr(a, b)
kgen.generator export @some_kernel<spec: struct<() memoryOnly>, type: dtype, rank>(%arg0: !kgen.pointer<struct<() memoryOnly>> mut, %arg1: !kgen.pointer<struct<() memoryOnly>> read_mem, %arg2: !kgen.pointer<struct<() memoryOnly>> read_mem) attributes {mogg.arg_params = [[#kgen.param.decl.ref<"type"> : !kgen.dtype, #kgen.param.decl.ref<"rank"> : index], [#kgen.param.decl.ref<"type"> : !kgen.dtype, #kgen.param.decl.ref<"rank"> : index], [#kgen.param.decl.ref<"type"> : !kgen.dtype, #kgen.param.decl.ref<"rank"> : index]], mogg.arg_src_names = ["c", "a", "b"], mogg.arg_type_names = ["tensor_utils::ManagedTensorSlice", "tensor_utils::ManagedTensorSlice", "tensor_utils::ManagedTensorSlice"], mogg.execute = "some_kernel" : !kgen.string, mogg.spec = "spec"} {
  %0 = kgen.call @function_with_attr<:dtype type, rank>(%arg1) : (!kgen.pointer<struct<() memoryOnly>> read_mem) -> !kgen.none
  %1 = kgen.call @function_calling_func_with_attr<:dtype type, rank>(%arg1, %arg2) : (!kgen.pointer<struct<() memoryOnly>> read_mem, !kgen.pointer<struct<() memoryOnly>> read_mem) -> !kgen.none

  kgen.return
}

}

// COM: Old kernel should have the mogg attrs stripped.
// CHECK: kgen.generator export @some_kernel

// CHECK: kgen.generator  @function_with_attr_{{[[0-9]]*}}<type: dtype, rank, __MOGG_SPEC0: struct<({{.*}}) memoryOnly>>
// CHECK-NEXT: kgen.param.declare *"spec`1": struct<({{.*}})> = <__MOGG_SPEC0>

// CHECK: kgen.generator  @function_calling_func_with_attr_{{[[0-9]]*}}<type: dtype, rank, __MOGG_SPEC1: struct<({{.*}}) memoryOnly>>
// CHECK-NEXT: kgen.call @function_with_attr_{{[[0-9]]*}}<:dtype type, rank, :struct<({{.*}})> __MOGG_SPEC1>

// CHECK: kgen.generator export @some_kernel_{{[[0-9]]*}}<spec: struct<() memoryOnly>, type: dtype, rank
// COM: Only arguments 1 and 2 have the tensor spec parameter
// CHECK-SAME: mogg.tensor_spec_params = [unit, {{.*}}__MOGG_SPEC1{{.*}}__MOGG_SPEC2{{.*}}
// CHECK-NEXT: kgen.call @function_with_attr_{{[[0-9]]*}}<:dtype type, rank, :struct<({{.*}})> __MOGG_SPEC1>
// CHECK-NEXT: kgen.call @function_calling_func_with_attr_{{[[0-9]]*}}<:dtype type, rank, :struct<({{.*}})> __MOGG_SPEC2>
