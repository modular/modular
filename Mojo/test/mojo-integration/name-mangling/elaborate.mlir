// RUN: kgen-opt %s -split-input-file -verify-parameters -elaborate-generators="use-parametric-interpret=false" \
// RUN:   -allow-unregistered-dialect | FileCheck %s
// RUN: kgen-opt %s -split-input-file -elaborate-generators="use-parametric-interpret=true" \
// RUN:   -allow-unregistered-dialect | FileCheck %s

kgen.generator @some_generator() {
  hlcf.return
} <sourceName = "foo">

// CHECK-LABEL: kgen.func export @get_source_name
kgen.generator export @get_source_name() {
  // CHECK-NEXT: constant: string = <"foo">
  %0 = kgen.param.constant: string = <#kgen.get_source_name<#kgen.gen<#kgen.func.symbol<@some_generator>> : !kgen.generator<() -> ()>>>
  hlcf.return
}

// -----

// A function value coming back from the comptime interpreter is the concrete
// function, whose name mangles the quotes in the generator's name. The source
// name must still be found through it.

kgen.generator @"load(\22\22)"(%arg0: !kgen.scalar<index>) -> !kgen.scalar<index> {
  hlcf.return %arg0 : !kgen.scalar<index>
} <sourceName = "load">

kgen.generator @identity(%arg0: !kgen.generator<(!kgen.scalar<index>) -> !kgen.scalar<index>>) -> !kgen.generator<(!kgen.scalar<index>) -> !kgen.scalar<index>> {
  hlcf.return %arg0 : !kgen.generator<(!kgen.scalar<index>) -> !kgen.scalar<index>>
}

kgen.generator @name_of<func_type: type, func: !kgen.param<func_type>>() {
  // CHECK-LABEL: kgen.func @"name_of,
  // CHECK-NEXT: constant: string = <"load">
  %0 = kgen.param.constant: string = <#kgen.get_source_name<#kgen.param.decl.ref<"func"> : !kgen.param<func_type>>>
  hlcf.return
}

kgen.generator export @get_source_name_through_interpreter() {
  kgen.param.apply *"f" = [(!kgen.generator<(!kgen.scalar<index>) -> !kgen.scalar<index>>) -> !kgen.generator<(!kgen.scalar<index>) -> !kgen.scalar<index>>: @identity](@"load(\22\22)")
  kgen.call @name_of<:type !kgen.generator<(!kgen.scalar<index>) -> !kgen.scalar<index>>, :!kgen.generator<(!kgen.scalar<index>) -> !kgen.scalar<index>> *"f">() : () -> ()
  hlcf.return
}

// -----

// A transparent thunk names the function it wraps, as it does for
// `get_linkage_name`. The callee resolves through the wrapper's witness.

kgen.generator @user_kernel_impl<n: i32>() {
  hlcf.return
} <sourceName = "user_kernel_impl">

kgen.struct.generator @ClosureWrapper<n: i32> =
    struct_inst<"ClosureWrapper"[n]<:i32 n>(field0: index)>
{
  kgen.conformance @Callable {
    kgen.witness "__call__" : () -> () = @user_kernel_impl<:i32 n>
  }
}

kgen.generator @callable_thunk<W: type>() {
  kgen.param.declare callee: () -> () =
      <#kgen.get_witness<W, @Callable, "__call__">>
  kgen.call_param[() -> () : callee]()
  hlcf.return
} <sourceName = "callable_thunk"> attributes {"kgen.transparent_thunk_callee_expr" =
    #kgen.get_witness<W, @Callable, "__call__"> : !kgen.generator<() -> ()>}

#wrapper128 = #kgen.type<
    typevalue<:!kgen.type #kgen.genref<@ClosureWrapper<:i32 128>>>,
    struct<(index)>
  > : !kgen.type

// CHECK-LABEL: kgen.func export @get_source_name_through_thunk
kgen.generator export @get_source_name_through_thunk() {
  // CHECK-NEXT: constant: string = <"user_kernel_impl">
  %0 = kgen.param.constant: string = <#kgen.get_source_name<#kgen.gen<#kgen.func.symbol<@callable_thunk<:type #wrapper128>>> : !kgen.generator<() -> ()>>>
  hlcf.return
}
