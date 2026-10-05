// RUN: kgen-opt %s -split-input-file -verify-parameters -elaborate-generators="use-parametric-interpret=false" \
// RUN:   -allow-unregistered-dialect | FileCheck %s
// RUN: kgen-opt %s -split-input-file -elaborate-generators="use-parametric-interpret=true" \
// RUN:   -allow-unregistered-dialect | FileCheck %s

kgen.generator @some_generator() attributes {sourceName = "foo"} {
  hlcf.return
}

// CHECK-LABEL: kgen.func export @get_source_name
kgen.generator export @get_source_name() {
  // CHECK-NEXT: constant: string = <"foo">
  %0 = kgen.param.constant: string = <#kgen.get_source_name<#kgen.symbol.constant<@some_generator> : !kgen.generator<() -> ()>>>
  hlcf.return
}

// -----

// A function value coming back from the comptime interpreter is the concrete
// function, whose name mangles the quotes in the generator's name. The source
// name must still be found through it.

kgen.generator @"load(\22\22)"(%arg0: !kgen.scalar<index>) -> !kgen.scalar<index> attributes {sourceName = "load"} {
  hlcf.return %arg0 : !kgen.scalar<index>
}

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
