// RUN: kgen-opt %s -elaborate-generators="max-depth=128 use-parametric-interpret=false" -verify-diagnostics -split-input-file -allow-unregistered-dialect

// Recursive expansions.

// expected-note @below {{function instantiation failed}}
// expected-note-re @below {{elaborator expansion is {{[0-9]+}} levels deep - infinite recursion?}}
// expected-note-re @below {{error recurses {{[0-9]+}} times}}
// expected-note @below {{remaining errors after}}
kgen.generator @genItf3<x>() {
  // expected-note @+1 {{call expansion failed}}
  kgen.call @genItf3<add(x, 1)>() : () -> ()
  kgen.return
}

// expected-error @+1 {{function instantiation failed}}
kgen.generator @use_Itf3two() {
  // expected-note @+1 {{call expansion failed}}
  kgen.call @genItf3<2>() : () -> ()
  kgen.return
}

// -----

// expected-error @+1 {{function instantiation failed}}
kgen.generator @unfoldableIndex() {
  kgen.param.declare x = <4>

  // Index type parameter expressions can only fold when they are known the
  // same on 32-bit and 64-bit systems or if target-specific information is
  // known.
  // expected-note @+1 {{could not simplify operator div(8000000000, 4)}}
  %1 = kgen.param.constant = <div(8000000000, x)> // 8B/4 differs on 32-bit.
  kgen.return
}

// -----

// expected-error @below {{function instantiation failed}}
kgen.generator @unfoldableIndex() {
  kgen.param.declare x = <4>

  // Index type parameter expressions can only fold when they are known the
  // same on 32-bit and 64-bit systems or if target-specific information is
  // known.
  // expected-note @+1 {{could not simplify operator div(8000000000, 4)}}
  %1 = kgen.param.constant = <add(1, div(8000000000, x))> // 8B/4 differs on 32-bit.
  kgen.return
}

// -----

#target = #kgen.target<triple="", arch="", features="", data_layout="", simd_bit_width=128> : !kgen.target

// expected-error @below {{function instantiation failed}}
kgen.generator @sizeof_unknown() {
  // expected-note @below {{could not simplify operator get_sizeof}}
  %0 = kgen.param.constant: index = <get_sizeof(!opaque<"type">, #target)>
  kgen.return
}

// -----

// expected-note @below {{failed to interpret function @cant_interpret}}
kgen.generator @cant_interpret(%arg0: index) -> index {
  // expected-note @below {{failed to fold operation some.op(1 : index)}}
  %0 = "some.op"(%arg0) : (index) -> index
  kgen.return %0 : index
}

// expected-error @below {{function instantiation failed}}
kgen.generator @interp_func() {
  // expected-note @below {{failed to compile-time evaluate function call}}
  %0 = kgen.param.constant = <apply(:(index) -> index @cant_interpret, 1)>
  kgen.return
}

// -----

// expected-note @below {{failed to interpret function @fails_to_interpret}}
kgen.generator @fails_to_interpret() {
  // expected-note @below {{failed to fold operation some.op()}}
  "some.op"() : () -> ()
  kgen.return
}

// expected-note @below {{failed to interpret function @passthrough}}
kgen.generator @passthrough() -> index {
  // expected-note @below {{failed to evaluate call}}
  kgen.call @fails_to_interpret() : () -> ()
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

// expected-error @below {{function instantiation failed}}
kgen.generator @call_it() {
  // expected-note @below {{failed to compile-time evaluate function call}}
  kgen.param.constant = <apply(:() -> index @passthrough)>
  kgen.return
}


// -----

// expected-error @below {{function instantiation failed}}
kgen.generator @brokenVLenAssert() {
  kgen.param.declare B : !kgen.string = <"foo">

  // expected-note @+1 {{constraint failed: foo}}
  kgen.param.assert <eq(2, 3)>, B
  kgen.return
}

// -----

// COM: Unused `kgen.param.declare` should not be ignored.

// expected-note @below {{function instantiation failed}}
kgen.generator @fail_if_zero<value>() -> index {
  %0 = index.constant 0
  // expected-note @below {{constraint failed: must not be zero!}}
  kgen.param.assert <ne(value, 0)>, "must not be zero!"
  kgen.return %0 : index
}

// expected-error @below {{function instantiation failed}}
kgen.generator @unused_param_declare() {
  kgen.param.declare unused = <apply(:() -> index bind_params(:<index>() -> index @fail_if_zero, 0))>
  kgen.return
}

// -----

// expected-error @below {{function instantiation failed}}
kgen.generator @invalid_rebind(%arg0: !pop.scalar<si32>) {
  kgen.param.declare dt: dtype = <ui32>
  // expected-note @below {{error: rebind input type '!pop.scalar<si32>' does not match result type '!pop.scalar<ui32>'}}
  %0 = kgen.rebind %arg0 : !pop.scalar<si32> to !pop.scalar<dt>
  kgen.return
}

// -----

// expected-note @below {{failed to interpret function @fails}}
kgen.generator @fails() -> index {
  // expected-note @below {{failed to fold operation kgen.unreachable()}}
  kgen.unreachable
}

// expected-error @below {{function instantiation failed}}
kgen.generator @failed_apply() {
  // expected-note @below {{failed to compile-time evaluate function call}}
  kgen.param.apply value = [() -> index: @fails]()
  kgen.param.constant = <value>
  kgen.return
}

// -----

// expected-error @below {{function instantiation failed}}
kgen.generator @failed_param_rebind() {
  // expected-note @below {{rebind input type 'i64' does not match result type 'i32'}}
  kgen.param.declare value: i32 = <rebind(:i64 2)>
  kgen.return
}

// -----

kgen.generator @function<param>() {
  kgen.return
}

// expected-error @below {{function instantiation failed}}
kgen.generator export @invalid_param_ref() {
  // expected-note @below {{cannot reference parametric function}}
  kgen.cost_of[<index>() -> (): @function]
  kgen.return
}

// -----

// expected-error @below {{function instantiation failed}}
kgen.generator export @recursive() -> index {
  // expected-note @below {{function instantiation in parameter domain that recursively requires itself}}
  // expected-note @below {{function recursively calls itself in the parameter domain}}
  kgen.param.apply x = [() -> index: @recursive]()
  %0 = kgen.param.constant = <x>
  kgen.return %0 : index
}

// -----

// expected-error @below {{function instantiation failed}}
kgen.generator export @recursive0() -> index {
  // expected-note @below {{function instantiation in parameter domain that recursively requires itself}}
  // expected-note @below {{back to parameter domain function call here}}
  kgen.param.apply x = [() -> index: @recursive1]()
  %0 = kgen.param.constant = <x>
  kgen.return %0 : index
}

kgen.generator @recursive1() -> index {
  // expected-note @below {{recursively instantiated through here}}
  %0 = kgen.call @recursive0() : () -> index
  kgen.return %0 : index
}


// -----
// COM: MOCO-964 fix.
// expected-error @below {{function instantiation failed}}
kgen.generator @will_fail() {
  kgen.param.declare B : !kgen.string = <"foo">

  // expected-note @+1 {{constraint failed: foo}}
  kgen.param.assert <eq(2, 3)>, B

  kgen.return
}

kgen.generator @will_pass<a, b>() -> (index, index) {
  %0 = kgen.param.constant = <a>
  %1 = kgen.param.constant = <b>
  kgen.return %0, %1 : index, index
}

!capture = !kgen.struct<(string, index, (!kgen.pointer<pointer<none>>) capturing -> !kgen.none)>

// expected-error @below {{function instantiation failed}}
kgen.generator export @main() {
  // expected-note @+1  {{failed to run the pass manager}}
  %0 = kgen.param.constant: !capture = <#kgen.compile_assembly<current_target(), =asm, "", false, :() -> () @will_fail>>
  %1 = kgen.param.constant: !capture = <#kgen.compile_assembly<current_target(), =asm, "", false, :() -> (index, index) @will_pass<3, 4>>>
  kgen.return
}

// -----

// Illegal recursion hidden behind struct type instantiation.

// expected-note @below {{function instantiation failed}}
kgen.generator @recursive() -> index {
  // expected-note @below {{function instantiation in parameter domain that recursively requires itself}}
  // expected-note @below {{function recursively calls itself in the parameter domain}}
  kgen.param.apply x = [() -> index: @recursive]()
  %0 = kgen.param.constant = <x>
  kgen.return %0 : index
}

// expected-note @below {{function instantiation failed}}
kgen.struct.generator @WeirdStruct<T: type> = struct_inst<"WeirdStruct"(data: array<apply(:() -> index @recursive), index>)>

kgen.generator @use_type<T: type>() {
  kgen.return
}

#weird_struct = #kgen.type<typevalue<#kgen.genref<@WeirdStruct<:type index>>>, struct<(array<2, index>)>> : !kgen.type

// expected-error @below {{function instantiation failed}}
kgen.generator export @gen_structs() {
  // expected-note @below {{call expansion failed}}
  kgen.call @use_type<:type #weird_struct>() : () -> ()
  kgen.return
}

// -----

// expected-note @below {{function instantiation failed}}
// expected-note @below {{cannot concretize name in 'llvm_metadata'}}
kgen.generator export @metadata<x>() attributes {LLVMMetadataArray = [
  #pop.array<x> : !pop.array<1, index>,  #pop.array<x> : !pop.array<1, index>
]}{
  kgen.return
}

// expected-error @below {{function instantiation failed}}
kgen.generator @metadata_caller() {
  // expected-note @below {{call expansion failed}}
  kgen.call @metadata<2>() : () -> ()
  kgen.return
}

// -----

// COM: test displaying trivial parameter values with call expansion failures.
// expected-note @below {{function instantiation failed}}
kgen.generator @fn1<a, b>() {
  // expected-note @+1  {{constraint failed: must be equal!}}
  kgen.param.assert <eq(a, b)>, "must be equal!"
  kgen.return
}

// expected-note @below {{function instantiation failed}}
kgen.generator @fn2<a, b>() {
  // expected-note @+1 {{call expansion failed with parameter value(s): ("a": 2, "b": 4)}}
  kgen.call @fn1<a, b>() : () -> ()
  kgen.return
}

// expected-note @below {{function instantiation failed}}
kgen.generator @fn3<a, b>() {
  // expected-note @+1 {{call expansion failed with parameter value(s): ("a": 2, "b": 4)}}
  kgen.call @fn2<a, b>() : () -> ()
  kgen.return
}

// expected-error @below {{function instantiation failed}}
kgen.generator export @main() {
  // expected-note @+1 {{call expansion failed with parameter value(s): ("a": 2, "b": 4)}}
  kgen.call @fn3<2, 4>() : () -> ()
  kgen.return
}


// -----

kgen.generator @g<T: i1>() -> index {
  // expected-note @+1 {{call expansion failed with parameter value(s): ("T": true)}}
  %0 = kgen.call @f<:i1 T>() : () -> index
  kgen.return %0 : index
}

kgen.generator @f<T: i1>() -> index {
  %0 = kgen.param.constant = <42>
  // expected-note @+1 {{codegen unreachable: materializing code that is not codegen reachable is not allowed}}
  kgen.codegen.reachable <not(T)>, "materializing code that is not codegen reachable is not allowed"
  kgen.return %0 : index
}

kgen.generator export @main() {
  // expected-error @+1 {{call expansion failed}}
  %0 = kgen.call @g<:i1 1>() : () -> index
  kgen.return
}
