// RUN: kgen-opt %s -elaborate-generators="enable-search=true max-depth=128" -verify-diagnostics -split-input-file -allow-unregistered-dialect

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

kgen.generator @use_Itf3two() {
  // expected-error @+1 {{call expansion failed}}
  kgen.call @genItf3<2>() : () -> ()
  kgen.return
}

// -----

kgen.generator @unfoldableIndex() {
  kgen.param.declare x = <4>

  // Index type parameter expressions can only fold when they are known the
  // same on 32-bit and 64-bit systems or if target-specific information is
  // known.
  // expected-error @+1 {{could not simplify operator div(8000000000, 4)}}
  %1 = kgen.param.constant = <div(8000000000, x)> // 8B/4 differs on 32-bit.
  kgen.return
}


// -----

#target = #kgen.target<triple="", arch="", features="", data_layout="", simd_bit_width=128> : !kgen.target

kgen.generator @sizeof_unknown() {
  // expected-error @below {{could not simplify operator get_sizeof}}
  %0 = kgen.param.constant: !kgen.int_literal = <get_sizeof(!opaque<"type">, #target)>
  kgen.return
}

// -----

// expected-note @below {{failed to interpret function @cant_interpret}}
kgen.generator @cant_interpret(%arg0: index) -> index {
  // expected-note @below {{failed to fold operation some.op(1 : index)}}
  %0 = "some.op"(%arg0) : (index) -> index
  kgen.return %0 : index
}

kgen.generator @interp_func() {
  // expected-error @below {{failed to compile-time evaluate function call}}
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

kgen.generator @call_it() {
  // expected-error @below {{failed to compile-time evaluate function call}}
  kgen.param.constant = <apply(:() -> index @passthrough)>
  kgen.return
}


// -----

kgen.generator @brokenVLenAssert() {
  kgen.param.declare B : !kgen.string = <"foo">

  // expected-error @+1 {{constraint failed: foo}}
  kgen.param.assert <eq(2, 3)>, B
  kgen.return
}

// -----

// COM: Unused `kgen.param.declare` should not be ignored.

// expected-error @below {{function instantiation failed}}
kgen.generator @fail_if_zero<value>() -> index {
  %0 = index.constant 0
  // expected-note @below {{constraint failed: must not be zero!}}
  kgen.param.assert <ne(value, 0)>, "must not be zero!"
  kgen.return %0 : index
}

kgen.generator @unused_param_declare() {
  kgen.param.declare unused = <apply(:() -> index bind_signature(:<index>() -> index @fail_if_zero, 0))>
  kgen.return
}

// -----

kgen.generator @invalid_rebind(%arg0: !pop.scalar<si32>) {
  kgen.param.declare dt: dtype = <ui32>
  // expected-error @below {{error: rebind input type '!pop.scalar<si32>' does not match result type '!pop.scalar<ui32>'}}
  %0 = kgen.rebind %arg0 : !pop.scalar<si32> to !pop.scalar<dt>
  kgen.return
}

// -----

// expected-note @below {{failed to interpret function @fails}}
kgen.generator @fails() -> index {
  // expected-note @below {{failed to fold operation kgen.unreachable()}}
  kgen.unreachable
}

kgen.generator @failed_apply() {
  // expected-error @below {{failed to compile-time evaluate function call}}
  kgen.param.apply value = [() -> index: @fails]()
  kgen.param.constant = <value>
  kgen.return
}

// -----

kgen.generator @failed_param_rebind() {
  // expected-error @below {{rebind input type 'i64' does not match result type 'i32'}}
  kgen.param.declare value: i32 = <rebind(:i64 2)>
  kgen.return
}

// -----

kgen.generator @function<param>() {
  kgen.return
}

kgen.generator export @invalid_param_ref() {
  // expected-error @below {{cannot reference parametric function}}
  kgen.cost_of[<index>() -> (): @function]
  kgen.return
}

// -----

kgen.generator export @recursive() -> index {
  // expected-error @below {{function requires parameter domain instantiation of recursive call that cannot be resolved}}
  kgen.param.apply x = [() -> index: @recursive]()
  %0 = kgen.param.constant = <x>
  kgen.return %0 : index
}
