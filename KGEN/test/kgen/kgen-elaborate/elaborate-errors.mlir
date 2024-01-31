// RUN: kgen-opt %s -elaborate-generators="enable-search=true max-depth=128" -verify-diagnostics -split-input-file -allow-unregistered-dialect

// Recursive expansions.

// expected-note @below {{no viable expansions found}}
// expected-note-re @below {{elaborator expansion is {{[0-9]+}} levels deep - infinite recursion?}}
// expected-note-re @below {{error recurses {{[0-9]+}} times}}
// expected-note @below {{remaining errors after}}
kgen.generator @genItf3<x>() {
  // expected-note @+1 {{call expansion failed}}
  kgen.call @genItf3<add(x, 1)>() : () -> ()
  kgen.return
}

// expected-error @below {{no viable expansions found}}
kgen.generator @use_Itf3two() {
  // expected-note @+1 {{call expansion failed}}
  kgen.call @genItf3<2>() : () -> ()
  kgen.return
}

// -----

kgen.generator @getSIMDLength<dt: dtype -> length>() {
  // This could be implemented as a constraint.
  kgen.param.assert <eq(:dtype dt, f32)>, "this only works for f32"
  kgen.param.result_bind<4>
  kgen.return
}

// expected-error @+1 {{no viable expansions found}}
kgen.generator @brokenVLenAssert() {
  kgen.call @getSIMDLength<:dtype f32 -> flen>() : () -> ()

  // expected-note @+1 {{vector length should be 3}}
  kgen.param.assert <eq(flen, 3)>, "vector length should be 3"
  kgen.return
}

// -----

// expected-error @+1 {{no viable expansions found}}
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

#target = #kgen.target<triple="", arch="", features="", data_layout="", simd_bit_width=128> : !kgen.target

// expected-error @below {{no viable expansions found}}
kgen.generator @sizeof_unknown() {
  // expected-note @below {{could not simplify operator get_sizeof}}
  %0 = kgen.param.constant = <get_sizeof(!opaque<"type">, #target)>
  kgen.return
}

// -----

// expected-note @below {{failed to interpret function @cant_interpret}}
kgen.generator @cant_interpret(%arg0: index) -> index {
  // expected-note @below {{failed to fold operation some.op(1 : index)}}
  %0 = "some.op"(%arg0) : (index) -> index
  kgen.return %0 : index
}

// expected-error @below {{no viable expansions found}}
kgen.generator @interp_func() {
  // expected-note @below {{failed to evaluate 'apply'}}
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

// expected-error @below {{no viable expansions found}}
kgen.generator @call_it() {
  // expected-note @below {{failed to evaluate 'apply'}}
  kgen.param.constant = <apply(:() -> index @passthrough)>
  kgen.return
}


// -----

// expected-error @+1 {{no viable expansions found}}
kgen.generator @brokenVLenAssert() {
  kgen.param.declare B : !kgen.string = <"foo">

  // expected-note @+1 {{constraint failed: foo}}
  kgen.param.assert <eq(2, 3)>, B
  kgen.return
}

// -----

// expected-note @below {{no viable expansions found}}
kgen.generator @paramRecurse<() -> out>() {
  // expected-note @below {{recursive call to function with result parameters}}
  kgen.call @paramRecurse<[] -> val>() : () -> ()
  kgen.param.result_bind<0>
  kgen.return
}

// expected-error @below {{no viable expansions found}}
kgen.generator export @caller() {
  // expected-note @below {{call expansion failed - no concrete specializations}}
  kgen.call @paramRecurse<[] -> v>() : () -> ()
  kgen.return
}

// -----

// expected-error @below {{no viable expansions found}}
kgen.generator @bad_recursion() {
  kgen.param.fork N = <[1, 2]>
  // expected-note @below {{recursive call to function with more than 1 implementation}}
  kgen.call @bad_recursion() : () -> ()
  kgen.return
}

// -----

// COM: Unused `kgen.param.declare` should not be ignored.

// expected-note @below {{no successful concrete nodes}}
kgen.generator @fail_if_zero<value>() -> index {
  %0 = index.constant 0
  // expected-note @below {{constraint failed: must not be zero!}}
  kgen.param.assert <ne(value, 0)>, "must not be zero!"
  kgen.return %0 : index
}

// expected-error @below {{no viable expansions found}}
kgen.generator @unused_param_declare() {
  kgen.param.declare unused = <apply(:() -> index bind_signature(:<index>() -> index @fail_if_zero, 0))>
  kgen.return
}

// -----

// expected-note @below {{failed to interpret function @fails_to_interpret_if_true}}
kgen.generator @fails_to_interpret_if_true<cond: i1>() -> index {
  kgen.param.if <cond> {
    // expected-note @below {{failed to fold operation}}
    "unknown.op"() : () -> ()
    kgen.param.yield
  } else {
    kgen.param.yield
  }
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

// expected-error @below {{no viable expansions found}}
kgen.generator @interpreter_state_owner() {
  // expected-note @below {{failed to evaluate 'apply'}}
  kgen.param.fork first_fails = <[
    apply(:() -> index @fails_to_interpret_if_true<:i1 1>),
    apply(:() -> index @fails_to_interpret_if_true<:i1 0>)
  ]>
  kgen.return
}

// -----

// expected-error @below {{no viable expansions found}}
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

// expected-error @below {{no viable expansions found}}
kgen.generator @failed_apply() {
  // expected-note @below {{failed to evaluate 'apply'}}
  kgen.param.apply value = [() -> index: @fails]()
  kgen.param.constant = <value>
  kgen.return
}

// -----

kgen.generator @evaluator(%fns: !kgen.pointer<() -> ()>, %size: index) -> index {
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

// expected-note @below {{no viable expansions found}}
kgen.generator @no_valid_specializations() {
  // expected-note @below {{constraint failed: none}}
  kgen.param.assert <0>, "none"
  kgen.return
}

// expected-error @below {{no viable expansions found}}
kgen.generator export @entry() {
  // expected-note @below {{call expansion failed - no concrete specialization}}
  kgen.param.evaluate f: () -> () = [@no_valid_specializations] with
    [(!kgen.pointer<() -> ()>, index) -> index: @evaluator]
  kgen.return
}

// -----

kgen.generator @evaluator(%fns: !kgen.pointer<() -> ()>, %size: index) -> index {
  %idx1 = index.constant 1
  kgen.return %idx1 : index
}

kgen.generator @one() {
  kgen.return
}

// expected-error @below {{no viable expansions found}}
kgen.generator export @entry() {
  // expected-note @below {{user-provided evaluator returned an out-of-bounds result: 1}}
  kgen.param.evaluate f: () -> () = [@one] with
    [(!kgen.pointer<() -> ()>, index) -> index: @evaluator]
  kgen.return
}

// -----

// expected-error @below {{primary generator with more than one successful implementation}}
// expected-note @below {{select one implementation using search or remove forks in the implementation}}
kgen.generator export @multiversioned() {
  kgen.param.fork N = <[1, 2]>
  kgen.return
}

// -----

// expected-note @below {{no viable expansions found}}
kgen.generator @no_impls() {
// expected-note @below {{constraint failed}}
  kgen.param.assert <0>, "none"
  kgen.return
}

// expected-error @below {{no viable expansions found}}
kgen.generator export @get_all_impls_none() {
  kgen.param.declare impls: variadic<!kgen.signature<() -> ()>> = <get_all_impls(@no_impls)>
  kgen.return
}

// -----

// expected-error @below {{no viable expansions found}}
kgen.generator @failed_param_rebind() {
  // expected-note @below {{rebind input type 'i64' does not match result type 'i32'}}
  kgen.param.declare value: i32 = <rebind(:i64 2)>
  kgen.return
}

// -----

// expected-error @below {{primary generator with more than one successful implementation}}
// expected-note @below {{select one implementation using search or remove forks in the implementation}}
kgen.generator @kernel() {
  kgen.param.fork a = <[1, 2]>
  kgen.return
}

// expected-error @below {{no viable expansions found}}
kgen.generator export @top() {
  // expected-note @below {{failed to run the pass manager}}
  kgen.param.constant: string = <compile_assembly(current_target(), asm, :() -> () @kernel)>
  kgen.return
}

// -----

kgen.generator @kernel() {
  kgen.return
}

kgen.generator export @top() {
  // expected-error @below {{custom op 'kgen.param.constant' the emission kind must be either llvm or asm}}
  kgen.param.constant: string = <compile_assembly(current_target(), something, :() -> () @kernel)>
  kgen.return
}

// -----

kgen.generator @function<param>() {
  kgen.return
}

// expected-error @below {{no viable expansions found}}
kgen.generator export @invalid_param_ref() {
  // expected-note @below {{cannot reference parametric function}}
  kgen.cost_of[<index>() -> (): @function]
  kgen.return
}
