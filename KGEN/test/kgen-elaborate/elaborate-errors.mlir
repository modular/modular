// RUN: kgen-opt %s -elaborate-generators="search-path=%S" -verify-diagnostics -split-input-file -allow-unregistered-dialect

kgen.include "library-test.mlir"

// expected-error @below {{interface redeclaration argument #0 has type 'f32' but previous interface declaration expected type 'si32'}}
kgen.generator.interface @unary_add<size>(f32) -> si32

// -----

kgen.include "struct-test.mlir"

kgen.struct.decl @FooStruct<T:type> {
  // expected-error @below {{struct @FooStruct field #0 redeclared with different name "y"}}
  kgen.struct.field y : !kgen.paramref<T>
}

// -----

// This yields a verification error when elaborated.
// expected-error @+1 {{no viable implementations found}}
kgen.generator @local_verif_error() {

  kgen.param.declare ty : dtype = <f32>
  %0 = pop.constant(1.0 : f32) : !pop.simd<1, ty>

  // expected-note @+1 {{verification error: 'pop.cast_to_builtin' op cannot convert from scalar dtype f32 to 'i8'}}
  %1 = pop.cast_to_builtin %0: !pop.simd<1, ty> to i8
  kgen.return
}

// -----

kgen.generator.interface @genItf2<x>()

// expected-note @below {{failed to expand this declaration}}
kgen.generator @genItf2_impl0<x>()
// expected-note @below {{constraint failed: x must be zarooo}}
  constraints <[eq(x, 0), "x must be zarooo"]> implements @genItf2 {
  "impl0"() : () -> ()
  kgen.return
}
// expected-note @+1 {{failed to expand this declaration}}
kgen.generator @genItf2_impl1<x>() implements @genItf2 {
  // expected-note @+1 {{unknown parameter-defining operator}}
  "impl1" () { paramDecls = #kgen<param.decls[badaram : index]> } : () -> ()
  kgen.return
}

// This has no expansions, so it should generate an error message.
// expected-error @+1 {{no viable implementations found}}
kgen.generator @use_Itf2two() {
  // expected-note @+1 {{call expansion failed}}
  kgen.call @genItf2<x = 2>() : () -> ()
  kgen.return
}

// -----

// Recursive expansions.

kgen.generator.interface @genItf3<x>()

// expected-note @+1 {{elaborator expansion is 129 levels deep - infinite recursion?}}
kgen.generator @genItf3_impl<x>() implements @genItf3 {
  // expected-note @+1 {{call expansion failed}}
  kgen.call @genItf3<x = 7>() : () -> ()
  kgen.return
}

// expected-error @+1 {{no viable implementations found}}
kgen.generator @use_Itf3two() {
  // expected-note @+1 {{call expansion failed}}
  kgen.call @genItf3<x = 2>() : () -> ()
  kgen.return
}

// -----

// Expansions of kernels with zero expansions.

// expected-note @+1 {{no implementations of interface 'itf' found}}
kgen.generator.interface @itf<x>()

// expected-error @+1 {{no viable implementations found}}
kgen.generator @k1() {
  // expected-note @+1 {{call expansion failed}}
  kgen.call @itf<x = 2>() : () -> ()
  kgen.return
}

// expected-error @+1 {{no viable implementations found}}
kgen.generator @k2() {
  // expected-note @+1 {{call expansion failed}}
  kgen.call @k1() : () -> ()
  kgen.return
}

// -----

kgen.generator.interface @getSIMDLength<dt: dtype -> index>()

kgen.generator @getSIMDLengthF32<dt: dtype -> index>()
     implements @getSIMDLength {
  // This could be implemented as a constraint.
  kgen.param.assert <eq(:dtype dt, f32)>, "this only works for f32"
  kgen.return <4>
}

// expected-error @+1 {{no viable implementations found}}
kgen.generator @brokenVLenAssert() {
  kgen.call @getSIMDLength<dt : dtype = f32 -> flen>() : () -> ()

  // expected-note @+1 {{vector length should be 3}}
  kgen.param.assert <eq(flen, 3)>, "vector length should be 3"
  kgen.return
}

// -----

// expected-error @+1 {{could not find file 'does-not-exist.mlir'}}
kgen.include "does-not-exist.mlir"

// -----

// expected-error @+1 {{no viable implementations found}}
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

kgen.generator.interface @genItf2<x>()

// expected-error @+1 {{unlowered lit.func discovered in KGEN elaborator}}
lit.func @genItf2_impl0<x>() implements @genItf2 {
  kgen.return
}

// -----


kgen.generator @call_with_42<fn: <value>()->()>() {
  // expected-note @below {{call expansion failed}}
  kgen.call_param[()->(): bind_signature(:<value>()->() fn, 42)]()
  kgen.return
}

// expected-error @below {{no viable implementations found}}
kgen.generator @test_region_constraints() {
  kgen.param.declare.region fn = <value>()
      // expected-note @below {{constraint failed: I insist index be twelve}}
      constraints<[eq(value, 12), "I insist index be twelve"]> {
    kgen.return
  }
  // expected-note @below {{call expansion failed}}
  kgen.call @call_with_42<fn: <value>()->() = fn>() : () -> ()
  kgen.return
}

// -----

// expected-error @below {{no viable implementations found}}
kgen.generator @recursiveEvaluator(%funcs: !pop.pointer<() -> index>, %size: index) -> index {
  %0 = kgen.call @itf() : () -> index
  kgen.return %0 : index
}

// expected-note @below {{evaluator should have 1 candidate}}
kgen.generator.interface @itf() -> index
  evaluator (!pop.pointer<() -> index>, index) -> index = @recursiveEvaluator

// -----

kgen.struct.decl @Unknown {
  kgen.struct.field value : !opaque<"type">
}

// expected-error @below {{no viable implementations found}}
kgen.generator @sizeof_unknown() {
  // expected-note @below {{could not simplify operator get_sizeof}}
  %0 = kgen.param.constant = <get_sizeof(!kgen.declref<@Unknown>, #kgen<target host>)>
  kgen.return
}

// -----

// expected-note @below {{failed to interpret function @cant_interpret}}
kgen.func @cant_interpret(%arg0: index) -> index {
  // expected-note @below {{failed to fold operation some.op(1 : index)}}
  %0 = "some.op"(%arg0) : (index) -> index
  kgen.return %0 : index
}

// expected-error @below {{no viable implementations found}}
kgen.generator @interp_func() {
  // expected-note @below {{failed to evaluate 'apply'}}
  %0 = kgen.param.constant = <apply(:(index) -> index @cant_interpret, 1)>
  kgen.return
}

// -----

// expected-note @below {{no implementations of interface 'no_impls' found}}
kgen.generator.interface @no_impls() -> index

// expected-error @below {{no viable implementations found}}
kgen.generator @call_it() {
  // expected-note @below {{unable to evaluate generator or interface}}
  kgen.param.constant = <apply(:() -> index @no_impls)>
  kgen.return
}

// -----

// expected-note @below {{failed to interpret function @fails_to_interpret}}
kgen.func @fails_to_interpret() {
  // expected-note @below {{failed to fold operation some.op()}}
  "some.op"() : () -> ()
  kgen.return
}

// expected-note @below {{failed to interpret function @passthrough}}
kgen.func @passthrough() -> index {
  // expected-note @below {{failed to evaluate call}}
  kgen.call @fails_to_interpret() : () -> ()
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

// expected-error @below {{no viable implementations found}}
kgen.generator @call_it() {
  // expected-note @below {{failed to evaluate 'apply'}}
  kgen.param.constant = <apply(:() -> index @passthrough)>
  kgen.return
}
