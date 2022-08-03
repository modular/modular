// RUN: kgen-elaborate %s -library=%S/library-test.mlir -verify-diagnostics -o /dev/null -split-input-file

// expected-error @+1 {{interface argument #0 has type 'f32' but library interface expected type 'si32'}}
kgen.generator.interface @unary_add<size>(f32) -> si32

// -----

// This yields a verification error when elaborated.
// expected-error @+1 {{failed to generate any kernels}}
kgen.kernel @local_verif_error() {

  kgen.param.bind ty : dtype = <f32>
  %c1 = arith.constant 1.0 : f32
  %0 = meta.cast_from_builtin %c1: f32 to !meta.scalar<ty>

  // expected-note @+1 {{verification error: 'meta.cast_to_builtin' op does not support casting %1 = "meta.cast_from_builtin"(%0) : (f32) -> !meta.scalar<f32> to 'i8'}}
  %1 = meta.cast_to_builtin %0: !meta.scalar<ty> to i8
  kgen.return
}

// -----

kgen.generator.interface @genItf2<x>()

// expected-note @below {{failed to expand this declaration}}
// expected-note @+1 {{constraint failed}}
kgen.generator @genItf2_impl0<x>()
  constraints <eq(x, 0)> implements @genItf2 {
  "impl0"() : () -> ()
  kgen.return
}
// expected-note @+1 {{failed to expand this declaration}}
kgen.generator @genItf2_impl1<x>() implements @genItf2 {
  // expected-note @+1 {{unknown parameter-defining operator}}
  "impl1" () { paramDecls = [#kgen.param.decl<"badaram" : index>] } : () -> ()
  kgen.return
}

// This has no expansions, so it should generate an error message.
// expected-error @+1 {{failed to generate any kernels}}
kgen.kernel @use_Itf2two() {
  // expected-note @+1 {{call expansion failed}}
  kgen.call @genItf2<x = 2>() : () -> ()
  kgen.return
}

// -----

// Recursive expansions.

kgen.generator.interface @genItf3<x>()

// expected-note @+2 {{back to this declaration}}
// expected-error @+1 {{declaration involved in recursive elaboration cycle}}
kgen.generator @genItf3_impl<x>() implements @genItf3 {
  // expected-note @+1 {{through this call}}
  kgen.call @genItf3<x = 7>() : () -> ()
  kgen.return
}

kgen.kernel @use_Itf3two() {
  kgen.call @genItf3<x = 2>() : () -> ()
  kgen.return
}

// -----

// Expansions of kernels with zero expansions.

// expected-note @+1 {{no implementations of interface 'itf' found}}
kgen.generator.interface @itf<x>()

// expected-error @+1 {{failed to generate any kernels}}
kgen.kernel @k1() {
  // expected-note @+1 {{call expansion failed}}
  kgen.call @itf<x = 2>() : () -> ()
  kgen.return
}

// expected-error @+1 {{failed to generate any kernels}}
kgen.kernel @k2() {
  // expected-note @+1 {{call expansion failed}}
  kgen.call @k1() : () -> ()
  kgen.return
}
