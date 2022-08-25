// RUN: kgen-opt %s -elaborate-kernels="search-path=%S" -verify-diagnostics -o /dev/null -split-input-file -allow-unregistered-dialect

kgen.include "library-test.mlir"

// expected-error @+1 {{interface argument #0 has type 'f32' but library interface expected type 'si32'}}
kgen.generator.interface @unary_add<size>(f32) -> si32

// -----

// This yields a verification error when elaborated.
// expected-error @+1 {{failed to generate any kernels}}
kgen.kernel @local_verif_error() {

  kgen.param.declare ty : dtype = <f32>
  %c1 = arith.constant 1.0 : f32
  %0 = meta.cast_from_builtin %c1: f32 to !meta.scalar<ty>

  // expected-note @+1 {{verification error: 'meta.cast_to_builtin' op does not support casting '!meta.scalar<f32>' to 'i8'}}
  %1 = meta.cast_to_builtin %0: !meta.scalar<ty> to i8
  kgen.return
}

// -----

kgen.generator.interface @genItf2<x>()

// expected-note @below {{failed to expand this declaration}}
// expected-note @+1 {{constraint failed: x must be zarooo}}
kgen.generator @genItf2_impl0<x>()
  constraints <[eq(x, 0), "x must be zarooo"]> implements @genItf2 {
  "impl0"() : () -> ()
  kgen.return
}
// expected-note @+1 {{failed to expand this declaration}}
kgen.generator @genItf2_impl1<x>() implements @genItf2 {
  // expected-note @+1 {{unknown parameter-defining operator}}
  "impl1" () { paramDecls = #kgen<param.decls["badaram" : index]> } : () -> ()
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

// -----

kgen.generator.interface @getSIMDLength<dt: dtype -> length>()

kgen.generator @getSIMDLengthF32<dt: dtype -> length>()
     implements @getSIMDLength {
  // This could be implemented as a constraint.
  kgen.param.assert <eq(:dtype dt, f32)>, "this only works for f32"
  kgen.return <length = 4>
}

// expected-error @+1 {{failed to generate any kernels}}
kgen.kernel @brokenVLenAssert() {
  kgen.call @getSIMDLength<dt : dtype = f32 -> flen>() : () -> ()

  // expected-note @+1 {{vector length should be 3}}
  kgen.param.assert <eq(flen, 3)>, "vector length should be 3"
  kgen.return
}

// -----

// expected-error @+1 {{could not find file 'does-not-exist.mlir'}}
kgen.include "does-not-exist.mlir"

// -----

// expected-error @+1 {{failed to generate any kernels}}
kgen.kernel @unfoldableIndex() {
  kgen.param.declare x = <4>

  // Index type parameter expressions can only fold when they are known the
  // same on 32-bit and 64-bit systems or if target-specific information is
  // known.
  // expected-note @+1 {{could not simplify operator mul(4, 2000000000)}}
  %1 = kgen.param.constant = <mul(2000000000, x)> // 2B*4 overflows on 32-bit.
  kgen.return
}

