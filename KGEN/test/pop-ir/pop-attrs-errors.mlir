// RUN: kgen-opt -verify-diagnostics -split-input-file %s

kgen.func @simd_constant() {
  // expected-error @below {{integer value doesn't fit into 4 bits: 128}}
  %0 = kgen.param.constant: !pop.scalar<ui4> = <#pop.simd<128>>
  kgen.return
}

// -----

kgen.func @simd_constant() {
  // expected-error @below {{failed to parse floating point value}}
  %0 = kgen.param.constant: !pop.scalar<f16> = <#pop.simd<"e">>
  kgen.return
}

// -----

kgen.func @simd_constant() {
  // expected-error @below {{cannot convert 0.1 to f16}}
  %0 = kgen.param.constant: !pop.scalar<f16> = <#pop.simd<"0.1">>
  kgen.return
}

// -----

kgen.func @simd_constant() {
  // expected-error @below {{expected 'true' or 'false' for bool literal}}
  %0 = kgen.param.constant: !pop.scalar<bool> = <#pop.simd<e>>
  kgen.return
}

// -----

kgen.generator @simd_constant<size>() {
  // expected-error @below {{SIMD constant requires a concrete type}}
  %0 = kgen.param.constant: !pop.simd<size, bool> = <#pop.simd<true>>
  kgen.return
}

// -----

kgen.generator @simd_constant<size>() {
  // expected-error @below {{only integer, float, and bool dtype constants can be parsed}}
  %0 = kgen.param.constant: !pop.scalar<index> = <#pop.simd<0>>
  kgen.return
}
