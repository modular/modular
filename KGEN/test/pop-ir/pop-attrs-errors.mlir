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
  // expected-error @below {{cannot convert 1e+100 to f16}}
  %0 = kgen.param.constant: !pop.scalar<f16> = <#pop.simd<"1e+100">>
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
  // expected-error @below {{only integer, float, bool, and index dtype constants can be parsed}}
  %0 = kgen.param.constant: !pop.scalar<f8> = <#pop.simd<0>>
  kgen.return
}

// -----

kgen.generator @array_constant<size>() {
  // expected-error @below {{array attribute expected a fully-resolved array type}}
  %0 = kgen.param.constant: !pop.array<size, index> = <#pop.array<0>>
}

// -----

kgen.generator @array_constant<T: type>() {
  // expected-error @below {{array attribute expected a fully-resolved array type}}
  %0 = kgen.param.constant: !pop.array<1, T> = <#pop.array<0>>
}

// -----

kgen.generator @struct_constant<T: type>() {
  // expected-error @below {{struct attribute expected a fully-resolved struct type}}
  %0 = kgen.param.constant: !pop.struct<T> = <#pop.struct<0>>
}

// -----

kgen.generator @variant_constant<value: i32>() {
  // expected-error @below {{variant attribute value type 'i32' is not a possible variant subtype}}
  %0 = kgen.param.constant: !pop.variant<f32, f64> = <#pop.variant<:i32 value>>
}
