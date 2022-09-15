// RUN: kgen-opt -elaborate-generators %s -verify-diagnostics -split-input-file

kgen.generator @invalid_signedness<type: dtype>() -> !meta.scalar<type> {
  // expected-note @below {{cannot change signfulness when converting from si8 to ui32}}
  %0 = pop.constant(4 : si8) : !meta.scalar<type>
  kgen.return %0 : !meta.scalar<type>
}

// expected-error @below {{no viable implementations}}
kgen.generator @impl() {
  // expected-note @below {{call expansion failed}}
  %0 = kgen.call @invalid_signedness<type: dtype = ui32>() : () -> !meta.scalar<ui32>
  kgen.return
}

// -----

kgen.generator @invalid_trunc<type: dtype>() -> !meta.scalar<type> {
  // expected-note @below {{integer constant does not fit into ui8}}
  %0 = pop.constant(600 : i32) : !meta.scalar<type>
  kgen.return %0 : !meta.scalar<type>
}

// expected-error @below {{no viable implementations}}
kgen.generator @impl() {
  // expected-note @below {{call expansion failed}}
  %0 = kgen.call @invalid_trunc<type: dtype = ui8>() : () -> !meta.scalar<ui8>
  kgen.return
}

// -----

kgen.generator @invalid_fp_type<type: dtype>() -> !meta.scalar<type> {
  // expected-note @below {{unsupported floating point type: f8}}
  %0 = pop.constant(600 : i32) : !meta.scalar<type>
  kgen.return %0 : !meta.scalar<type>
}

// expected-error @below {{no viable implementations}}
kgen.generator @impl() {
  // expected-note @below {{call expansion failed}}
  %0 = kgen.call @invalid_fp_type<type: dtype = f8>() : () -> !meta.scalar<f8>
  kgen.return
}

// -----

kgen.generator @invalid_fp_type<type: dtype>() -> !meta.scalar<type> {
  // expected-note @below {{integer constant could not be exactly converted to f32}}
  %0 = pop.constant(16777217 : i32) : !meta.scalar<type>
  kgen.return %0 : !meta.scalar<type>
}

// expected-error @below {{no viable implementations}}
kgen.generator @impl() {
  // expected-note @below {{call expansion failed}}
  %0 = kgen.call @invalid_fp_type<type: dtype = f32>() : () -> !meta.scalar<f32>
  kgen.return
}

// -----

kgen.generator @invalid_bool<type: dtype>() -> !meta.scalar<type> {
  // expected-note @below {{cannot coerce constant value to bool}}
  %0 = pop.constant(1 : i32) : !meta.scalar<type>
  kgen.return %0 : !meta.scalar<type>
}

// expected-error @below {{no viable implementations}}
kgen.generator @impl() {
  // expected-note @below {{call expansion failed}}
  %0 = kgen.call @invalid_bool<type: dtype = bool>() : () -> !meta.scalar<bool>
  kgen.return
}

// -----

kgen.generator @inexact_int<type: dtype>() -> !meta.scalar<type> {
  // expected-note @below {{only exact integer floats can be converted to integers}}
  %0 = pop.constant(1.2) : !meta.scalar<type>
  kgen.return %0 : !meta.scalar<type>
}

// expected-error @below {{no viable implementations}}
kgen.generator @impl() {
  // expected-note @below {{call expansion failed}}
  %0 = kgen.call @inexact_int<type: dtype = si8>() : () -> !meta.scalar<si8>
  kgen.return
}

// -----

kgen.generator @invalid_fp_type<type: dtype>() -> !meta.scalar<type> {
  // expected-note @below {{unsupported floating point type}}
  %0 = pop.constant(1.2) : !meta.scalar<type>
  kgen.return %0 : !meta.scalar<type>
}

// expected-error @below {{no viable implementations}}
kgen.generator @impl() {
  // expected-note @below {{call expansion failed}}
  %0 = kgen.call @invalid_fp_type<type: dtype = f8>() : () -> !meta.scalar<f8>
  kgen.return
}

// -----

kgen.generator @invalid_bitcast<size, type: dtype>(%a: !meta.simd<4, f32>) -> !meta.simd<size, type> {
  // expected-note @below {{'!meta.simd<4, f32>' and result type '!meta.simd<2, ui32>' are cast incompatible}}
  %0 = pop.bitcast %a : !meta.simd<4, f32> to !meta.simd<size, type>
  kgen.return %0 : !meta.simd<size, type>
}

// expected-error @below {{no viable implementations}}
kgen.generator @impl(%a: !meta.simd<4, f32>) {
  // expected-note @below {{call expansion failed}}
  %0 = kgen.call @invalid_bitcast<size = 2, type: dtype = ui32>(%a) : (!meta.simd<4, f32>) -> (!meta.simd<2, ui32>)
  kgen.return
}
