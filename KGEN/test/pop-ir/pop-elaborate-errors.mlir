// RUN: kgen-opt -elaborate-generators %s -verify-diagnostics -split-input-file

kgen.generator @invalid_signedness<type: dtype>() -> !pop.simd<1, type> {
  // expected-note @below {{cannot change signfulness when converting from si8 to ui32}}
  %0 = pop.constant(4 : si8) : !pop.simd<1, type>
  kgen.return %0 : !pop.simd<1, type>
}

// expected-error @below {{no viable implementations}}
kgen.generator @impl() {
  // expected-note @below {{call expansion failed}}
  %0 = kgen.call @invalid_signedness<type: dtype = ui32>() : () -> !pop.simd<1, ui32>
  kgen.return
}

// -----

kgen.generator @invalid_trunc<type: dtype>() -> !pop.simd<1, type> {
  // expected-note @below {{integer constant does not fit into ui8}}
  %0 = pop.constant(600 : i32) : !pop.simd<1, type>
  kgen.return %0 : !pop.simd<1, type>
}

// expected-error @below {{no viable implementations}}
kgen.generator @impl() {
  // expected-note @below {{call expansion failed}}
  %0 = kgen.call @invalid_trunc<type: dtype = ui8>() : () -> !pop.simd<1, ui8>
  kgen.return
}

// -----

kgen.generator @invalid_fp_type<type: dtype>() -> !pop.simd<1, type> {
  // expected-note @below {{unsupported floating point type: f8}}
  %0 = pop.constant(600 : i32) : !pop.simd<1, type>
  kgen.return %0 : !pop.simd<1, type>
}

// expected-error @below {{no viable implementations}}
kgen.generator @impl() {
  // expected-note @below {{call expansion failed}}
  %0 = kgen.call @invalid_fp_type<type: dtype = f8>() : () -> !pop.simd<1, f8>
  kgen.return
}

// -----

kgen.generator @invalid_fp_type<type: dtype>() -> !pop.simd<1, type> {
  // expected-note @below {{integer constant could not be exactly converted to f32}}
  %0 = pop.constant(16777217 : i32) : !pop.simd<1, type>
  kgen.return %0 : !pop.simd<1, type>
}

// expected-error @below {{no viable implementations}}
kgen.generator @impl() {
  // expected-note @below {{call expansion failed}}
  %0 = kgen.call @invalid_fp_type<type: dtype = f32>() : () -> !pop.simd<1, f32>
  kgen.return
}

// -----

kgen.generator @inexact_int<type: dtype>() -> !pop.simd<1, type> {
  // expected-note @below {{only exact integer floats can be converted to integers}}
  %0 = pop.constant(1.2) : !pop.simd<1, type>
  kgen.return %0 : !pop.simd<1, type>
}

// expected-error @below {{no viable implementations}}
kgen.generator @impl() {
  // expected-note @below {{call expansion failed}}
  %0 = kgen.call @inexact_int<type: dtype = si8>() : () -> !pop.simd<1, si8>
  kgen.return
}

// -----

kgen.generator @invalid_fp_type<type: dtype>() -> !pop.simd<1, type> {
  // expected-note @below {{unsupported floating point type}}
  %0 = pop.constant(1.2) : !pop.simd<1, type>
  kgen.return %0 : !pop.simd<1, type>
}

// expected-error @below {{no viable implementations}}
kgen.generator @impl() {
  // expected-note @below {{call expansion failed}}
  %0 = kgen.call @invalid_fp_type<type: dtype = f8>() : () -> !pop.simd<1, f8>
  kgen.return
}

// -----

kgen.generator @invalid_bitcast<size, type: dtype>(%a: !pop.simd<4, f32>) -> !pop.simd<size, type> {
  // expected-note @below {{'!pop.simd<4, f32>' and result type '!pop.simd<2, ui32>' are cast incompatible}}
  %0 = pop.bitcast %a : !pop.simd<4, f32> to !pop.simd<size, type>
  kgen.return %0 : !pop.simd<size, type>
}

// expected-error @below {{no viable implementations}}
kgen.generator @impl(%a: !pop.simd<4, f32>) {
  // expected-note @below {{call expansion failed}}
  %0 = kgen.call @invalid_bitcast<size = 2, type: dtype = ui32>(%a) : (!pop.simd<4, f32>) -> (!pop.simd<2, ui32>)
  kgen.return
}
