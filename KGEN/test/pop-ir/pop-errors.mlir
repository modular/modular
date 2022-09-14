// RUN: kgen-opt %s -verify-diagnostics -split-input-file -o /dev/null

kgen.func @pop_constant() -> !meta.scalar<si64> {
  // expected-error @below {{incompatible scalar data type}}
  // expected-error @below {{is incompatible with value type}}
  %0 = pop.constant(32.0 : f32) : !meta.scalar<si64>
  kgen.return %0 : !meta.scalar<si64>
}

// -----

kgen.func @pop_constant() -> !meta.scalar<f32> {
  // expected-error @below {{incompatible scalar data type}}
  // expected-error @below {{is incompatible with value type}}
  %0 = pop.constant(16777217 : i32) : !meta.scalar<f32>
  kgen.return %0 : !meta.scalar<f32>
}

// -----

kgen.func @pop_constant() {
  // expected-error @below {{incompatible scalar data type}}
  // expected-error @below {{is incompatible with value type}}
  %0 = pop.constant(dense<0> : vector<1xi32>) : !meta.scalar<si32>
  kgen.return
}

// -----

kgen.func @pop_constant() {
  // expected-error @below {{expected a vector type}}
  // expected-error @below {{is incompatible with value type}}
  %0 = pop.constant(0 : i32) : !meta.simd<2, si32>
  kgen.return
}

// -----

kgen.func @pop_constant() {
  // expected-error @below {{element types do not match}}
  // expected-error @below {{result type ('!meta.simd<2, f32>') is incompatible with value type ('vector<2xi32>')}}
  %0 = pop.constant(dense<16777217> : vector<2xi32>) : !meta.simd<2, f32>
  kgen.return
}

// -----

// COM: copysign is not defined on non-floating point types

kgen.func @pop_copysign(%arg0 : !meta.scalar<si32>, %arg1 : !meta.scalar<si32>) -> !meta.scalar<si32> {
  // expected-error @below {{whose value is either unbound or a floating-point dtype}}
  %0 = pop.copysign %arg0, %arg1 : !meta.scalar<si32>
  kgen.return %0 : !meta.scalar<si32>
}

// -----

// COM: copysign is not defined on non-floating point types

kgen.func @pop_copysign(%arg0 : !meta.simd<4, si32>, %arg1 : !meta.simd<4, si32>) -> !meta.simd<4, si32> {
  // expected-error @below {{whose element type is either unbound or a floating-point dtype}}
  %0 = pop.copysign %arg0, %arg1 : !meta.simd<4, si32>
  kgen.return %0 : !meta.simd<4, si32>
}

// -----

kgen.func @pop_select_simd(
    // expected-note @below {{prior use here}}
    %arg0: !meta.scalar<bool>,
    %arg1: !meta.simd<4, si32>,
    %arg2: !meta.simd<4, si32>
  ) -> !meta.simd<4, si32> {
  // expected-error @below {{use of value '%arg0' expects different type than prior uses: '!meta.simd<4, bool>' vs '!meta.scalar<bool>'}}
  %0 = pop.select %arg0, %arg1, %arg2 : !meta.simd<4, si32>
  kgen.return %0 : !meta.simd<4, si32>
}

// -----

kgen.func @pop_select_simd(
    // expected-note @below {{prior use here}}
    %arg0: !meta.simd<8, bool>,
    %arg1: !meta.simd<4, si32>,
    %arg2: !meta.simd<4, si32>
  ) -> !meta.simd<4, si32> {
  // expected-error @below {{use of value '%arg0' expects different type than prior uses: '!meta.simd<4, bool>' vs '!meta.simd<8, bool>'}}
  %0 = pop.select %arg0, %arg1, %arg2 : !meta.simd<4, si32>
  kgen.return %0 : !meta.simd<4, si32>
}

// -----

kgen.generator @bitcast_scalar(%a: !meta.scalar<f32>) {
  // expected-error @below {{'pop.bitcast' op operand type '!meta.scalar<f32>' and result type '!meta.scalar<si8>' are cast incompatible}}
  %0 = pop.bitcast %a : !meta.scalar<f32> to !meta.scalar<si8>
  kgen.return
}

// -----

kgen.generator @bitcast_simd(%a: !meta.simd<4, f32>) {
  // expected-error @below {{'pop.bitcast' op operand type '!meta.simd<4, f32>' and result type '!meta.simd<8, f32>' are cast incompatible}}
  %0 = pop.bitcast %a : !meta.simd<4, f32> to !meta.simd<8, f32>
  kgen.return
}

// -----

kgen.generator @bitcast_simd(%a: !meta.simd<4, f32>) {
  // expected-error @below {{'pop.bitcast' op operand type '!meta.simd<4, f32>' and result type '!meta.simd<4, f64>' are cast incompatible}}
  %0 = pop.bitcast %a : !meta.simd<4, f32> to !meta.simd<4, f64>
  kgen.return
}

// -----

kgen.generator @cast_scalar_to_simd<size, type: dtype>(%a: !meta.scalar<type>) {
  // expected-error @below {{cannot cast between a scalar type and SIMD type}}
  %0 = pop.cast %a : !meta.scalar<type> to !meta.simd<size, type>
  kgen.return
}

// -----

kgen.generator @cast_simd_to_scalar<size, type: dtype>(%a: !meta.simd<size, type>) {
  // expected-error @below {{cannot cast between a scalar type and SIMD type}}
  %0 = pop.cast %a : !meta.simd<size, type> to !meta.scalar<type>
  kgen.return
}

// -----

kgen.generator @cast_simd_size<type: dtype>(%a: !meta.simd<2, type>) {
  // expected-error @below {{cannot cast between SIMD types of different sizes}}
  %0 = pop.cast %a : !meta.simd<2, type> to !meta.simd<4, type>
  kgen.return
}

// -----

kgen.generator @cast_simd_size<size, type: dtype>(%a: !meta.simd<size, type>) {
  // expected-error @below {{cannot cast between SIMD types of different sizes}}
  %0 = pop.cast %a : !meta.simd<size, type> to !meta.simd<add(size, 1), type>
  kgen.return
}

// -----

kgen.generator @buffer_stack_allocation() {
  // expected-error @below {{'pop.buffer.stack_allocation' op cannot stack allocate a buffer of unknown size}}
  %0 = pop.buffer.stack_allocation : !meta.buffer<?, f32>
  kgen.return
}

// -----

kgen.generator @buffer_stack_allocation<size>() {
  // expected-error @below {{'pop.buffer.stack_allocation' op result #0 must be buffer with known dtype, but got '!meta.buffer<size, ?>'}}
  %0 = pop.buffer.stack_allocation : !meta.buffer<size, ?>
  kgen.return
}

// -----

kgen.generator @simd_shuffle(%a: !meta.simd<2, f32>) {
  // expected-error @below {{expected result dtype to match operand dtypes}}
  %0 = pop.simd.shuffle %a, %a [1] : !meta.simd<2, f32> -> !meta.simd<1, f64>
  kgen.return
}

// -----

kgen.generator @simd_shuffle<type: dtype>(%a: !meta.simd<2, f32>) {
  // expected-error @below {{expected result dtype to match operand dtypes}}
  %0 = pop.simd.shuffle %a, %a [1] : !meta.simd<2, f32> -> !meta.simd<1, type>
  kgen.return
}

// -----

kgen.generator @simd_shuffle<size>(%a: !meta.simd<2, f32>) {
  // expected-error @below {{expected result to be a vector of 1 elements}}
  %0 = pop.simd.shuffle %a, %a [1] : !meta.simd<2, f32> -> !meta.simd<size, f32>
  kgen.return
}

// -----

kgen.generator @simd_shuffle<size>(%a: !meta.simd<2, f32>) {
  // expected-error @below {{mask element 4 is out of bounds}}
  %0 = pop.simd.shuffle %a, %a [4] : !meta.simd<2, f32> -> !meta.simd<1, f32>
  kgen.return
}
