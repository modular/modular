// RUN: kgen-opt %s -verify-diagnostics -split-input-file

kgen.generator @pop_constant<type: type>() {
  // expected-error @below {{expected integer or float attribute for unspecified result type}}
  %0 = pop.constant(dense<0> : vector<1xi32>) : !kgen.paramref<type>
  kgen.return
}

// -----

kgen.func @pop_constant() {
  // expected-error @below {{cannot convert from attribute type 'f32' to dtype si64}}
  %0 = pop.constant(32.0 : f32) : !meta.scalar<si64>
  kgen.return
}

// -----

kgen.func @pop_constant() {
  // expected-error @below {{scalar constant expected integer or float attribute for constant value}}
  %0 = pop.constant(dense<0> : vector<1xi32>) : !meta.scalar<si32>
  kgen.return
}

// -----

kgen.func @pop_constant() {
  // expected-error @below {{expected dense elements attribute for vector constant with known size}}
  %0 = pop.constant(0 : i32) : !meta.simd<2, si32>
  kgen.return
}

// -----

kgen.func @pop_constant() {
  // expected-error @below {{expected attribute type to be vector<2xT>}}
  %0 = pop.constant(dense<0.0> : tensor<2xf32>) : !meta.simd<2, f32>
  kgen.return
}

// -----

kgen.func @pop_constant() {
  // expected-error @below {{expected attribute type to be vector<2xT>}}
  %0 = pop.constant(dense<0> : vector<2x2xsi32>) : !meta.simd<2, si32>
  kgen.return
}

// -----

kgen.func @pop_constant() {
  // expected-error @below {{expected attribute type to be vector<2xT>}}
  %0 = pop.constant(dense<0> : vector<1xsi32>) : !meta.simd<2, si32>
  kgen.return
}

// -----

kgen.func @pop_constant() {
  // expected-error @below {{cannot convert from attribute type 'i32' to dtype si32}}
  %0 = pop.constant(dense<0> : vector<2xi32>) : !meta.simd<2, si32>
  kgen.return
}

// -----

kgen.generator @pop_constant<size>() {
  // expected-error @below {{expected integer or float attribute for vector constant of unspecified size}}
  %0 = pop.constant(dense<0> : vector<2xsi32>) : !meta.simd<size, si32>
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

// You are not allowed to bitcast a pointer to a scalar.
kgen.generator @bitcast_pointer(%a: !meta.pointer<!meta.scalar<f32>>) {
  // expected-error @below {{'pop.bitcast' op operand type '!meta.pointer<!meta.scalar<f32>>' and result type '!meta.scalar<f32>' are cast incompatible}}
  %0 = pop.bitcast %a : !meta.pointer<!meta.scalar<f32>> to !meta.scalar<f32>
  kgen.return
}

// -----

// You are not allowed to bitcast a pointer to a scalar.
kgen.generator @bitcast_pointer(%a: !meta.scalar<f32>) {
  // expected-error @below {{'pop.bitcast' op operand type '!meta.scalar<f32>' and result type '!meta.pointer<!meta.scalar<f32>>' are cast incompatible}}
  %1 = pop.bitcast %a : !meta.scalar<f32> to !meta.pointer<!meta.scalar<f32>>
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

// -----

kgen.func @global_constant() {
  // expected-error @below {{cannot convert from attribute type 'f32' to dtype f64}}
  %0 = pop.global_constant(0.0 : f32) : !meta.scalar<f64>
  kgen.return
}

// -----

kgen.func @global_constant() {
  // expected-error @below {{expected dense elements attribute for array constant with known size}}
  %0 = pop.global_constant(0.0 : f32) : !pop.array<4, !meta.scalar<f32>>
  kgen.return
}

// -----

kgen.func @global_constant() {
  // expected-error @below {{expected attribute type to be tensor<2xT>}}
  %0 = pop.global_constant(dense<0.0> : vector<2xf32>) : !pop.array<2, !meta.scalar<f32>>
  kgen.return
}

// -----

kgen.func @global_constant() {
  // expected-error @below {{expected attribute type to be tensor<2xT>}}
  %0 = pop.global_constant(dense<0.0> : tensor<2x2xf32>) : !pop.array<2, !meta.scalar<f32>>
  kgen.return
}

// -----

kgen.func @global_constant() {
  // expected-error @below {{expected attribute type to be tensor<2xT>}}
  %0 = pop.global_constant(dense<0.0> : tensor<1xf32>) : !pop.array<2, !meta.scalar<f32>>
  kgen.return
}

// -----

kgen.generator @global_constant<size>() {
  // expected-error @below {{expected integer or float attribute for array constant of unspecified size}}
  %0 = pop.global_constant(dense<0.0> : tensor<1xf32>) : !pop.array<size, !meta.scalar<f32>>
  kgen.return
}

// -----

kgen.func @global_constant() {
  // expected-error @below {{array constant must have scalar elements}}
  %0 = pop.global_constant(dense<0.0> : tensor<2xf32>) : !pop.array<2, !meta.simd<1, f32>>
  kgen.return
}

// -----

kgen.func @global_constant() {
  // expected-error @below {{convert from attribute type 'f64' to dtype f32}}
  %0 = pop.global_constant(dense<0.0> : tensor<2xf64>) : !pop.array<2, !meta.scalar<f32>>
  kgen.return
}

// -----

kgen.func @cast_from_builtin_type(%arg0: !meta.scalar<si32>) {
  // expected-error @below {{expected an integer or float type}}
  %0 = pop.type_lower %arg0: !meta.scalar<si32> to vector<1xsi32>
  kgen.return
}

// -----

kgen.func @cast_from_builtin_type(%arg0: si32) {
  // expected-error @below {{cannot convert from scalar dtype ui32 to 'si32'}}
  %0 = pop.type_raise %arg0 : si32 to !meta.scalar<ui32>
  kgen.return
}

// -----

kgen.func @cast_simd_to_vector(%arg0: !meta.simd<4, f32>) {
  // expected-error @below {{expected a rank 1 non-scalable vector}}
  %0 = pop.type_lower %arg0 : !meta.simd<4, f32> to f32
  kgen.return
}

// -----

kgen.generator @cast_simd_to_vector<size>(%arg0: !meta.simd<size, f32>) {
  // expected-error @below {{cannot convert from SIMD dtype f32 to vector element 'i32'}}
  %0 = pop.type_lower %arg0 : !meta.simd<size, f32> to vector<4xi32>
  kgen.return
}

// -----

kgen.func @cast_simd_to_vector(%arg0: !meta.simd<4, f32>) {
  // expected-error @below {{expected vector<4xT>}}
  %0 = pop.type_lower %arg0 : !meta.simd<4, f32> to vector<8xf32>
  kgen.return
}
