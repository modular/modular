// RUN: kgen-opt %s -verify-diagnostics -split-input-file -o /dev/null

kgen.func @cast_from_builtin_type(%arg0: !meta.scalar<si32>) {
  // expected-error @below {{incompatible scalar data type}}
  %0 = meta.cast_to_builtin %arg0: !meta.scalar<si32> to ui32
  kgen.return
}

// -----

kgen.func @cast_from_builtin_type(%arg0: si32) {
  // expected-error @below {{incompatible scalar data type}}
  %0 = meta.cast_from_builtin %arg0 : si32 to !meta.scalar<ui32>
  kgen.return
}

// -----

kgen.func @cast_from_builtin_type(%arg0: !meta.scalar<f32>) {
  // expected-error @+1 {{'meta.cast_to_builtin' op does not support casting '!meta.scalar<f32>' to 'i8'}}
  %0 = meta.cast_to_builtin %arg0: !meta.scalar<f32> to i8
  kgen.return
}

// -----

kgen.func @cast_from_builtin_type(%arg0: !meta.scalar<f32>) {
  // expected-error @+1 {{'meta.cast_to_builtin' op does not support casting '!meta.scalar<f32>' to 'f64'}}
  %0 = meta.cast_to_builtin %arg0: !meta.scalar<f32> to f64
  kgen.return
}

// -----

kgen.func @cast_simd_to_vector(%arg0: !meta.simd<4, f32>) {
  // expected-error @+1 {{does not support casting '!meta.simd<4, f32>' to 'f32'}}
  %0 = meta.cast_to_builtin %arg0 : !meta.simd<4, f32> to f32
  kgen.return
}

// -----

kgen.func @cast_simd_to_vector(%arg0: !meta.simd<4, f32>) {
  // expected-error @+1 {{vector type should not be scalable}}
  %0 = meta.cast_to_builtin %arg0 : !meta.simd<4, f32> to vector<[4]xf32>
  kgen.return
}

// -----

kgen.func @cast_simd_to_vector(%arg0: !meta.simd<4, f32>) {
  // expected-error @+1 {{expected a rank 1 vector}}
  %0 = meta.cast_to_builtin %arg0 : !meta.simd<4, f32> to vector<4x4xf32>
  kgen.return
}

// -----

kgen.func @cast_simd_to_vector(%arg0: !meta.simd<4, f32>) {
  // expected-error @+1 {{dimensions do not match}}
  %0 = meta.cast_to_builtin %arg0 : !meta.simd<4, f32> to vector<5xf32>
  kgen.return
}

// -----

kgen.func @cast_simd_to_vector(%arg0: !meta.simd<4, f32>) {
  // expected-error @+1 {{element types do not match}}
  %0 = meta.cast_to_builtin %arg0 : !meta.simd<4, f32> to vector<4xf64>
  kgen.return
}

// -----

kgen.func @cast_from_meta_type(%arg0: f64) {
  // expected-error @+1 {{'meta.cast_from_builtin' op does not support casting 'f64' to '!meta.scalar<f32>'}}
  %0 = meta.cast_from_builtin %arg0: f64 to !meta.scalar<f32>
  kgen.return
}

// -----

kgen.func @meta_buffer_construct(%ptr: !meta.pointer<!meta.scalar<f32>>) {
  // expected-error @below {{requires a size operand}}
  %0 = meta.buffer.construct %ptr : !meta.buffer<?, f32>
  kgen.return
}

// -----

kgen.func @meta_buffer_construct(%ptr: !meta.pointer<?>) {
  // expected-error @below {{requires a dtype operand}}
  %0 = meta.buffer.construct %ptr : !meta.buffer<4, ?>
  kgen.return
}

// -----

// expected-error @+1 {{expected attribute value}}
kgen.func @unknown_size_simd(%arg0: !meta.simd<?, f32>) -> !meta.simd<?, f32> {
  kgen.return %arg0 : !meta.simd<?, f32>
}

// -----

// expected-error @+1 {{expected attribute value}}
kgen.func @unknown_type_simd(%arg0: !meta.simd<4, ?>) -> !meta.simd<4, ?> {
  kgen.return %arg0 : !meta.simd<4, ?>
}
