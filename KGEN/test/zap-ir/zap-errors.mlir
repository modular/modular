// RUN: kgen-opt %s -verify-diagnostics -split-input-file -o /dev/null

kgen.func @zap_buffer_construct(%ptr: !meta.pointer<!meta.scalar<f32>>) {
  // expected-error @below {{either a size operand or a buffer type with static size}}
  %0 = zap.buffer.construct %ptr : !zap.buffer<?, f32>
  kgen.return
}

// -----

kgen.func @meta_buffer_construct(%ptr: !meta.pointer<!meta.scalar<f32>>, %size: index) {
  // expected-error @below {{either a size operand or a buffer type with static size}}
  %0 = zap.buffer.construct %ptr[%size] : !zap.buffer<4, f32>
  kgen.return
}

// -----

kgen.func @meta_buffer_construct(%ptr: !meta.pointer<?>) {
  // expected-error @below {{either a dtype operand or a buffer type with static dtype}}
  %0 = zap.buffer.construct %ptr : !zap.buffer<4, ?>
  kgen.return
}

// -----

kgen.func @meta_buffer_construct(%ptr: !meta.pointer<!meta.scalar<f32>>, %dtype: !kgen.dtype) {
  // expected-error @below {{either a dtype operand or a buffer type with static dtype}}
  %0 = zap.buffer.construct %ptr of %dtype : !zap.buffer<4, f32>
  kgen.return
}

// -----

kgen.func @simd_load(%buff: !zap.buffer<4, si32>) {
  %idx = index.constant 0
  // expected-error @below {{'zap.simd.load' op the buffer type ('!zap.buffer<4, si32>') must have the same element type as the result simd type ('!meta.simd<4, f32>')}}
  %0 = zap.simd.load %buff[%idx]: !zap.buffer<4, si32>, !meta.simd<4, f32>
}

// -----

kgen.func @simd_store(%val : !meta.simd<4, f32>, %buff: !zap.buffer<4, si32>) {
  %idx = index.constant 0
  // expected-error @below {{'zap.simd.store' op the buffer type ('!zap.buffer<4, si32>') must have the same element type as the value simd type ('!meta.simd<4, f32>')}}
  zap.simd.store %val, %buff[%idx]: !meta.simd<4, f32>, !zap.buffer<4, si32>
}

// -----

kgen.generator @buffer_stack_allocation() {
  // expected-error @below {{'zap.buffer.stack_allocation' op result #0 must be buffer with concrete size and dtype, but got '!zap.buffer<?, f32>'}}
  %0 = zap.buffer.stack_allocation : !zap.buffer<?, f32>
  kgen.return
}

// -----

kgen.generator @buffer_stack_allocation<size>() {
  // expected-error @below {{'zap.buffer.stack_allocation' op result #0 must be buffer with concrete size and dtype, but got '!zap.buffer<size, ?>'}}
  %0 = zap.buffer.stack_allocation : !zap.buffer<size, ?>
  kgen.return
}
