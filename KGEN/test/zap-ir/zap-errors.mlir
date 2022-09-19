// RUN: kgen-opt %s -verify-diagnostics -split-input-file -o /dev/null

kgen.func @simd_load(%buff: !meta.buffer<4, si32>) {
  %idx = index.constant 0
  // expected-error @below {{'zap.simd.load' op the buffer type ('!meta.buffer<4, si32>') must have the same element type as the result simd type ('!meta.simd<4, f32>')}}
  %0 = zap.simd.load %buff[%idx]: !meta.buffer<4, si32>, !meta.simd<4, f32>
}

// -----

kgen.func @simd_store(%val : !meta.simd<4, f32>, %buff: !meta.buffer<4, si32>) {
  %idx = index.constant 0
  // expected-error @below {{'zap.simd.store' op the buffer type ('!meta.buffer<4, si32>') must have the same element type as the value simd type ('!meta.simd<4, f32>')}}
  zap.simd.store %val, %buff[%idx]: !meta.simd<4, f32>, !meta.buffer<4, si32>
}

// -----

kgen.generator @buffer_stack_allocation() {
  // expected-error @below {{'zap.buffer.stack_allocation' op result #0 must be buffer with concrete size and dtype, but got '!meta.buffer<?, f32>'}}
  %0 = zap.buffer.stack_allocation : !meta.buffer<?, f32>
  kgen.return
}

// -----

kgen.generator @buffer_stack_allocation<size>() {
  // expected-error @below {{'zap.buffer.stack_allocation' op result #0 must be buffer with concrete size and dtype, but got '!meta.buffer<size, ?>'}}
  %0 = zap.buffer.stack_allocation : !meta.buffer<size, ?>
  kgen.return
}
