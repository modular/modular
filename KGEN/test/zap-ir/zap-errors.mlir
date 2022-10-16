// RUN: kgen-opt %s -verify-diagnostics -split-input-file -o /dev/null

kgen.func @zap_buffer_construct(%ptr: !pop.pointer<!pop.scalar<f32>>) {
  // expected-error @below {{either a size operand or a buffer type with static size}}
  %0 = zap.buffer.construct %ptr : !zap.buffer<?, f32>
  kgen.return
}

// -----

kgen.func @zap_buffer_construct(%ptr: !pop.pointer<!pop.scalar<f32>>, %size: index) {
  // expected-error @below {{either a size operand or a buffer type with static size}}
  %0 = zap.buffer.construct %ptr[%size] : !zap.buffer<4, f32>
  kgen.return
}

// -----

kgen.func @zap_buffer_construct(%ptr: !pop.pointer<!pop.scalar<invalid>>) {
  // expected-error @below {{either a dtype operand or a buffer type with static dtype}}
  %0 = zap.buffer.construct %ptr : !zap.buffer<4, ?>
  kgen.return
}

// -----

kgen.func @zap_buffer_construct(%ptr: !pop.pointer<!pop.scalar<f32>>, %dtype: !kgen.dtype) {
  // expected-error @below {{either a dtype operand or a buffer type with static dtype}}
  %0 = zap.buffer.construct %ptr of %dtype : !zap.buffer<4, f32>
  kgen.return
}

// -----

kgen.func @simd_load(%buff: !zap.buffer<4, si32>) {
  %idx = index.constant 0
  // expected-error @below {{'zap.buffer.simd_load' op the type ('!zap.buffer<4, si32>') must have the same element type as the simd type ('!pop.simd<4, f32>')}}
  %0 = zap.buffer.simd_load %buff[%idx]: !zap.buffer<4, si32>, !pop.simd<4, f32>
}

// -----

kgen.func @simd_store(%val : !pop.simd<4, f32>, %buff: !zap.buffer<4, si32>) {
  %idx = index.constant 0
  // expected-error @below {{'zap.buffer.simd_store' op the type ('!zap.buffer<4, si32>') must have the same element type as the simd type ('!pop.simd<4, f32>')}}
  zap.buffer.simd_store %val, %buff[%idx]: !pop.simd<4, f32>, !zap.buffer<4, si32>
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

// -----

// expected-error @below {{shape parameter for ndbuffer must not be empty}}
kgen.generator @zap_ndbuffer(%arg0 : !zap.ndbuffer<[], f32>) {
  kgen.return
}
// -----

// expected-error @below {{size parameter for ndbuffer must be positive}}
kgen.generator @zap_ndbuffer(%arg0 : !zap.ndbuffer<[0], f32>) {
  kgen.return
}
// -----

// expected-error @below {{size parameter for ndbuffer must be positive}}
kgen.generator @zap_ndbuffer(%arg0 : !zap.ndbuffer<[-1], f32>) {
  kgen.return
}

// -----

// expected-error @below {{shape parameter exceeds the maximum rank of the ndbuffer type}}
kgen.generator @zap_ndbuffer(%arg0 : !zap.ndbuffer<[1,2,3,4,5,6], f32>) {
  kgen.return
}

// -----

kgen.generator @zap_ndbuffer_load(%arg0 : !zap.ndbuffer<[3], f32>, %idx : index) {
  // expected-error @below {{'zap.ndbuffer.load' op requires the number of input positions (2) to match the rank of the ndbuffer type (1)}}
  %val = zap.ndbuffer.load %arg0[%idx, %idx] : !zap.ndbuffer<[3], f32>
  kgen.return
}

// -----

kgen.generator @zap_ndbuffer_store(%val : !pop.scalar<f32>,
                                %arg0 : !zap.ndbuffer<[3], f32>,
                                %idx : index) {
  // expected-error @below {{'zap.ndbuffer.store' op requires the number of input positions (2) to match the rank of the ndbuffer type (1)}}
  zap.ndbuffer.store %val, %arg0[%idx, %idx] : !zap.ndbuffer<[3], f32>
  kgen.return
}

// -----

kgen.generator @zap_ndbuffer_simd_load(%arg0 : !zap.ndbuffer<[3], f32>, %idx : index) {
  // expected-error @below {{'zap.ndbuffer.simd_load' op the type ('!zap.ndbuffer<[3], f32>') must have the same element type as the simd type ('!pop.simd<4, si32>')}}
  %val = zap.ndbuffer.simd_load %arg0[%idx, %idx] : !zap.ndbuffer<[3], f32>, !pop.simd<4, si32>
  kgen.return
}

// -----

kgen.generator @zap_ndbuffer_simd_load(%arg0 : !zap.ndbuffer<[3], f32>, %idx : index) {
  // expected-error @below {{'zap.ndbuffer.simd_load' op requires the number of input positions (2) to match the rank of the ndbuffer type (1)}}
  %val = zap.ndbuffer.simd_load %arg0[%idx, %idx] : !zap.ndbuffer<[3], f32>, !pop.simd<4, f32>
  kgen.return
}

// -----

kgen.generator @zap_ndbuffer_simd_store(%val : !pop.simd<4, si32>,
                                      %arg0 : !zap.ndbuffer<[3], f32>,
                                      %idx : index) {
  // expected-error @below {{'zap.ndbuffer.simd_store' op the type ('!zap.ndbuffer<[3], f32>') must have the same element type as the simd type ('!pop.simd<4, si32>')}}
  zap.ndbuffer.simd_store %val, %arg0[%idx] : !pop.simd<4, si32>, !zap.ndbuffer<[3], f32>
  kgen.return
}


// -----

kgen.generator @zap_ndbuffer_simd_store(%val : !pop.simd<4, f32>,
                                      %arg0 : !zap.ndbuffer<[3], f32>,
                                      %idx : index) {
  // expected-error @below {{'zap.ndbuffer.simd_store' op requires the number of input positions (2) to match the rank of the ndbuffer type (1)}}
  zap.ndbuffer.simd_store %val, %arg0[%idx, %idx] : !pop.simd<4, f32>, !zap.ndbuffer<[3], f32>
  kgen.return
}

// -----

kgen.generator @zap_ndbuffer_dim(%arg0 : !zap.ndbuffer<[3], f32>) {
  // expected-error @below {{'zap.ndbuffer.dim' op requires the '1' index to be less than the rank of the ndbuffer's rank of '1'}}
  zap.ndbuffer.dim %arg0[1] : !zap.ndbuffer<[3], f32>
  kgen.return
}

// -----

kgen.func @string_wrong_array_size() {
  // expected-error @below {{expected array result to have 6 elements but got 1}}
  %0 = zap.global_string "foobar"[1]
  kgen.return
}

// -----

kgen.func @not_si8_array() {
  // expected-error @below {{result #0 must be pointer to array of scalar `si8`}}
  %0 = "zap.global_string"() {value = "foobar"} : () -> (!pop.scalar<si8>)
  kgen.return
}
