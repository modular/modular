// RUN: kgen-opt %s -verify-diagnostics -split-input-file -o /dev/null

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

kgen.generator @zap_ndbuffer_stack_allocate() {
  // expected-error @below {{'zap.ndbuffer.stack_allocation' op result #0 must be ndbuffer with concrete size and dtype, but got '!zap.ndbuffer<[?], f32>'}}
  %0 = zap.ndbuffer.stack_allocation : !zap.ndbuffer<[?], f32>
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
  %val = zap.ndbuffer.load %arg0[%idx, %idx] : !zap.ndbuffer<[3], f32>, !pop.simd<1, f32>
  kgen.return
}

// -----

kgen.generator @zap_ndbuffer_store(%val : !pop.simd<1, f32>,
                                %arg0 : !zap.ndbuffer<[3], f32>,
                                %idx : index) {
  // expected-error @below {{'zap.ndbuffer.store' op requires the number of input positions (2) to match the rank of the ndbuffer type (1)}}
  zap.ndbuffer.store %val, %arg0[%idx, %idx] : !pop.simd<1, f32>, !zap.ndbuffer<[3], f32>
  kgen.return
}

// -----

kgen.generator @zap.ndbuffer.load(%arg0 : !zap.ndbuffer<[3], f32>, %idx : index) {
  // expected-error @below {{'zap.ndbuffer.load' op the type ('!zap.ndbuffer<[3], f32>') must have the same element type as the simd type ('!pop.simd<4, si32>')}}
  %val = zap.ndbuffer.load %arg0[%idx, %idx] : !zap.ndbuffer<[3], f32>, !pop.simd<4, si32>
  kgen.return
}

// -----

kgen.generator @zap.ndbuffer.load(%arg0 : !zap.ndbuffer<[3], f32>, %idx : index) {
  // expected-error @below {{'zap.ndbuffer.load' op requires the number of input positions (2) to match the rank of the ndbuffer type (1)}}
  %val = zap.ndbuffer.load %arg0[%idx, %idx] : !zap.ndbuffer<[3], f32>, !pop.simd<4, f32>
  kgen.return
}

// -----

kgen.generator @zap.ndbuffer.store(%val : !pop.simd<4, si32>,
                                      %arg0 : !zap.ndbuffer<[3], f32>,
                                      %idx : index) {
  // expected-error @below {{'zap.ndbuffer.store' op the type ('!zap.ndbuffer<[3], f32>') must have the same element type as the simd type ('!pop.simd<4, si32>')}}
  zap.ndbuffer.store %val, %arg0[%idx] : !pop.simd<4, si32>, !zap.ndbuffer<[3], f32>
  kgen.return
}


// -----

kgen.generator @zap.ndbuffer.store(%val : !pop.simd<4, f32>,
                                      %arg0 : !zap.ndbuffer<[3], f32>,
                                      %idx : index) {
  // expected-error @below {{'zap.ndbuffer.store' op requires the number of input positions (2) to match the rank of the ndbuffer type (1)}}
  zap.ndbuffer.store %val, %arg0[%idx, %idx] : !pop.simd<4, f32>, !zap.ndbuffer<[3], f32>
  kgen.return
}

// -----

kgen.generator @zap_ndbuffer_dim(%arg0 : !zap.ndbuffer<[3], f32>) {
  // expected-error @below {{'zap.ndbuffer.dim' op requires the '1' index to be less than the rank of the ndbuffer's rank of '1'}}
  zap.ndbuffer.dim %arg0[1] : !zap.ndbuffer<[3], f32>
  kgen.return
}

// -----

kgen.generator @zap_ndbuffer_dim(%arg0 : !zap.ndbuffer<[3], f32>) {
  // expected-error @below {{'zap.ndbuffer.dim' op attribute 'index' failed to satisfy constraint: index attribute whose value is non-negative}}
  zap.ndbuffer.dim %arg0[-1] : !zap.ndbuffer<[3], f32>
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
  %0 = "zap.global_string"() {value = "foobar"} : () -> (!pop.simd<1, si8>)
  kgen.return
}
