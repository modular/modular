// RUN: support-dialect-opt %s -split-input-file -allow-unregistered-dialect -verify-diagnostics

// expected-error @below {{zero-width element type unsupported}}
"M"() {a = #M.primitives_array<i0>} : () -> ()

// -----

// expected-error @below {{expected integer or float element type}}
"M"() {a = #M.primitives_array<vector<2xi2>>} : () -> ()

// -----

// expected-error @below {{expected a shaped type}}
"M"() {a = #M.dense_array<1> : i32} : () -> ()

// -----

// expected-error @below {{shaped type must have static shape}}
"M"() {a = #M.dense_array<1> : tensor<*xi32>} : () -> ()

// -----

// expected-error @below {{attribute type indicates 2 elements, but array has 1}}
"M"() {a = #M.dense_array<1> : tensor<2xi32>} : () -> ()

// -----

// expected-error@+1 {{invalid hex string for aligned_bytes}}
"M"() {a = #M.aligned_bytes<16: "0xg0010204">} : () -> ()

// -----

// expected-error@+1 {{alignment must be a power of two.}}
"M"() {a = #M.aligned_bytes<15: "0x01020304">} : () -> ()
