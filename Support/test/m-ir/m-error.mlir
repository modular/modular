// RUN: index-opt %s -split-input-file -allow-unregistered-dialect -verify-diagnostics

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
