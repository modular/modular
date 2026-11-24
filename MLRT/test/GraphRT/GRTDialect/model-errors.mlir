// RUN: graphrt-opt -allow-unregistered-dialect -split-input-file --verify-diagnostics %s

// Test that init output types must be a prefix of execute input types.

// expected-error @+1 {{'mef.model' op init output types 'f32' are not a prefix of execute input types 'i32', 'tensor<?xf32>'}}
mef.model @init_output_not_prefix_of_execute_input : mof (tensor<?xf32>) -> tensor<?xf32> devices [] inputDeviceIndices [] resultDeviceIndices [] argumentNames [] resultNames [] init (%arg0: i32) -> (f32) {
    %f = "test.constant"() : () -> f32
    mef.output %f : f32
  } execute (%arg0: i32, %arg1: tensor<?xf32>) -> (tensor<?xf32>) {
    mef.output %arg1 : tensor<?xf32>
  }

// -----

// Test that init output must match execute input prefix exactly.

// expected-error @+1 {{'mef.model' op init output types 'i32', 'i32' are not a prefix of execute input types 'i32', 'f32', 'tensor<?xf32>'}}
mef.model @init_output_type_mismatch : mof (tensor<?xf32>) -> tensor<?xf32> devices [] inputDeviceIndices [] resultDeviceIndices [] argumentNames [] resultNames [] init (%arg0: i32) -> (i32, i32) {
    mef.output %arg0, %arg0 : i32, i32
  } execute (%arg0: i32, %arg1: f32, %arg2: tensor<?xf32>) -> (tensor<?xf32>) {
    mef.output %arg2 : tensor<?xf32>
  }

// -----

// Test that init outputs cannot be longer than execute inputs.

// expected-error @+1 {{'mef.model' op init output types have 3 elements, but execute input types only have 2 elements; init outputs must be a prefix of execute inputs}}
mef.model @init_output_too_long : mof (f32) -> f32 devices [] inputDeviceIndices [] resultDeviceIndices [] argumentNames [] resultNames [] init (%arg0: i32) -> (i32, f32, !mef.chain) {
    %f = "test.constant"() : () -> f32
    %ch = grt.chain.create ()
    mef.output %arg0, %f, %ch : i32, f32, !mef.chain
  } execute (%arg0: i32, %arg1: f32) -> (f32) {
    mef.output %arg1 : f32
  }

// -----

// Test that init region output types must match init_type results.

// expected-error @+1 {{'mef.model' op init region output types 'f32' do not match init_type results 'i32'}}
mef.model @init_output_type_mismatch_with_attr : mof () -> i32 devices [] inputDeviceIndices [] resultDeviceIndices [] argumentNames [] resultNames [] init (%arg0: i32) -> (i32) {
    %f = "test.constant"() : () -> f32
    mef.output %f : f32
  } execute (%arg0: i32) -> (i32) {
    mef.output %arg0 : i32
  }

// -----

// Test that execute region output types must match execute_type results.

// expected-error @+1 {{'mef.model' op execute region output types 'i32' do not match execute_type results 'f32'}}
mef.model @execute_output_type_mismatch_with_attr : mof () -> f32 devices [] inputDeviceIndices [] resultDeviceIndices [] argumentNames [] resultNames [] init () -> () {
    mef.output
  } execute (%arg0: i32) -> (f32) {
    mef.output %arg0 : i32
  }
