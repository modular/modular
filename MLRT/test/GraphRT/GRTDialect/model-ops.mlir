// RUN: graphrt-opt -allow-unregistered-dialect %s | graphrt-opt -allow-unregistered-dialect | FileCheck %s

// Test basic mef.model parsing and printing.

// CHECK-LABEL: mef.model @simple_model
// CHECK-SAME: init
// CHECK-SAME: (%[[INIT_ARG0:.*]]: i32) -> i32
mef.model @simple_model : mof (i32, tensor<?xf32>) -> tensor<?xf32> devices [] inputDeviceIndices [] resultDeviceIndices [] argumentNames [] resultNames [] init (%arg0: i32) -> (i32) {
    // CHECK: mef.output %[[INIT_ARG0]] : i32
    mef.output %arg0 : i32
  // CHECK: execute
  // CHECK-SAME: (%[[EXEC_ARG0:.*]]: i32, %[[EXEC_ARG1:.*]]: tensor<?xf32>) -> tensor<?xf32>
  } execute (%arg0: i32, %arg1: tensor<?xf32>) -> (tensor<?xf32>) {
    // CHECK: mef.output %[[EXEC_ARG1]] : tensor<?xf32>
    mef.output %arg1 : tensor<?xf32>
  }

// CHECK-LABEL: mef.model @model_with_chain
mef.model @model_with_chain : mof (tensor<4xf32>) -> tensor<4xf32> devices [] inputDeviceIndices [] resultDeviceIndices [] argumentNames [] resultNames [] init (%ch: !mef.chain) -> (!mef.chain) {
    // CHECK: init (%[[INIT_CH:.*]]: !mef.chain) -> !mef.chain
    // CHECK: mef.output %[[INIT_CH]] : !mef.chain
    mef.output %ch : !mef.chain
  // CHECK: execute (%[[EXEC_CH:.*]]: !mef.chain, %[[EXEC_IN:.*]]: tensor<4xf32>) -> tensor<4xf32>
  } execute (%ch: !mef.chain, %in: tensor<4xf32>) -> (tensor<4xf32>) {
    // CHECK: mef.output %[[EXEC_IN]] : tensor<4xf32>
    mef.output %in : tensor<4xf32>
  }

// CHECK-LABEL: mef.model @model_multiple_init_outputs
mef.model @model_multiple_init_outputs : mof (tensor<?xf32>) -> tensor<?xf32> devices [] inputDeviceIndices [] resultDeviceIndices [] argumentNames [] resultNames [] init (%a: i32, %b: f32) -> (i32, f32, !mef.chain) {
    // CHECK: init (%[[INIT_A:.*]]: i32, %[[INIT_B:.*]]: f32) -> (i32, f32, !mef.chain)
    // CHECK: %[[INIT_CH:.*]] = grt.chain.create ()
    %ch = grt.chain.create ()
    // CHECK: mef.output %[[INIT_A]], %[[INIT_B]], %[[INIT_CH]] : i32, f32, !mef.chain
    mef.output %a, %b, %ch : i32, f32, !mef.chain
  // CHECK: execute (%[[EXEC_A:.*]]: i32, %[[EXEC_B:.*]]: f32, %[[EXEC_CH:.*]]: !mef.chain, %[[EXEC_IN:.*]]: tensor<?xf32>) -> tensor<?xf32>
  } execute (%a: i32, %b: f32, %ch: !mef.chain, %in: tensor<?xf32>) -> (tensor<?xf32>) {
    // CHECK: mef.output %[[EXEC_IN]] : tensor<?xf32>
    mef.output %in : tensor<?xf32>
  }

// CHECK-LABEL: mef.model @model_no_extra_execute_inputs
mef.model @model_no_extra_execute_inputs : mof () -> i32 devices [] inputDeviceIndices [] resultDeviceIndices [] argumentNames [] resultNames [] init () -> (i32) {
    // CHECK: init () -> i32
    // CHECK: %[[CONST:.*]] = "test.constant"()
    %c = "test.constant"() : () -> i32
    // CHECK: mef.output %[[CONST]] : i32
    mef.output %c : i32
  // CHECK: execute (%[[EXEC_A:.*]]: i32) -> i32
  } execute (%a: i32) -> (i32) {
    // CHECK: mef.output %[[EXEC_A]] : i32
    mef.output %a : i32
  }

// CHECK-LABEL: mef.model @model_empty_init_outputs
mef.model @model_empty_init_outputs : mof (tensor<2xf32>) -> tensor<2xf32> devices [] inputDeviceIndices [] resultDeviceIndices [] argumentNames [] resultNames [] init (%a: i32) -> () {
    // CHECK: init (%[[INIT_A:.*]]: i32) -> ()
    // CHECK: mef.output
    mef.output
  // CHECK: execute (%[[EXEC_IN:.*]]: tensor<2xf32>) -> tensor<2xf32>
  } execute (%in: tensor<2xf32>) -> (tensor<2xf32>) {
    // CHECK: mef.output %[[EXEC_IN]] : tensor<2xf32>
    mef.output %in : tensor<2xf32>
  }

// CHECK-LABEL: mef.model @model_with_ops_in_regions
mef.model @model_with_ops_in_regions : mof (tensor<?xf32>) -> tensor<?xf32> devices [] inputDeviceIndices [] resultDeviceIndices [] argumentNames [] resultNames [] init (%ch: !mef.chain) -> (!mef.chain) {
    // CHECK: init (%[[INIT_CH:.*]]: !mef.chain) -> !mef.chain
    // CHECK: %[[NEW_CH:.*]] = grt.chain.create ()
    %new_ch = grt.chain.create ()
    // CHECK: mef.output %[[NEW_CH]] : !mef.chain
    mef.output %new_ch : !mef.chain
  // CHECK: execute (%[[EXEC_CH:.*]]: !mef.chain, %[[EXEC_IN:.*]]: tensor<?xf32>) -> tensor<?xf32>
  } execute (%ch: !mef.chain, %in: tensor<?xf32>) -> (tensor<?xf32>) {
    // CHECK: %[[RESULT:.*]] = "test.identity"(%[[EXEC_IN]])
    %result = "test.identity"(%in) : (tensor<?xf32>) -> tensor<?xf32>
    // CHECK: mef.output %[[RESULT]] : tensor<?xf32>
    mef.output %result : tensor<?xf32>
  }
