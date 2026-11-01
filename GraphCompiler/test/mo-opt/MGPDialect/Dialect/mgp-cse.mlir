// RUN: mo-opt %s --allow-unregistered-dialect --cse --canonicalize --split-input-file | FileCheck %s
// RUN: mo-opt %s --allow-unregistered-dialect --canonicalize --cse --split-input-file | FileCheck %s --check-prefix=CSE-PLAN

// Original graph
// mo.graph @oneOp() -> (!mo.tensor<[8], f32>) {
//  %iota = "mo.constant"() {value = #M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0> : tensor<8xf32>} : () -> !mo.tensor<[8],f32>
//  %tens = "mo.constant"() {value = #M.dense_array<10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0> : tensor<8xf32>} : () -> !mo.tensor<[8], f32>  %1 = mo.mul(%iota, %tens) : !mo.tensor<[8], f32>
//  %2 = mo.mul(%1, %tens) : !mo.tensor<[8], f32>
//  %3 = mo.add(%2, %tens) : !mo.tensor<[8], f32>
//  mo.output %3 : !mo.tensor<[8], f32>
//}

module {
  mgp.model @basicMoGraph : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
    %0 = grt.chain.create ()
    %1 = mgp.context.create 0
    mef.output %0, %1, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
  } execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.tensor<8xf32>) {
    %0 = mgp.buffer.constant #M.aligned_bytes<"0x0000803F0000004000004040000080400000A0400000C0400000E04000000041", align 32>
    %1 = mgp.tensor_spec.create  : !mgp.tensor_spec<8xf32>
    %2 = mgp.tensor.create %arg0, %0, %1 : !mgp.tensor<8xf32>
    %3 = mgp.buffer.constant #M.aligned_bytes<"0x0000204100002041000020410000204100002041000020410000204100002041", align 32>
    %4 = mgp.tensor_spec.create  : !mgp.tensor_spec<8xf32>
    %5 = mgp.tensor.create %arg0, %3, %4 : !mgp.tensor<8xf32>
    %bufferResult = mgp.tensor.extract.buffer %2 : !mgp.tensor<8xf32>
    %tensorSpecResult = mgp.tensor.extract.tensor_spec %2 : <8xf32>
    %bufferResult_0 = mgp.tensor.extract.buffer %5 : !mgp.tensor<8xf32>
    %tensorSpecResult_1 = mgp.tensor.extract.tensor_spec %5 : <8xf32>
    %size = index.constant 32
    %6 = mgp.buffer.alloc %arg0, %size, %cpu : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
    %7 = mgp.tensor_spec.create  : !mgp.tensor_spec<8xf32>
    %8 = "mop.kernels.mul"(%arg0, %bufferResult, %tensorSpecResult, %bufferResult_0, %tensorSpecResult_1, %6, %7) : (!mef.chain, !mgp.buffer, !mgp.tensor_spec<8xf32>, !mgp.buffer, !mgp.tensor_spec<8xf32>, !mgp.buffer, !mgp.tensor_spec<8xf32>) -> !mef.chain
    %9 = mgp.tensor.create %8, %6, %7 : !mgp.tensor<8xf32>
    %bufferResult_2 = mgp.tensor.extract.buffer %9 : !mgp.tensor<8xf32>
    %tensorSpecResult_3 = mgp.tensor.extract.tensor_spec %9 : <8xf32>
    %bufferResult_4 = mgp.tensor.extract.buffer %5 : !mgp.tensor<8xf32>
    %tensorSpecResult_5 = mgp.tensor.extract.tensor_spec %5 : <8xf32>
    %10 = mgp.buffer.alloc %arg0, %size, %cpu : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
    %11 = mgp.tensor_spec.create  : !mgp.tensor_spec<8xf32>
    %12 = "mop.kernels.mul"(%8, %bufferResult_2, %tensorSpecResult_3, %bufferResult_4, %tensorSpecResult_5, %10, %11) : (!mef.chain, !mgp.buffer, !mgp.tensor_spec<8xf32>, !mgp.buffer, !mgp.tensor_spec<8xf32>, !mgp.buffer, !mgp.tensor_spec<8xf32>) -> !mef.chain
    %13 = mgp.tensor.create %12, %10, %11 : !mgp.tensor<8xf32>
    %bufferResult_6 = mgp.tensor.extract.buffer %13 : !mgp.tensor<8xf32>
    %tensorSpecResult_7 = mgp.tensor.extract.tensor_spec %13 : <8xf32>
    %bufferResult_8 = mgp.tensor.extract.buffer %5 : !mgp.tensor<8xf32>
    %tensorSpecResult_9 = mgp.tensor.extract.tensor_spec %5 : <8xf32>
    %14 = mgp.buffer.alloc %arg0, %size, %cpu : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
    %15 = mgp.tensor_spec.create  : !mgp.tensor_spec<8xf32>
    %16 = "mop.kernels.add"(%12, %bufferResult_6, %tensorSpecResult_7, %bufferResult_8, %tensorSpecResult_9, %14, %15) : (!mef.chain, !mgp.buffer, !mgp.tensor_spec<8xf32>, !mgp.buffer, !mgp.tensor_spec<8xf32>, !mgp.buffer, !mgp.tensor_spec<8xf32>) -> !mef.chain
    %17 = mgp.tensor.create %16, %14, %15 : !mgp.tensor<8xf32>
    mef.output %16, %17 : !mef.chain, !mgp.tensor<8xf32>
  }
}

// Check that we remove all but one of the spec creates.

// CHECK-LABEL: mgp.model @basicMoGraph
// CHECK: mgp.tensor_spec.create  : !mgp.tensor_spec<8xf32>
// CHECK-NOT: mgp.tensor_spec.create  : !mgp.tensor_spec<8xf32>

// -----

mgp.model @tensor_extract_fold : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 0
  mef.output %0, %1, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %in_spec: !mgp.tensor_spec<si32>, %in_buf: !mgp.buffer) -> (!mef.chain, !mgp.buffer, !mgp.tensor_spec<si32>) {
  %0 = mgp.tensor.create %arg0, %in_buf, %in_spec  : !mgp.tensor<si32>
  %buf = mgp.tensor.extract.buffer %0 : !mgp.tensor<si32>
  %spec = mgp.tensor.extract.tensor_spec %0 : <si32>
  mef.output %arg0, %buf, %spec : !mef.chain, !mgp.buffer, !mgp.tensor_spec<si32>
}

// CHECK-LABEL: mgp.model @tensor_extract_fold
// CHECK: } execute (%[[CHAIN:.*]]: !mef.chain,{{.*}}%[[SPEC:.*]]: !mgp.tensor_spec<si32>, %[[BUF:.*]]: !mgp.buffer) ->
// CHECK-NEXT: mef.output %[[CHAIN]], %[[BUF]], %[[SPEC]] :

// -----

// Test that two BufferPlanOps with the same sharing matrix and operands are
// deduplicated by CSE. The sharing matrix is the canonical representation, so
// no canonicalization pass is needed before CSE.
//
// CSE-PLAN-LABEL: mgp.model @buffer_plan_cse_after_canonicalize
// CSE-PLAN: } execute (
// CSE-PLAN: %[[HWM:.*]], %[[OFFS:.*]]:2 = mgp.buffer.plan
// CSE-PLAN-SAME: canShare = #M.dense_array<0, 0, 0, 0>
// CSE-PLAN-NOT: mgp.buffer.plan
mgp.model @buffer_plan_cse_after_canonicalize : mof devices [#M.device_ref<"gpu", 0>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"gpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  mef.output %0, %1, %arg0, %arg1 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.opaque<"weights_registry">, %arg3: !mgp.device_context<<"gpu", 0>>, %arg4: index, %arg5: index) -> !mef.chain {
  // Both ops have the same canShare matrix and operands — CSE deduplicates.
  %hwm0, %off0a, %off0b = mgp.buffer.plan(%arg4, %arg5) <
    alignments = #M.dense_array<16, 16> : !M.array<2xui64>,
    canShare = #M.dense_array<0, 0, 0, 0> : !M.array<4xui8>
  > : (index, index) -> (index, index, index)
  %hwm1, %off1a, %off1b = mgp.buffer.plan(%arg4, %arg5) <
    alignments = #M.dense_array<16, 16> : !M.array<2xui64>,
    canShare = #M.dense_array<0, 0, 0, 0> : !M.array<4xui8>
  > : (index, index) -> (index, index, index)
  %ch = "use"(%arg0, %off0a, %off0b, %off1a, %off1b) : (!mef.chain, index, index, index, index) -> !mef.chain
  mef.output %ch : !mef.chain
}

// -----

// Test that two BufferPlanOps with identical sharing matrices are deduplicated
// even without a canonicalization step.
//
// CSE-PLAN-LABEL: mgp.model @buffer_plan_cse_identical_sharing
// CSE-PLAN: } execute (
// CSE-PLAN: %[[HWM:.*]], %[[OFFS:.*]]:3 = mgp.buffer.plan
// CSE-PLAN-SAME: canShare = #M.dense_array<0, 0, 0, 0, 0, 0, 0, 0, 0>
// CSE-PLAN-NOT: mgp.buffer.plan
mgp.model @buffer_plan_cse_identical_sharing : mof devices [#M.device_ref<"gpu", 0>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"gpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  mef.output %0, %1, %arg0, %arg1 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.opaque<"weights_registry">, %arg3: !mgp.device_context<<"gpu", 0>>, %arg4: index, %arg5: index, %arg6: index) -> !mef.chain {
  %hwm0, %off0a, %off0b, %off0c = mgp.buffer.plan(%arg4, %arg5, %arg6) <
    alignments = #M.dense_array<16, 16, 16> : !M.array<3xui64>,
    canShare = #M.dense_array<0, 0, 0, 0, 0, 0, 0, 0, 0> : !M.array<9xui8>
  > : (index, index, index) -> (index, index, index, index)
  %hwm1, %off1a, %off1b, %off1c = mgp.buffer.plan(%arg4, %arg5, %arg6) <
    alignments = #M.dense_array<16, 16, 16> : !M.array<3xui64>,
    canShare = #M.dense_array<0, 0, 0, 0, 0, 0, 0, 0, 0> : !M.array<9xui8>
  > : (index, index, index) -> (index, index, index, index)
  %ch = "use"(%arg0, %off0a, %off0b, %off0c, %off1a, %off1b, %off1c) : (!mef.chain, index, index, index, index, index, index) -> !mef.chain
  mef.output %ch : !mef.chain
}
