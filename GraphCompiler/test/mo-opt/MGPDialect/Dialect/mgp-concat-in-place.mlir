// RUN: mo-opt %s --allow-unregistered-dialect --mgp-concat-in-place --canonicalize --cse | FileCheck %s

//
// @dynamic_concat
//

// CHECK-LABEL: mgp.model @dynamic_concat
// CHECK: } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context
// CHECK: %[[IDX_0:.*]] = index.constant 0
// CHECK: %[[TOTAL_SIZE:.*]] = index.constant 128
// CHECK: %[[SPLIT_SIZE:.*]] = index.constant 64
// CHECK: %[[OUT_BUF:.*]] = mgp.buffer.alloc %[[CH0]], %[[TOTAL_SIZE]]

// CHECK: %[[SLICE0:.*]] = mgp.buffer.slice %[[OUT_BUF]], %[[IDX_0]], %[[SPLIT_SIZE]]

// CHECK: %[[CH2:.*]] = "add"(%[[CH0]], %{{.*}}, %{{.*}}, %[[SLICE0]])
// CHECK: %[[SLICE1:.*]] = mgp.buffer.slice %[[OUT_BUF]], %[[SPLIT_SIZE]], %[[SPLIT_SIZE]]
// CHECK: %[[CH3:.*]] = "sub"(%[[CH0]], %{{.*}}, %{{.*}}, %[[SLICE1]])
// CHECK: %[[CH4:.*]] = grt.chain.create (%[[CH2]], %[[CH3]])
// CHECK: %[[TEN:.*]] = mgp.tensor.create %[[CH4]], %[[OUT_BUF]]
// CHECK: mef.output %[[CH0]], %[[TEN]]

mgp.model @dynamic_concat : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.tensor<32xf32>) {
  %0 = mgp.buffer.constant #M.aligned_bytes<"0x0000803F0000004000004040000080400000A0400000C0400000E04000000041", align 32>
  %2 = mgp.buffer.constant #M.aligned_bytes<"0x0000204100002041000020410000204100002041000020410000204100002041", align 32>
  %const_64 = index.constant 64
  %4 = mgp.buffer.alloc %ch0, %const_64, %cpu : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %const_8 = index.constant 8
  %5 = mgp.tensor_spec.create %const_8 : !mgp.tensor_spec<?xf32>
  %6 = "add"(%ch0, %0, %2, %4) {tempInOutSig = #M.inout_sig<"iiio">} : (!mef.chain, !mgp.buffer, !mgp.buffer, !mgp.buffer) -> !mef.chain
  %10 = mgp.buffer.alloc %ch0, %const_64, %cpu : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %12 = "sub"(%ch0, %0, %2, %10) {tempInOutSig = #M.inout_sig<"iiio">} : (!mef.chain, !mgp.buffer, !mgp.buffer, !mgp.buffer) -> !mef.chain
  %14 = mgp.tensor_spec.create : !mgp.tensor_spec<32xf32>
  %const_128 = index.constant 128
  %15 = mgp.buffer.alloc %ch0, %const_128, %cpu : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %16 = grt.chain.create(%6: !mef.chain, %12: !mef.chain)
  %17 = mgp.buffer.concat(%16, %15, %4, %10, %cpu) : (!mef.chain, !mgp.buffer, !mgp.buffer, !mgp.buffer, !mgp.device_context<<"cpu", 0>>) -> !mef.chain
  %18 = mgp.tensor.create %17, %15, %14 : !mgp.tensor<32xf32>
  mef.output %ch0, %18 : !mef.chain, !mgp.tensor<32xf32>
}

//
// @concat_with_shared_input_buffer
//

// CHECK-LABEL: mgp.model @concat_with_shared_input_buffer
// CHECK: %{{.*}} = mgp.buffer.slice
// CHECK: %{{.*}} = mgp.buffer.slice

mgp.model @concat_with_shared_input_buffer : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.tensor<32xf32>) {
  %0 = mgp.buffer.constant #M.aligned_bytes<"0x0000803F0000004000004040000080400000A0400000C0400000E04000000041", align 32>
  %2 = mgp.buffer.constant #M.aligned_bytes<"0x0000204100002041000020410000204100002041000020410000204100002041", align 32>
  %size_64 = index.constant 64
  %4 = mgp.buffer.alloc %ch0, %size_64, %cpu : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %6 = "add"(%ch0, %0, %2, %4) {tempInOutSig = #M.inout_sig<"iiio">} : (!mef.chain, !mgp.buffer, !mgp.buffer, !mgp.buffer) -> !mef.chain
  %10 = mgp.buffer.alloc %ch0, %size_64, %cpu : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %12 = "sub"(%6, %4, %2, %10) {tempInOutSig = #M.inout_sig<"iiio">} : (!mef.chain, !mgp.buffer, !mgp.buffer, !mgp.buffer) -> !mef.chain
  %size_128  = index.constant 128
  %14 = mgp.buffer.alloc %ch0, %size_128, %cpu : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %15 = mgp.tensor_spec.create : !mgp.tensor_spec<32xf32>
  %16 = grt.chain.create(%6: !mef.chain, %12: !mef.chain)
  %17 = mgp.buffer.concat(%16, %14, %4, %10, %cpu) : (!mef.chain, !mgp.buffer, !mgp.buffer, !mgp.buffer, !mgp.device_context<<"cpu", 0>>) -> !mef.chain
  %18 = mgp.tensor.create %17, %14, %15 : !mgp.tensor<32xf32>
  mef.output %ch0, %18 : !mef.chain, !mgp.tensor<32xf32>
}

//
// @non_trivial_chains
//

// CHECK-LABEL: mgp.model @non_trivial_chains
// CHECK: %{{.*}} = mgp.buffer.slice
// CHECK: %{{.*}} = mgp.buffer.slice

mgp.model @non_trivial_chains : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.tensor<32xf32>) {
  %0 = mgp.buffer.constant #M.aligned_bytes<"0x0000803F0000004000004040000080400000A0400000C0400000E04000000041", align 32>
  %2 = mgp.buffer.constant #M.aligned_bytes<"0x0000204100002041000020410000204100002041000020410000204100002041", align 32>
  %size_64 = index.constant 64
  %4 = mgp.buffer.alloc %ch0, %size_64, %cpu : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %6 = "add"(%ch0, %0, %2, %4) {tempInOutSig = #M.inout_sig<"iiio">} : (!mef.chain, !mgp.buffer, !mgp.buffer, !mgp.buffer) -> !mef.chain
  %10 = mgp.buffer.alloc %6, %size_64, %cpu : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %12 = "sub"(%6, %0, %2, %10) {numBacOutputBuffers = 1 : ui32} : (!mef.chain, !mgp.buffer, !mgp.buffer, !mgp.buffer) -> !mef.chain
  %size_128 = index.constant 128
  %14 = mgp.buffer.alloc %12, %size_128, %cpu : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %15 = mgp.tensor_spec.create : !mgp.tensor_spec<32xf32>
  %17 = mgp.buffer.concat(%12, %14, %4, %10, %cpu) : (!mef.chain, !mgp.buffer, !mgp.buffer, !mgp.buffer, !mgp.device_context<<"cpu", 0>>) -> !mef.chain
  %18 = mgp.tensor.create %17, %14, %15 : !mgp.tensor<32xf32>
  mef.output %ch0, %18 : !mef.chain, !mgp.tensor<32xf32>
}

//
// @dont_trigger_since_duplicate_buffers
//

// CHECK-LABEL: mgp.model @dont_trigger_since_duplicate_buffers
// CHECK: %{{.*}} = mgp.buffer.concat

mgp.model @dont_trigger_since_duplicate_buffers : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.tensor<32xf32>) {
  %0 = mgp.buffer.constant #M.aligned_bytes<"0x0000803F0000004000004040000080400000A0400000C0400000E04000000041", align 32>
  %2 = mgp.buffer.constant #M.aligned_bytes<"0x0000204100002041000020410000204100002041000020410000204100002041", align 32>
  %size_64 = index.constant 64
  %4 = mgp.buffer.alloc %ch0, %size_64, %cpu : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %6 = "add"(%ch0, %0, %2, %4) {tempInOutSig = #M.inout_sig<"iiio">} : (!mef.chain, !mgp.buffer, !mgp.buffer, !mgp.buffer) -> !mef.chain
  %size_128 = index.constant 128
  %14 = mgp.buffer.alloc %ch0, %size_128, %cpu : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %15 = mgp.tensor_spec.create : !mgp.tensor_spec<32xf32>
  %17 = mgp.buffer.concat(%6, %14, %4, %4, %cpu) : (!mef.chain, !mgp.buffer, !mgp.buffer, !mgp.buffer, !mgp.device_context<<"cpu", 0>>) -> !mef.chain
  %18 = mgp.tensor.create %17, %14, %15 : !mgp.tensor<32xf32>
  mef.output %ch0, %18 : !mef.chain, !mgp.tensor<32xf32>
}

// A dynamic extent scaled by a per-row byte stride the alignment divides keeps
// every slice offset aligned, so the rewrite still fires.

// CHECK-LABEL: mgp.model @dynamic_same_size
// CHECK: } execute ({{.*}})
// CHECK-DAG: %[[ROWS:.*]] = "dynamic_size"()
// CHECK-DAG: %[[SIZE:.*]] = index.mul %[[ROWS]], %{{.*}}
// CHECK-DAG: %[[OFFSET2:.*]] = index.mul %[[ROWS]], %{{.*}}
// CHECK-DAG: %{{.*}} = mgp.buffer.slice %{{.*}}, %[[SIZE]], %[[SIZE]]
// CHECK-DAG: %{{.*}} = mgp.buffer.slice %{{.*}}, %[[OFFSET2]], %[[SIZE]]
// CHECK: mef.output

mgp.model @dynamic_same_size : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.tensor<32xf32>) {
  // Dynamic sizes for the allocations
  %rows = "dynamic_size"() : () -> (index)
  %row_bytes = index.constant 128
  %alloc_size = index.mul %rows, %row_bytes

  // Allocations to concat
  %alloc_0 = mgp.buffer.alloc %ch0, %alloc_size, %cpu : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %alloc_1 = mgp.buffer.alloc %ch0, %alloc_size, %cpu : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %alloc_2 = mgp.buffer.alloc %ch0, %alloc_size, %cpu : !mgp.buffer, !mgp.device_context<<"cpu", 0>>

  // Need a use to keep the generated slice ops around
  %eff0 = "populate"(%alloc_0) : (!mgp.buffer) -> (!mef.chain)
  %eff1 = "populate"(%alloc_1) : (!mgp.buffer) -> (!mef.chain)
  %eff2 = "populate"(%alloc_2) : (!mgp.buffer) -> (!mef.chain)

  %merged = grt.chain.create(%ch0, %eff0, %eff1, %eff2)

  %spec = mgp.tensor_spec.create : !mgp.tensor_spec<32xf32>
  %const_64 = index.constant 64

  // Allocation for the concat result
  %total_size = "total_size"() : () -> (index)
  %out_buf = mgp.buffer.alloc %merged, %total_size, %cpu : !mgp.buffer, !mgp.device_context<<"cpu", 0>>

  %ch1 = mgp.buffer.concat(%merged, %out_buf, %alloc_0, %alloc_1, %alloc_2, %cpu) : (!mef.chain, !mgp.buffer, !mgp.buffer, !mgp.buffer, !mgp.buffer, !mgp.device_context<<"cpu", 0>>) -> !mef.chain

  %tensor = mgp.tensor.create %ch1, %out_buf, %spec : !mgp.tensor<32xf32>
  mef.output %ch1, %tensor : !mef.chain, !mgp.tensor<32xf32>
}

//
// @dont_trigger_since_offset_alignment_unknown
//
// Slicing an operand to an offset the device preferred alignment does not
// divide would hand its producer a base weaker than the alignment every kernel
// tensor is told it has, which faults rather than merely slowing down.

// CHECK-LABEL: mgp.model @dont_trigger_since_offset_alignment_unknown
// CHECK: %{{.*}} = mgp.buffer.concat
// CHECK-NOT: %{{.*}} = mgp.buffer.slice

mgp.model @dont_trigger_since_offset_alignment_unknown : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.tensor<32xf32>) {
  %alloc_size = "dynamic_size"() : () -> (index)
  %alloc_0 = mgp.buffer.alloc %ch0, %alloc_size, %cpu : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %alloc_1 = mgp.buffer.alloc %ch0, %alloc_size, %cpu : !mgp.buffer, !mgp.device_context<<"cpu", 0>>

  %eff0 = "populate"(%alloc_0) : (!mgp.buffer) -> (!mef.chain)
  %eff1 = "populate"(%alloc_1) : (!mgp.buffer) -> (!mef.chain)
  %merged = grt.chain.create(%ch0, %eff0, %eff1)

  %spec = mgp.tensor_spec.create : !mgp.tensor_spec<32xf32>
  %total_size = "total_size"() : () -> (index)
  %out_buf = mgp.buffer.alloc %merged, %total_size, %cpu : !mgp.buffer, !mgp.device_context<<"cpu", 0>>

  %ch1 = mgp.buffer.concat(%merged, %out_buf, %alloc_0, %alloc_1, %cpu) : (!mef.chain, !mgp.buffer, !mgp.buffer, !mgp.buffer, !mgp.device_context<<"cpu", 0>>) -> !mef.chain

  %tensor = mgp.tensor.create %ch1, %out_buf, %spec : !mgp.tensor<32xf32>
  mef.output %ch1, %tensor : !mef.chain, !mgp.tensor<32xf32>
}
