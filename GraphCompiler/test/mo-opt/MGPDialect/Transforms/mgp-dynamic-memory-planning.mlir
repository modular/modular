// RUN: mo-opt %s --mgp-dynamic-memory-planning --split-input-file --allow-unregistered-dialect | FileCheck %s

// Test basic memory planning with non-overlapping lifetimes
// CHECK-LABEL: mgp.model @basic_memory_planning
mgp.model @basic_memory_planning : mof devices [#M.device_ref<"gpu", 0>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"gpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  mef.output %0, %1, %arg0, %arg1 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.opaque<"weights_registry">, %arg3: !mgp.device_context<<"gpu", 0>>, %arg4: index) -> !mef.chain {
  // CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, %[[WEIGHTS:.*]]: !mgp.opaque<"weights_registry">, %[[GPU_DEV:.*]]: !mgp.device_context<<"gpu", 0>>, %[[SIZE:.*]]: index) -> !mef.chain {

  // These allocations should be merged into a single pool
  // CHECK: %[[HIGH_WATER:.*]], %[[OFFSETS:.*]]:2 = mgp.buffer.plan
  // CHECK: %[[POOL:.*]] = mgp.buffer.alloc %[[CHAIN]], %[[HIGH_WATER]], %[[GPU_DEV]], align 16 : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  // CHECK: %[[SLICES:.*]]:2 = mgp.buffer.bulk_slice %[[POOL]], %[[GPU_DEV]] offsets(%[[OFFSETS]]#0, %[[OFFSETS]]#1) sizes(%[[SIZE]], %[[SIZE]]) : <<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  // CHECK-NOT: mgp.buffer.slice

  %alloc0 = mgp.buffer.alloc %arg0, %arg4, %arg3, align 16 : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %alloc1 = mgp.buffer.alloc %arg0, %arg4, %arg3, align 16 : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>

  // Sequential use pattern
  %ch0 = "use"(%arg0, %alloc0) : (!mef.chain, !mgp.buffer<<"gpu", 0>>) -> !mef.chain
  %ch1 = "use"(%ch0, %alloc1) : (!mef.chain, !mgp.buffer<<"gpu", 0>>) -> !mef.chain

  mef.output %ch1 : !mef.chain
}

// -----

// Test that CPU allocations are ignored
// CHECK-LABEL: mgp.model @cpu_allocations_ignored
mgp.model @cpu_allocations_ignored : mof devices [#M.device_ref<"cpu", 0>, #M.device_ref<"gpu", 0>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"cpu", 0>>, %arg2: !mgp.device_context<<"gpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  mef.output %0, %1, %arg0, %arg1, %arg2 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>, !mgp.device_context<<"gpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.opaque<"weights_registry">, %arg3: !mgp.device_context<<"cpu", 0>>, %arg4: !mgp.device_context<<"gpu", 0>>, %arg5: index) -> !mef.chain {
  // CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, %[[WEIGHTS:.*]]: !mgp.opaque<"weights_registry">, %[[CPU_DEV:.*]]: !mgp.device_context<<"cpu", 0>>, %[[GPU_DEV:.*]]: !mgp.device_context<<"gpu", 0>>, %[[SIZE:.*]]: index) -> !mef.chain {
  // GPU allocations should be planned
  // CHECK: %[[GPU_HIGH_WATER:.*]], %[[GPU_OFFSETS:.*]]:2 = mgp.buffer.plan
  // CHECK: %[[GPU_POOL:.*]] = mgp.buffer.alloc %[[CHAIN]], %[[GPU_HIGH_WATER]], %[[GPU_DEV]], align 16 : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  // CHECK: %[[GPU_SLICES:.*]]:2 = mgp.buffer.bulk_slice %[[GPU_POOL]], %[[GPU_DEV]] offsets(%[[GPU_OFFSETS]]#0, %[[GPU_OFFSETS]]#1) sizes(%[[SIZE]], %[[SIZE]]) : <<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %gpu_alloc0 = mgp.buffer.alloc %arg0, %arg5, %arg4, align 16 : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %gpu_alloc1 = mgp.buffer.alloc %arg0, %arg5, %arg4, align 16 : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>

  // CPU allocations should remain untouched
  // CHECK: %[[CPU_ALLOC:.*]] = mgp.buffer.alloc %[[CHAIN]], %[[SIZE]], %[[CPU_DEV]], align 16 : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %cpu_alloc = mgp.buffer.alloc %arg0, %arg5, %arg3, align 16 : !mgp.buffer, !mgp.device_context<<"cpu", 0>>

  %ch0 = "use"(%arg0, %cpu_alloc) : (!mef.chain, !mgp.buffer) -> !mef.chain
  %ch1 = "use"(%arg0, %gpu_alloc0) : (!mef.chain, !mgp.buffer<<"gpu", 0>>) -> !mef.chain
  %ch2 = "use"(%ch1, %gpu_alloc1) : (!mef.chain, !mgp.buffer<<"gpu", 0>>) -> !mef.chain
  %ch3 = grt.chain.create (%ch0, %ch2)

  mef.output %ch3 : !mef.chain
}

// -----

// Test single allocation is not planned
// CHECK-LABEL: mgp.model @single_allocation_not_planned
mgp.model @single_allocation_not_planned : mof devices [#M.device_ref<"gpu", 0>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"gpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  mef.output %0, %1, %arg0, %arg1 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.opaque<"weights_registry">, %arg3: !mgp.device_context<<"gpu", 0>>, %arg4: index) -> !mef.chain {
  // CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, %[[WEIGHTS:.*]]: !mgp.opaque<"weights_registry">, %[[GPU_DEV:.*]]: !mgp.device_context<<"gpu", 0>>, %[[SIZE:.*]]: index) -> !mef.chain {
  // Single allocation should remain as-is
  // CHECK-NOT: mgp.buffer.plan
  // CHECK: %[[SINGLE_ALLOC:.*]] = mgp.buffer.alloc %[[CHAIN]], %[[SIZE]], %[[GPU_DEV]], align 16 : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %alloc = mgp.buffer.alloc %arg0, %arg4, %arg3, align 16 : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>

  %ch = "use"(%arg0, %alloc) : (!mef.chain, !mgp.buffer<<"gpu", 0>>) -> !mef.chain

  mef.output %ch : !mef.chain
}

// -----

// Test different alignments
// CHECK-LABEL: mgp.model @different_alignments
mgp.model @different_alignments : mof devices [#M.device_ref<"gpu", 0>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"gpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  mef.output %0, %1, %arg0, %arg1 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.opaque<"weights_registry">, %arg3: !mgp.device_context<<"gpu", 0>>, %arg4: index) -> !mef.chain {
  // CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, %[[WEIGHTS:.*]]: !mgp.opaque<"weights_registry">, %[[GPU_DEV:.*]]: !mgp.device_context<<"gpu", 0>>, %[[SIZE:.*]]: index) -> !mef.chain {
  // Pool should use LCM alignment (64)
  // CHECK: %[[ALIGN_HIGH_WATER:.*]], %[[ALIGN_OFFSETS:.*]]:3 = mgp.buffer.plan
  // CHECK-SAME: alignments = #M.dense_array<16, 32, 64>
  // CHECK: %[[ALIGN_POOL:.*]] = mgp.buffer.alloc %[[CHAIN]], %[[ALIGN_HIGH_WATER]], %[[GPU_DEV]], align 64 : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  // CHECK: %[[ALIGN_SLICES:.*]]:3 = mgp.buffer.bulk_slice %[[ALIGN_POOL]], %[[GPU_DEV]] offsets(%[[ALIGN_OFFSETS]]#0, %[[ALIGN_OFFSETS]]#1, %[[ALIGN_OFFSETS]]#2) sizes(%[[SIZE]], %[[SIZE]], %[[SIZE]]) : <<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %alloc1 = mgp.buffer.alloc %arg0, %arg4, %arg3, align 16 : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %alloc2 = mgp.buffer.alloc %arg0, %arg4, %arg3, align 32 : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %alloc3 = mgp.buffer.alloc %arg0, %arg4, %arg3, align 64 : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>

  %ch0 = "use"(%arg0, %alloc1) : (!mef.chain, !mgp.buffer<<"gpu", 0>>) -> !mef.chain
  %ch1 = "use"(%ch0, %alloc2) : (!mef.chain, !mgp.buffer<<"gpu", 0>>) -> !mef.chain
  %ch2 = "use"(%ch1, %alloc3) : (!mef.chain, !mgp.buffer<<"gpu", 0>>) -> !mef.chain

  mef.output %ch2 : !mef.chain
}

// -----

// Test multiple devices are handled separately
// CHECK-LABEL: mgp.model @multiple_devices
mgp.model @multiple_devices : mof devices [#M.device_ref<"gpu", 0>, #M.device_ref<"gpu", 1>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"gpu", 0>>, %arg2: !mgp.device_context<<"gpu", 1>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  mef.output %0, %1, %arg0, %arg1, %arg2 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>, !mgp.device_context<<"gpu", 1>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.opaque<"weights_registry">, %arg3: !mgp.device_context<<"gpu", 0>>, %arg4: !mgp.device_context<<"gpu", 1>>, %arg5: index) -> !mef.chain {
  // CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, %[[WEIGHTS:.*]]: !mgp.opaque<"weights_registry">, %[[GPU0_DEV:.*]]: !mgp.device_context<<"gpu", 0>>, %[[GPU1_DEV:.*]]: !mgp.device_context<<"gpu", 1>>, %[[SIZE:.*]]: index) -> !mef.chain {
  // Each device should get its own memory pool
  // CHECK-DAG: %[[GPU0_HIGH_WATER:.*]], %[[GPU0_OFFSETS:.*]]:2 = mgp.buffer.plan
  // CHECK-DAG: %[[GPU0_POOL:.*]] = mgp.buffer.alloc %[[CHAIN]], %[[GPU0_HIGH_WATER]], %[[GPU0_DEV]], align 16 : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  // CHECK-DAG: %[[GPU1_HIGH_WATER:.*]], %[[GPU1_OFFSETS:.*]]:2 = mgp.buffer.plan
  // CHECK-DAG: %[[GPU1_POOL:.*]] = mgp.buffer.alloc %[[CHAIN]], %[[GPU1_HIGH_WATER]], %[[GPU1_DEV]], align 16 : !mgp.buffer<<"gpu", 1>>, !mgp.device_context<<"gpu", 1>>

  %alloc0_gpu0 = mgp.buffer.alloc %arg0, %arg5, %arg3, align 16 : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %alloc1_gpu0 = mgp.buffer.alloc %arg0, %arg5, %arg3, align 16 : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>

  %alloc0_gpu1 = mgp.buffer.alloc %arg0, %arg5, %arg4, align 16 : !mgp.buffer<<"gpu", 1>>, !mgp.device_context<<"gpu", 1>>
  %alloc1_gpu1 = mgp.buffer.alloc %arg0, %arg5, %arg4, align 16 : !mgp.buffer<<"gpu", 1>>, !mgp.device_context<<"gpu", 1>>

  // CHECK-DAG: mgp.buffer.bulk_slice %[[GPU0_POOL]], %[[GPU0_DEV]] offsets(%[[GPU0_OFFSETS]]#0, %[[GPU0_OFFSETS]]#1) sizes(%[[SIZE]], %[[SIZE]]) : <<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  // CHECK-DAG: mgp.buffer.bulk_slice %[[GPU1_POOL]], %[[GPU1_DEV]] offsets(%[[GPU1_OFFSETS]]#0, %[[GPU1_OFFSETS]]#1) sizes(%[[SIZE]], %[[SIZE]]) : <<"gpu", 1>>, !mgp.device_context<<"gpu", 1>>

  %ch0 = "use"(%arg0, %alloc0_gpu0) : (!mef.chain, !mgp.buffer<<"gpu", 0>>) -> !mef.chain
  %ch1 = "use"(%ch0, %alloc1_gpu0) : (!mef.chain, !mgp.buffer<<"gpu", 0>>) -> !mef.chain
  %ch2 = "use"(%arg0, %alloc0_gpu1) : (!mef.chain, !mgp.buffer<<"gpu", 1>>) -> !mef.chain
  %ch3 = "use"(%ch2, %alloc1_gpu1) : (!mef.chain, !mgp.buffer<<"gpu", 1>>) -> !mef.chain
  %ch4 = grt.chain.create (%ch1, %ch3)

  mef.output %ch4 : !mef.chain
}

// -----

// A GPU buffer reached only through a tensor view (tensor.create ->
// tensor.extract.buffer) must stay live until the view's consumers, not just
// the view ops. Here %allocA is used before (%ch0) and after (%ch2) %allocB's
// use (%ch1), so their lifetimes overlap and the planner must not share their
// slot: canShare stays all-zero.
// CHECK-LABEL: mgp.model @tensor_view_extends_lifetime
mgp.model @tensor_view_extends_lifetime : mof devices [#M.device_ref<"gpu", 0>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"gpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  mef.output %0, %1, %arg0, %arg1 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.opaque<"weights_registry">, %arg3: !mgp.device_context<<"gpu", 0>>, %arg4: index) -> !mef.chain {
  // CHECK: mgp.buffer.plan
  // CHECK-SAME: canShare = #M.dense_array<0, 0, 0, 0>
  %allocA = mgp.buffer.alloc %arg0, %arg4, %arg3, align 16 : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %spec = mgp.tensor_spec.create : !mgp.tensor_spec<4xsi64>
  %tA = mgp.tensor.create %arg0, %allocA, %spec : !mgp.tensor<4xsi64, <"gpu", 0>>
  %bA = mgp.tensor.extract.buffer %tA : <4xsi64, <"gpu", 0>>

  %ch0 = "use"(%arg0, %bA) : (!mef.chain, !mgp.buffer<<"gpu", 0>>) -> !mef.chain
  %allocB = mgp.buffer.alloc %ch0, %arg4, %arg3, align 16 : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %ch1 = "use"(%ch0, %allocB) : (!mef.chain, !mgp.buffer<<"gpu", 0>>) -> !mef.chain
  %ch2 = "use"(%ch1, %bA) : (!mef.chain, !mgp.buffer<<"gpu", 0>>) -> !mef.chain

  mef.output %ch2 : !mef.chain
}
