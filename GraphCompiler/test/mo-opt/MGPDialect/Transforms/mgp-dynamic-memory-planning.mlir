// RUN: mo-opt %s --mgp-dynamic-memory-planning --split-input-file --allow-unregistered-dialect | FileCheck %s

// Test basic memory planning with non-overlapping lifetimes
// CHECK-LABEL: mgp.model @basic_memory_planning
mgp.model @basic_memory_planning devices [#M.device_ref<"gpu", 0>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"gpu", 0>>) {
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
mgp.model @cpu_allocations_ignored devices [#M.device_ref<"cpu", 0>, #M.device_ref<"gpu", 0>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"cpu", 0>>, %arg2: !mgp.device_context<<"gpu", 0>>) {
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
mgp.model @single_allocation_not_planned devices [#M.device_ref<"gpu", 0>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"gpu", 0>>) {
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
mgp.model @different_alignments devices [#M.device_ref<"gpu", 0>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"gpu", 0>>) {
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
mgp.model @multiple_devices devices [#M.device_ref<"gpu", 0>, #M.device_ref<"gpu", 1>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"gpu", 0>>, %arg2: !mgp.device_context<<"gpu", 1>>) {
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
mgp.model @tensor_view_extends_lifetime devices [#M.device_ref<"gpu", 0>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"gpu", 0>>) {
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

// -----

// A grt.if branch's allocations share one pool, allocated on the branch's
// chain argument.
// CHECK-LABEL: mgp.model @plan_in_control_flow_region
mgp.model @plan_in_control_flow_region devices [#M.device_ref<"gpu", 0>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"gpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  mef.output %0, %1, %arg0, %arg1 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.opaque<"weights_registry">, %arg3: !mgp.device_context<<"gpu", 0>>, %arg4: index, %arg5: i1) -> !mef.chain {
  // CHECK: grt.if %{{.*}} (%[[CHAIN:[^:]*]]: !mef.chain, %[[GPU_DEV:[^:]*]]: !mgp.device_context<<"gpu", 0>>, %[[SIZE:[^:]*]]: index)
  // CHECK: %[[HIGH_WATER:.*]], %[[OFFSETS:.*]]:2 = mgp.buffer.plan
  // CHECK: %[[POOL:.*]] = mgp.buffer.alloc %[[CHAIN]], %[[HIGH_WATER]], %[[GPU_DEV]], align 16
  // CHECK: mgp.buffer.bulk_slice %[[POOL]], %[[GPU_DEV]] offsets(%[[OFFSETS]]#0, %[[OFFSETS]]#1) sizes(%[[SIZE]], %[[SIZE]])
  // CHECK-NOT: mgp.buffer.alloc
  // CHECK: } else {
  %ifchain = grt.if %arg5 (%arg0: !mef.chain, %arg3: !mgp.device_context<<"gpu", 0>>, %arg4: index) -> (!mef.chain) {
    %alloc0 = mgp.buffer.alloc %arg0, %arg4, %arg3, align 16 : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
    %alloc1 = mgp.buffer.alloc %arg0, %arg4, %arg3, align 16 : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
    %ch0 = "use"(%arg0, %alloc0) : (!mef.chain, !mgp.buffer<<"gpu", 0>>) -> !mef.chain
    %ch1 = "use"(%ch0, %alloc1) : (!mef.chain, !mgp.buffer<<"gpu", 0>>) -> !mef.chain
    mef.output %ch1 : !mef.chain
  } else {
    mef.output %arg0 : !mef.chain
  }
  mef.output %ifchain : !mef.chain
}

// -----

// An if nested in a branch plans its own block: its pool is allocated on the
// inner branch's chain, and the outer branch's single allocation is left as is.
// CHECK-LABEL: mgp.model @plan_in_nested_control_flow_region
mgp.model @plan_in_nested_control_flow_region devices [#M.device_ref<"gpu", 0>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"gpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  mef.output %0, %1, %arg0, %arg1 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.opaque<"weights_registry">, %arg3: !mgp.device_context<<"gpu", 0>>, %arg4: index, %arg5: i1) -> !mef.chain {
  // CHECK: grt.if %{{.*}} (%[[OUTER_CHAIN:[^:]*]]: !mef.chain
  // CHECK-NOT: mgp.buffer.plan
  // CHECK: mgp.buffer.alloc %[[OUTER_CHAIN]]
  // CHECK: grt.if %{{.*}} (%[[INNER_CHAIN:[^:]*]]: !mef.chain, %[[GPU_DEV:[^:]*]]: !mgp.device_context<<"gpu", 0>>
  // CHECK: %[[HIGH_WATER:.*]], %{{.*}}:2 = mgp.buffer.plan
  // CHECK: mgp.buffer.alloc %[[INNER_CHAIN]], %[[HIGH_WATER]], %[[GPU_DEV]], align 16
  // CHECK: mgp.buffer.bulk_slice
  // CHECK-NOT: mgp.buffer.alloc
  // CHECK: } else {
  %ifchain = grt.if %arg5 (%arg0: !mef.chain, %arg3: !mgp.device_context<<"gpu", 0>>, %arg4: index, %arg5: i1) -> (!mef.chain) {
    %solo = mgp.buffer.alloc %arg0, %arg4, %arg3, align 16 : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
    %ch0 = "use"(%arg0, %solo) : (!mef.chain, !mgp.buffer<<"gpu", 0>>) -> !mef.chain
    %inner = grt.if %arg5 (%ch0: !mef.chain, %arg3: !mgp.device_context<<"gpu", 0>>, %arg4: index) -> (!mef.chain) {
      %alloc0 = mgp.buffer.alloc %ch0, %arg4, %arg3, align 16 : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
      %alloc1 = mgp.buffer.alloc %ch0, %arg4, %arg3, align 16 : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
      %ch1 = "use"(%ch0, %alloc0) : (!mef.chain, !mgp.buffer<<"gpu", 0>>) -> !mef.chain
      %ch2 = "use"(%ch1, %alloc1) : (!mef.chain, !mgp.buffer<<"gpu", 0>>) -> !mef.chain
      mef.output %ch2 : !mef.chain
    } else {
      mef.output %ch0 : !mef.chain
    }
    mef.output %inner : !mef.chain
  } else {
    mef.output %arg0 : !mef.chain
  }
  mef.output %ifchain : !mef.chain
}

// -----

// A branch with allocations on two devices gets one pool per device, each on
// its device's context.
// CHECK-LABEL: mgp.model @plan_per_device_in_control_flow_region
mgp.model @plan_per_device_in_control_flow_region devices [#M.device_ref<"gpu", 0>, #M.device_ref<"gpu", 1>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"gpu", 0>>, %arg2: !mgp.device_context<<"gpu", 1>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  mef.output %0, %1, %arg0, %arg1, %arg2 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>, !mgp.device_context<<"gpu", 1>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.opaque<"weights_registry">, %arg3: !mgp.device_context<<"gpu", 0>>, %arg4: !mgp.device_context<<"gpu", 1>>, %arg5: index, %arg6: i1) -> !mef.chain {
  // CHECK: grt.if %{{.*}} (%[[CHAIN:[^:]*]]: !mef.chain, %[[GPU0:[^:]*]]: !mgp.device_context<<"gpu", 0>>, %[[GPU1:[^:]*]]: !mgp.device_context<<"gpu", 1>>
  // CHECK-DAG: mgp.buffer.alloc %[[CHAIN]], %{{.*}}, %[[GPU0]], align 16
  // CHECK-DAG: mgp.buffer.alloc %[[CHAIN]], %{{.*}}, %[[GPU1]], align 16
  // CHECK-DAG: mgp.buffer.bulk_slice %{{.*}}, %[[GPU0]]
  // CHECK-DAG: mgp.buffer.bulk_slice %{{.*}}, %[[GPU1]]
  // CHECK: } else {
  %ifchain = grt.if %arg6 (%arg0: !mef.chain, %arg3: !mgp.device_context<<"gpu", 0>>, %arg4: !mgp.device_context<<"gpu", 1>>, %arg5: index) -> (!mef.chain) {
    %a0 = mgp.buffer.alloc %arg0, %arg5, %arg3, align 16 : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
    %a1 = mgp.buffer.alloc %arg0, %arg5, %arg3, align 16 : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
    %b0 = mgp.buffer.alloc %arg0, %arg5, %arg4, align 16 : !mgp.buffer<<"gpu", 1>>, !mgp.device_context<<"gpu", 1>>
    %b1 = mgp.buffer.alloc %arg0, %arg5, %arg4, align 16 : !mgp.buffer<<"gpu", 1>>, !mgp.device_context<<"gpu", 1>>
    %ch0 = "use"(%arg0, %a0, %b0) : (!mef.chain, !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 1>>) -> !mef.chain
    %ch1 = "use"(%ch0, %a1, %b1) : (!mef.chain, !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 1>>) -> !mef.chain
    mef.output %ch1 : !mef.chain
  } else {
    mef.output %arg0 : !mef.chain
  }
  mef.output %ifchain : !mef.chain
}

// -----

// A grt.while body's allocations share one pool, allocated on the body's
// chain argument.
// CHECK-LABEL: mgp.model @plan_in_while_body
mgp.model @plan_in_while_body devices [#M.device_ref<"gpu", 0>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"gpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  mef.output %0, %1, %arg0, %arg1 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.opaque<"weights_registry">, %arg3: !mgp.device_context<<"gpu", 0>>, %arg4: index, %arg5: i1) -> !mef.chain {
  // The body's block arguments print as %arg0 (chain) and %arg3 (device).
  // CHECK: grt.while
  // CHECK: } do {
  // CHECK-NEXT: %[[HIGH_WATER:.*]], %{{.*}}:2 = mgp.buffer.plan
  // CHECK-NEXT: mgp.buffer.alloc %arg0, %[[HIGH_WATER]], %arg3, align 16
  // CHECK-NEXT: mgp.buffer.bulk_slice
  // CHECK-NOT: mgp.buffer.alloc
  // CHECK: mef.output
  %ch, %dev, %size, %flag = grt.while (%arg0: !mef.chain, %arg3: !mgp.device_context<<"gpu", 0>>, %arg4: index, %arg5: i1) {
    mef.output %arg0, %arg3, %arg4, %arg5, %arg5 : !mef.chain, !mgp.device_context<<"gpu", 0>>, index, i1, i1
  } do {
    %alloc0 = mgp.buffer.alloc %arg0, %arg4, %arg3, align 16 : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
    %alloc1 = mgp.buffer.alloc %arg0, %arg4, %arg3, align 16 : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
    %ch0 = "use"(%arg0, %alloc0) : (!mef.chain, !mgp.buffer<<"gpu", 0>>) -> !mef.chain
    %ch1 = "use"(%ch0, %alloc1) : (!mef.chain, !mgp.buffer<<"gpu", 0>>) -> !mef.chain
    mef.output %ch1, %arg3, %arg4, %arg5 : !mef.chain, !mgp.device_context<<"gpu", 0>>, index, i1
  }
  mef.output %ch : !mef.chain
}
