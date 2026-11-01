// RUN: mo-opt -allow-unregistered-dialect --mgp-lower-primitives %s | FileCheck %s

//--- Test lowering of MGP

// CHECK-LABEL: mgp.model @mgp_assert
// CHECK: mogg.routine.async
// CHECK: mgp.assert
// CHECK: mogg.output
mgp.model @mgp_assert : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
    %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, i1) {
  %ch0 = grt.chain.create ()
  %cond = index.bool.constant true
  %ch = mgp.assert %cond, "ASSERT"
  mef.output %ch, %cond: !mef.chain, i1
}

// CHECK-LABEL: mgp.model @mgp_alloc_cpu
// CHECK: mogg.routine.async
// CHECK: mgp.buffer.alloc
// CHECK: mogg.output
mgp.model @mgp_alloc_cpu : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
    %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {
  %size = index.constant 4
  %buf0 = mgp.buffer.alloc %arg0, %size, %cpu : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  mef.output %arg0: !mef.chain
}

// CHECK-LABEL: mgp.model @mgp_tensor_spec_create_static
// CHECK: mogg.routine.async
// CHECK: mgp.tensor_spec.create
// CHECK: mogg.output
mgp.model @mgp_tensor_spec_create_static : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain) {
  %spec = mgp.tensor_spec.create : !mgp.tensor_spec<2x3x2xui8>
  mef.output %ch0 : !mef.chain
}

// CHECK-LABEL: mgp.model @mgp_tensor_spec_create_dynamic
// CHECK: mogg.routine.async
// CHECK: mgp.tensor_spec.create
// CHECK: mogg.output
mgp.model @mgp_tensor_spec_create_dynamic : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {
  %size = index.constant 2
  %spec = mgp.tensor_spec.create %size : !mgp.tensor_spec<2x?xui8>
  mef.output %ch0 : !mef.chain
}

// CHECK-LABEL: mgp.model @mgp_device_profile
mgp.model @mgp_device_profile : mof devices [#M.device_ref<"cpu", 0>, #M.device_ref<"gpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>, !mgp.device_context<<"gpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %gpu: !mgp.device_context<<"gpu", 0>>) -> (!mef.chain) {
  %ch1 = mgp.device.context.profile.start %arg0, %gpu, cTag = "", dFilePath = "" : !mgp.device_context<<"gpu", 0>>
  %ch2 = mgp.device.context.profile.start %ch1, %gpu, cTag = "", dFilePath = "" : !mgp.device_context<<"gpu", 0>>
  mef.output %ch2 : !mef.chain
}

// CHECK-LABEL: mgp.model @mgp_buffer_to_bool
// CHECK: mogg.routine.async (
// CHECK: mgp.buffer.to_bool
// CHECK: mogg.output
mgp.model @mgp_buffer_to_bool : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute(%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, i1) {
  %buf1 = mgp.buffer.constant #M.dense_array<1> : !M.array<1xui8>
  %b1 = mgp.buffer.to_bool %ch0, %buf1 : !mgp.buffer
  mef.output %ch0, %b1 : !mef.chain, i1
}

// CHECK-LABEL: mgp.model @buffer_concat
// CHECK: mogg.routine.async
// CHECK: mgp.buffer.concat
// CHECK-NEXT: mogg.async.ready
// CHECK: mogg.output
mgp.model @buffer_concat : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute(%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %arg3: !mgp.buffer, %arg4: !mgp.buffer) -> (!mef.chain) {
  %size = index.constant 12
  %6 = mgp.buffer.alloc %ch0, %size, %cpu, align 64 : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %8 = mgp.buffer.concat(%ch0, %6, %arg4, %arg4, %cpu) : (!mef.chain, !mgp.buffer, !mgp.buffer, !mgp.buffer, !mgp.device_context<<"cpu", 0>>) -> !mef.chain
  mef.output %ch0 : !mef.chain
}

// CHECK-LABEL: mgp.model @mgp_device_to_host
// CHECK: mogg.routine.async
// CHECK: mgp.buffer.device_to_host
// CHECK-NEXT: mogg.async.ready
// CHECK: mogg.output
mgp.model @mgp_device_to_host : mof devices [#M.device_ref<"cpu", 0>, #M.device_ref<"gpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  %ch3 = mgp.runtime.create %ch2, %ctx[2]
  mef.output %ch2, %ctx, %weights, %cpu, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>, !mgp.device_context<<"gpu", 0>>
} execute(%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %gpu: !mgp.device_context<<"gpu", 0>>, %src: !mgp.buffer<<"gpu", 0>>) -> !mef.chain {
  %size = index.constant 1
  %dst = mgp.buffer.alloc %ch0, %size, %cpu : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %ch = mgp.buffer.device_to_host %ch0, %src, %dst, %gpu : !mgp.buffer<<"gpu", 0>> -> !mgp.buffer, !mgp.device_context<<"gpu", 0>>
  mef.output %ch : !mef.chain
}

// CHECK-LABEL: mgp.model @mgp_device_to_device
// CHECK: mogg.routine.async
// CHECK: mgp.buffer.device_to_device
// CHECK-NEXT: mogg.async.ready
// CHECK: mogg.output
mgp.model @mgp_device_to_device : mof devices [#M.device_ref<"cpu", 0>, #M.device_ref<"gpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  %ch3 = mgp.runtime.create %ch2, %ctx[2]
  mef.output %ch2, %ctx, %weights, %cpu, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>, !mgp.device_context<<"gpu", 0>>
} execute(%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %gpu: !mgp.device_context<<"gpu", 0>>, %src: !mgp.buffer<<"gpu", 0>>) -> !mef.chain {
  %size = index.constant 1
  %dst = mgp.buffer.alloc %ch0, %size, %gpu : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %ch = mgp.buffer.device_to_device %ch0, %src, %gpu, %dst, %gpu : !mgp.buffer<<"gpu", 0>> -> !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  mef.output %ch : !mef.chain
}

// CHECK-LABEL: mgp.model @mgp_host_to_device
// CHECK: mogg.routine.async
// CHECK: mgp.buffer.host_to_device
// CHECK-NEXT: mogg.async.ready
// CHECK: mogg.output
mgp.model @mgp_host_to_device : mof devices [#M.device_ref<"cpu", 0>, #M.device_ref<"gpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  %ch3 = mgp.runtime.create %ch2, %ctx[2]
  mef.output %ch2, %ctx, %weights, %cpu, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>, !mgp.device_context<<"gpu", 0>>
} execute(%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %gpu: !mgp.device_context<<"gpu", 0>>) -> !mef.chain {
  %src = mgp.buffer.constant #M.dense_array<1> : !M.array<1xui8>
  %size = index.constant 1
  %dst = mgp.buffer.alloc %ch0, %size, %gpu : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %ch = mgp.buffer.host_to_device %ch0, %src, %dst, %gpu : !mgp.buffer -> !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  mef.output %ch : !mef.chain
}


// An event sync runs as its own task.
// CHECK-LABEL: mgp.model @mgp_sync
// CHECK: mogg.routine.async dispatch(deferred)
// CHECK: mgp.sync
// CHECK-NEXT: %[[OUT:.*]] = mogg.async.ready
// CHECK: mogg.output %[[OUT]] : !mef.chain
mgp.model @mgp_sync : mof devices [#M.device_ref<"cpu", 0>, #M.device_ref<"gpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>, !mgp.device_context<<"gpu", 0>>
} execute(%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %gpu: !mgp.device_context<<"gpu", 0>>) -> !mef.chain {
  %out_ch = mgp.sync(%ch0, %ctx, %gpu) device <"gpu", 0> {aRuntimeSlot = 1 : ui64} : (!mef.chain, !mgp.context, !mgp.device_context<<"gpu", 0>>) -> !mef.chain
  mef.output %out_ch : !mef.chain
}
