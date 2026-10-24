// RUN: mo-opt %s --allow-unregistered-dialect --mgp-exec-invariant-code-motion --canonicalize --cse --split-input-file | FileCheck %s

//
// @liftable_op
//

// CHECK-LABEL: mgp.model @liftable_op : mof
// CHECK-DAG:   %[[CH0:.*]] = grt.chain.create ()
// CHECK-DAG:   %[[CTX:.*]] = mgp.context.create 3
// CHECK-DAG:   %[[CH1:.*]] = mgp.runtime.create %[[CH0]], %1[1]
// CHECK-DAG:   %[[BUF:.*]] = mgp.buffer.constant.external %[[CH1]]
// CHECK-DAG:   %[[CACHE_CH:.*]] = mgp.buffer.cache %[[CH0]], %[[CTX]][2], %[[BUF]]
// CHECK-DAG:   %[[CH_OUT:.*]] = grt.chain.create (%[[CH1]], %[[CACHE_CH]])
// CHECK:       mef.output %[[CH_OUT]], %[[CTX]]
// CHECK-NEXT:  } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context

// CHECK:   %[[BUF:.*]] = mgp.buffer.get_cached %[[CH0]], %[[CTX]][{{.*}}] : !mgp.buffer
// CHECK:   mef.output %[[CH0]], %[[BUF]] : !mef.chain, !mgp.buffer
// CHECK: }

mgp.model @liftable_op : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.buffer) {
  %name = mgp.string.create "foo" : !mgp.opaque<"mojo_string">
  %out_buf = mgp.buffer.constant.external %ch0, %weights[%name, 37632, align 16] : !mgp.opaque<"weights_registry">, !mgp.opaque<"mojo_string"> -> !mgp.buffer
  mef.output %ch0, %out_buf : !mef.chain, !mgp.buffer
}

// -----

// CHECK-LABEL: mgp.model @constant_external_mutable : mof
// CHECK-NOT:   mgp.buffer.constant.external
// CHECK:     } execute (
// CHECK:       mgp.buffer.constant.external

mgp.model @constant_external_mutable : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.buffer) {
  %name = mgp.string.create "foo" : !mgp.opaque<"mojo_string">
  %out_buf = mgp.buffer.constant.external %ch0, %weights[%name, 4096, align 4 has_alias true] : !mgp.opaque<"weights_registry">, !mgp.opaque<"mojo_string"> -> !mgp.buffer
  mef.output %ch0, %out_buf : !mef.chain, !mgp.buffer
}

// -----

//
// @non_pure_op
//

// CHECK-LABEL: mgp.model @non_pure_op : mof
// CHECK-NEXT:  %{{.*}} = grt.chain.create ()
// Nothing was lifted.
// CHECK:       %[[CTX:.*]] = mgp.context.create 2

mgp.model @non_pure_op : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>

} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.tensor<?x?xf32>) {
  %const_buf = mgp.buffer.constant #M.aligned_bytes<"0x0000803F0000803F0000803F0000803F", align 16>
  %const_spec = mgp.tensor_spec.create  : !mgp.tensor_spec<2x2xf32>
  %out_spec = "foo.side_effect_op"(%const_spec) : (!mgp.tensor_spec<2x2xf32>) -> !mgp.tensor_spec<?x?xf32>
  %out_dim0 = mgp.tensor_spec.get_dim[0] %out_spec : !mgp.tensor_spec<?x?xf32>
  %out_dim1 = mgp.tensor_spec.get_dim[1] %out_spec : !mgp.tensor_spec<?x?xf32>
  %c8 = index.constant 8
  %tmp = index.mul %out_dim0, %out_dim1
  %bytes = index.mul %c8, %tmp
  %out_buf = mgp.buffer.alloc %ch0, %bytes, %cpu, align 16 : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %ch2 = "bar.side_effect_op"(%ch0, %const_buf, %const_spec, %out_buf, %out_spec) : (!mef.chain, !mgp.buffer, !mgp.tensor_spec<2x2xf32>, !mgp.buffer, !mgp.tensor_spec<?x?xf32>) -> !mef.chain
  %out_tensor = mgp.tensor.create %ch2, %out_buf, %out_spec  : !mgp.tensor<?x?xf32>
  mef.output %ch0, %out_tensor : !mef.chain, !mgp.tensor<?x?xf32>
}

// -----

//
// @dont_lift_if_no_computation_saved
//

// CHECK-LABEL: mgp.model @dont_lift_if_no_computation_saved : mof
// CHECK:  %{{.*}} = grt.chain.create ()
// Nothing changed
// CHECK:       %[[CTX:.*]] = mgp.context.create 3
// CHECK:       mef.output %{{.*}}, %[[CTX]]
//
// CHECK-NEXT:  } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context
// CHECK:       mef.output %[[CH0]], %{{.*}}
// CHECK-NEXT:  }

mgp.model @dont_lift_if_no_computation_saved : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>

} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.buffer) {
  %size = index.constant 0
  %out_buf = mgp.buffer.alloc %ch0, %size, %cpu, align 16 : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  mef.output %ch0, %out_buf : !mef.chain, !mgp.buffer
}

// -----

// CHECK-LABEL: mgp.model @generic_execute : mof
// CHECK-DAG:     %[[CH0:.*]] = grt.chain.create ()
// CHECK-DAG:     %[[CTX:.*]] = mgp.context.create 3
// CHECK-DAG:     %[[CH1:.*]] = mgp.runtime.create %[[CH0]], %1[1]
// CHECK-DAG:     %[[BUF:.*]] = mgp.buffer.alloc %[[CH1]]
// CHECK-DAG:     %[[CH3:.*]] = mgp.buffer.cache %[[CH0]], %[[CTX]][2], %[[BUF]]
// CHECK-DAG:     %[[MERGE:.*]] = grt.chain.create (%[[CH1]], %[[CH3]])
// CHECK:         mef.output %[[MERGE]], %[[CTX]]
// CHECK-NEXT:  } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context
// CHECK:         %[[BUF:.*]] = mgp.buffer.get_cached %[[CH0]]
// CHECK:         %[[OUT:.*]] = "mgp.generic.execute"({{.*}}, %[[BUF]])

mgp.model @generic_execute : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>

} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.buffer) {
  %weight_buf = mgp.buffer.constant dense_resource<onnx__Conv_497> : !M.aligned_bytes<37632, align 16>

  %size = index.constant 37632
  %out_buf = mgp.buffer.alloc %ch0, %size, %cpu, align 16 : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %out_spec = mgp.tensor_spec.create  : !mgp.tensor_spec<7x7x3x64xf32>

  %i42 = index.constant 42

  %ch2 = "mgp.generic.execute"(%ctx, %cpu, %weight_buf, %out_buf) {
    aRuntimeSlot = 0 : ui64,
    bModuleIndex = 0 : ui64,
    cFunctionIndex = 0 : ui64,
    dSummary = "mogg.routine.async:14",
    eIsAsync = false,
    fConstantResource =[]
  } : (!mgp.context, !mgp.device_context<<"cpu", 0>>, !mgp.buffer, !mgp.buffer) -> (!mef.chain)

  mef.output %ch2, %out_buf : !mef.chain, !mgp.buffer
}

// -----

//
// @while
//

// CHECK-LABEL: mgp.model @while : mof
// Nothing was lifted
// CHECK-NEXT:  %{{.*}} = grt.chain.create ()
// CHECK:       %{{.*}} = mgp.context.create 2

mgp.model @while : mof devices [#M.device_ref<"cpu", 0>, #M.device_ref<"gpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu, %gpu: !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>, !mgp.device_context<<"gpu", 0>>

} execute (%ch: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %gpu: !mgp.device_context<<"gpu", 0>>) -> (!mef.chain) {

  %i, %j, %k = grt.while (%ch: !mef.chain, %ctx: !mgp.context, %gpu: !mgp.device_context<<"gpu", 0>>) {
    // Sync is a liftable op
    %ch2 = mgp.sync(%ch, %ctx, %gpu) device <"gpu", 0> {aRuntimeSlot = 2 : ui64} : (!mef.chain, !mgp.context, !mgp.device_context<<"gpu", 0>>) -> !mef.chain

    %cond = "some.op"() : () -> i1
    mef.output %ch2, %ctx, %gpu, %cond : !mef.chain, !mgp.context, !mgp.device_context<<"gpu", 0>>, i1
  } do {
    mef.output %ch, %ctx, %gpu: !mef.chain, !mgp.context, !mgp.device_context<<"gpu", 0>>
  }

  mef.output %i : !mef.chain
}

// CHECK-LABEL: mgp.model @extend_lifetime : mof
// CHECK: {{.*}} = mgp.extend.lifetime
// CHECK:  } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context
mgp.model @extend_lifetime : mof devices [#M.device_ref<"cpu", 0>, #M.device_ref<"gpu", 0>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"cpu", 0>>, %arg2: !mgp.device_context<<"gpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  %2 = mgp.runtime.create %0, %1[0]
  mef.output %2, %1, %arg0, %arg1, %arg2 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>, !mgp.device_context<<"gpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.opaque<"weights_registry">, %arg3: !mgp.device_context<<"cpu", 0>>, %arg4: !mgp.device_context<<"gpu", 0>>, %arg5: !mgp.tensor<4xf32>) -> !mef.chain {
  %idx16 = index.constant 16
  %0 = mgp.tensor_spec.create  : !mgp.tensor_spec<4xf32>
  %1 = mgp.buffer.constant #M.aligned_bytes<"0x000000000000803F0000004000004040", align 64> : !M.array<16xui8>
  %2 = mgp.buffer.alloc %arg0, %idx16, %arg4, align 256 : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %3 = mgp.buffer.host_to_device %arg0, %1, %2, %arg4 : !mgp.buffer -> !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %4 = mgp.sync(%3, %arg1, %arg4) device <"gpu", 0> {aRuntimeSlot = 1 : ui64} : (!mef.chain, !mgp.context, !mgp.device_context<<"gpu", 0>>) -> !mef.chain
  %5 = mgp.extend.lifetime  %4, %1 : (!mef.chain, !mgp.buffer) -> !mef.chain
  %6 = mgp.tensor.create %4, %2, %0 : !mgp.tensor<4xf32,<"gpu", 0>>
  %7 = grt.chain.create(%5, %6: !mgp.tensor<4xf32, <"gpu", 0>>)
  mef.output %7 : !mef.chain
}

// -----

// A compiled region whose only extra results are dimensions moves to init.
// The step keeps the root chain and reads the dimension back.

// CHECK-LABEL: mgp.model @compiled_region_chain_and_dim : mof
// CHECK-DAG: mogg.routine.async
// CHECK-DAG: mgp.int.cache
// CHECK: } execute (%[[CH:.*]]: !mef.chain
// CHECK-NOT: mogg.routine.async
// CHECK: mgp.int.get_cached %[[CH]]
// CHECK: mef.output %[[CH]], {{%.*}} : !mef.chain, !kgen.scalar<si64>

mgp.model @compiled_region_chain_and_dim : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !kgen.scalar<si64>) {
  %ch, %dim = mogg.routine.async (%ch0) : (%in: !mef.chain) -> (!mef.chain, !kgen.scalar<si64>) {
    %ready = mogg.async.ready
    %c4 = index.constant 4
    %si = index.casts %c4 : index to si64
    %dim = pop.cast_from_builtin %si : si64 to !kgen.scalar<si64>
    %packed = mogg.async.pack(%dim) : !kgen.scalar<si64>
    mogg.output %ready, %packed : !mef.chain, !mogg.async<!kgen.scalar<si64>>
  } inOutSignature = <"i"> {device = #M.device_ref<"cpu", 0>}
  mef.output %ch, %dim : !mef.chain, !kgen.scalar<si64>
}

// -----

// A tensor result has no cache slot, so the region stays in the step.

// CHECK-LABEL: mgp.model @compiled_region_uncacheable_result : mof
// CHECK: } execute
// CHECK: mogg.routine.async

mgp.model @compiled_region_uncacheable_result : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.tensor<2x2xf32>) {
  %buf = mgp.buffer.constant #M.aligned_bytes<"0x0000803F0000803F0000803F0000803F", align 16>
  %ch, %tensor = mogg.routine.async (%ch0, %buf) : (%in: !mef.chain, %arg: !mgp.buffer) -> (!mef.chain, !mgp.tensor<2x2xf32>) {
    %ready = mogg.async.ready
    %unpacked = mogg.async.unpack(%arg) : !mgp.buffer
    %spec = mgp.tensor_spec.create : !mgp.tensor_spec<2x2xf32>
    %created = mgp.tensor.create %ready, %unpacked, %spec : !mgp.tensor<2x2xf32>
    %packed = mogg.async.pack(%created) : !mgp.tensor<2x2xf32>
    mogg.output %ready, %packed : !mef.chain, !mogg.async<!mgp.tensor<2x2xf32>>
  } inOutSignature = <"ii"> {device = #M.device_ref<"cpu", 0>}
  mef.output %ch, %tensor : !mef.chain, !mgp.tensor<2x2xf32>
}
