// RUN: mo-opt %s --allow-unregistered-dialect --mgp-predicate-asserts-deps --mgp-simplify-chains --cse | FileCheck %s
//

// CHECK-LABEL: mgp.model @simple_case : mof devices [#M.device_ref<"cpu", 0>] init
// CHECK: execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context,
// CHECK: %[[CH2:.*]] = mgp.assert %{{.*}}, "input shapes must match"
// CHECK: %[[CH3:.*]] = grt.chain.create (%[[CH0]], %[[CH2]])
// CHECK: %[[BUF:.*]] = mgp.buffer.alloc %[[CH3]], %{{.*}}, %{{.*}}, align 16 : !mgp.buffer
// CHECK: %{{.*}} = "kernel_execute"(%[[CH3]], %{{.*}}, %{{.*}}, %[[BUF]])

mgp.model @simple_case : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>

} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %arg2: !mgp.tensor<?x?xf32>, %arg3: !mgp.tensor<?x?xf32>) -> (!mef.chain, !mgp.tensor<?x?xf32>) {
  %bufferResult = mgp.tensor.extract.buffer %arg2 : !mgp.tensor<?x?xf32>
  %tensorSpecResult = mgp.tensor.extract.tensor_spec %arg2 : <?x?xf32>
  %dim0 = mgp.tensor_spec.get_dim[0] %tensorSpecResult : !mgp.tensor_spec<?x?xf32>
  %dim1 = mgp.tensor_spec.get_dim[1] %tensorSpecResult : !mgp.tensor_spec<?x?xf32>
  %bufferResult_0 = mgp.tensor.extract.buffer %arg3 : !mgp.tensor<?x?xf32>
  %tensorSpecResult_1 = mgp.tensor.extract.tensor_spec %arg3 : <?x?xf32>
  %dim2 = mgp.tensor_spec.get_dim[0] %tensorSpecResult_1 : !mgp.tensor_spec<?x?xf32>
  %dim3 = mgp.tensor_spec.get_dim[1] %tensorSpecResult_1 : !mgp.tensor_spec<?x?xf32>
  %2 = index.cmp eq(%dim0, %dim2)
  %3 = index.cmp eq(%dim1, %dim3)
  %4 = pop.and %2, %3
  %5 = mgp.assert %4, "input shapes must match"
  %bufferResult_2 = mgp.tensor.extract.buffer %arg2 : !mgp.tensor<?x?xf32>
  %bufferResult_4 = mgp.tensor.extract.buffer %arg3 : !mgp.tensor<?x?xf32>
  %6 = mgp.tensor_spec.create %dim0, %dim1 : !mgp.tensor_spec<?x?xf32>
  %c8 = index.constant 8
  %tmp = index.mul %dim0, %dim1
  %7 = index.mul %c8, %tmp
  %8 = mgp.buffer.alloc %ch0, %7, %cpu, align 16 : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %9 = "kernel_execute"(%ch0, %bufferResult_2, %bufferResult_4, %8) : (!mef.chain, !mgp.buffer, !mgp.buffer, !mgp.buffer) -> !mef.chain
  %10 = mgp.tensor.create %9, %8, %6  : !mgp.tensor<?x?xf32>
  mef.output %ch0, %10 : !mef.chain, !mgp.tensor<?x?xf32>
}

// CHECK-LABEL: mgp.model @two_asserts : mof devices [#M.device_ref<"cpu", 0>] init
// CHECK: execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context,
// CHECK: %[[CH2:.*]] = mgp.assert %{{.*}}, "input shapes must match"
// CHECK: %[[CH3:.*]] = mgp.assert %{{.*}}, "input dimension 0 must match"
// CHECK: %[[CH4:.*]] = grt.chain.create (%[[CH0]], %[[CH2]], %[[CH3]])
// CHECK: %[[BUF:.*]] = mgp.buffer.alloc %[[CH4]], %{{.*}}, %{{.*}}, align 16 : !mgp.buffer
// CHECK: %{{.*}} = "kernel_execute"(%[[CH4]], %{{.*}}, %{{.*}}, %[[BUF]])

mgp.model @two_asserts : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>

} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %arg2: !mgp.tensor<?x?xf32>, %arg3: !mgp.tensor<?x?xf32>) -> (!mef.chain, !mgp.tensor<?x?xf32>) {
  %bufferResult = mgp.tensor.extract.buffer %arg2 : !mgp.tensor<?x?xf32>
  %tensorSpecResult = mgp.tensor.extract.tensor_spec %arg2 : <?x?xf32>
  %dim0 = mgp.tensor_spec.get_dim[0] %tensorSpecResult : !mgp.tensor_spec<?x?xf32>
  %dim1 = mgp.tensor_spec.get_dim[1] %tensorSpecResult : !mgp.tensor_spec<?x?xf32>
  %bufferResult_0 = mgp.tensor.extract.buffer %arg3 : !mgp.tensor<?x?xf32>
  %tensorSpecResult_1 = mgp.tensor.extract.tensor_spec %arg3 : <?x?xf32>
  %dim2 = mgp.tensor_spec.get_dim[0] %tensorSpecResult_1 : !mgp.tensor_spec<?x?xf32>
  %dim3 = mgp.tensor_spec.get_dim[1] %tensorSpecResult_1 : !mgp.tensor_spec<?x?xf32>
  %2 = index.cmp eq(%dim0, %dim2)
  %3 = index.cmp eq(%dim1, %dim3)
  %4 = pop.and %2, %3
  %5 = mgp.assert %4, "input shapes must match"
  %added_assert = mgp.assert %2, "input dimension 0 must match"
  %bufferResult_2 = mgp.tensor.extract.buffer %arg2 : !mgp.tensor<?x?xf32>
  %bufferResult_4 = mgp.tensor.extract.buffer %arg3 : !mgp.tensor<?x?xf32>
  %6 = mgp.tensor_spec.create %dim0, %dim1 : !mgp.tensor_spec<?x?xf32>
  %c8 = index.constant 8
  %tmp = index.mul %dim0, %dim1
  %7 = index.mul %tmp, %c8
  %8 = mgp.buffer.alloc %ch0, %7, %cpu, align 16 : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %9 = "kernel_execute"(%ch0, %bufferResult_2, %bufferResult_4, %8) : (!mef.chain, !mgp.buffer, !mgp.buffer, !mgp.buffer) -> !mef.chain
  %10 = mgp.tensor.create %9, %8, %6  : !mgp.tensor<?x?xf32>
  mef.output %ch0, %10 : !mef.chain, !mgp.tensor<?x?xf32>
}

// CHECK-LABEL: mgp.model @assert_depends_on_control_flow : mof devices [#M.device_ref<"cpu", 0>] init
// CHECK: execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context,
// CHECK: %[[CH2:.*]] = mgp.assert %{{.*}}, "input shapes must match"
// CHECK: %[[CH3:.*]] = grt.chain.create (%[[CH0]], %[[CH2]]
// CHECK: %[[BUF:.*]] = mgp.buffer.alloc %[[CH3]], %{{.*}}, %{{.*}}, align 16 : !mgp.buffer
// CHECK: %{{.*}} = "kernel_execute"(%[[CH3]], %{{.*}}, %{{.*}}, %[[BUF]])

mgp.model @assert_depends_on_control_flow : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>

} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %arg2: !mgp.tensor<?x?xf32>, %arg3: !mgp.tensor<?x?xf32>) -> (!mef.chain, !mgp.tensor<?x?xf32>) {
  %bufferResult = mgp.tensor.extract.buffer %arg2 : !mgp.tensor<?x?xf32>
  %tensorSpecResult = mgp.tensor.extract.tensor_spec %arg2 : <?x?xf32>
  %dim0 = mgp.tensor_spec.get_dim[0] %tensorSpecResult : !mgp.tensor_spec<?x?xf32>
  %dim1 = mgp.tensor_spec.get_dim[1] %tensorSpecResult : !mgp.tensor_spec<?x?xf32>
  %bufferResult_0 = mgp.tensor.extract.buffer %arg3 : !mgp.tensor<?x?xf32>
  %tensorSpecResult_1 = mgp.tensor.extract.tensor_spec %arg3 : <?x?xf32>
  %dim2 = mgp.tensor_spec.get_dim[0] %tensorSpecResult_1 : !mgp.tensor_spec<?x?xf32>
  %dim3 = mgp.tensor_spec.get_dim[1] %tensorSpecResult_1 : !mgp.tensor_spec<?x?xf32>
  %2 = index.cmp eq(%dim0, %dim2)
  %3 = index.cmp eq(%dim1, %dim3)
  %4 = pop.and %2, %3
  %cf:2 = grt.if %4 (%ch0: !mef.chain, %2: i1, %3: i1) -> (!mef.chain, i1) {
    mef.output %ch0, %2 : !mef.chain, i1
  } else {
    mef.output %ch0, %3 : !mef.chain, i1
  }
  %5 = mgp.assert %cf#1, "input shapes must match"
  %bufferResult_2 = mgp.tensor.extract.buffer %arg2 : !mgp.tensor<?x?xf32>
  %bufferResult_4 = mgp.tensor.extract.buffer %arg3 : !mgp.tensor<?x?xf32>
  %6 = mgp.tensor_spec.create %dim0, %dim1 : !mgp.tensor_spec<?x?xf32>
  %7 = index.mul %dim0, %dim1 {constInt = 8 : si64}
  %8 = mgp.buffer.alloc %ch0, %7, %cpu, align 16 : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %9 = "kernel_execute"(%ch0, %bufferResult_2, %bufferResult_4, %8) : (!mef.chain, !mgp.buffer, !mgp.buffer, !mgp.buffer) -> !mef.chain
  %10 = mgp.tensor.create %9, %8, %6  : !mgp.tensor<?x?xf32>
  mef.output %ch0, %10 : !mef.chain, !mgp.tensor<?x?xf32>
}

// NOTE: Currently the assert is ignored since nested blocks within control flow
//       ops are ignored!
// CHECK-LABEL: mgp.model @assert_in_control_flow : mof devices [#M.device_ref<"cpu", 0>] init
// CHECK: execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context,
// CHECK: %[[CH2:.*]] = grt.if %{{.*}} (%{{.*}}: i1, %{{.*}}: i1, %{{.*}}: !mef.chain) -> !mef.chain {
// CHECK:   %[[CH3:.*]] = mgp.assert %{{.*}}, "dimensions must match"
// CHECK:   mef.output %[[CH3]] : !mef.chain
// CHECK: } else {
// CHECK:   mef.output %{{.*}} : !mef.chain
// CHECK: }
// CHECK: %[[BUF:.*]] = mgp.buffer.alloc %[[CH0]], %{{.*}}, %{{.*}}, align 16 : !mgp.buffer
// CHECK: %{{.*}} = "kernel_execute"(%[[CH0]], %{{.*}}, %{{.*}}, %[[BUF]])

mgp.model @assert_in_control_flow : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>

} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %arg2: !mgp.tensor<?x?xf32>, %arg3: !mgp.tensor<?x?xf32>) -> (!mef.chain, !mgp.tensor<?x?xf32>) {
  %bufferResult = mgp.tensor.extract.buffer %arg2 : !mgp.tensor<?x?xf32>
  %tensorSpecResult = mgp.tensor.extract.tensor_spec %arg2 : <?x?xf32>
  %dim0 = mgp.tensor_spec.get_dim[0] %tensorSpecResult : !mgp.tensor_spec<?x?xf32>
  %dim1 = mgp.tensor_spec.get_dim[1] %tensorSpecResult : !mgp.tensor_spec<?x?xf32>
  %bufferResult_0 = mgp.tensor.extract.buffer %arg3 : !mgp.tensor<?x?xf32>
  %tensorSpecResult_1 = mgp.tensor.extract.tensor_spec %arg3 : <?x?xf32>
  %dim2 = mgp.tensor_spec.get_dim[0] %tensorSpecResult_1 : !mgp.tensor_spec<?x?xf32>
  %dim3 = mgp.tensor_spec.get_dim[1] %tensorSpecResult_1 : !mgp.tensor_spec<?x?xf32>
  %2 = index.cmp eq(%dim0, %dim2)
  %3 = index.cmp eq(%dim1, %dim3)
  %4 = pop.and %2, %3
  %cf = grt.if %4 (%2: i1, %3: i1, %ch0: !mef.chain) -> (!mef.chain) {
    %5 = mgp.assert %2, "dimensions must match"
    mef.output %5 : !mef.chain
  } else {
    mef.output %ch0 : !mef.chain
  }
  %bufferResult_2 = mgp.tensor.extract.buffer %arg2 : !mgp.tensor<?x?xf32>
  %bufferResult_4 = mgp.tensor.extract.buffer %arg3 : !mgp.tensor<?x?xf32>
  %6 = mgp.tensor_spec.create %dim0, %dim1 : !mgp.tensor_spec<?x?xf32>
  %c8 = index.constant 8
  %tmp = index.mul %dim0, %dim1
  %7 = index.mul %tmp, %c8
  %8 = mgp.buffer.alloc %ch0, %7, %cpu, align 16 : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %9 = "kernel_execute"(%ch0, %bufferResult_2, %bufferResult_4, %8) : (!mef.chain, !mgp.buffer, !mgp.buffer, !mgp.buffer) -> !mef.chain
  %10 = mgp.tensor.create %9, %8, %6  : !mgp.tensor<?x?xf32>
  mef.output %ch0, %10 : !mef.chain, !mgp.tensor<?x?xf32>
}

// CHECK-LABEL: mgp.model @no_predication : mof devices [#M.device_ref<"cpu", 0>] init
// CHECK: execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context,
// CHECK: %{{.*}} = mgp.assert %{{.*}}, "input shapes must match"
// CHECK-NEXT: mef.output %[[CH0]] : !mef.chain

mgp.model @no_predication : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>

} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %arg2: !mgp.tensor<?x?xf32>, %arg3: !mgp.tensor<?x?xf32>) -> (!mef.chain) {
  %bufferResult = mgp.tensor.extract.buffer %arg2 : !mgp.tensor<?x?xf32>
  %tensorSpecResult = mgp.tensor.extract.tensor_spec %arg2 : <?x?xf32>
  %dim0 = mgp.tensor_spec.get_dim[0] %tensorSpecResult : !mgp.tensor_spec<?x?xf32>
  %dim1 = mgp.tensor_spec.get_dim[1] %tensorSpecResult : !mgp.tensor_spec<?x?xf32>
  %bufferResult_0 = mgp.tensor.extract.buffer %arg3 : !mgp.tensor<?x?xf32>
  %tensorSpecResult_1 = mgp.tensor.extract.tensor_spec %arg3 : <?x?xf32>
  %dim2 = mgp.tensor_spec.get_dim[0] %tensorSpecResult_1 : !mgp.tensor_spec<?x?xf32>
  %dim3 = mgp.tensor_spec.get_dim[1] %tensorSpecResult_1 : !mgp.tensor_spec<?x?xf32>
  %2 = index.cmp eq(%dim0, %dim2)
  %3 = index.cmp eq(%dim1, %dim3)
  %4 = pop.and %2, %3
  %5 = mgp.assert %4, "input shapes must match"
  mef.output %ch0 : !mef.chain
}

// CHECK-LABEL: mgp.model @on_output_op : mof devices [#M.device_ref<"cpu", 0>] init
// CHECK: execute (%[[CH0:.*]]: !mef.chain,
// CHECK: %[[CH1:.*]] = mgp.assert %{{.*}}, "Tensor type mismatch for argument 0"
// CHECK: %[[CH2:.*]] = mgp.assert %{{.*}}, "Tensor type mismatch for argument 1"
// CHECK: %[[CH3:.*]] = grt.chain.create (%[[CH0]], %[[CH1]], %[[CH2]])
// CHECK: mef.output %[[CH3]]

mgp.model @on_output_op : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>

} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %arg2: !mgp.tensor<2x4xf32>, %arg3: !mgp.tensor<2x4xf32>) -> (!mef.chain, !mgp.tensor<2x4xf32>, !mgp.tensor<2x4xf32>) {
  %0 = mgp.tensor.extract.tensor_spec %arg2 : <2x4xf32>
  %idx2 = index.constant 2
  %idx4 = index.constant 4
  %arg0dim0 = mgp.tensor_spec.get_dim[0] %0 : <2x4xf32>
  %arg0dim1 = mgp.tensor_spec.get_dim[1] %0 : <2x4xf32>
  %arg0dim0eq = index.cmp eq(%arg0dim0, %idx2)
  %arg0dim1eq = index.cmp eq(%arg0dim1, %idx4)
  %arg0eq = pop.and %arg0dim0eq, %arg0dim1eq
  %2 = mgp.assert %arg0eq, "Tensor type mismatch for argument 0"
  %3 = mgp.tensor.extract.tensor_spec %arg3 : <2x4xf32>
  %arg1dim0 = mgp.tensor_spec.get_dim[0] %3 : <2x4xf32>
  %arg1dim1 = mgp.tensor_spec.get_dim[1] %3 : <2x4xf32>
  %arg1dim0eq = index.cmp eq(%arg1dim0, %idx2)
  %arg1dim1eq = index.cmp eq(%arg1dim1, %idx4)
  %arg1eq = pop.and %arg1dim0eq, %arg1dim1eq
  %5 = mgp.assert %arg1eq, "Tensor type mismatch for argument 1"
  mef.output %ch0, %arg2, %arg3 : !mef.chain, !mgp.tensor<2x4xf32>, !mgp.tensor<2x4xf32>
}
