// RUN: mo-opt --mgp-verify-model-arguments --cse %s | FileCheck %s

// CHECK-LABEL:  mgp.model @static
// CHECK:        } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context,
// CHECK: %[[ARG2:[a-z0-9]*]]: !mgp.tensor<2x4xf32>, %[[ARG3:.*]]: !mgp.tensor<2x4xf32>) -> (!mef.chain, !mgp.tensor<2x4xf32>, !mgp.tensor<2x4xf32>) {
// CHECK-NEXT:     %[[EXTRACT1:.*]] = mgp.tensor.extract.tensor_spec %[[ARG2]] : <2x4xf32>
// CHECK-NEXT:     %[[IDX2:.*]] = index.constant 2
// CHECK-NEXT:     %[[DIM0:.*]] = mgp.tensor_spec.get_dim[0] %[[EXTRACT1]] : <2x4xf32>
// CHECK-NEXT:     %[[CMP1:.*]] = index.cmp eq(%[[DIM0]], %[[IDX2]])
// CHECK-NEXT:     %[[IDX4:.*]] = index.constant 4
// CHECK-NEXT:     %[[DIM1:.*]] = mgp.tensor_spec.get_dim[1] %[[EXTRACT1]] : <2x4xf32>
// CHECK-NEXT:     %[[CMP2:.*]] = index.cmp eq(%[[DIM1]], %[[IDX4]])
// CHECK-NEXT:     %[[EQUAL1:.*]] = pop.and %[[CMP1]], %[[CMP2]]
// CHECK-NEXT:     %[[ASSERT1:.*]] = mgp.assert %[[EQUAL1:.*]]
// CHECK-NEXT:     %[[EXTRACT2:.*]] = mgp.tensor.extract.tensor_spec %[[ARG3]] : <2x4xf32>
// CHECK-NEXT:     %[[DIM2:.*]] = mgp.tensor_spec.get_dim[0] %[[EXTRACT2]] : <2x4xf32>
// CHECK-NEXT:     %[[CMP3:.*]] = index.cmp eq(%[[DIM2]], %[[IDX2]])
// CHECK-NEXT:     %[[DIM4:.*]] = mgp.tensor_spec.get_dim[1] %[[EXTRACT2]] : <2x4xf32>
// CHECK-NEXT:     %[[CMP4:.*]] = index.cmp eq(%[[DIM4]], %[[IDX4]])
// CHECK-NEXT:     %[[EQUAL2:.*]] = pop.and %[[CMP3]], %[[CMP4]]
// CHECK-NEXT:     %[[ASSERT2:.*]] = mgp.assert %[[EQUAL2]]
// CHECK-NEXT:     mef.output %[[CH0]], %[[ARG2]], %[[ARG3]] : !mef.chain, !mgp.tensor<2x4xf32>, !mgp.tensor<2x4xf32>
// CHECK-NEXT:   }

mgp.model @static : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %arg2: !mgp.tensor<2x4xf32>, %arg3: !mgp.tensor<2x4xf32>) -> (!mef.chain, !mgp.tensor<2x4xf32>, !mgp.tensor<2x4xf32>) {
  mef.output %arg0, %arg2, %arg3 : !mef.chain, !mgp.tensor<2x4xf32>, !mgp.tensor<2x4xf32>
}

// Expecting no check because all dimensions are unknown.
// CHECK-LABEL: mgp.model @dynamic
// CHECK:       } execute
// CHECK-NEXT:    mef.output
mgp.model @dynamic : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %arg2: !mgp.tensor<?x?xf32>, %arg3: !mgp.tensor<?x?xf32>) -> (!mef.chain, !mgp.tensor<?x?xf32>, !mgp.tensor<?x?xf32>) {
  mef.output %arg0, %arg2, %arg3 : !mef.chain, !mgp.tensor<?x?xf32>, !mgp.tensor<?x?xf32>
}

// CHECK-LABEL:  mgp.model @mixed
// CHECK:        } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context,
// CHECK: %[[ARG2:[a-z0-9]*]]: !mgp.tensor<?x4xf32>, %[[ARG3:.*]]: !mgp.tensor<?x4xf32>) -> (!mef.chain, !mgp.tensor<?x4xf32>, !mgp.tensor<?x4xf32>) {
// CHECK-NEXT:     %[[EXTRACT1:.*]] = mgp.tensor.extract.tensor_spec %[[ARG2]] : <?x4xf32>
// CHECK-NEXT:     %[[IDX4:.*]] = index.constant 4
// CHECK-NEXT:     %[[DIM1:.*]] = mgp.tensor_spec.get_dim[1] %[[EXTRACT1]] : <?x4xf32>
// CHECK-NEXT:     %[[EQUAL1:.*]] = index.cmp eq(%[[DIM1]], %[[IDX4]])
// CHECK-NEXT:     %[[ASSERT1:.*]] = mgp.assert %[[EQUAL1:.*]]
// CHECK-NEXT:     %[[EXTRACT2:.*]] = mgp.tensor.extract.tensor_spec %[[ARG3]] : <?x4xf32>
// CHECK-NEXT:     %[[DIM2:.*]] = mgp.tensor_spec.get_dim[1] %[[EXTRACT2]] : <?x4xf32>
// CHECK-NEXT:     %[[EQUAL2:.*]] = index.cmp eq(%[[DIM2]], %[[IDX4]])
// CHECK-NEXT:     %[[ASSERT2:.*]] = mgp.assert %[[EQUAL2]]
// CHECK-NEXT:     mef.output %[[CH0]], %[[ARG2]], %[[ARG3]] : !mef.chain, !mgp.tensor<?x4xf32>, !mgp.tensor<?x4xf32>
// CHECK-NEXT:   }
mgp.model @mixed : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %arg2: !mgp.tensor<?x4xf32>, %arg3: !mgp.tensor<?x4xf32>) -> (!mef.chain, !mgp.tensor<?x4xf32>, !mgp.tensor<?x4xf32>) {
  mef.output %arg0, %arg2, %arg3 : !mef.chain, !mgp.tensor<?x4xf32>, !mgp.tensor<?x4xf32>
}
