// RUN: mo-opt --gc-pipeline %s -o %t.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

// CHECK-LABEL: --- Running 'basic_test':
// CHECK: 'basic_test' returned tensor<2xsi64> [29, 13]
// CHECK: , tensor<2x2xui8> [1, 1, 1, 1]
mgp.model @basic_test : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute(%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %in_tensor: !mgp.tensor<2x2xui8>) -> (!mef.chain, !mgp.tensor<2xsi64>, !mgp.tensor<2x2xui8>) {

  %idx0 = index.constant 29
  %idx1 = index.constant 13

  %size = index.constant 16
  %buf = mgp.buffer.alloc %ch0, %size, %cpu : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %ch2 = mgp.buffer.set_with_index %buf, %idx0, %idx1 : !mgp.buffer
  %spec = mgp.tensor_spec.create : !mgp.tensor_spec<2xsi64>
  %tensor = mgp.tensor.create %ch2, %buf, %spec : !mgp.tensor<2xsi64>

  %guarded = mgp.guard %ch0, %tensor : !mgp.tensor<2xsi64>

  %results:2 = mgp.guard %ch0, %tensor, %in_tensor : !mgp.tensor<2xsi64>, !mgp.tensor<2x2xui8>

  mef.output %ch0, %results#0, %results#1 : !mef.chain, !mgp.tensor<2xsi64>, !mgp.tensor<2x2xui8>
}
