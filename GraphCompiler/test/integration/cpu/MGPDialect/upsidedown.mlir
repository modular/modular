// RUN: mo-opt -gc-pipeline %s -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s

// CHECK-LABEL: --- Running 'upsidedown'
// CHECK:'upsidedown' returned tensor<1x4xf32> [1.000000e+02, 2.990000e+00, 4.564640e+00, 6.924320e+00]
mgp.model @upsidedown : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %arg2: !mgp.tensor<1x4xf32>) -> (!mef.chain, !mgp.tensor<1x4xf32>) {
  %2 = mgp.tensor.create %ch0, %0, %1  : !mgp.tensor<1x4xf32>
  %1 = mgp.tensor_spec.create  : !mgp.tensor_spec<1x4xf32>
  %0 = mgp.buffer.constant #M.aligned_bytes<"0x0000C842295C3F40881192400894DD40", align 16> : !M.array<16xui8>
  mef.output %ch0, %2 : !mef.chain, !mgp.tensor<1x4xf32>
}
