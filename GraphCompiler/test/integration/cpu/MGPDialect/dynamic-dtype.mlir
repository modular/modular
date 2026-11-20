// RUN: mt --execute --model-inputs=ones --input-specs=2x3xf32 --result-output-style=compact %s | FileCheck %s

// CHECK: --- Running 'test':
// CHECK-NEXT: 'test' returned tensor({{..}}1.0000, 1.0000, 1.0000],
// CHECK-NEXT: [1.0000, 1.0000, 1.0000]], dtype=f32, shape=[2,3])

mef.callable @test$init(%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 0
  mef.output %0, %1, %arg0, %arg1 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
}
mef.callable @test$exec(%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.opaque<"weights_registry">, %arg3: !mgp.device_context<<"cpu", 0>>, %arg4: !mgp.tensor<?x?xunk>) -> (!mef.chain, !mgp.tensor<?x?xunk>) {
  mef.output %arg0, %arg4 : !mef.chain, !mgp.tensor<?x?xunk>
}
mef.model @test : mof (!mgp.tensor<?x?xunk>) -> !mgp.tensor<?x?xunk> devices [#M.device_ref<"cpu", 0>] inputDeviceIndices [] resultDeviceIndices [] argumentNames [] resultNames [] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>) {
  %0:4 = grt.invoke @test$init(%arg0, %arg1) : (!mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>)
  mef.output %0#0, %0#1, %0#2, %0#3 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.opaque<"weights_registry">, %arg3: !mgp.device_context<<"cpu", 0>>, %arg4: !mgp.tensor<?x?xunk>) -> (!mef.chain, !mgp.tensor<?x?xunk>) {
  %0:2 = grt.invoke @test$exec(%arg0, %arg1, %arg2, %arg3, %arg4) : (!mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>, !mgp.tensor<?x?xunk>) -> (!mef.chain, !mgp.tensor<?x?xunk>)
  mef.output %0#0, %0#1 : !mef.chain, !mgp.tensor<?x?xunk>
}
