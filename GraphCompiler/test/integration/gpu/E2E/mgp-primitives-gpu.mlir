// RUN: mo-opt --mogg-import-packages --mgp-lower-primitives --mogg-sequence-fusion --mogg-outline-async-routines --mogg-add-try-except-blocks --mogg-emit-mojo --jit --mogg-strip-kernel-decls --mogg-lower-compiled-routines --mgp-outline-models-to-callables %s -o %t.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

// CHECK-LABEL: --- Running 'device_to_device'
// CHECK-NEXT: 'device_to_device' returned tensor<4x4xui8> [1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0]
mgp.model @device_to_device : mof devices [#M.device_ref<"cpu", 0>, #M.device_ref<"gpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 4
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu, %gpu: !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>, !mgp.device_context<<"gpu", 0>>
} execute (%ch: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %gpu: !mgp.device_context<<"gpu", 0>>) -> (!mef.chain, !mgp.tensor<4x4xui8>) {
  %size = index.constant 16
  %buf0 = mgp.buffer.alloc %ch, %size, %cpu : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %idx0 = index.constant 1
  %idx1 = index.constant 2
  %ch1 = mgp.buffer.set_with_index %buf0, %idx0, %idx1 : !mgp.buffer
  %buf1 = mgp.buffer.alloc %ch, %size, %gpu: !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %ch4 = mgp.buffer.host_to_device %ch1, %buf0, %buf1, %gpu : !mgp.buffer -> !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %buf2 = mgp.buffer.alloc %ch4, %size, %gpu: !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %ch5 = mgp.buffer.device_to_device %ch4, %buf1, %gpu, %buf2, %gpu: !mgp.buffer<<"gpu", 0>> -> !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %buf3 = mgp.buffer.alloc %ch, %size, %cpu: !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %ch6 = mgp.buffer.device_to_host %ch5, %buf2, %buf3, %gpu : !mgp.buffer<<"gpu", 0>> -> !mgp.buffer, !mgp.device_context<<"gpu", 0>>
  %spec = mgp.tensor_spec.create  : !mgp.tensor_spec<4x4xui8>
  %out = mgp.tensor.create %ch6, %buf3, %spec : !mgp.tensor<4x4xui8>
  mef.output %ch6, %out : !mef.chain, !mgp.tensor<4x4xui8>
}

// CHECK-LABEL: --- Running 'sync'
mgp.model @sync : mof devices [#M.device_ref<"cpu", 0>, #M.device_ref<"gpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %0 = grt.chain.create ()
  %ctx = mgp.context.create 3
  %2 = mgp.runtime.create %0, %ctx[0]
  mef.output %2, %ctx, %weights, %cpu, %gpu: !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>, !mgp.device_context<<"gpu", 0>>
} execute(%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %gpu: !mgp.device_context<<"gpu", 0>>) -> (!mef.chain) {
  %out_chain = mgp.sync(%ch0, %ctx, %gpu) device <"gpu", 0> {aRuntimeSlot = 1 : ui64} : (!mef.chain, !mgp.context, !mgp.device_context<<"gpu", 0>>) -> !mef.chain
  mef.output %out_chain : !mef.chain
}

{-#
  dialect_resources: {
    builtin: {
      resource: "0x80000000fffefdfc"
    }
  }
#-}
