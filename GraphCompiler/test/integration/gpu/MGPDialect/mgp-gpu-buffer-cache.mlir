// RUN: mo-opt --mogg-import-packages --mgp-lower-primitives --mogg-sequence-fusion --mogg-outline-async-routines --mogg-add-try-except-blocks --canonicalize --mogg-emit-mojo --jit --mogg-strip-kernel-decls --mogg-lower-compiled-routines --mgp-outline-models-to-callables %s -o %t.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

//
// mgp.runtime.create is implicitly tested in all the following.
//

//
// mgp.buffer.{cache, get_cached}
//

// CHECK-LABEL: --- Running 'cache'
// CHECK-NEXT: tensor = tensor<4xui8> [255, 254, 253, 252]
mgp.model @cache : mof devices [#M.device_ref<"cpu", 0>, #M.device_ref<"gpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 4
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu, %gpu: !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>, !mgp.device_context<<"gpu", 0>>
} execute(%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %gpu: !mgp.device_context<<"gpu", 0>>) -> (!mef.chain) {
  %size = index.constant 4
  %cpu_buf0 = mgp.buffer.constant dense_resource<resource> : !M.aligned_bytes<4, align 128>
  %buf0 = mgp.buffer.alloc %ch0, %size, %gpu: !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %dev_chain = mgp.buffer.host_to_device %ch0, %cpu_buf0, %buf0, %gpu : !mgp.buffer -> !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %sync = mgp.sync(%dev_chain, %ctx, %gpu) device <"gpu", 0> {aRuntimeSlot = 2 : ui64} : (!mef.chain, !mgp.context, !mgp.device_context<<"gpu", 0>>) -> !mef.chain
  %ch1 = mgp.buffer.cache %sync, %ctx[3], %buf0 : (!mef.chain, !mgp.context, !mgp.buffer<<"gpu", 0>>) -> !mef.chain
  %idx = index.constant 3
  %buf1 = mgp.buffer.get_cached %ch1, %ctx[%idx] : !mgp.buffer<<"gpu", 0>>
  %spec = mgp.tensor_spec.create : !mgp.tensor_spec<4xui8>
  %buf_print = mgp.buffer.alloc %ch1, %size, %cpu, align 64 : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %ch_buf_d2h = mgp.buffer.device_to_host %sync, %buf0, %buf_print, %gpu : !mgp.buffer<<"gpu", 0>> -> !mgp.buffer, !mgp.device_context<<"gpu", 0>>
  %ch_d2h_sync = mgp.sync(%ch_buf_d2h, %ctx, %gpu) device <"gpu", 0> {aRuntimeSlot = 1 : ui64} : (!mef.chain, !mgp.context, !mgp.device_context<<"gpu", 0>>) -> !mef.chain
  %ch_extend = mgp.extend.lifetime %ch_d2h_sync : (!mef.chain) -> !mef.chain
  %ch2 = mgp.debug.tensor.print %ch_d2h_sync, %buf_print, %spec, "tensor" : (!mef.chain, !mgp.buffer, !mgp.tensor_spec<4xui8>) -> !mef.chain
  %ch3 = mgp.sync(%ch2, %ctx, %gpu) device <"gpu", 0> {aRuntimeSlot = 2 : ui64} : (!mef.chain, !mgp.context, !mgp.device_context<<"gpu", 0>>) -> !mef.chain
  mef.output %ch3 : !mef.chain
}
// CHECK-LABEL: --- Running 'cache_in_init'
// CHECK-NEXT: tensor = tensor<4xui8> [255, 254, 253, 252]
mgp.model @cache_in_init : mof devices [#M.device_ref<"cpu", 0>, #M.device_ref<"gpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 4
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  %size = index.constant 4
  %cpu_buf0 = mgp.buffer.constant dense_resource<resource> : !M.aligned_bytes<4, align 128>
  %buf0 = mgp.buffer.alloc %ch2, %size, %gpu: !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %dev_chain = mgp.buffer.host_to_device %ch2, %cpu_buf0, %buf0, %gpu : !mgp.buffer -> !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %sync = mgp.sync(%dev_chain, %ctx, %gpu) device <"gpu", 0> {aRuntimeSlot = 2 : ui64} : (!mef.chain, !mgp.context, !mgp.device_context<<"gpu", 0>>) -> !mef.chain
  %ch4 = mgp.buffer.cache %sync, %ctx[3], %buf0 : (!mef.chain, !mgp.context, !mgp.buffer<<"gpu", 0>>) -> !mef.chain
  mef.output %ch4, %ctx, %weights, %cpu, %gpu: !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>, !mgp.device_context<<"gpu", 0>>
} execute(%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %gpu: !mgp.device_context<<"gpu", 0>>) -> (!mef.chain) {
  %idx = index.constant 3
  %buf1 = mgp.buffer.get_cached %ch0, %ctx[%idx] : !mgp.buffer<<"gpu", 0>>
  %spec = mgp.tensor_spec.create : !mgp.tensor_spec<4xui8>
  %size = index.constant 4
  %buf_print = mgp.buffer.alloc %ch0, %size, %cpu, align 64 : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %ch_buf_d2h = mgp.buffer.device_to_host %ch0, %buf1, %buf_print, %gpu : !mgp.buffer<<"gpu", 0>> -> !mgp.buffer, !mgp.device_context<<"gpu", 0>>
  %ch_d2h_sync = mgp.sync(%ch_buf_d2h, %ctx, %gpu) device <"gpu", 0> {aRuntimeSlot = 1 : ui64} : (!mef.chain, !mgp.context, !mgp.device_context<<"gpu", 0>>) -> !mef.chain
  %ch_extend = mgp.extend.lifetime %ch_d2h_sync : (!mef.chain) -> !mef.chain
  %ch1 = mgp.debug.tensor.print %ch_d2h_sync, %buf_print, %spec, "tensor" : (!mef.chain, !mgp.buffer, !mgp.tensor_spec<4xui8>) -> !mef.chain
  %ch2 = mgp.sync(%ch1, %ctx, %gpu) device <"gpu", 0> {aRuntimeSlot = 2 : ui64} : (!mef.chain, !mgp.context, !mgp.device_context<<"gpu", 0>>) -> !mef.chain
  mef.output %ch2 : !mef.chain
}


{-#
  dialect_resources: {
    builtin: {
      resource: "0x80000000fffefdfc"
    }
  }
#-}
