// RUN: mo-opt --mogg-import-packages --mgp-lower-primitives --mogg-sequence-fusion --mogg-outline-async-routines --mogg-add-try-except-blocks --canonicalize --mogg-emit-mojo --jit --mogg-strip-kernel-decls --mogg-lower-compiled-routines --mgp-outline-models-to-callables %s -o %t.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

//
// mgp.runtime.create is implicitly tested in all the following.
//

//
// mgp.module.load will be tested by constant_load below.
//

//
// mgp.tensor.{create, extract.buffer, extract.tensor_spec}
//

// CHECK-LABEL: --- Running 'tensor_create_extract'
// CHECK-NEXT: tensor = tensor<1x4xui8> [255, 254, 253, 252]
// CHECK-NEXT: [mgp_test] debug string
mgp.model @tensor_create_extract : mof devices [#M.device_ref<"cpu", 0>, #M.device_ref<"gpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 4
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu, %gpu: !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>, !mgp.device_context<<"gpu", 0>>
} execute(%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %gpu: !mgp.device_context<<"gpu", 0>>) -> (!mef.chain) {
  %size = index.constant 4
  %cpu_buf0 = mgp.buffer.constant dense_resource<resource> : !M.aligned_bytes<4, align 128>
  %const0 = mgp.buffer.alloc %ch0, %size, %gpu: !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %dev_chain0 = mgp.buffer.host_to_device %ch0, %cpu_buf0, %const0, %gpu : !mgp.buffer -> !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %sync0 = mgp.sync(%dev_chain0, %ctx, %gpu) device <"gpu", 0> {aRuntimeSlot = 2 : ui64} : (!mef.chain, !mgp.context, !mgp.device_context<<"gpu", 0>>) -> !mef.chain
  %sizeless_const0 = mgp.buffer.alloc %ch0, %size, %gpu: !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %dev_chain1 = mgp.buffer.host_to_device %ch0, %cpu_buf0, %sizeless_const0, %gpu : !mgp.buffer -> !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %sync1 = mgp.sync(%dev_chain1, %ctx, %gpu) device <"gpu", 0> {aRuntimeSlot = 2 : ui64} : (!mef.chain, !mgp.context, !mgp.device_context<<"gpu", 0>>) -> !mef.chain
  %sync = grt.chain.create (%sync0, %sync1)
  %spec0 = mgp.tensor_spec.create %size : !mgp.tensor_spec<1x?xui8>
  %tensor0 = mgp.tensor.create %sync, %sizeless_const0, %spec0 : !mgp.tensor<1x?xui8, <"gpu", 0>>
  %buf1 = mgp.tensor.extract.buffer %tensor0 : <1x?xui8, <"gpu", 0>>
  %spec1 = mgp.tensor.extract.tensor_spec %tensor0 : !mgp.tensor<1x?xui8, <"gpu", 0>>
  %buf_print = mgp.buffer.alloc %ch0, %size, %cpu, align 64 : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %ch_buf_d2h = mgp.buffer.device_to_host %sync, %buf1, %buf_print, %gpu : !mgp.buffer<<"gpu", 0>> -> !mgp.buffer, !mgp.device_context<<"gpu", 0>>
  %ch_d2h_sync = mgp.sync(%ch_buf_d2h, %ctx, %gpu) device <"gpu", 0> {aRuntimeSlot = 1 : ui64} : (!mef.chain, !mgp.context, !mgp.device_context<<"gpu", 0>>) -> !mef.chain
  %ch_extend = mgp.extend.lifetime %ch_d2h_sync, %buf1 : (!mef.chain, !mgp.buffer<<"gpu", 0>>) -> !mef.chain
  %ch2 = mgp.debug.tensor.print %ch_extend, %buf_print, %spec1, "tensor" : (!mef.chain, !mgp.buffer, !mgp.tensor_spec<1x?xui8>) -> !mef.chain
  %ch3 = mgp.debug.print %ch2, %ctx, "debug string", "mgp_test"
  mef.output %ch3 : !mef.chain
}

//
// mgp.debug.tensor.print and mgp.debug.print already tested above.
//

{-#
  dialect_resources: {
    builtin: {
      resource: "0x80000000fffefdfc"
    }
  }
#-}
