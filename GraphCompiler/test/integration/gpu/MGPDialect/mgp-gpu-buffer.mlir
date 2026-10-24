// RUN: mo-opt --mogg-import-packages --mgp-lower-primitives --mogg-sequence-fusion --mogg-outline-async-routines --mogg-add-try-except-blocks --canonicalize --mogg-emit-mojo --jit --mogg-strip-kernel-decls --mogg-lower-compiled-routines --mgp-outline-models-to-callables %s -o %t.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

//
// mgp.runtime.create is implicitly tested in all the following.
//

//
// mgp.module.load will be tested by constant_load below.
//

//
// mgp.buffer.constant
//

// CHECK-LABEL: --- Running 'constant'
// CHECK-NEXT: tensor0 = tensor<1x4xui8> [255, 254, 253, 252]
// CHECK-NEXT: tensor1 = tensor<0xui8> []
mgp.model @constant : mof devices [#M.device_ref<"cpu", 0>, #M.device_ref<"gpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 3
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu, %gpu: !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>, !mgp.device_context<<"gpu", 0>>
} execute(%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %gpu: !mgp.device_context<<"gpu", 0>>) -> (!mef.chain) {
  %size = index.constant 4
  %cpu_buf0 = mgp.buffer.constant dense_resource<resource> : !M.aligned_bytes<4, align 128>
  %buf0 = mgp.buffer.alloc %ch0, %size, %gpu: !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %dev_chain = mgp.buffer.host_to_device %ch0, %cpu_buf0, %buf0, %gpu : !mgp.buffer -> !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %sync = mgp.sync(%dev_chain, %ctx, %gpu) device <"gpu", 0> {aRuntimeSlot = 2 : ui64} : (!mef.chain, !mgp.context, !mgp.device_context<<"gpu", 0>>) -> !mef.chain
  %spec0 = mgp.tensor_spec.create : !mgp.tensor_spec<1x4xui8>
  %buf_print = mgp.buffer.alloc %ch0, %size, %cpu, align 64 : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %ch_buf_d2h = mgp.buffer.device_to_host %sync, %buf0, %buf_print, %gpu : !mgp.buffer<<"gpu", 0>> -> !mgp.buffer, !mgp.device_context<<"gpu", 0>>
  %ch_d2h_sync = mgp.sync(%ch_buf_d2h, %ctx, %gpu) device <"gpu", 0> {aRuntimeSlot = 1 : ui64} : (!mef.chain, !mgp.context, !mgp.device_context<<"gpu", 0>>) -> !mef.chain
  %ch_extend = mgp.extend.lifetime %ch_d2h_sync : (!mef.chain) -> !mef.chain
  %ch1 = mgp.debug.tensor.print %ch_d2h_sync, %buf_print, %spec0, "tensor0" : (!mef.chain, !mgp.buffer, !mgp.tensor_spec<1x4xui8>) -> !mef.chain
  %zero_size = index.constant 0
  %cpu_buf1 = mgp.buffer.constant #M.aligned_bytes<"", align 128>
  %buf1 = mgp.buffer.alloc %ch0, %zero_size, %gpu: !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %dev_chain1 = mgp.buffer.host_to_device %ch1, %cpu_buf1, %buf1, %gpu : !mgp.buffer -> !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %sync1 = mgp.sync(%dev_chain1, %ctx, %gpu) device <"gpu", 0> {aRuntimeSlot = 2 : ui64} : (!mef.chain, !mgp.context, !mgp.device_context<<"gpu", 0>>) -> !mef.chain
  %spec1 = mgp.tensor_spec.create : !mgp.tensor_spec<0xui8>
  %buf_print1 = mgp.buffer.alloc %ch0, %zero_size, %cpu, align 64 : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %ch_buf_d2h1 = mgp.buffer.device_to_host %sync1, %buf1, %buf_print1, %gpu : !mgp.buffer<<"gpu", 0>> -> !mgp.buffer, !mgp.device_context<<"gpu", 0>>
  %ch_d2h_sync1 = mgp.sync(%ch_buf_d2h1, %ctx, %gpu) device <"gpu", 0> {aRuntimeSlot = 1 : ui64} : (!mef.chain, !mgp.context, !mgp.device_context<<"gpu", 0>>) -> !mef.chain
  %ch_extend1 = mgp.extend.lifetime %ch_d2h_sync1 : (!mef.chain) -> !mef.chain
  %ch2 = mgp.debug.tensor.print %ch_d2h_sync1, %buf_print1, %spec1, "tensor1" : (!mef.chain, !mgp.buffer, !mgp.tensor_spec<0xui8>) -> !mef.chain
  mef.output %ch2 : !mef.chain
}

//
// mgp.buffer.{host_to_device, device_to_host, device_to_device}
//

// CHECK-LABEL: --- Running 'device_to_device'
// CHECK-NEXT: 'device_to_device' returned tensor<4x4xui8> [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
mgp.model @device_to_device : mof devices [#M.device_ref<"cpu", 0>, #M.device_ref<"gpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 4
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu, %gpu: !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %gpu: !mgp.device_context<<"gpu", 0>>, %tensor: !mgp.tensor<4x4xui8>) -> (!mef.chain, !mgp.tensor<4x4xui8>) {
  %buf0 = mgp.tensor.extract.buffer %tensor : <4x4xui8>
  %size = index.constant 16
  %buf1 = mgp.buffer.alloc %ch0, %size, %gpu : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %ch3 = mgp.buffer.host_to_device %ch0, %buf0, %buf1, %gpu : !mgp.buffer -> !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %buf2 = mgp.buffer.alloc %ch3, %size, %gpu : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %ch4 = mgp.buffer.device_to_device %ch3, %buf1, %gpu, %buf2, %gpu : !mgp.buffer<<"gpu", 0>> -> !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %buf3 = mgp.buffer.alloc %ch0, %size, %cpu : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %ch5 = mgp.buffer.device_to_host %ch4, %buf2, %buf3, %gpu : !mgp.buffer<<"gpu", 0>> -> !mgp.buffer, !mgp.device_context<<"gpu", 0>>
  %spec = mgp.tensor_spec.create  : !mgp.tensor_spec<4x4xui8>
  %out = mgp.tensor.create %ch5, %buf3, %spec : !mgp.tensor<4x4xui8>
  mef.output %ch0, %out : !mef.chain, !mgp.tensor<4x4xui8>
}


//
// mgp.buffer.concat is not implemented
//

//
// mgp.buffer.split.borrow is not implemented
//

// CHECK-LABEL: --- Running 'buffer_slice'
// CHECK-NEXT: Tensor(6xsi32, TensorBufferRef(0x{{.*}}, 24))
// CHECK-NEXT: Tensor(3xsi32, TensorBufferRef(0x{{.*}}, 12))
// CHECK-NEXT: 'buffer_slice' returned tensor<3xsi32> [4, 5, 6]
mgp.model @buffer_slice : mof devices [#M.device_ref<"cpu", 0>, #M.device_ref<"gpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 4
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu, %gpu: !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>, !mgp.device_context<<"gpu", 0>>
} execute(%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %gpu: !mgp.device_context<<"gpu", 0>>) -> (!mef.chain, !mgp.tensor<?xsi32>) {
  %idx0 = index.constant 12
  %idx1 = index.constant 12
  %idx2 = index.constant 3
  %idx3 = index.constant 6
  %size0 = index.constant 24
  %cpu_buf0 = mgp.buffer.constant #M.dense_array<1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0, 6, 0, 0, 0> : !M.array<24xui8>
  %buf0 = mgp.buffer.alloc %ch0, %size0, %gpu: !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %dev_chain = mgp.buffer.host_to_device %ch0, %cpu_buf0, %buf0, %gpu : !mgp.buffer -> !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %sync = mgp.sync(%dev_chain, %ctx, %gpu) device <"gpu", 0> {aRuntimeSlot = 2 : ui64} : (!mef.chain, !mgp.context, !mgp.device_context<<"gpu", 0>>) -> !mef.chain
  %sliced = mgp.buffer.slice %buf0, %idx0, %idx1, %gpu : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %spec = mgp.tensor_spec.create : !mgp.tensor_spec<6xsi32>
  %slicedSpec = mgp.tensor_spec.create %idx2 : !mgp.tensor_spec<?xsi32>
  %ch3 = mgp.debug.tensor.print %sync, %buf0, %spec : (!mef.chain, !mgp.buffer<<"gpu", 0>>, !mgp.tensor_spec<6xsi32>) -> !mef.chain
  %ch4 = mgp.debug.tensor.print %ch3, %sliced, %slicedSpec : (!mef.chain, !mgp.buffer<<"gpu", 0>>, !mgp.tensor_spec<?xsi32>) -> !mef.chain
  %hostBuf = mgp.buffer.alloc %ch3, %idx1, %cpu, align 16 : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %ch6 = mgp.buffer.device_to_host %ch3, %sliced, %hostBuf, %gpu : !mgp.buffer<<"gpu", 0>> -> !mgp.buffer, !mgp.device_context<<"gpu", 0>>
  %tensor = mgp.tensor.create %ch6, %hostBuf, %slicedSpec : !mgp.tensor<?xsi32>
  mef.output %ch4, %tensor : !mef.chain, !mgp.tensor<?xsi32>
}


//
// mgp.buffer.{to_bool, set_with_index} are not implemented
//


{-#
  dialect_resources: {
    builtin: {
      resource: "0x80000000fffefdfc"
    }
  }
#-}
