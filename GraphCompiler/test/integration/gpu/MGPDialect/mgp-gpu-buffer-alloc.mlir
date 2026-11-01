// RUN: mo-opt --mogg-import-packages --mgp-lower-primitives --mogg-sequence-fusion --mogg-outline-async-routines --mogg-add-try-except-blocks --canonicalize --mogg-emit-mojo --jit --mogg-strip-kernel-decls --mogg-lower-compiled-routines --mgp-outline-models-to-callables %s -o %t.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

//
// mgp.runtime.create is implicitly tested in all the following.
//

//
// mgp.buffer.alloc
//

// CHECK-LABEL: --- Running 'alloc'
// CHECK-NEXT: tensor0 = tensor<1x4xui8> [123, 254, 253, 252]
// CHECK-NEXT: tensor1 = tensor<1x4xui8> [123, 254, 253, 252]
mgp.model @alloc : mof devices [#M.device_ref<"cpu", 0>, #M.device_ref<"gpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 3
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu, %gpu: !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>, !mgp.device_context<<"gpu", 0>>
} execute(%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %gpu: !mgp.device_context<<"gpu", 0>>) -> (!mef.chain) {
  %size = index.constant 4
  // Load the bottom 4 bytes of the resource from dialect_resources on to the host. This is used for both validations in this test.
  %expected = mgp.buffer.constant dense_resource<resource> : !M.aligned_bytes<4, align 128>

  // ------------------------------------------------------------------------------------------------
  // Validation 0: Can we allocate a buffer on the GPU and print it with a static tensor spec?
  // ------------------------------------------------------------------------------------------------
  // Allocate a buffer on the GPU.
  %buf0 = mgp.buffer.alloc %ch0, %size, %gpu : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %spec0 = mgp.tensor_spec.create : !mgp.tensor_spec<1x4xui8>

  // Copy the expected values to the GPU buffer.
  %ch0_expected_h2d = mgp.buffer.host_to_device %ch0, %expected, %buf0, %gpu : !mgp.buffer -> !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>

  // Now validate that we can print the data from the GPU buffer. To do that, we need to copy the data back to the host.
  %cpubuf0 = mgp.buffer.alloc %ch0, %size, %cpu, align 64 : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %ch_buf0_d2h = mgp.buffer.device_to_host %ch0_expected_h2d, %buf0, %cpubuf0, %gpu : !mgp.buffer<<"gpu", 0>> -> !mgp.buffer, !mgp.device_context<<"gpu", 0>>
  %ch_d2h_sync0 = mgp.sync(%ch_buf0_d2h, %ctx, %gpu) device <"gpu", 0> {aRuntimeSlot = 1 : ui64} : (!mef.chain, !mgp.context, !mgp.device_context<<"gpu", 0>>) -> !mef.chain
  %ch_extend0 = mgp.extend.lifetime %ch_d2h_sync0 : (!mef.chain) -> !mef.chain
  %ch2 = mgp.debug.tensor.print %ch_d2h_sync0, %cpubuf0, %spec0, "tensor0" : (!mef.chain, !mgp.buffer, !mgp.tensor_spec<1x4xui8>) -> !mef.chain

  // ------------------------------------------------------------------------------------------------
  // Validation 1: Can we allocate a buffer on the GPU and print it with a dynamic tensor spec?
  // ------------------------------------------------------------------------------------------------

  // Allocate a buffer on the GPU. Note that this allocates waits for the first debug print to complete, so that we don't get a race condition between the two prints.
  %buf1 = mgp.buffer.alloc %ch2, %size, %gpu : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
  %spec1 = mgp.tensor_spec.create %size : !mgp.tensor_spec<1x?xui8>

  // Copy the expected values to the GPU buffer.
  %ch1_expected_h2d = mgp.buffer.host_to_device %ch0, %expected, %buf1, %gpu : !mgp.buffer -> !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>

  // Now validate that we can print the data from the GPU buffer. To do that, we need to copy the data back to the host.
  %cpubuf1 = mgp.buffer.alloc %ch0, %size, %cpu, align 64 : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %ch_buf1_d2h = mgp.buffer.device_to_host %ch1_expected_h2d, %buf1, %cpubuf1, %gpu : !mgp.buffer<<"gpu", 0>> -> !mgp.buffer, !mgp.device_context<<"gpu", 0>>
  %ch_d2h_sync1 = mgp.sync(%ch_buf1_d2h, %ctx, %gpu) device <"gpu", 0> {aRuntimeSlot = 1 : ui64} : (!mef.chain, !mgp.context, !mgp.device_context<<"gpu", 0>>) -> !mef.chain
  %ch_extend1 = mgp.extend.lifetime %ch_d2h_sync1 : (!mef.chain) -> !mef.chain
  %ch4 = mgp.debug.tensor.print %ch_d2h_sync1, %cpubuf1, %spec1, "tensor1" : (!mef.chain, !mgp.buffer, !mgp.tensor_spec<1x?xui8>) -> !mef.chain

  mef.output %ch4 : !mef.chain
}


{-#
  dialect_resources: {
    builtin: {
      resource: "0x800000007bfefdfc"
    }
  }
#-}
