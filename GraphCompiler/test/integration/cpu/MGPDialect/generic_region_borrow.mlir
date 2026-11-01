// TODO(GEX-2747): Fix MGPPrimitives tests with EmitMojo
// XFAIL: *

// RUN: mo-opt --mgp-lower-primitives --jit --mogg-lower-compiled-routines %s -o %t.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

// CHECK-LABEL:--- Running 'buffer_borrow_intrinsic':

// Output should have the pointer of buf1 with storage of buf2

// CHECK-NEXT: Tensor(4xui8, TensorBufferRef(0x[[PTR_VAL_0:[[:xdigit:]]*]],
// CHECK-NEXT: Tensor(4xui8, TensorBufferRef(0x[[PTR_VAL_1:[[:xdigit:]]*]],
// CHECK-NEXT: Tensor(4xui8, TensorBufferRef(0x[[PTR_VAL_0]]

mgp.model @buffer_borrow_intrinsic : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %arg0: !mgp.device_context<<"cpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  %2 = mgp.runtime.create %0, %1[1]
  mef.output %2, %1, %weights, %arg0 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %arg2: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {

  %size = index.constant 4
  %buf0 = mgp.buffer.alloc %arg0, %size, %arg2: !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %buf1 = mgp.buffer.alloc %arg0, %size, %arg2: !mgp.buffer, !mgp.device_context<<"cpu", 0>>

  %spec1 = mgp.tensor_spec.create : !mgp.tensor_spec<4xui8>
  %ch2 = mgp.debug.tensor.print %arg0, %buf0, %spec1 : (!mef.chain, !mgp.buffer, !mgp.tensor_spec<4xui8>) -> !mef.chain
  %ch3 = mgp.debug.tensor.print %ch2, %buf1, %spec1 : (!mef.chain, !mgp.buffer, !mgp.tensor_spec<4xui8>) -> !mef.chain

  %1 = mogg.routine.async (%arg1, %buf0, %buf1) : (%ctx: !mgp.context, %b1: !mgp.buffer, %b2: !mgp.buffer)  -> !mgp.buffer{
    %3 = mogg.buffer.borrow (%b1, %b2) : (!mgp.buffer, !mgp.buffer) -> !mgp.buffer
    mogg.output %3 : !mgp.buffer
  } fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mgp.buffer.alloc">> canBeFusedInto = true canFuseIntoOther = true inOutSignature = <""> {device = #M.device_ref<"cpu", 0>}


  %ch4 = mgp.debug.tensor.print %ch3, %1, %spec1 : (!mef.chain, !mgp.buffer, !mgp.tensor_spec<4xui8>) -> !mef.chain
  mef.output %ch4 : !mef.chain
}


// CHECK-LABEL:--- Running 'buffer_borrow_constant':

// Baseline pointers.
// CHECK-NEXT: Tensor(3xf32, TensorBufferRef(0x[[ALLOC_PTR_VAL:[[:xdigit:]]*]], 12))
// CHECK-NEXT: Tensor(3xf32, TensorBufferRef(0x[[CONST_PTR_VAL:[[:xdigit:]]*]], 12))

// Output1 should have the pointer of buf1 with storage of buf2 (empty as it's constant)
// CHECK-NEXT: Tensor(3xf32, TensorBufferRef(0x[[ALLOC_PTR_VAL]], 12))

// Output2 should have the pointer of buf2 with storage of buf1
// CHECK-NEXT: Tensor(3xf32, TensorBufferRef(0x[[CONST_PTR_VAL]], 12)

mgp.model @buffer_borrow_constant : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %arg0: !mgp.device_context<<"cpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  %2 = mgp.runtime.create %0, %1[1]
  mef.output %2, %1, %weights, %arg0 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %arg2: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {

  %size = index.constant 12

  %buf0 = mgp.buffer.alloc %arg0, %size, %arg2: !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %buf1 = mgp.buffer.constant #M.aligned_bytes<"0x0000E0400000004100001041", align 64> : !M.array<12xui8>

  %spec1 = mgp.tensor_spec.create : !mgp.tensor_spec<3xf32>

  %ch2 = mgp.debug.tensor.print %arg0, %buf0, %spec1 : (!mef.chain, !mgp.buffer, !mgp.tensor_spec<3xf32>) -> !mef.chain
  %ch3 = mgp.debug.tensor.print %ch2, %buf1, %spec1 : (!mef.chain, !mgp.buffer, !mgp.tensor_spec<3xf32>) -> !mef.chain

  %1 = mogg.routine.async (%arg1, %buf0, %buf1) : (%ctx: !mgp.context, %b1: !mgp.buffer, %b2: !mgp.buffer)  -> !mgp.buffer{
    %3 = mogg.buffer.borrow (%b1, %b2) : (!mgp.buffer, !mgp.buffer) -> !mgp.buffer
    mogg.output %3 : !mgp.buffer
  } fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mgp.buffer.alloc">> canBeFusedInto = true canFuseIntoOther = true inOutSignature = <""> {device = #M.device_ref<"cpu", 0>}

  %ch4 = mgp.debug.tensor.print %ch3, %1, %spec1 : (!mef.chain, !mgp.buffer, !mgp.tensor_spec<3xf32>) -> !mef.chain

  %2 = mogg.routine.async (%arg1, %buf1, %buf0) : (%ctx: !mgp.context, %b1: !mgp.buffer, %b2: !mgp.buffer)  -> !mgp.buffer{
    %3 = mogg.buffer.borrow (%b1, %b2) : (!mgp.buffer, !mgp.buffer) -> !mgp.buffer
    mogg.output %3 : !mgp.buffer
  } fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mgp.buffer.alloc">> canBeFusedInto = true canFuseIntoOther = true inOutSignature = <""> {device = #M.device_ref<"cpu", 0>}

  %ch5 = mgp.debug.tensor.print %ch4, %2, %spec1 : (!mef.chain, !mgp.buffer, !mgp.tensor_spec<3xf32>) -> !mef.chain

  mef.output %ch5 : !mef.chain
}
