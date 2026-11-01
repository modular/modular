// Regression guard for "path D": when the graph compiler emits Mojo for
// mgp.buffer.alloc, the primitive must return the owning OwnedByteBuffer
// composite -- whose live DeviceBuffer the async pack site transfers net-zero
// into the runtime via take_handle() -- and NOT a bare MutByteBuffer minted by
// severing the buffer with take_ptr(). If these checks fail, mgp.buffer.alloc
// has regressed to the old take_ptr() ownership model.
//
// This inspects the emitted Mojo interface (stop at --mogg-emit-mojo; no --jit)
// rather than executing; the end-to-end behavior is covered by
// mgp-cpu-buffer.mlir.

// RUN: mo-opt %s --mogg-import-packages --mgp-lower-primitives --mogg-sequence-fusion --mogg-outline-async-routines --mogg-add-try-except-blocks --canonicalize --mogg-emit-mojo | FileCheck %s

// The composite type is referenced as the alloc primitive's result type...
// CHECK-DAG: !OwnedByteBuffer = !lit.struct<{{.*}}@OwnedByteBuffer>

// ...and the emitted mgp.buffer.alloc primitive declaration returns it.
// CHECK: mogg.primitive.decl{{.*}}@builtin_primitives.primitives.mgp_buffer_alloc
// CHECK-SAME: "__result__": !lit.ref<!OwnedByteBuffer

mgp.model @mgp_buffer_alloc_emits_owning : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, !mgp.tensor<1x?xui8>) {
  %size = index.constant 4
  %buf0 = mgp.buffer.alloc %arg0, %size, %cpu : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %spec0 = mgp.tensor_spec.create %size : !mgp.tensor_spec<1x?xui8>
  %idx0 = index.constant 42
  %ch2 = mgp.buffer.set_with_index %buf0, %idx0 : !mgp.buffer
  %tensor0 = mgp.tensor.create %ch2, %buf0, %spec0 : !mgp.tensor<1x?xui8>
  mef.output %arg0, %tensor0: !mef.chain, !mgp.tensor<1x?xui8>
}
