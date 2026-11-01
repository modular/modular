// RUN: mo-opt -allow-unregistered-dialect --canonicalize --mgp-lower-primitives %s | FileCheck %s

// Test that mgp.buffer.bulk_slice is properly lowered with emit-mojo enabled.
// The operation should be wrapped in mogg.routine.async with async
// unpack for the base buffer and index operands, and async pack suppressed for
// the output buffers (resultsToPack returns all-false).

// CHECK-LABEL: mgp.model @bulk_slice_basic
mgp.model @bulk_slice_basic : mof devices [#M.device_ref<"gpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>, %base: !mgp.buffer<<"gpu", 0>>, %off0: index, %off1: index, %sz0: index, %sz1: index) -> (!mef.chain, !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>) {
  // The operation is lowered to mogg.routine.async
  // CHECK: mogg.routine.async
  // CHECK: mogg.async.unpack
  // CHECK: mogg.async.unpack
  // CHECK: mogg.async.unpack
  // CHECK: mogg.async.unpack
  // CHECK: mogg.async.unpack
  // CHECK: mgp.buffer.bulk_slice
  // CHECK: mogg.output
  %slice0, %slice1 = mgp.buffer.bulk_slice %base, %gpu offsets(%off0, %off1) sizes(%sz0, %sz1) : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>

  mef.output %ch0, %slice0, %slice1 : !mef.chain, !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>
}
