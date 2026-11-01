// RUN: mo-opt -allow-unregistered-dialect --canonicalize --mgp-lower-primitives %s | FileCheck %s

// Test that mgp.buffer.plan operation is properly lowered with emit-mojo enabled.
// The operation should be wrapped in mogg.routine.async with async pack/unpack.

// CHECK-LABEL: mgp.model @buffer_plan_basic
mgp.model @buffer_plan_basic : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  mef.output %ch0, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %size0: index, %size1: index, %size2: index) -> (!mef.chain, index, index, index, index) {
  // The operation is lowered to mogg.routine.async with async unpack/pack
  // CHECK: mogg.routine.async
  // CHECK: mogg.async.unpack
  // CHECK: mogg.async.unpack
  // CHECK: mogg.async.unpack
  // CHECK: mgp.buffer.plan
  // CHECK-SAME: alignments =
  // CHECK-SAME: canShare =
  // CHECK: mogg.async.pack
  // CHECK: mogg.async.pack
  // CHECK: mogg.output
  %high_watermark, %offset0, %offset1, %offset2 = mgp.buffer.plan(%size0, %size1, %size2) <
    alignments = #M.dense_array<64, 128, 64> : !M.array<3xui64>,
    canShare = #M.dense_array<0, 0, 0, 0, 0, 0, 0, 0, 0> : !M.array<9xui8>
  > : (index, index, index) -> (index, index, index, index)

  mef.output %ch0, %high_watermark, %offset0, %offset1, %offset2 : !mef.chain, index, index, index, index
}

// -----

// Test that an op with pre-existing staticSizes and a constant runtime operand
// is canonicalized (constant promoted to staticSizes) and then lowered with
// only one mogg.async.unpack for the remaining dynamic runtime size.

// CHECK-LABEL: mgp.model @buffer_plan_existing_static_with_constant
mgp.model @buffer_plan_existing_static_with_constant : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %dynamic_size: index) -> (!mef.chain, index, index, index, index) {
  // After canonicalization, the constant 512 is merged into staticSizes alongside
  // the pre-existing 256. Only the truly dynamic size remains as a runtime operand,
  // so there should be exactly one mogg.async.unpack.
  // CHECK: mogg.routine.async
  // CHECK: mogg.async.unpack
  // CHECK-NOT: mogg.async.unpack
  // CHECK: mgp.buffer.plan
  // CHECK-SAME: staticSizes =
  // CHECK: mogg.async.pack
  // CHECK: mogg.async.pack
  // CHECK: mogg.output
  %c512 = index.constant 512
  %high_watermark, %offset0, %offset1, %offset2 = mgp.buffer.plan(%c512, %dynamic_size) <
    alignments = #M.dense_array<64, 64, 64> : !M.array<3xui64>,
    canShare = #M.dense_array<0, 0, 0, 0, 0, 0, 0, 0, 0> : !M.array<9xui8>,
    staticSizes = #M.dense_array<256> : !M.array<1xsi64>
  > : (index, index) -> (index, index, index, index)

  mef.output %ch0, %high_watermark, %offset0, %offset1, %offset2 : !mef.chain, index, index, index, index
}
