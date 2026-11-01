// RUN: mo-opt %s --mogg-import-packages --mgp-lower-primitives --mogg-sequence-fusion --mogg-outline-async-routines --mogg-add-try-except-blocks --canonicalize --mogg-emit-mojo --mogg-dump-mojo 2>&1 | FileCheck %s

// Test that when ops with different source note metadata (FusedLoc) are
// sequence-fused, the outer MGP region wraps errors with the region name.
// See GEX-2678.
//
// After --mgp-lower-primitives, each mgp.assert becomes a
// AsyncRoutineOp with the op's FusedLoc metadata.
// --mogg-sequence-fusion merges adjacent regions into one kernel.
// The resulting outer-level stub uses AsyncErrorOp to wrap the error with the
// region name via mogg_format_region_error.

// This test has no inner MOGG ABI stubs with isThrows (the ops are inlined
// directly into the outer region), so only mogg_format_region_error appears.
// CHECK:       mogg_format_region_error(
// CHECK:       mogg_async_error_value(

mgp.model @fused_source_notes : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {
  %cond = index.bool.constant true
  // Chain the second assert through the first to guarantee sequential
  // dependency and sequence fusion.
  %ch1 = mgp.assert %cond, "alpha failed" loc(fused<"  File \22model.py\22, line 10\n    trace_alpha\n">["alpha":0:0])
  %buf = mgp.buffer.constant #M.dense_array<1> : !M.array<1xui8>
  %flag = mgp.buffer.to_bool %ch1, %buf : !mgp.buffer
  %ch2 = mgp.assert %flag, "beta failed" loc(fused<"  File \22model.py\22, line 20\n    trace_beta\n">["beta":0:0])
  mef.output %ch2 : !mef.chain
}
