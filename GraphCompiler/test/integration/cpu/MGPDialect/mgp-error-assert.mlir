// RUN: mo-opt --mogg-import-packages --mgp-lower-primitives --mogg-sequence-fusion --mogg-outline-async-routines --mogg-add-try-except-blocks --canonicalize --mogg-emit-mojo --mogg-strip-kernel-decls --jit --mogg-lower-compiled-routines --mgp-outline-models-to-callables %s -o %t.mlir
// RUN: not mt --execute --model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

// CHECK-LABEL: --- Running 'assert_false'
// CHECK: <error:
// CHECK: the buffer must not contain 0>
mgp.model @assert_false : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute(%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {
  %buf = mgp.buffer.constant #M.dense_array<0> : !M.array<1xui8>
  %0 = mgp.buffer.to_bool %ch0, %buf : !mgp.buffer
  %ch2 = mgp.assert %0, "the buffer must not contain 0"
  mef.output %ch2 : !mef.chain
}
