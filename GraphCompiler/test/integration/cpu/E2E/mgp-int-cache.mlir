// RUN: mo-opt --mogg-import-packages --mgp-lower-primitives --mogg-sequence-fusion --mogg-outline-async-routines --mogg-add-try-except-blocks --canonicalize --mogg-emit-mojo --mogg-strip-kernel-decls --jit --jit-pipeline %s -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s

// CHECK-LABEL: --- Running 'int_cache'
// CHECK-NEXT: [val] 42
mgp.model @int_cache : mof devices [#M.device_ref<"cpu", 0>]
    init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 3
  %ch1 = mgp.runtime.create %ch0, %ctx[1]
  %c42 = index.constant 42
  %ch2 = mgp.int.cache %ch1, %ctx[2], %c42
  mef.output %ch2, %ctx, %weights, %cpu
    : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context,
           %weights: !mgp.opaque<"weights_registry">,
           %cpu: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {
  %slot = index.constant 2
  %val = mgp.int.get_cached %ch0, %ctx[%slot]
  %ch1 = mgp.debug.print.int %ch0, %ctx, %val, "val"
  mef.output %ch1 : !mef.chain
}
