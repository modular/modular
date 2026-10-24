// RUN: mo-opt --mogg-import-packages --mgp-lower-primitives --mogg-sequence-fusion --mogg-outline-async-routines --mogg-add-try-except-blocks --canonicalize --mogg-emit-mojo --mogg-strip-kernel-decls --jit --jit-pipeline %s -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s

// CHECK-LABEL: --- Running 'debug_print_int'
// CHECK-NEXT: 42
// CHECK-NEXT: [myval] 99
mgp.model @debug_print_int : mof devices [#M.device_ref<"cpu", 0>]
    init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  %ch1 = mgp.runtime.create %ch0, %ctx[0]
  mef.output %ch1, %ctx, %weights, %cpu
    : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context,
           %weights: !mgp.opaque<"weights_registry">,
           %cpu: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {
  %c42 = index.constant 42
  %ch1 = mgp.debug.print.int %ch0, %ctx, %c42
  %c99 = index.constant 99
  %ch2 = mgp.debug.print.int %ch1, %ctx, %c99, "myval"
  mef.output %ch2 : !mef.chain
}
