// RUN: mo-opt --canonicalize --cse %s | FileCheck %s

// The kernel fills a buffer its operands point to, so neither an unused
// result nor an identical earlier call makes the call removable.
// CHECK-LABEL: func.func @fill_twice
// CHECK:         mogg.compiled.call @fill
// CHECK:         mogg.compiled.call @fill
// CHECK:         return
func.func @fill_twice(%ch: !mef.chain, %buf: !mgp.buffer, %ctx: !mgp.device_context<<"cpu", 0>>) {
  %0 = mogg.compiled.call @fill(%ch, %buf, %ctx) dispatch(deferred) constantResources = [], usedParams = [] {device = #M.device_ref<"cpu", 0>} : (!mef.chain, !mgp.buffer, !mgp.device_context<<"cpu", 0>>) -> !mef.chain
  %1 = mogg.compiled.call @fill(%ch, %buf, %ctx) dispatch(deferred) constantResources = [], usedParams = [] {device = #M.device_ref<"cpu", 0>} : (!mef.chain, !mgp.buffer, !mgp.device_context<<"cpu", 0>>) -> !mef.chain
  return
}
