// RUN: mo-opt --mogg-import-packages %s --mgp-lower-primitives --mogg-outline-async-routines --mogg-add-try-except-blocks --canonicalize --mogg-emit-mojo --jit --mogg-strip-kernel-decls --mogg-lower-compiled-routines --mgp-outline-models-to-callables -o %t
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t

// CHECK-LABEL: --- Running 'sync'

mgp.model @sync : mof devices [#M.device_ref<"cpu", 0>, #M.device_ref<"gpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %0 = grt.chain.create ()
  %ctx = mgp.context.create 3
  %2 = mgp.runtime.create %0, %ctx[0]
  mef.output %2, %ctx, %weights, %cpu, %gpu: !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>, !mgp.device_context<<"gpu", 0>>
} execute(%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %gpu: !mgp.device_context<<"gpu", 0>>) -> (!mef.chain) {
  %out_chain = mgp.sync(%ch0, %ctx, %gpu) device <"gpu", 0> {aRuntimeSlot = 1 : ui64} : (!mef.chain, !mgp.context, !mgp.device_context<<"gpu", 0>>) -> !mef.chain
  mef.output %out_chain : !mef.chain
}
