// RUN: mo-opt -mogg-lift-buffer-cache-slots %s | FileCheck %s
// RUN: mo-opt -mogg-lift-buffer-cache-slots -mogg-merge-identical-funcs -symbol-dce %s | FileCheck %s --check-prefix=MERGED

// GEX-3685: two MGP-ABI kernels that differ only in the integer slot operand
// of `mgp.buffer.get_cached`. Without the lift the kernels are not merge
// candidates because `MergeIdenticalFuncs` hashes the inline
// `index.constant` value as part of the body. After the lift their bodies
// are identical and the merge pass collapses them.

// First kernel: slot 197 lifted to a new index argument; body now reads
// from the new arg instead of an inline constant.

// CHECK-LABEL: mogg.func @lift_a private
// CHECK-SAME:    (%[[CHAIN_A:[^:]+]]: !mef.chain, %[[CTX_A:[^:]+]]: !mgp.context, %[[SLOT_A:[^:]+]]: index)
// CHECK-NOT:     index.constant
// CHECK:         mgp.buffer.get_cached %[[CHAIN_A]], %[[CTX_A]][%[[SLOT_A]]]
mogg.func @lift_a private(%arg0: !mef.chain, %arg1: !mgp.context) -> (!mgp.buffer<<"gpu", 0>>)
  attributes {abi = #mogg.abi<MGP>, constantResources = [], device = "gpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"lift_a">>} {
  %slot = index.constant 197
  %buf = mgp.buffer.get_cached %arg0, %arg1[%slot] : !mgp.buffer<<"gpu", 0>>
  mogg.output %buf : !mgp.buffer<<"gpu", 0>>
}

// Second kernel: differs only in the slot value; same lift treatment.

// CHECK-LABEL: mogg.func @lift_b private
// CHECK-SAME:    (%[[CHAIN_B:[^:]+]]: !mef.chain, %[[CTX_B:[^:]+]]: !mgp.context, %[[SLOT_B:[^:]+]]: index)
// CHECK-NOT:     index.constant
// CHECK:         mgp.buffer.get_cached %[[CHAIN_B]], %[[CTX_B]][%[[SLOT_B]]]
mogg.func @lift_b private(%arg0: !mef.chain, %arg1: !mgp.context) -> (!mgp.buffer<<"gpu", 0>>)
  attributes {abi = #mogg.abi<MGP>, constantResources = [], device = "gpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"lift_b">>} {
  %slot = index.constant 261
  %buf = mgp.buffer.get_cached %arg0, %arg1[%slot] : !mgp.buffer<<"gpu", 0>>
  mogg.output %buf : !mgp.buffer<<"gpu", 0>>
}

// MOGG-ABI kernels (abi = 0) must be skipped: their bodies don't legally
// contain `mgp.buffer.get_cached`, but if a synthetic input had a benign
// `index.constant` (no `get_cached` use) we must not touch them. They also
// aren't called via CompiledCallOp, so caller rewriting wouldn't be sound
// even if we tried.

// CHECK-LABEL: mogg.func @mogg_abi_kernel private
// CHECK-SAME:    () attributes
// CHECK:         %{{.+}} = index.constant 7
mogg.func @mogg_abi_kernel private()
  attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "cpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mogg_abi_kernel">>} {
  %0 = index.constant 7
  mogg.output
}

// Each CompiledCallOp call site gets a fresh `index.constant` with the
// original kernel-internal value as a new trailing operand.

// CHECK-LABEL: mgp.model @top_level
// CHECK:         %[[V197:[^ ]+]] = index.constant 197
// CHECK:         mogg.compiled.call @lift_a({{[^,]+}}, {{[^,]+}}, %[[V197]])
// CHECK-SAME:      (!mef.chain, !mgp.context, index) -> !mgp.buffer<<"gpu", 0>>
// CHECK:         %[[V261:[^ ]+]] = index.constant 261
// CHECK:         mogg.compiled.call @lift_b({{[^,]+}}, {{[^,]+}}, %[[V261]])
// CHECK-SAME:      (!mef.chain, !mgp.context, index) -> !mgp.buffer<<"gpu", 0>>

// With merge + symbol-DCE chained, only one kernel body survives.
// MergeIdenticalFuncs picks the lexicographically-smallest name as the
// canonical one (`@lift_a`); both call sites get rewritten to point at it.
// MERGED:     mogg.func @lift_a private
// MERGED-NOT: mogg.func @lift_b private
// MERGED:     mogg.compiled.call @lift_a
// MERGED:     mogg.compiled.call @lift_a
mgp.model @top_level subgraph : mof devices [#M.device_ref<"cpu", 0>, #M.device_ref<"gpu", 0>]
init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"cpu", 0>>, %arg2: !mgp.device_context<<"gpu", 0>>) {
    %0 = grt.chain.create ()
    %1 = mgp.context.create 2
    %2 = mgp.runtime.create %0, %1[1]
    mef.output %0, %1, %arg0, %arg1, %arg2 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>, !mgp.device_context<<"gpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.opaque<"weights_registry">, %arg3: !mgp.device_context<<"cpu", 0>>, %arg4: !mgp.device_context<<"gpu", 0>>, %idx: index) -> (!mef.chain) {
    %a = mogg.compiled.call @lift_a(%arg0, %arg1) fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mogg.routine.async">> constantResources = [], usedParams = [] {device = #M.device_ref<"gpu", 0>}
        : (!mef.chain, !mgp.context) -> (!mgp.buffer<<"gpu", 0>>)
    %b = mogg.compiled.call @lift_b(%arg0, %arg1) fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mogg.routine.async">> constantResources = [], usedParams = [] {device = #M.device_ref<"gpu", 0>}
        : (!mef.chain, !mgp.context) -> (!mgp.buffer<<"gpu", 0>>)
    mef.output %arg0 : !mef.chain
}
