// RUN: mo-opt -allow-unregistered-dialect --canonicalize --split-input-file %s | FileCheck %s

mo.graph @genericRegionChainCreate(%arg0: index, %arg1: index) {
  %ch = grt.chain.create(%arg0 : index, %arg1 : index)

  mogg.routine.async (%ch) : (%in : !mef.chain) {
    mogg.output
  } inOutSignature = <"."> {device = #M.device_ref<"cpu", 0>}

  mo.output
}

// CHECK-LABEL: mo.graph @genericRegionChainCreate
// CHECK:       (%[[ARG0:.*]]: index, %[[ARG1:.*]]: index)
// CHECK:       mogg.routine.async (%[[ARG0]], %[[ARG1]])
// CHECK:       inOutSignature = <"..">

// -----

mo.graph @genericRegionChainCreateNoInline(%arg0: index, %arg1: index) {
  %ch = grt.chain.create(%arg0 : index, %arg1 : index)

  mogg.routine.async (%ch) : (%in : !mef.chain) {
    mogg.output
  } inOutSignature = <"."> {device = #M.device_ref<"cpu", 0>}

  mogg.routine.async (%ch) : (%in : !mef.chain) {
    mogg.output
  } inOutSignature = <"."> {device = #M.device_ref<"cpu", 0>}

  mo.output
}

// CHECK-LABEL: mo.graph @genericRegionChainCreateNoInline
// CHECK:       (%[[ARG0:.*]]: index, %[[ARG1:.*]]: index)
// CHECK:       %[[CH:.*]] = grt.chain.create (%[[ARG0]]: index, %[[ARG1]]: index)
// CHECK:       mogg.routine.async (%[[CH]])
// CHECK:       inOutSignature = <".">
// CHECK:       mogg.routine.async (%[[CH]])
// CHECK:       inOutSignature = <".">

// -----

mo.graph @genericRegionEmptyChain() {
  %ch = grt.chain.create()

   mogg.routine.async (%ch) : (%in : !mef.chain) {
    mogg.output
  } inOutSignature = <"."> {device = #M.device_ref<"cpu", 0>}

  mo.output
}

// CHECK-LABEL: mo.graph @genericRegionEmptyChain
// CHECK:       mogg.routine.async ()
// CHECK:       inOutSignature = <"">

// -----

// Duplicate operands are collapsed to the first occurrence, the duplicate body
// argument is erased, and uses of it are redirected to the first occurrence's
// argument. The signature drops the corresponding entry. The resulting
// `mogg.async.pack(mogg.async.unpack(x))` chain then folds to `x`, leaving
// just `mogg.output %[[A]]`.
mo.graph @genericRegionDedupOperands(%arg0: index, %arg1: index) {
  %0 = mogg.routine.async (%arg0, %arg1, %arg0) : (%a: index, %b: index, %c: index) -> index {
    %u = mogg.async.unpack(%c) : index
    %p = mogg.async.pack(%u) : index
    mogg.output %p : !mogg.async<index>
  } inOutSignature = <"..."> {device = #M.device_ref<"cpu", 0>}

  mo.output
}

// CHECK-LABEL: mo.graph @genericRegionDedupOperands
// CHECK:       (%[[ARG0:.*]]: index, %[[ARG1:.*]]: index)
// CHECK:       mogg.routine.async (%[[ARG0]], %[[ARG1]]) :
// CHECK-SAME:  (%[[A:[^,]*]]: index, %[[B:[^,]*]]: index)
// CHECK:         mogg.output %[[A]] : !mogg.async<index>
// CHECK:       inOutSignature = <"..">

// -----

// `mogg.async.pack(mogg.async.unpack(x))` folds to `x`. The unpack's input is
// the region body arg (`!mogg.async<index>` printed as `index`), so the new
// pack fold elides the round-trip; the now-dead unpack drops out via DCE,
// leaving a single `mogg.output %[[A]]`.
mo.graph @fold_pack_unpack_round_trip(%arg0: index) {
  %0 = mogg.routine.async (%arg0) : (%a: index) -> index {
    %u = mogg.async.unpack(%a) : index
    %p = mogg.async.pack(%u) : index
    mogg.output %p : !mogg.async<index>
  } inOutSignature = <"."> {device = #M.device_ref<"cpu", 0>}

  mo.output
}

// CHECK-LABEL: mo.graph @fold_pack_unpack_round_trip
// CHECK:       (%[[ARG0:.*]]: index)
// CHECK:       mogg.routine.async (%[[ARG0]]) :
// CHECK-SAME:  (%[[A:[^,)]*]]: index)
// CHECK:         mogg.output %[[A]] : !mogg.async<index>
// CHECK:       inOutSignature = <".">

// -----

// When duplicate operands carry different in/out semantics, the surviving
// operand absorbs the union: `kIn` ('i') + `kMut` ('m') -> `kMut` ('m').
mo.graph @genericRegionDedupUnionSemantics(%arg0: index) {
  mogg.routine.async (%arg0, %arg0) : (%a: index, %b: index) {
    mogg.output
  } inOutSignature = <"im"> {device = #M.device_ref<"cpu", 0>}

  mo.output
}

// CHECK-LABEL: mo.graph @genericRegionDedupUnionSemantics
// CHECK:       mogg.routine.async (%[[ARG0:[^,)]*]]) :
// CHECK-SAME:  (%{{[^,]*}}: index)
// CHECK:       inOutSignature = <"m">

// -----

// Unused mogg.try results are dropped: the try/except mogg.yield ops drop the
// corresponding operands, and the now-dead mogg.async.error in the except
// block is erased.
func.func @tryUnusedResult(%arg0: index) -> index {
  %0:2 = mogg.try {
    %a = mogg.async.pack(%arg0) : index
    mogg.yield %a, %a : !mogg.async<index>, !mogg.async<index>
  } except(%err) {
    %e0 = mogg.async.error %err : !mogg.async<index>
    %e1 = mogg.async.error %err : !mogg.async<index>
    mogg.yield %e0, %e1 : !mogg.async<index>, !mogg.async<index>
  } : !mogg.async<index>, !mogg.async<index>
  %u = mogg.async.unpack(%0#0) : index
  return %u : index
}

// CHECK-LABEL: func.func @tryUnusedResult
// CHECK:       %[[TRY:.*]] = mogg.try {
// CHECK:         mogg.yield %{{[^,]*}} : !mogg.async<index>
// CHECK:       } except(%{{[^)]+}}) {
// CHECK:         %[[ERR:.*]] = mogg.async.error
// CHECK-NOT:     mogg.async.error
// CHECK:         mogg.yield %[[ERR]] : !mogg.async<index>
// CHECK:       } : !mogg.async<index>
// CHECK:       mogg.async.unpack(%[[TRY]])

// -----

// Duplicate `!mef.chain` outputs are collapsed to a single output, and uses of
// the removed chain results are rewired to the surviving one.
mgp.model @remove_redundant_chains : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"cpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  %2 = mgp.runtime.create %0, %1[1]
  mef.output %2, %1, %arg0, %arg1 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.opaque<"weights_registry">, %arg3: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain) {
  %true = index.bool.constant true
  %true_0 = index.bool.constant true
  %0:2 = mogg.routine.async (%true, %true_0) : (%arg4: i1, %arg5: i1)  -> (!mef.chain, !mef.chain) {
    %1 = mogg.async.ready
    %2 = mogg.async.ready
    mogg.output %1, %2 : !mef.chain, !mef.chain
  } fusionInfo = #mogg.fusion<#mogg.fusion_sequence<[#mogg.fusion_leaf<"mgp.assert">, #mogg.fusion_leaf<"mgp.assert">]>> canBeFusedInto = true canFuseIntoOther = true inOutSignature = <""> {device = #M.device_ref<"cpu", 0>, emitsMojo = false}
  %res = "test.sink"(%0#0, %0#1) : (!mef.chain, !mef.chain) -> !mef.chain
  mef.output %res : !mef.chain
}

// CHECK-LABEL: mgp.model @remove_redundant_chains
// CHECK:         %[[OUT:[^ ]*]] = mogg.routine.async ({{.*}}) -> !mef.chain {
// CHECK:           mogg.output %{{[^ ]*}} : !mef.chain
// CHECK:         "test.sink"(%[[OUT]], %[[OUT]])

// -----

// Op attributes (e.g. emitsMojo) are preserved when a region is left unchanged.
mgp.model @preserve_kernel_attributes : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"cpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  %2 = mgp.runtime.create %0, %1[1]
  mef.output %2, %1, %arg0, %arg1 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.opaque<"weights_registry">, %arg3: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain) {
  %true = index.bool.constant true
  %true_0 = index.bool.constant true
  mogg.routine.async (%true, %true_0) : (%arg4: i1, %arg5: i1)  -> () {
    mogg.output
  } fusionInfo = #mogg.fusion<#mogg.fusion_sequence<[#mogg.fusion_leaf<"mgp.assert">, #mogg.fusion_leaf<"mgp.assert">]>> canBeFusedInto = true canFuseIntoOther = true inOutSignature = <""> {device = #M.device_ref<"cpu", 0>, emitsMojo = true}
  mef.output %arg0 : !mef.chain
}

// CHECK-LABEL: mgp.model @preserve_kernel_attributes
// CHECK:           emitsMojo = true
