// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-outline-structured-kernel-bodies --map-to-mogg --mogg-to-mgp | FileCheck %s
// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-outline-structured-kernel-bodies --map-to-mogg --mogg-to-mgp --mgp-exec-invariant-code-motion | FileCheck %s --check-prefix=ICM

mo.graph @sub(%arg0: !mo.tensor<[2], f32>) -> !mo.tensor<[2], f32> isSubgraph {
  %0 = mo.relu(%arg0) : !mo.tensor<[2], f32>
  mo.output %0 : !mo.tensor<[2], f32>
}

// A pure op after an mo.call that reads none of its results keeps the chain
// from before the call, so it lifts to init. Only the output waits on the
// call.

// CHECK-LABEL: mgp.model @pure_op_after_call
// CHECK:       } execute (%[[CH:[^:]*]]: !mef.chain
// CHECK:       %[[CALL:.*]]:2 = mgp.call @sub(%[[CH]],
// CHECK-NOT:   %[[CALL]]#0
// CHECK:       mogg.routine.async
// CHECK:       mef.output %[[CALL]]#0,

// ICM-LABEL:   mgp.model @pure_op_after_call
// ICM:         mogg.routine.async
// ICM:         } execute (
// ICM:         mgp.call @sub(
// ICM-NOT:     mogg.routine.async
// ICM:         mef.output
mo.graph @pure_op_after_call(%x: !mo.tensor<[2], f32>) -> (!mo.tensor<[2], f32>, !mo.tensor<[2], f32>) {
  %c = mo.call @sub(%x) : (!mo.tensor<[2], f32>) -> !mo.tensor<[2], f32>
  %w = mo.constant <#M.dense_array<1.0, 2.0> : tensor<2xf32>> : !mo.tensor<[2], f32>
  %r = mo.negative(%w) : !mo.tensor<[2], f32>
  mo.output %c, %r : !mo.tensor<[2], f32>, !mo.tensor<[2], f32>
}

// An op after an mo.call that reads its result waits on the call's chain.

// CHECK-LABEL: mgp.model @use_after_call
// CHECK:       %[[CALL:.*]]:2 = mgp.call @sub(
// CHECK:       %[[GUARD:.*]] = grt.chain.create (%[[CALL]]#0
// CHECK:       mogg.routine.async {{(dispatch\([a-z]+\) )?}}(%[[GUARD]],
// CHECK:       mef.output %[[CALL]]#0,
mo.graph @use_after_call(%x: !mo.tensor<[2], f32>) -> !mo.tensor<[2], f32> {
  %c = mo.call @sub(%x) : (!mo.tensor<[2], f32>) -> !mo.tensor<[2], f32>
  %r = mo.negative(%c) : !mo.tensor<[2], f32>
  mo.output %r : !mo.tensor<[2], f32>
}

// A second call waits on the first, and a pure op after both still keeps the
// chain from before them.

// CHECK-LABEL: mgp.model @call_after_call
// CHECK:       } execute (%[[CH:[^:]*]]: !mef.chain
// CHECK:       %[[FIRST:.*]]:2 = mgp.call @sub(%[[CH]],
// CHECK:       %[[SECOND:.*]]:2 = mgp.call @sub(%[[FIRST]]#0,
// CHECK-NOT:   %[[SECOND]]#0
// CHECK:       mogg.routine.async
// CHECK:       mef.output %[[SECOND]]#0,

// ICM-LABEL:   mgp.model @call_after_call
// ICM:         mogg.routine.async
// ICM:         } execute (
// ICM-NOT:     mogg.routine.async
// ICM:         mef.output
mo.graph @call_after_call(%x: !mo.tensor<[2], f32>, %y: !mo.tensor<[2], f32>) -> (!mo.tensor<[2], f32>, !mo.tensor<[2], f32>, !mo.tensor<[2], f32>) {
  %a = mo.call @sub(%x) : (!mo.tensor<[2], f32>) -> !mo.tensor<[2], f32>
  %b = mo.call @sub(%y) : (!mo.tensor<[2], f32>) -> !mo.tensor<[2], f32>
  %w = mo.constant <#M.dense_array<1.0, 2.0> : tensor<2xf32>> : !mo.tensor<[2], f32>
  %r = mo.negative(%w) : !mo.tensor<[2], f32>
  mo.output %a, %b, %r : !mo.tensor<[2], f32>, !mo.tensor<[2], f32>, !mo.tensor<[2], f32>
}
