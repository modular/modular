// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-outline-structured-kernel-bodies --map-to-mogg --mogg-to-mgp | FileCheck %s
// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-outline-structured-kernel-bodies --map-to-mogg --mogg-to-mgp --mgp-exec-invariant-code-motion | FileCheck %s --check-prefix=ICM

// A pure op after an mo.while that reads none of its results keeps the chain
// from before the while, so it lifts to init. Only the output waits on the
// while.

// CHECK-LABEL: mgp.model @pure_op_after_while
// CHECK:       } execute (%[[CH:[^:]*]]: !mef.chain
// CHECK:       %[[WHILE:.*]]:4 = grt.while (%[[CH]]: !mef.chain,
// CHECK:       } do {
// CHECK:       mef.output
// CHECK-NEXT:  }
// CHECK-NOT:   %[[WHILE]]#
// CHECK:       %[[PRE:.*]] = grt.chain.create (%[[CH]], %[[CH]])
// CHECK-NOT:   %[[WHILE]]#
// CHECK:       mogg.routine.async {{(dispatch\([a-z]+\) )?}}(%[[PRE]],
// CHECK:       mef.output %[[WHILE]]#0, %[[WHILE]]#1,

// ICM-LABEL:   mgp.model @pure_op_after_while
// ICM:         mogg.routine.async
// ICM:         } execute (
// ICM:         grt.while
// ICM:         } do {
// ICM:         mef.output
// ICM-NEXT:    }
// ICM-NOT:     mogg.routine.async
// ICM:         mef.output
mo.graph @pure_op_after_while(%x: !mo.tensor<[], f32>, %y: !mo.tensor<[], f32>) -> (!mo.tensor<[], f32>, !mo.tensor<[2], f32>) {
  %res = mo.while (%x as %xi: !mo.tensor<[], f32>) {
    %cond = mo.greater (%xi, %y) : (!mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[], bool>
    mo.while.condition(%cond) %xi : !mo.tensor<[], f32>
  } do {
    %new_x = mo.add (%xi, %xi) : !mo.tensor<[], f32>
    mo.yield %new_x : !mo.tensor<[], f32>
  }
  %w = mo.constant <#M.dense_array<1.0, 2.0> : tensor<2xf32>> : !mo.tensor<[2], f32>
  %r = mo.negative(%w) : !mo.tensor<[2], f32>
  mo.output %res, %r : !mo.tensor<[], f32>, !mo.tensor<[2], f32>
}

// An op after an mo.while that reads its result waits on the while's chain.

// CHECK-LABEL: mgp.model @use_after_while
// CHECK:       } execute (%[[CH:[^:]*]]: !mef.chain
// CHECK:       %[[WHILE:.*]]:4 = grt.while
// CHECK:       } do {
// CHECK:       mef.output
// CHECK-NEXT:  }
// CHECK:       %[[GUARD:.*]] = grt.chain.create (%[[WHILE]]#0, %[[CH]])
// CHECK:       mogg.routine.async {{(dispatch\([a-z]+\) )?}}(%[[GUARD]],
mo.graph @use_after_while(%x: !mo.tensor<[], f32>, %y: !mo.tensor<[], f32>) -> !mo.tensor<[], f32> {
  %res = mo.while (%x as %xi: !mo.tensor<[], f32>) {
    %cond = mo.greater (%xi, %y) : (!mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[], bool>
    mo.while.condition(%cond) %xi : !mo.tensor<[], f32>
  } do {
    %new_x = mo.add (%xi, %xi) : !mo.tensor<[], f32>
    mo.yield %new_x : !mo.tensor<[], f32>
  }
  %r = mo.negative(%res) : !mo.tensor<[], f32>
  mo.output %r : !mo.tensor<[], f32>
}
