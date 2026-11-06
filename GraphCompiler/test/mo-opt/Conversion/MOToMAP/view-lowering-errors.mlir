// RUN: not mo-opt %s --mo-to-map 2>&1 | FileCheck %s

// A failure inside a view op's lowering surfaces as a pass failure, not a
// crash or a half-lowered graph. `mo.slice` must preserve rank, so a rank-3
// output from a rank-2 input fails the lowering's own check.

// CHECK: error: 'mo.slice' op MOToMAP: slice must preserve rank
mo.graph @sliceRankMismatch(%input: !mo.tensor<[10, 64], f32>) -> !mo.tensor<[3, 32, 1], f32> {
  %start = mo.constant <#M.dense_array<1, 32> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop = mo.constant <#M.dense_array<4, 64> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %res = mo.slice(%input, %start, %stop, %step) : (
    !mo.tensor<[10, 64], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[3, 32, 1], f32>
  mo.output %res : !mo.tensor<[3, 32, 1], f32>
}
