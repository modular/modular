// RUN: mo-opt %s --mo-shape-inference --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

// A concat of concats fuses into one kernel whose select holds a select in
// each of its regions; each nested select's closure needs its own name. The
// inner concats mix widths 1 and 2.

// CHECK: --- Running 'concat_of_concats':
// CHECK-NEXT: 'concat_of_concats' returned tensor<4x3xsi32> [1, 3, 4, 1, 5, 6, 7, 9, 10, 8, 11, 12]
mo.graph @concat_of_concats(%a: !mo.tensor<[2, 1], si32>) -> !mo.tensor<[4, 3], si32> {
  %b = mo.constant <#M.dense_array<3, 4, 5, 6> : tensor<2x2xsi32>> : !mo.tensor<[2, 2], si32>
  %c = mo.constant <#M.dense_array<7, 8> : tensor<2x1xsi32>> : !mo.tensor<[2, 1], si32>
  %d = mo.constant <#M.dense_array<9, 10, 11, 12> : tensor<2x2xsi32>> : !mo.tensor<[2, 2], si32>
  %ab = mo.concat[1](%a, %b) : (!mo.tensor<[2, 1], si32>, !mo.tensor<[2, 2], si32>) -> !mo.tensor<[2, 3], si32>
  %cd = mo.concat[1](%c, %d) : (!mo.tensor<[2, 1], si32>, !mo.tensor<[2, 2], si32>) -> !mo.tensor<[2, 3], si32>
  %out = mo.concat[0](%ab, %cd) : (!mo.tensor<[2, 3], si32>, !mo.tensor<[2, 3], si32>) -> !mo.tensor<[4, 3], si32>
  mo.output %out : !mo.tensor<[4, 3], si32>
}
