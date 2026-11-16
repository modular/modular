// RUN: mo-opt %s --mo-shape-inference --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

// A concat feeding a reduction: fused into the fold, or kept a kernel of its
// own where the select can't run (a per-element phase, a transposed read).
// `%a` is all ones; `%b` holds 1 to 8.

// The fold reads the concat beside another input, so the select stays in the
// fold and the row's tile holds `%c`.
// CHECK: --- Running 'sum_of_concat_plus':
// CHECK-NEXT: 'sum_of_concat_plus' returned tensor<2x1xsi32> [94, 110]
mo.graph @sum_of_concat_plus(%a: !mo.tensor<[2, 4], si32>) -> !mo.tensor<[2, 1], si32> {
  %b = mo.constant <#M.dense_array<1, 2, 3, 4, 5, 6, 7, 8> : tensor<2x4xsi32>> : !mo.tensor<[2, 4], si32>
  %c = mo.constant <#M.dense_array<10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10> : tensor<2x8xsi32>> : !mo.tensor<[2, 8], si32>
  %ab = mo.concat[1](%a, %b) : (!mo.tensor<[2, 4], si32>, !mo.tensor<[2, 4], si32>) -> !mo.tensor<[2, 8], si32>
  %x = mo.add(%ab, %c) : !mo.tensor<[2, 8], si32>
  %r = mo.reduce.add(%x) : (!mo.tensor<[2, 8], si32>) -> !mo.tensor<[2, 1], si32>
  mo.output %r : !mo.tensor<[2, 1], si32>
}

// CHECK: --- Running 'concat_plus_its_sum':
// CHECK-NEXT: 'concat_plus_its_sum' returned tensor<2x8xsi32> [15, 15, 15, 15, 15, 16, 17, 18, 31, 31, 31, 31, 35, 36, 37, 38]
mo.graph @concat_plus_its_sum(%a: !mo.tensor<[2, 4], si32>) -> !mo.tensor<[2, 8], si32> {
  %b = mo.constant <#M.dense_array<1, 2, 3, 4, 5, 6, 7, 8> : tensor<2x4xsi32>> : !mo.tensor<[2, 4], si32>
  %ab = mo.concat[1](%a, %b) : (!mo.tensor<[2, 4], si32>, !mo.tensor<[2, 4], si32>) -> !mo.tensor<[2, 8], si32>
  %s = mo.reduce.add(%ab) : (!mo.tensor<[2, 8], si32>) -> !mo.tensor<[2, 1], si32>
  %sb = mo.static.broadcast_to(%s) : !mo.tensor<[2, 1], si32> -> !mo.tensor<[2, 8], si32>
  %y = mo.add(%ab, %sb) : !mo.tensor<[2, 8], si32>
  mo.output %y : !mo.tensor<[2, 8], si32>
}

// CHECK: --- Running 'sum_of_transposed_concat':
// CHECK-NEXT: 'sum_of_transposed_concat' returned tensor<8x1xsi32> [2, 2, 2, 2, 6, 8, 10, 12]
mo.graph @sum_of_transposed_concat(%a: !mo.tensor<[2, 4], si32>) -> !mo.tensor<[8, 1], si32> {
  %b = mo.constant <#M.dense_array<1, 2, 3, 4, 5, 6, 7, 8> : tensor<2x4xsi32>> : !mo.tensor<[2, 4], si32>
  %perm = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %ab = mo.concat[1](%a, %b) : (!mo.tensor<[2, 4], si32>, !mo.tensor<[2, 4], si32>) -> !mo.tensor<[2, 8], si32>
  %t = mo.transpose(%ab, %perm) : (!mo.tensor<[2, 8], si32>, !mo.tensor<[2], si64>) -> !mo.tensor<[8, 2], si32>
  %r = mo.reduce.add(%t) : (!mo.tensor<[8, 2], si32>) -> !mo.tensor<[8, 1], si32>
  mo.output %r : !mo.tensor<[8, 1], si32>
}
