// RUN: mo-opt %s --mo-shape-inference --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: python3 %mo_vs_numpy_py -n Split -f %t.mlir | FileCheck %s

// CHECK: All MO values match numpy

mo.graph @serving_default<D0, D1, D2 -> D3, D4, D5, D6, D7, D8, D9, D10, D11, D12, D13, D14, D15, D16, D17, D18, D19, D20>(%arg0: !mo.tensor<[D0, D1, D2], si32>, %arg2: !mo.tensor<[6], si32>, %arg1: !mo.tensor<[1], si32>) -> (!mo.tensor<[D3, D4, D5], si32>, !mo.tensor<[D6, D7, D8], si32>, !mo.tensor<[D9, D10, D11], si32>, !mo.tensor<[D12, D13, D14], si32>, !mo.tensor<[D15, D16, D17], si32>, !mo.tensor<[D18, D19, D20], si32>) attributes {argument_names = ["input_1", "input_2", "input_3"], result_names = ["lambda", "lambda_1", "lambda_2", "lambda_3", "lambda_4", "lambda_5"]} {
  // The split axis is now a compile-time attribute; the unused %arg1 axis input
  // is retained only to keep the graph signature the numpy harness feeds.
  %1:6 = mo.split<() -> D21, D22, D23, D24, D25, D26, D27, D28, D29, D30, D31, D32, D33, D34, D35, D36, D37, D38>(%arg0, %arg2) [axis = 0] : (!mo.tensor<[D0, D1, D2], si32>, !mo.tensor<[6], si32>) -> (!mo.tensor<[D21, D22, D23], si32>, !mo.tensor<[D24, D25, D26], si32>, !mo.tensor<[D27, D28, D29], si32>, !mo.tensor<[D30, D31, D32], si32>, !mo.tensor<[D33, D34, D35], si32>, !mo.tensor<[D36, D37, D38], si32>)
  mo.output<D21, D22, D23, D24, D25, D26, D27, D28, D29, D30, D31, D32, D33, D34, D35, D36, D37, D38> %1#0, %1#1, %1#2, %1#3, %1#4, %1#5 : !mo.tensor<[D21, D22, D23], si32>, !mo.tensor<[D24, D25, D26], si32>, !mo.tensor<[D27, D28, D29], si32>, !mo.tensor<[D30, D31, D32], si32>, !mo.tensor<[D33, D34, D35], si32>, !mo.tensor<[D36, D37, D38], si32>
}
