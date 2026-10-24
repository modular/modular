// RUN: mo-opt %s --mo-shape-inference --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: python3 %mo_vs_numpy_py -n GatherReduceConstAxis -f %t.mlir | FileCheck %s

// CHECK: All MO values match numpy

// This is the original gather reduce pattern.

mo.graph @serving_default<D0, D1, D2, D3 -> D4, D6>(%arg0: !mo.tensor<[D0, D1], f32>, %arg1: !mo.tensor<[D2, D3], si32>) -> !mo.tensor<[D4, 1, D6], f32> attributes {argument_names = ["input_1", "input_2"], result_names = ["lambda"]} {
  %0 = mo.gather<() -> D7, D8, D9>(%arg0, %arg1) [axis = 0] : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D2, D3], si32>) -> !mo.tensor<[D7, D8, D9], f32>
  %1 = mo.reduce.add(%0) : (!mo.tensor<[D7, D8, D9], f32>) -> !mo.tensor<[D7, 1, D9], f32>
  mo.output<D7, D9> %1 : !mo.tensor<[D7, 1, D9], f32>
}
