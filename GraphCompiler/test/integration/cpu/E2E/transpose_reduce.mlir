// RUN: mo-opt %s --mo-shape-inference --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: python3 %mo_vs_numpy_py -n TransposeReduce -f %t.mlir | FileCheck %s

// CHECK: All MO values match numpy

module {
  mo.graph @serving_default<D0, D1, D2, D3 -> D4, D5, D6>(%arg0: !mo.tensor<[D0, D1, D2, D3], f32>, %arg2: !mo.tensor<[1], si32>, %arg1: !mo.tensor<[4], si32>) -> !mo.tensor<[D4, D5, D6, 1], f32> attributes {argument_names = ["input_1", "input_3", "input_2"], result_names = ["lambda"]} {
    %0 = mo.transpose<() -> D8, D9, D10, D11>(%arg0, %arg1) : (!mo.tensor<[D0, D1, D2, D3], f32>, !mo.tensor<[4], si32>) -> !mo.tensor<[D8, D9, D10, D11], f32>
    %3 = mo.reduce.add(%0) : (!mo.tensor<[D8, D9, D10, D11], f32>) -> !mo.tensor<[D8, D9, D10, 1], f32>
    mo.output<D8, D9, D10> %3 : !mo.tensor<[D8, D9, D10, 1], f32>
  }
}
