// RUN: mo-opt %s --mo-shape-inference --gc-pipeline -o %t.mlir
// RUN: python3 %mo_vs_numpy_py -n MultiViewFusion -f %t.mlir | FileCheck %s

// CHECK: All MO values match numpy

// NOTE: I have manually demangled the order of the TF inputs & outputs to be
// in a sane order. (#18187)

mo.graph @serving_default<D0, D1, D2 -> D3, D4, D5>(%arg0: !mo.tensor<[D0, D1, D2], f32>, %arg1: !mo.tensor<[3], si32>, %arg2: !mo.tensor<[3], si32>, %arg3: !mo.tensor<[3], si32>, %arg4: !mo.tensor<[3], si32>) -> !mo.tensor<[D3, D4, D5], f32> attributes {argument_names = ["input_1", "input_4", "input_3", "input_2", "input_5"], result_names = ["lambda"]} {
  %0 = mo.slice<() -> D6, D7, D8>(%arg0, %arg1, %arg2, %arg3) : (!mo.tensor<[D0, D1, D2], f32>, !mo.tensor<[3], si32>, !mo.tensor<[3], si32>, !mo.tensor<[3], si32>) -> !mo.tensor<[D6, D7, D8], f32>
  %1 = mo.transpose<() -> D9, D10, D11>(%0, %arg4) : (!mo.tensor<[D6, D7, D8], f32>, !mo.tensor<[3], si32>) -> !mo.tensor<[D9, D10, D11], f32>
  %2 = mo.shape_of(%1) : (!mo.tensor<[D9, D10, D11], f32>) -> !mo.tensor<[3], si64>
  %3 = mo.shape_of(%1) : (!mo.tensor<[D9, D10, D11], f32>) -> !mo.tensor<[3], si64>
  %4 = mo.broadcast_shape(%2, %3) : (!mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[3], si64>
  %5 = mo.broadcast_to(%1, %4) : (!mo.tensor<[D9, D10, D11], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[D9, D10, D11], f32>
  %6 = mo.broadcast_to(%1, %4) : (!mo.tensor<[D9, D10, D11], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[D9, D10, D11], f32>
  %7 = mo.add(%5, %6) : !mo.tensor<[D9, D10, D11], f32>
  mo.output<D9, D10, D11> %7 : !mo.tensor<[D9, D10, D11], f32>
}
