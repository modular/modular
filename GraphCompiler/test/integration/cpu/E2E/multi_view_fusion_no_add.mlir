// RUN: mo-opt %s --mo-shape-inference --gc-pipeline -o %t.mlir
// RUN: python3 %mo_vs_numpy_py -n MultiViewFusionNoAdd -f %t.mlir | FileCheck %s

// CHECK: All MO values match numpy

mo.graph @serving_default<D0, D1, D2 -> D3, D4, D5>(%arg0: !mo.tensor<[D0, D1, D2], f32>, %arg3: !mo.tensor<[3], si32>, %arg2: !mo.tensor<[3], si32>, %arg1: !mo.tensor<[3], si32>, %arg4: !mo.tensor<[3], si32>) -> !mo.tensor<[D3, D4, D5], f32> attributes {argument_names = ["input_1", "input_4", "input_3", "input_2", "input_5"], result_names = ["lambda"]} {
    %0 = mo.slice<() -> D6, D7, D8>(%arg0, %arg3, %arg2, %arg1) : (!mo.tensor<[D0, D1, D2], f32>, !mo.tensor<[3], si32>, !mo.tensor<[3], si32>, !mo.tensor<[3], si32>) -> !mo.tensor<[D6, D7, D8], f32>
    %1 = mo.transpose<() -> D9, D10, D11>(%0, %arg4) : (!mo.tensor<[D6, D7, D8], f32>, !mo.tensor<[3], si32>) -> !mo.tensor<[D9, D10, D11], f32>
    mo.output<D9, D10, D11> %1 : !mo.tensor<[D9, D10, D11], f32>
}
