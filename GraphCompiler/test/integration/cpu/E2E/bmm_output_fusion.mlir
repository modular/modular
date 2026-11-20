// RUN: mo-opt %s --mo-shape-inference --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: python3 %mo_vs_numpy_py -n BatchMatmulRelu -f %t.mlir | FileCheck %s

// CHECK: All MO values match numpy

module {
  mo.graph @serving_default<D0, D1, D2, D3, D4, D5 -> D6, D7, D8>(%arg0: !mo.tensor<[D0, D1, D2], f32>, %arg1: !mo.tensor<[D3, D4, D5], f32>) -> !mo.tensor<[D6, D7, D8], f32> attributes {argument_names = ["input_1", "input_2"], result_names = ["lambda"]} {
    %0 = mo.constant <#M.dense_array<0.000000e+00> : tensor<f32>> : !mo.tensor<[], f32>
    %1 = mo.shape_of(%arg0) : (!mo.tensor<[D0, D1, D2], f32>) -> !mo.tensor<[3], si64>
    %2 = mo.shape_of(%arg1) : (!mo.tensor<[D3, D4, D5], f32>) -> !mo.tensor<[3], si64>
    %3 = mo.constant <#M.dense_array<0> : tensor<1xsi64>> : !mo.tensor<[1], si64>
    %4 = mo.constant <#M.dense_array<1> : tensor<1xsi64>> : !mo.tensor<[1], si64>
    %5 = mo.constant <#M.dense_array<3> : tensor<1xsi64>> : !mo.tensor<[1], si64>
    %6 = mo.slice(%1, %3, %4, %4) : (!mo.tensor<[3], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[1], si64>
    %7 = mo.slice(%1, %4, %5, %4) : (!mo.tensor<[3], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[2], si64>
    %8 = mo.slice(%2, %3, %4, %4) : (!mo.tensor<[3], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[1], si64>
    %9 = mo.slice(%2, %4, %5, %4) : (!mo.tensor<[3], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[2], si64>
    %10 = mo.broadcast_shape(%6, %8) : (!mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[1], si64>
    %11 = mo.constant <#M.dense_array<0> : tensor<si64>> : !mo.tensor<[], si64>
    %12 = mo.concat[0] (%10, %7) : (!mo.tensor<[1], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[3], si64>
    %13 = mo.concat[0] (%10, %9) : (!mo.tensor<[1], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[3], si64>
    %14 = mo.broadcast_to<() -> D9, D10, D11>(%arg0, %12) : (!mo.tensor<[D0, D1, D2], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[D9, D10, D11], f32>
    %15 = mo.broadcast_to<() -> D12, D13, D14>(%arg1, %13) : (!mo.tensor<[D3, D4, D5], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[D12, D13, D14], f32>
    %16 = mo.batch_matmul<() -> D15, D16, D17>(%14, %15) : (!mo.tensor<[D9, D10, D11], f32>, !mo.tensor<[D12, D13, D14], f32>) -> !mo.tensor<[D15, D16, D17], f32>
    %17 = mo.shape_of(%16) : (!mo.tensor<[D15, D16, D17], f32>) -> !mo.tensor<[3], si64>
    %18 = mo.shape_of(%0) : (!mo.tensor<[], f32>) -> !mo.tensor<[0], si64>
    %19 = mo.broadcast_shape(%17, %18) : (!mo.tensor<[3], si64>, !mo.tensor<[0], si64>) -> !mo.tensor<[3], si64>
    %20 = mo.broadcast_to(%16, %19) : (!mo.tensor<[D15, D16, D17], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[D15, D16, D17], f32>
    %21 = mo.broadcast_to(%0, %19) : (!mo.tensor<[], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[D15, D16, D17], f32>
    %22 = mo.max(%20, %21) : !mo.tensor<[D15, D16, D17], f32>
    mo.output<D15, D16, D17> %22 : !mo.tensor<[D15, D16, D17], f32>
  }
}
