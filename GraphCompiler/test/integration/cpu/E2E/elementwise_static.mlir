// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: python3 %mo_vs_numpy_py -n ElementwiseStatic -f %t.mlir | FileCheck %s

// CHECK: All MO values match numpy

module {
  mo.graph @serving_default(%arg0: !mo.tensor<[20, 5, 1, 6], f32>, %arg1: !mo.tensor<[20, 5, 1, 6], f32>) -> !mo.tensor<[20, 5, 1, 6], f32> attributes {argument_names = ["input_1", "input_2"], result_names = ["lambda"]} {
    %0 = mo.constant <#M.dense_array<-5.000000e-01> : tensor<f32>> : !mo.tensor<[], f32>
    %1 = mo.constant <#M.dense_array<5.000000e-01> : tensor<f32>> : !mo.tensor<[], f32>
    %2 = mo.abs(%arg0) : !mo.tensor<[20, 5, 1, 6], f32>
    %3 = mo.exp(%2) : !mo.tensor<[20, 5, 1, 6], f32>
    %4 = mo.greater(%arg0, %3) : (!mo.tensor<[20, 5, 1, 6], f32>, !mo.tensor<[20, 5, 1, 6], f32>) -> !mo.tensor<[20, 5, 1, 6], bool>
    %5 = mo.select(%4, %arg0, %3) : (!mo.tensor<[20, 5, 1, 6], bool>, !mo.tensor<[20, 5, 1, 6], f32>, !mo.tensor<[20, 5, 1, 6], f32>) -> !mo.tensor<[20, 5, 1, 6], f32>
    %6 = mo.tanh(%arg1) : !mo.tensor<[20, 5, 1, 6], f32>
    %7 = mo.sub(%arg0, %6) : !mo.tensor<[20, 5, 1, 6], f32>
    %8 = mo.mul(%7, %arg0) : !mo.tensor<[20, 5, 1, 6], f32>
    %9 = mo.erf(%8) : !mo.tensor<[20, 5, 1, 6], f32>
    %10 = mo.constant <#M.dense_array<20, 5, 1, 6> : !M.array<4xsi64>> : !mo.tensor<[4], si64>
    %11 = mo.broadcast_to(%1, %10) : (!mo.tensor<[], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[20, 5, 1, 6], f32>
    %12 = mo.max(%arg0, %11) : !mo.tensor<[20, 5, 1, 6], f32>
    %13 = mo.add(%3, %12) : !mo.tensor<[20, 5, 1, 6], f32>
    %14 = mo.constant <#M.dense_array<20, 5, 1, 6> : !M.array<4xsi64>> : !mo.tensor<[4], si64>
    %15 = mo.broadcast_to(%0, %14) : (!mo.tensor<[], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[20, 5, 1, 6], f32>
    %16 = mo.min(%arg0, %15) : !mo.tensor<[20, 5, 1, 6], f32>
    %17 = mo.add(%13, %16) : !mo.tensor<[20, 5, 1, 6], f32>
    %18 = mo.add(%5, %17) : !mo.tensor<[20, 5, 1, 6], f32>
    %19 = mo.ceil(%18) : !mo.tensor<[20, 5, 1, 6], f32>
    %20 = mo.floor(%18) : !mo.tensor<[20, 5, 1, 6], f32>
    %21 = mo.log(%18) : !mo.tensor<[20, 5, 1, 6], f32>
    %22 = mo.log1p(%18) : !mo.tensor<[20, 5, 1, 6], f32>
    %23 = mo.rsqrt(%18) : !mo.tensor<[20, 5, 1, 6], f32>
    %24 = mo.add(%18, %21) : !mo.tensor<[20, 5, 1, 6], f32>
    %25 = mo.add(%24, %23) : !mo.tensor<[20, 5, 1, 6], f32>
    %26 = mo.add(%25, %22) : !mo.tensor<[20, 5, 1, 6], f32>
    %27 = mo.add(%26, %19) : !mo.tensor<[20, 5, 1, 6], f32>
    %28 = mo.add(%27, %20) : !mo.tensor<[20, 5, 1, 6], f32>
    %29 = mo.add(%9, %28) : !mo.tensor<[20, 5, 1, 6], f32>
    mo.output %29 : !mo.tensor<[20, 5, 1, 6], f32>
  }
}
