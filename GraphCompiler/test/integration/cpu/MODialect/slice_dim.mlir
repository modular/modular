// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute %t.mlir --result-output-style=full | FileCheck %s

mo.graph @basic_slice_dim(%arg0: !mo.tensor<[3, 5], f32>) -> !mo.tensor<[2, 3], f32> {
  %input = mo.constant <#M.dense_array<0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0> : tensor<15xf32>> : !mo.tensor<[3, 5], f32>
  %in = mo.add(%input, %arg0): !mo.tensor<[3, 5], f32>
  %start = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop = mo.constant <#M.dense_array<3, 5> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<1, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %res = mo.slice(%in, %start, %stop, %step) : (
    !mo.tensor<[3, 5], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[2, 3], f32>
  mo.output %res : !mo.tensor<[2, 3], f32>
}

// CHECK{LITERAL}: 'basic_slice_dim' returned tensor<2x3xf32> [5.000000e+00, 7.000000e+00, 9.000000e+00, 1.000000e+01, 1.200000e+01, 1.400000e+01]

mo.graph @slice_dim_neg_step(%arg0: !mo.tensor<[5, 3], f32>) -> !mo.tensor<[2, 2], f32> {
  %input = mo.constant <#M.dense_array<0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0> : tensor<15xf32>> : !mo.tensor<[5, 3], f32>
  %in = mo.add(%input, %arg0): !mo.tensor<[5, 3], f32>
  %start = mo.constant <#M.dense_array<-1, -1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop = mo.constant <#M.dense_array<-1, -1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step = mo.constant <#M.dense_array<-2, -1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %res = mo.slice(%in, %start, %stop, %step) : (
    !mo.tensor<[5, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[2, 2], f32>
  mo.output %res : !mo.tensor<[2, 2], f32>
}

// CHECK{LITERAL}: 'slice_dim_neg_step' returned tensor<2x2xf32> [1.400000e+01, 1.300000e+01, 8.000000e+00, 7.000000e+00]
