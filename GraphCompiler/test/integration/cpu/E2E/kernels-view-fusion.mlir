// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s

// CHECK-LABEL: Running 'reduce_fused_cst':
// CHECK: 'reduce_fused_cst' returned tensor<1x3xf32> [3.000000e+00, 7.000000e+00, 1.100000e+01]
mo.graph @reduce_fused_cst() -> (!mo.tensor<[1, 3], f32>) {
  %0 = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0> : tensor<6xf32>> : !mo.tensor<[3, 2], f32>
  %perm  = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %1 = mo.transpose(%0, %perm) : (!mo.tensor<[3, 2], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 3], f32>
  %2 = mo.reduce.add(%1) : (!mo.tensor<[2, 3], f32>) -> !mo.tensor<[1, 3], f32>
  mo.output %2 : !mo.tensor<[1, 3], f32>
}

// CHECK-LABEL: Running 'add_fused_cst':
// CHECK: 'add_fused_cst' returned tensor<2x3xf32> [2.000000e+00, 5.000000e+00, 8.000000e+00, 6.000000e+00, 9.000000e+00, 1.200000e+01]
mo.graph @add_fused_cst() -> (!mo.tensor<[2, 3], f32>) {
  %0 = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0> : tensor<6xf32>> : !mo.tensor<[3, 2], f32>
  %1 = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0> : tensor<6xf32>> : !mo.tensor<[2, 3], f32>
  %perm  = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %2 = mo.transpose(%0, %perm) : (!mo.tensor<[3, 2], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 3], f32>
  %3 = mo.add(%1, %2) : !mo.tensor<[2, 3], f32>
  mo.output %3 : !mo.tensor<[2, 3], f32>
}
