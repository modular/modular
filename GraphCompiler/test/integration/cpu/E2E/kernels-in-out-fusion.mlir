// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s

// CHECK-LABEL: Running 'matmul':
// CHECK: 'matmul' returned tensor<2x2xf32> [1.900000e+01, 2.200000e+01, 4.300000e+01, 5.000000e+01]
mo.graph @matmul() -> !mo.tensor<[2, 2], f32, cpu:0> {
  %0 = "mo.constant"() {value = #M.dense_array<1.0, 2.0, 3.0, 4.0> : tensor<4xf32>} : () -> !mo.tensor<[2, 2], f32>
  %1 = "mo.constant"() {value = #M.dense_array<5.0, 6.0, 7.0, 8.0> : tensor<4xf32>} : () -> !mo.tensor<[2, 2], f32>
  %2 = mo.matmul(%0, %1) : (!mo.tensor<[2, 2], f32, cpu:0>, !mo.tensor<[2, 2], f32, cpu:0>) -> !mo.tensor<[2, 2], f32, cpu:0>
  mo.output %2 : !mo.tensor<[2, 2], f32, cpu:0>
}

// CHECK-LABEL: Running 'reduce_in_fusion':
// CHECK: 'reduce_in_fusion' returned tensor<4x1xf32> [3.000000e+00, 7.000000e+00, 1.100000e+01, 1.500000e+01]
mo.graph @reduce_in_fusion() -> (!mo.tensor<[4, 1], f32>) {
  %0 = mo.constant <#M.dense_array<2.0, 1.0, 3.0, 1.0, 3.0, 4.0, 3.0, 5.0> : tensor<8xf32>> : !mo.tensor<[4, 2], f32>
  %1 = mo.constant <#M.dense_array<-1.0, 1.0, 0.0, 3.0, 2.0, 2.0, 4.0, 3.0> : tensor<8xf32>> : !mo.tensor<[4, 2], f32>
  %2 = mo.add(%0, %1) : !mo.tensor<[4, 2], f32>
  %res = mo.reduce.add(%2) : (!mo.tensor<[4, 2], f32>) -> !mo.tensor<[4, 1], f32>
  mo.output %res : !mo.tensor<[4, 1], f32>
}

// CHECK-LABEL: Running 'reduce_out_fusion':
// CHECK: 'reduce_out_fusion' returned tensor<2x1x2xf32> [1.000000e+00, 2.000000e+00, 3.000000e+00, 7.000000e+00]
mo.graph @reduce_out_fusion() -> (!mo.tensor<[2, 1, 2], f32>) {
  %0 = mo.constant <#M.dense_array<2.0, 1.0, 3.0, 1.0, 3.0, 4.0, 3.0, 5.0> : tensor<8xf32>> : !mo.tensor<[2, 2, 2], f32>
  %1 = mo.constant <#M.dense_array<-1.0, 1.0, 0.0, 3.0> : tensor<4xf32>> : !mo.tensor<[2, 1, 2], f32>
  %2 = mo.reduce.min(%0) : (!mo.tensor<[2, 2, 2], f32>) -> !mo.tensor<[2, 1, 2], f32>
  %3 = mo.add(%2, %1) : !mo.tensor<[2, 1, 2], f32>
  mo.output %3 : !mo.tensor<[2, 1, 2], f32>
}

// CHECK-LABEL: Running 'reduce_fused_cast':
// CHECK: 'reduce_fused_cast' returned tensor<2x1x2xf32> [6.000000e+00, 4.000000e+00, 1.000000e+01, -1.000000e+01]
mo.graph @reduce_fused_cast() -> (!mo.tensor<[2, 1, 2], f32>) {
  %0 = mo.constant <#M.dense_array<2, 1, 3, 1, 3, 4, 3, 5> : tensor<8xsi32>> : !mo.tensor<[2, 2, 2], si32>
  %1 = mo.constant <#M.dense_array<3, 4, 5, 6, 7, 8, 9, -12> : tensor<8xsi32>> : !mo.tensor<[2, 2, 2], si32>
  %2 = mo.constant <#M.dense_array<-1.0, 1.0, 0.0, 3.0> : tensor<4xf32>> : !mo.tensor<[2, 1, 2], f32>
  %3 = mo.add(%0, %1) : !mo.tensor<[2, 2, 2], si32>
  %4 = mo.cast(%3) : (!mo.tensor<[2, 2, 2], si32>) -> !mo.tensor<[2, 2, 2], f32>
  %5 = mo.reduce.min(%4) : (!mo.tensor<[2, 2, 2], f32>) -> !mo.tensor<[2, 1, 2], f32>
  %6 = mo.sub(%5, %2) : !mo.tensor<[2, 1, 2], f32>
  mo.output %6 : !mo.tensor<[2, 1, 2], f32>
}

// CHECK-LABEL: Running 'reduce_out_fusion_cast':
// CHECK: 'reduce_out_fusion_cast' returned tensor<2x1x2xsi32> [1, 2, 3, 7]
mo.graph @reduce_out_fusion_cast() -> (!mo.tensor<[2, 1, 2], si32>) {
  %0 = mo.constant <#M.dense_array<2.0, 1.0, 3.0, 1.0, 3.0, 4.0, 3.0, 5.0> : tensor<8xf32>> : !mo.tensor<[2, 2, 2], f32>
  %1 = mo.constant <#M.dense_array<-1.0, 1.0, 0.0, 3.0> : tensor<4xf32>> : !mo.tensor<[2, 1, 2], f32>
  %2 = mo.reduce.min(%0) : (!mo.tensor<[2, 2, 2], f32>) -> !mo.tensor<[2, 1, 2], f32>
  %3 = mo.add(%2, %1) : !mo.tensor<[2, 1, 2], f32>
  %4 = mo.cast(%3) : (!mo.tensor<[2, 1, 2], f32>) -> !mo.tensor<[2, 1, 2], si32>
  mo.output %4 : !mo.tensor<[2, 1, 2], si32>
}
