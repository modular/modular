// RUN: mo-opt %s --gc-pipeline="constant-fold=false" -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s

// An elementwise producer read through two slices by one consumer: the
// producer is computed at each slice's own index.

// CHECK-LABEL: Running 'negate_through_slices':
// CHECK: 'negate_through_slices' returned tensor<2x2xf32> [0.000000e+00, 3.000000e+00, 2.400000e+01, 3.500000e+01]
mo.graph @negate_through_slices() -> (!mo.tensor<[2, 2], f32>) {
  %x = mo.constant <#M.dense_array<0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0> : tensor<8xf32>> : !mo.tensor<[2, 4], f32>
  %w = mo.negative(%x) : !mo.tensor<[2, 4], f32>
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %start0 = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop0 = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %start1 = mo.constant <#M.dense_array<0, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop1 = mo.constant <#M.dense_array<2, 4> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %a = mo.slice(%w, %start0, %stop0, %step) : (
    !mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[2, 2], f32>
  %b = mo.slice(%w, %start1, %stop1, %step) : (
    !mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[2, 2], f32>
  %r = mo.mul(%a, %b) : !mo.tensor<[2, 2], f32>
  mo.output %r : !mo.tensor<[2, 2], f32>
}

// The same through a transpose.

// CHECK-LABEL: Running 'negate_through_transpose':
// CHECK: 'negate_through_transpose' returned tensor<3x2xf32> [1.000000e+01, 1.700000e+01, 2.900000e+01, 3.600000e+01, 4.800000e+01, 5.500000e+01]
mo.graph @negate_through_transpose() -> (!mo.tensor<[3, 2], f32>) {
  %x = mo.constant <#M.dense_array<0.0, 1.0, 2.0, 3.0, 4.0, 5.0> : tensor<6xf32>> : !mo.tensor<[2, 3], f32>
  %y = mo.constant <#M.dense_array<10.0, 20.0, 30.0, 40.0, 50.0, 60.0> : tensor<6xf32>> : !mo.tensor<[3, 2], f32>
  %w = mo.negative(%x) : !mo.tensor<[2, 3], f32>
  %perm = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %t = mo.transpose(%w, %perm) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[3, 2], f32>
  %r = mo.add(%t, %y) : !mo.tensor<[3, 2], f32>
  mo.output %r : !mo.tensor<[3, 2], f32>
}
