// RUN: mo-opt %s --mo-constant-fold | FileCheck %s

// CHECK-LABEL: mo.graph @test_fold
// CHECK-NEXT: [[FOLDED:%.*]] = mo.constant <#M.dense_array<0.000000e+00, 2.000000e+00, 4.000000e+00, 6.000000e+00, 8.000000e+00> : !M.array<5xf32>> : !mo.tensor<[5], f32>
// CHECK-NEXT: mo.output [[FOLDED]]

mo.graph @test_fold() -> (!mo.tensor<[5], f32>) {
  %0 = mo.constant <#M.dense_array<0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00> : tensor<5xf32>> : !mo.tensor<[5], f32>
  %output = mo.add(%0, %0) : !mo.tensor<[5], f32>
  mo.output %output : !mo.tensor<[5], f32>
}
