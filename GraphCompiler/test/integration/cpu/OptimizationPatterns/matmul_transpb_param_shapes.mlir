// RUN: mo-opt %s --gc-pipeline="constant-fold=false" -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s --check-prefix=CHECK-IR
// RUN: mt --execute --input-shapes=4x2 --input-shapes=4x2 -model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s



mo.graph @matmul_transpb_param_shapes<M, N, K>(%x: !mo.tensor<[M, K], f32>, %w: !mo.tensor<[N, K], f32>) -> (!mo.tensor<[M, N], f32>) {
  %axes = mo.constant <#M.dense_array<1, 0> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %0 = mo.transpose(%w, %axes) : (!mo.tensor<[N, K], f32>, !mo.tensor<[2], si32>) -> !mo.tensor<[K, N], f32>
  %1 = mo.matmul(%x, %0) : (!mo.tensor<[M, K], f32>, !mo.tensor<[K, N], f32>) -> !mo.tensor<[M, N], f32>
  mo.output %1 : !mo.tensor<[M, N], f32>
}

// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.matmul{{.*}}"
// CHECK-NOT-IR: "mgp.generic.execute"{{.*}}dSummary = {{.*}}mo.transpose{{.*}}"
// CHECK{LITERAL}: 'matmul_transpb_param_shapes' returned tensor<4x4xf32> [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]
