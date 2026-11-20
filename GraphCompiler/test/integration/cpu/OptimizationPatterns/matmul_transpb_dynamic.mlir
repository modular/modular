// RUN: mo-opt %s --mo-shape-inference --mo-infer-layouts --mo-pattern-fusion | FileCheck %s --check-prefix=CHECK-PATTERN
// RUN: mo-opt %s --mo-shape-inference --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --input-shapes=4x2 --input-shapes=4x2 -model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s


// CHECK-EXEC-LABEL: --- Running 'matmul_transpb':
mo.graph @matmul_transpb<M, K, N>(%x: !mo.tensor<[M, K], f32>, %w: !mo.tensor<[N, K], f32>) -> !mo.tensor<[M, N], f32> {
  // CHECK-PATTERN: mo.matmul({{.*}}) [transpose_b = true] : (!mo.tensor<{{.*}}, f32>, !mo.tensor<{{.*}}, f32, {layout = #mo.layout<NK>}>)

  %axes = mo.constant <#M.dense_array<1, 0> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %0 = mo.transpose(%w, %axes) : (!mo.tensor<[N, K], f32>, !mo.tensor<[2], si32>) -> !mo.tensor<[K, N], f32>
  %1 = mo.matmul(%x, %0) : (!mo.tensor<[M, K], f32>, !mo.tensor<[K, N], f32>) -> !mo.tensor<[M, N], f32>
  mo.output %1 : !mo.tensor<[M, N], f32>
}

// CHECK{LITERAL}: 'matmul_transpb' returned tensor<4x4xf32> [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]


// CHECK-EXEC-LABEL: --- Running 'matmul_reused_transpb':
mo.graph @matmul_reused_transpb<M, K, N>(%x: !mo.tensor<[M, K], f32>, %w: !mo.tensor<[N, K], f32>) -> (!mo.tensor<[M, N], f32>, !mo.tensor<[M, N], f32>) {
  // CHECK-PATTERN: mo.matmul({{.*}}) [transpose_b = true] : (!mo.tensor<{{.*}}, f32>, !mo.tensor<{{.*}}, f32, {layout = #mo.layout<NK>}>)
  // CHECK-PATTERN: mo.matmul({{.*}}) [transpose_b = true] : (!mo.tensor<{{.*}}, f32>, !mo.tensor<{{.*}}, f32, {layout = #mo.layout<NK>}>)

  %axes = mo.constant <#M.dense_array<1, 0> : tensor<2xsi32>> : !mo.tensor<[2], si32>

  %w_t = mo.transpose(%w, %axes) : (!mo.tensor<[N, K], f32>, !mo.tensor<[2], si32>) -> !mo.tensor<[K, N], f32>
  %mm1 = mo.matmul(%x, %w_t) : (!mo.tensor<[M, K], f32>, !mo.tensor<[K, N], f32>) -> !mo.tensor<[M, N], f32>

  %x2 = mo.add(%x, %x) : !mo.tensor<[M, K], f32>
  %mm2 = mo.matmul(%x2, %w_t) : (!mo.tensor<[M, K], f32>, !mo.tensor<[K, N], f32>) -> !mo.tensor<[M, N], f32>

  mo.output %mm1, %mm2 : !mo.tensor<[M, N], f32>, !mo.tensor<[M, N], f32>
}

// CHECK{LITERAL}: 'matmul_reused_transpb' returned tensor<4x4xf32> [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]
// CHECK{LITERAL}: , tensor<4x4xf32> [4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00]


// CHECK-EXEC-LABEL: --- Running 'matmul_mul_transpb':
mo.graph @matmul_mul_transpb<M, K, N>(%x: !mo.tensor<[M, K], f32>, %w: !mo.tensor<[N, K], f32>) -> !mo.tensor<[M, N], f32> {
  // CHECK-PATTERN: mo.matmul({{.*}}) : (!mo.tensor<{{.*}}, f32>, !mo.tensor<{{.*}}, f32, {layout = #mo.layout<KN>}>)

  %axes = mo.constant <#M.dense_array<1, 0> : tensor<2xsi32>> : !mo.tensor<[2], si32>

  %ww  = mo.mul(%w, %w) : !mo.tensor<[N, K], f32>
  %w_t = mo.transpose(%ww, %axes) : (!mo.tensor<[N, K], f32>, !mo.tensor<[2], si32>) -> !mo.tensor<[K, N], f32>
  %mm = mo.matmul(%x, %w_t) : (!mo.tensor<[M, K], f32>, !mo.tensor<[K, N], f32>) -> !mo.tensor<[M, N], f32>

  mo.output %mm : !mo.tensor<[M, N], f32>
}

// CHECK{LITERAL}: 'matmul_mul_transpb' returned tensor<4x4xf32> [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]


// CHECK-EXEC-LABEL: --- Running 'matmul_transpb_multi_layout_weight':
mo.graph @matmul_transpb_multi_layout_weight(
  %x: !mo.tensor<[4, 2], f32>,
  %w: !mo.tensor<[4, 2], f32>
) -> (!mo.tensor<[4, 4], f32>, !mo.tensor<[2, 2], f32>) {
  // CHECK-PATTERN: mo.matmul({{.*}}) [transpose_b = true] : (!mo.tensor<{{.*}}, f32, {layout = #mo.layout<KN>}>, !mo.tensor<{{.*}}, f32, {layout = #mo.layout<NK>}>)
  // CHECK-PATTERN: mo.matmul({{.*}}) : (!mo.tensor<{{.*}}, f32>, !mo.tensor<{{.*}}, f32, {layout = #mo.layout<KN>}>)

  // Getting input to a specific shape so we can pass w and w_t to matmuls
  %axes = mo.constant <#M.dense_array<1, 0> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %x_t = mo.transpose(%x, %axes) : (!mo.tensor<[4, 2], f32>, !mo.tensor<[2], si32>) -> !mo.tensor<[2, 4], f32>
  %w_t = mo.transpose(%w, %axes) : (!mo.tensor<[4, 2], f32>, !mo.tensor<[2], si32>) -> !mo.tensor<[2, 4], f32>

  %mm1 = mo.matmul(%w, %w_t) : (!mo.tensor<[4, 2], f32>, !mo.tensor<[2, 4], f32>) -> !mo.tensor<[4, 4], f32>
  %mm2 = mo.matmul(%x_t, %w) : (!mo.tensor<[2, 4], f32>, !mo.tensor<[4, 2], f32>) -> !mo.tensor<[2, 2], f32>

  mo.output %mm1, %mm2 : !mo.tensor<[4, 4], f32>, !mo.tensor<[2, 2], f32>
}

// CHECK{LITERAL}: 'matmul_transpb_multi_layout_weight' returned tensor<4x4xf32> [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]
// CHECK{LITERAL}: , tensor<2x2xf32> [4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00]
