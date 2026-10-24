// Test on GPU we prefer matmul(A, B, transpose_b=true) on GPU
// RUN: mo-opt %s --mo-infer-layouts --mo-pattern-fusion --mo-constant-fold | FileCheck %s

// Check the existing transposes are absorbed by pattern for gpu
// CHECK-LABEL: mo.graph @matmul_existing_transb
// CHECK: mo.matmul({{.*}}) [transpose_b = true] :
// CHECK-SAME: (!mo.tensor<[4, 2], f32, gpu:0>, !mo.tensor<[4, 2], f32, gpu:0, {layout = #mo.layout<NK>}>)
// CHECK-SAME: -> !mo.tensor<[4, 4], f32, gpu:0>
mo.graph @matmul_existing_transb(
  %x: !mo.tensor<[4, 2], f32, gpu:0>,
  %w: !mo.tensor<[4, 2], f32, gpu:0>
) -> !mo.tensor<[4, 4], f32, gpu:0> {
  %axes = mo.constant <#M.dense_array<1, 0> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %w_t = mo.transpose(%w, %axes) : (!mo.tensor<[4, 2], f32, gpu:0>, !mo.tensor<[2], si32>) -> !mo.tensor<[2, 4], f32, gpu:0>
  %mm1 = mo.matmul(%w, %w_t) : (!mo.tensor<[4, 2], f32, gpu:0>, !mo.tensor<[2, 4], f32, gpu:0>) -> !mo.tensor<[4, 4], f32, gpu:0>
  mo.output %mm1 : !mo.tensor<[4, 4], f32, gpu:0>
}


// Check the existing matmuls are replaced with transposed matmuls.
// This includes an additional transpose we expect to be constant-folded
// CHECK-LABEL: mo.graph @gpu_matmul_prefers_transpose

// Note: constant is transposed from what it was before
// CHECK: mo.constant <#M.dense_array<1.000000e+00, 4.000000e+00, 2.000000e+00, 5.000000e+00, 3.000000e+00, 6.000000e+00> : !M.array<6xf32>> : !mo.tensor<[3, 2], f32, gpu:0, {layout = #mo.layout<NK>}>
// CHECK: mo.matmul({{.*}}) [transpose_b = true] : (!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[3, 2], f32, gpu:0, {layout = #mo.layout<NK>}>) -> !mo.tensor<[2, 3], f32, gpu:0>
// CHECK: mo.matmul({{.*}}) [transpose_b = true] : (!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[3, 2], f32, gpu:0, {layout = #mo.layout<NK>}>) -> !mo.tensor<[2, 3], f32, gpu:0>
// CHECK-NOT: mo.matmul
mo.graph @gpu_matmul_prefers_transpose(%arg0: !mo.tensor<[2, 2], f32, gpu:0>, %arg1: !mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 3], f32, gpu:0> {
  %0 = mo.constant <#M.dense_array<1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00> : !M.array<6xf32>> : !mo.tensor<[2, 3], f32, gpu:0>
  %1 = mo.matmul(%arg0, %0) : (!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:0>) -> !mo.tensor<[2, 3], f32, gpu:0>
  %2 = mo.matmul(%arg1, %0) : (!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:0>) -> !mo.tensor<[2, 3], f32, gpu:0>
  %3 = mo.add(%1, %2) : !mo.tensor<[2, 3], f32, gpu:0>
  mo.output %3 : !mo.tensor<[2, 3], f32, gpu:0>
}

// We don't transform the matmul's weights since they have non-matmul uses
// CHECK-LABEL: mo.graph @gpu_matmul_no_transform_non_matmul_uses
// CHECK: mo.constant <#M.dense_array<1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00> : !M.array<6xf32>> : !mo.tensor<[2, 3], f32, gpu:0, {layout = #mo.layout<KN>}>

// No more transforms or constants so all matmuls must be used the above constant
// CHECK-NOT: mo.transpose
// CHECK-NOT: mo.constant
mo.graph @gpu_matmul_no_transform_non_matmul_uses(%arg0: !mo.tensor<[2, 2], f32, gpu:0>, %arg1: !mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 3], f32, gpu:0> {
  %0 = mo.constant <#M.dense_array<1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00> : !M.array<6xf32>> : !mo.tensor<[2, 3], f32, gpu:0>
  %1 = mo.matmul(%arg0, %0) : (!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:0>) -> !mo.tensor<[2, 3], f32, gpu:0>
  %2 = mo.matmul(%arg1, %0) : (!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:0>) -> !mo.tensor<[2, 3], f32, gpu:0>
  %3 = mo.add(%1, %2) : !mo.tensor<[2, 3], f32, gpu:0>
  %4 = mo.add(%3, %0) : !mo.tensor<[2, 3], f32, gpu:0>
  mo.output %4 : !mo.tensor<[2, 3], f32, gpu:0>
}

// We don't transform the matmul's weights since a matmul is not run on gpu
// CHECK-LABEL: mo.graph @gpu_matmul_no_transform_non_gpu_uses
// CHECK: mo.constant <#M.dense_array<1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00> : !M.array<6xf32>> : !mo.tensor<[2, 3], f32, {layout = #mo.layout<KN>}>

// No more transforms or constants so all matmuls must be used the above constant
// CHECK-NOT: mo.transpose
// CHECK-NOT: mo.constant
mo.graph @gpu_matmul_no_transform_non_gpu_uses(%arg0: !mo.tensor<[2, 2], f32>, %arg1: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 3], f32> {
  %0 = mo.constant <#M.dense_array<1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00> : !M.array<6xf32>> : !mo.tensor<[2, 3], f32>
  %1 = mo.matmul(%arg0, %0) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  %2 = mo.matmul(%arg1, %0) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  %3 = mo.add(%1, %2) : !mo.tensor<[2, 3], f32>
  mo.output %3 : !mo.tensor<[2, 3], f32>
}

// This tests we apply optimization to "constant-like" weights.
//
// "constant-like" in this case means derived from pure operations with only
// "constant-like" operands. Base case is mo.constant* ops are "constant-like".
//
// CHECK-LABEL: mo.graph @cuda_matmul_prefers_transpose_constant_like
//
// CHECK: mo.matmul({{.*}}) [transpose_b = true] : (!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[3, 2], f32, gpu:0, {layout = #mo.layout<NK>}>) -> !mo.tensor<[2, 3], f32, gpu:0>
// CHECK: mo.matmul({{.*}}) [transpose_b = true] : (!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[3, 2], f32, gpu:0, {layout = #mo.layout<NK>}>) -> !mo.tensor<[2, 3], f32, gpu:0>
// CHECK-NOT: mo.matmul
mo.graph @cuda_matmul_prefers_transpose_constant_like(%arg0: !mo.tensor<[2, 2], f32, gpu:0>, %arg1: !mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 3], f32, gpu:0> {
  %0 = mo.constant.external [name = "weight", align = 32, device = #M.device_ref<"gpu", 0>] : !mo.tensor<[1, 1], f32, gpu:0>

  // This is still constant-like since it is a pure op derived with %0 as operand
  // and %0 is constant.
  %weight = mo.static.broadcast_to(%0): !mo.tensor<[1, 1], f32, gpu:0> -> !mo.tensor<[2, 3], f32, gpu:0>
  %1 = mo.matmul(%arg0, %weight) : (!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:0>) -> !mo.tensor<[2, 3], f32, gpu:0>
  %2 = mo.matmul(%arg1, %weight) : (!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:0>) -> !mo.tensor<[2, 3], f32, gpu:0>
  %3 = mo.add(%1, %2) : !mo.tensor<[2, 3], f32, gpu:0>
  mo.output %3 : !mo.tensor<[2, 3], f32, gpu:0>
}
