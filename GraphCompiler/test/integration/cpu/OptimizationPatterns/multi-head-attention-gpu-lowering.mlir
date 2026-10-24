// RUN: mo-opt --mo-pattern-fusion %s | FileCheck %s --enable-var-scope

// CHECK-LABEL: @multihead_flash_attention_rank3_mask
// CHECK: mo.composite.masked_flash_attention_gpu
// Checks if bcasting a mask of rank 3 works with the pattern
mo.graph @multihead_flash_attention_rank3_mask() -> !mo.tensor<[1, 1024, 32, 128], f32, gpu:0> {
  %0 = mo.constant <#M.dense_array<2.000000e+00> : tensor<1xf32>> : !mo.tensor<[], f32, gpu:0>
  %1 = mo.constant <#M.dense_array<3.000000e+00> : tensor<1xf32>> : !mo.tensor<[], f32, gpu:0>
  %2 = mo.constant <#M.dense_array<4.000000e+00> : tensor<1xf32>> : !mo.tensor<[], f32, gpu:0>
  %3 = mo.constant <#M.dense_array<2.000000e+00> : tensor<1xf32>> : !mo.tensor<[], f32, gpu:0>
  %4 = mo.constant <#M.dense_array<8.000000e+00> : tensor<1xf32>> : !mo.tensor<[], f32, gpu:0>
  %5 = mo.constant <#M.dense_array<0, 2, 1, 3> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %6 = mo.constant <#M.dense_array<0, 2, 3, 1> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %7 = mo.constant <#M.dense_array<0, 0, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %8 = mo.constant <#M.dense_array<1, 1024, 1024> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %9 = mo.constant <#M.dense_array<1, 1, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %10 = mo.static.broadcast_to(%0) : !mo.tensor<[], f32, gpu:0> -> !mo.tensor<[1, 1024, 32, 128], f32, gpu:0>
  %11 = mo.static.broadcast_to(%1) : !mo.tensor<[], f32, gpu:0> -> !mo.tensor<[1, 1024, 32, 128], f32, gpu:0>
  %12 = mo.static.broadcast_to(%2) : !mo.tensor<[], f32, gpu:0> -> !mo.tensor<[1, 1024, 32, 128], f32, gpu:0>
  %13 = mo.transpose(%10, %5) : (!mo.tensor<[1, 1024, 32, 128], f32, gpu:0>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 32, 1024, 128], f32, gpu:0>
  %14 = mo.transpose(%11, %6) : (!mo.tensor<[1, 1024, 32, 128], f32, gpu:0>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 32, 128, 1024], f32, gpu:0>
  %15 = mo.transpose(%12, %5) : (!mo.tensor<[1, 1024, 32, 128], f32, gpu:0>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 32, 1024, 128], f32, gpu:0>
  %16 = mo.batch_matmul(%13, %14) : (!mo.tensor<[1, 32, 1024, 128], f32, gpu:0>, !mo.tensor<[1, 32, 128, 1024], f32, gpu:0>) -> !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>
  %17 = mo.static.broadcast_to(%4) : !mo.tensor<[], f32, gpu:0> -> !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>
  %18 = mo.div(%16, %17) : !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>
  %19 = mo.static.broadcast_to(%3) : !mo.tensor<[], f32, gpu:0> -> !mo.tensor<[1, 2048, 2048], f32, gpu:0>
  %20 = mo.slice(%19, %7, %8, %9) : (!mo.tensor<[1, 2048, 2048], f32, gpu:0>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[1, 1024, 1024], f32, gpu:0>
  %21 = mo.static.broadcast_to(%20) : !mo.tensor<[1, 1024, 1024], f32, gpu:0> -> !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>
  %22 = mo.add(%18, %21) : !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>
  %23 = mo.reduce.softmax(%22) [axis = 3] : (!mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>) -> !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>
  %24 = mo.static.broadcast_to(%23) : !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0> -> !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>
  %25 = mo.batch_matmul(%24, %15) : (!mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>, !mo.tensor<[1, 32, 1024, 128], f32, gpu:0>) -> !mo.tensor<[1, 32, 1024, 128], f32, gpu:0>
  %26 = mo.transpose(%25, %5) : (!mo.tensor<[1, 32, 1024, 128], f32, gpu:0>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 1024, 32, 128], f32, gpu:0>
  mo.output %26 : !mo.tensor<[1, 1024, 32, 128], f32, gpu:0>
}

// CHECK-LABEL: @multihead_flash_attention_rank4_mask
// CHECK: mo.composite.masked_flash_attention_gpu
// Checks if bcasting a mask of rank 4 works with the pattern
mo.graph @multihead_flash_attention_rank4_mask() -> !mo.tensor<[1, 1024, 32, 128], f32, gpu:0> {
  %0 = mo.constant <#M.dense_array<2.000000e+00> : tensor<1xf32>> : !mo.tensor<[], f32, gpu:0>
  %1 = mo.constant <#M.dense_array<3.000000e+00> : tensor<1xf32>> : !mo.tensor<[], f32, gpu:0>
  %2 = mo.constant <#M.dense_array<4.000000e+00> : tensor<1xf32>> : !mo.tensor<[], f32, gpu:0>
  %3 = mo.constant <#M.dense_array<2.000000e+00> : tensor<1xf32>> : !mo.tensor<[], f32, gpu:0>
  %4 = mo.constant <#M.dense_array<8.000000e+00> : tensor<1xf32>> : !mo.tensor<[], f32, gpu:0>
  %5 = mo.constant <#M.dense_array<0, 2, 1, 3> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %6 = mo.constant <#M.dense_array<0, 2, 3, 1> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %7 = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %8 = mo.constant <#M.dense_array<1, 1, 1024, 1024> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %9 = mo.constant <#M.dense_array<1, 1, 1, 1> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %10 = mo.static.broadcast_to(%0) : !mo.tensor<[], f32, gpu:0> -> !mo.tensor<[1, 1024, 32, 128], f32, gpu:0>
  %11 = mo.static.broadcast_to(%1) : !mo.tensor<[], f32, gpu:0> -> !mo.tensor<[1, 1024, 32, 128], f32, gpu:0>
  %12 = mo.static.broadcast_to(%2) : !mo.tensor<[], f32, gpu:0> -> !mo.tensor<[1, 1024, 32, 128], f32, gpu:0>
  %13 = mo.transpose(%10, %5) : (!mo.tensor<[1, 1024, 32, 128], f32, gpu:0>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 32, 1024, 128], f32, gpu:0>
  %14 = mo.transpose(%11, %6) : (!mo.tensor<[1, 1024, 32, 128], f32, gpu:0>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 32, 128, 1024], f32, gpu:0>
  %15 = mo.transpose(%12, %5) : (!mo.tensor<[1, 1024, 32, 128], f32, gpu:0>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 32, 1024, 128], f32, gpu:0>
  %16 = mo.batch_matmul(%13, %14) : (!mo.tensor<[1, 32, 1024, 128], f32, gpu:0>, !mo.tensor<[1, 32, 128, 1024], f32, gpu:0>) -> !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>
  %17 = mo.static.broadcast_to(%4) : !mo.tensor<[], f32, gpu:0> -> !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>
  %18 = mo.div(%16, %17) : !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>
  %19 = mo.static.broadcast_to(%3) : !mo.tensor<[], f32, gpu:0> -> !mo.tensor<[1, 1, 2048, 2048], f32, gpu:0>
  %20 = mo.slice(%19, %7, %8, %9) : (!mo.tensor<[1, 1, 2048, 2048], f32, gpu:0>, !mo.tensor<[4], si64>, !mo.tensor<[4], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 1, 1024, 1024], f32, gpu:0>
  %21 = mo.static.broadcast_to(%20) : !mo.tensor<[1, 1, 1024, 1024], f32, gpu:0> -> !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>
  %22 = mo.add(%18, %21) : !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>
  %23 = mo.reduce.softmax(%22) [axis = 3] : (!mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>) -> !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>
  %24 = mo.static.broadcast_to(%23) : !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0> -> !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>
  %25 = mo.batch_matmul(%24, %15) : (!mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>, !mo.tensor<[1, 32, 1024, 128], f32, gpu:0>) -> !mo.tensor<[1, 32, 1024, 128], f32, gpu:0>
  %26 = mo.transpose(%25, %5) : (!mo.tensor<[1, 32, 1024, 128], f32, gpu:0>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 1024, 32, 128], f32, gpu:0>
  mo.output %26 : !mo.tensor<[1, 1024, 32, 128], f32, gpu:0>
}

// Make sure supports BF16 pattern which usually has some random casts inserted
// CHECK-LABEL: @multihead_flash_attention_rank4_mask_bf16_with_casts
// CHECK: mo.composite.masked_flash_attention_gpu
// CHECK-SAME: !mo.tensor<[1, 1024, 32, 128], bf16, gpu:0>, !mo.tensor<[1, 1024, 32, 128], bf16, gpu:0>, !mo.tensor<[1, 1024, 32, 128], bf16, gpu:0>,
// CHECK-SAME: !mo.tensor<[1, 32, 1024, 1024], bf16, gpu:0>, !mo.tensor<[], f32>) -> !mo.tensor<[1, 1024, 32, 128], bf16, gpu:0>
// Checks if bcasting a mask of rank 4 works with the pattern
  mo.graph @multihead_flash_attention_rank4_mask_bf16_with_casts() -> !mo.tensor<[1, 1024, 32, 128], bf16, gpu:0> {
    %0 = mo.constant <#M.dense_array<2.000000e+00> : tensor<1xbf16>> : !mo.tensor<[], bf16, gpu:0>
    %1 = mo.constant <#M.dense_array<3.000000e+00> : tensor<1xbf16>> : !mo.tensor<[], bf16, gpu:0>
    %2 = mo.constant <#M.dense_array<4.000000e+00> : tensor<1xbf16>> : !mo.tensor<[], bf16, gpu:0>
    %3 = mo.constant <#M.dense_array<2.000000e+00> : tensor<1xbf16>> : !mo.tensor<[], bf16, gpu:0>
    %4 = mo.constant <#M.dense_array<8.000000e+00> : tensor<1xbf16>> : !mo.tensor<[], bf16, gpu:0>
    %5 = mo.constant <#M.dense_array<0, 2, 1, 3> : tensor<4xsi64>> : !mo.tensor<[4], si64>
    %6 = mo.constant <#M.dense_array<0, 2, 3, 1> : tensor<4xsi64>> : !mo.tensor<[4], si64>
    %7 = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
    %8 = mo.constant <#M.dense_array<1, 1, 1024, 1024> : tensor<4xsi64>> : !mo.tensor<[4], si64>
    %9 = mo.constant <#M.dense_array<1, 1, 1, 1> : tensor<4xsi64>> : !mo.tensor<[4], si64>
      %10 = mo.static.broadcast_to(%0) : !mo.tensor<[], bf16, gpu:0> -> !mo.tensor<[1, 1024, 32, 128], bf16, gpu:0>
    %11 = mo.static.broadcast_to(%1) : !mo.tensor<[], bf16, gpu:0> -> !mo.tensor<[1, 1024, 32, 128], bf16, gpu:0>
    %12 = mo.static.broadcast_to(%2) : !mo.tensor<[], bf16, gpu:0> -> !mo.tensor<[1, 1024, 32, 128], bf16, gpu:0>
    %13 = mo.transpose(%10, %5) : (!mo.tensor<[1, 1024, 32, 128], bf16, gpu:0>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 32, 1024, 128], bf16, gpu:0>
    %14 = mo.transpose(%11, %6) : (!mo.tensor<[1, 1024, 32, 128], bf16, gpu:0>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 32, 128, 1024], bf16, gpu:0>
    %15 = mo.transpose(%12, %5) : (!mo.tensor<[1, 1024, 32, 128], bf16, gpu:0>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 32, 1024, 128], bf16, gpu:0>
    %16 = mo.batch_matmul(%13, %14) : (!mo.tensor<[1, 32, 1024, 128], bf16, gpu:0>, !mo.tensor<[1, 32, 128, 1024], bf16, gpu:0>) -> !mo.tensor<[1, 32, 1024, 1024], bf16, gpu:0>
    %17 = mo.static.broadcast_to(%4) : !mo.tensor<[], bf16, gpu:0> -> !mo.tensor<[1, 32, 1024, 1024], bf16, gpu:0>
    %18 = mo.div(%16, %17) : !mo.tensor<[1, 32, 1024, 1024], bf16, gpu:0>
    %19 = mo.static.broadcast_to(%3) : !mo.tensor<[], bf16, gpu:0> -> !mo.tensor<[1, 1, 2048, 2048], bf16, gpu:0>
    %20 = mo.slice(%19, %7, %8, %9) : (!mo.tensor<[1, 1, 2048, 2048], bf16, gpu:0>, !mo.tensor<[4], si64>, !mo.tensor<[4], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 1, 1024, 1024], bf16, gpu:0>
    %21 = mo.static.broadcast_to(%20) : !mo.tensor<[1, 1, 1024, 1024], bf16, gpu:0> -> !mo.tensor<[1, 32, 1024, 1024], bf16, gpu:0>
    %22 = mo.add(%18, %21) : !mo.tensor<[1, 32, 1024, 1024], bf16, gpu:0>
    %23 = mo.reduce.softmax(%22) [axis = 3] : (!mo.tensor<[1, 32, 1024, 1024], bf16, gpu:0>) -> !mo.tensor<[1, 32, 1024, 1024], bf16, gpu:0>

    // extra casts are in the model for some reason
    %24 = mo.cast(%23) : (!mo.tensor<[1, 32, 1024, 1024], bf16, gpu:0>) -> !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>
    %25 = mo.cast(%24) : (!mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>) -> !mo.tensor<[1, 32, 1024, 1024], bf16, gpu:0>
    %26 = mo.static.broadcast_to(%25) : !mo.tensor<[1, 32, 1024, 1024], bf16, gpu:0> -> !mo.tensor<[1, 32, 1024, 1024], bf16, gpu:0>
    %27 = mo.batch_matmul(%26, %15) : (!mo.tensor<[1, 32, 1024, 1024], bf16, gpu:0>, !mo.tensor<[1, 32, 1024, 128], bf16, gpu:0>) -> !mo.tensor<[1, 32, 1024, 128], bf16, gpu:0>
    %28 = mo.transpose(%27, %5) : (!mo.tensor<[1, 32, 1024, 128], bf16, gpu:0>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 1024, 32, 128], bf16, gpu:0>
    mo.output %28 : !mo.tensor<[1, 1024, 32, 128], bf16, gpu:0>
  }

// Check the pattern and replacement with the custom op doesn't "absorb" the transpose
// CHECK-LABEL: @multihead_flash_attention_rank3_mask_does_not_absorb_wrong_transpose
// CHECK: mo.composite.masked_flash_attention_gpu
// Because we don't absorb the transpose, we have two transpose:
//  - a new transpose to transform BHSD -> BSDH
//  - the old unabsorbed transpose
// CHECK-NEXT: mo.transpose
// CHECK-NEXT: mo.transpose
mo.graph @multihead_flash_attention_rank3_mask_does_not_absorb_wrong_transpose() -> !mo.tensor<[128, 1024, 32, 1], f32, gpu:0> {
  %0 = mo.constant <#M.dense_array<2.000000e+00> : tensor<1xf32>> : !mo.tensor<[], f32, gpu:0>
  %1 = mo.constant <#M.dense_array<3.000000e+00> : tensor<1xf32>> : !mo.tensor<[], f32, gpu:0>
  %2 = mo.constant <#M.dense_array<4.000000e+00> : tensor<1xf32>> : !mo.tensor<[], f32, gpu:0>
  %3 = mo.constant <#M.dense_array<2.000000e+00> : tensor<1xf32>> : !mo.tensor<[], f32, gpu:0>
  %4 = mo.constant <#M.dense_array<8.000000e+00> : tensor<1xf32>> : !mo.tensor<[], f32, gpu:0>
  %5 = mo.constant <#M.dense_array<0, 2, 1, 3> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %6 = mo.constant <#M.dense_array<0, 2, 3, 1> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %7 = mo.constant <#M.dense_array<0, 0, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %8 = mo.constant <#M.dense_array<1, 1024, 1024> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %9 = mo.constant <#M.dense_array<1, 1, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %10 = mo.static.broadcast_to(%0) : !mo.tensor<[], f32, gpu:0> -> !mo.tensor<[1, 1024, 32, 128], f32, gpu:0>
  %11 = mo.static.broadcast_to(%1) : !mo.tensor<[], f32, gpu:0> -> !mo.tensor<[1, 1024, 32, 128], f32, gpu:0>
  %12 = mo.static.broadcast_to(%2) : !mo.tensor<[], f32, gpu:0> -> !mo.tensor<[1, 1024, 32, 128], f32, gpu:0>
  %13 = mo.transpose(%10, %5) : (!mo.tensor<[1, 1024, 32, 128], f32, gpu:0>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 32, 1024, 128], f32, gpu:0>
  %14 = mo.transpose(%11, %6) : (!mo.tensor<[1, 1024, 32, 128], f32, gpu:0>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 32, 128, 1024], f32, gpu:0>
  %15 = mo.transpose(%12, %5) : (!mo.tensor<[1, 1024, 32, 128], f32, gpu:0>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 32, 1024, 128], f32, gpu:0>
  %16 = mo.batch_matmul(%13, %14) : (!mo.tensor<[1, 32, 1024, 128], f32, gpu:0>, !mo.tensor<[1, 32, 128, 1024], f32, gpu:0>) -> !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>
  %17 = mo.static.broadcast_to(%4) : !mo.tensor<[], f32, gpu:0> -> !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>
  %18 = mo.div(%16, %17) : !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>
  %19 = mo.static.broadcast_to(%3) : !mo.tensor<[], f32, gpu:0> -> !mo.tensor<[1, 2048, 2048], f32, gpu:0>
  %20 = mo.slice(%19, %7, %8, %9) : (!mo.tensor<[1, 2048, 2048], f32, gpu:0>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[1, 1024, 1024], f32, gpu:0>
  %21 = mo.static.broadcast_to(%20) : !mo.tensor<[1, 1024, 1024], f32, gpu:0> -> !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>
  %22 = mo.add(%18, %21) : !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>
  %23 = mo.reduce.softmax(%22) [axis = 3] : (!mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>) -> !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>
  %24 = mo.static.broadcast_to(%23) : !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0> -> !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>
  %25 = mo.batch_matmul(%24, %15) : (!mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>, !mo.tensor<[1, 32, 1024, 128], f32, gpu:0>) -> !mo.tensor<[1, 32, 1024, 128], f32, gpu:0>

  // The pattern can absorb a transpose after the batch matmul as long as it can't prove it doesn't reshape BHSD -> BSHD
  // This one does something weird so we don't expect to absorb it.
  %26 = mo.constant <#M.dense_array<3, 2, 1, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %27 = mo.transpose(%25, %26) : (!mo.tensor<[1, 32, 1024, 128], f32, gpu:0>, !mo.tensor<[4], si64>) -> !mo.tensor<[128, 1024, 32, 1], f32, gpu:0>
  mo.output %27 : !mo.tensor<[128, 1024, 32, 1], f32, gpu:0>
}



// Make sure we have the custom operator and the K and V have head dimension 8
// CHECK-LABEL: @multihead_flash_attention_rank4_mask
// CHECK: mo.composite.masked_flash_attention_gpu
// CHECK-SAME: (
// CHECK-SAME: !mo.tensor<[1, 1024, 32, 128], f32, gpu:0>,
// CHECK-SAME: !mo.tensor<[1, 1024, 8, 128], f32, gpu:0>,
// CHECK-SAME: !mo.tensor<[1, 1024, 8, 128], f32, gpu:0>,
// CHECK-SAME: !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>,
// CHECK-SAME: !mo.tensor<[], f32>) -> !mo.tensor<[1, 1024, 32, 128], f32, gpu:0>
// Checks if bcasting a mask of rank 4 works with the pattern
mo.graph @multihead_flash_attention_rank4_mask_with_gqa() -> !mo.tensor<[1, 1024, 32, 128], f32, gpu:0> {
  %0 = mo.constant <#M.dense_array<2.000000e+00> : tensor<1xf32>> : !mo.tensor<[], f32, gpu:0>
  %1 = mo.constant <#M.dense_array<3.000000e+00> : tensor<1xf32>> : !mo.tensor<[], f32, gpu:0>
  %2 = mo.constant <#M.dense_array<4.000000e+00> : tensor<1xf32>> : !mo.tensor<[], f32, gpu:0>
  %3 = mo.constant <#M.dense_array<2.000000e+00> : tensor<1xf32>> : !mo.tensor<[], f32, gpu:0>
  %4 = mo.constant <#M.dense_array<8.000000e+00> : tensor<1xf32>> : !mo.tensor<[], f32, gpu:0>
  %5 = mo.constant <#M.dense_array<0, 2, 1, 3> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %6 = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %7 = mo.constant <#M.dense_array<1, 1, 1024, 1024> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %8 = mo.constant <#M.dense_array<1, 1, 1, 1> : tensor<4xsi64>> : !mo.tensor<[4], si64>

  // Groups = 4, so key and value have an initial head dimension of 8
  %9 = mo.static.broadcast_to(%0) : !mo.tensor<[], f32, gpu:0> -> !mo.tensor<[1, 1024, 32, 128], f32, gpu:0>
  %10 = mo.static.broadcast_to(%1) : !mo.tensor<[], f32, gpu:0> -> !mo.tensor<[1, 8, 128, 1024], f32, gpu:0>
  %11 = mo.static.broadcast_to(%2) : !mo.tensor<[], f32, gpu:0> -> !mo.tensor<[1, 1024, 8, 128], f32, gpu:0>

  // Perform GQA transform for key_t, duplicating weights 4x along head dimension.
  // Key is initially BHDS already.
  %12 = mo.static.reshape(%10) : !mo.tensor<[1, 8, 128, 1024], f32, gpu:0> -> !mo.tensor<[1, 8, 1, 128, 1024], f32, gpu:0>
  %13 = mo.static.broadcast_to(%12) : !mo.tensor<[1, 8, 1, 128, 1024], f32, gpu:0> -> !mo.tensor<[1, 8, 4, 128, 1024], f32, gpu:0>
  %14 = mo.static.reshape(%13) : !mo.tensor<[1, 8, 4, 128, 1024], f32, gpu:0> -> !mo.tensor<[1, 32, 128, 1024], f32, gpu:0>

  // Perform GQA transform for value, duplicating weights 4x along head dimension
  %15 = mo.static.reshape(%11) : !mo.tensor<[1, 1024, 8, 128], f32, gpu:0> -> !mo.tensor<[1, 1024, 8, 1, 128], f32, gpu:0>
  %16 = mo.static.broadcast_to(%15) : !mo.tensor<[1, 1024, 8, 1, 128], f32, gpu:0> -> !mo.tensor<[1, 1024, 8, 4, 128], f32, gpu:0>
  %17 = mo.static.reshape(%16) : !mo.tensor<[1, 1024, 8, 4, 128], f32, gpu:0> -> !mo.tensor<[1, 1024, 32, 128], f32, gpu:0>
  %18 = mo.transpose(%9, %5) : (!mo.tensor<[1, 1024, 32, 128], f32, gpu:0>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 32, 1024, 128], f32, gpu:0>
  %19 = mo.transpose(%17, %5) : (!mo.tensor<[1, 1024, 32, 128], f32, gpu:0>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 32, 1024, 128], f32, gpu:0>
  %20 = mo.batch_matmul(%18, %14) : (!mo.tensor<[1, 32, 1024, 128], f32, gpu:0>, !mo.tensor<[1, 32, 128, 1024], f32, gpu:0>) -> !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>
  %21 = mo.static.broadcast_to(%4) : !mo.tensor<[], f32, gpu:0> -> !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>
  %22 = mo.div(%20, %21) : !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>
  %23 = mo.static.broadcast_to(%3) : !mo.tensor<[], f32, gpu:0> -> !mo.tensor<[1, 1, 2048, 2048], f32, gpu:0>
  %24 = mo.slice(%23, %6, %7, %8) : (!mo.tensor<[1, 1, 2048, 2048], f32, gpu:0>, !mo.tensor<[4], si64>, !mo.tensor<[4], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 1, 1024, 1024], f32, gpu:0>
  %25 = mo.static.broadcast_to(%24) : !mo.tensor<[1, 1, 1024, 1024], f32, gpu:0> -> !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>
  %26 = mo.add(%22, %25) : !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>
  %27 = mo.reduce.softmax(%26) [axis = 3] : (!mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>) -> !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>
  %28 = mo.static.broadcast_to(%27) : !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0> -> !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>
  %29 = mo.batch_matmul(%28, %19) : (!mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>, !mo.tensor<[1, 32, 1024, 128], f32, gpu:0>) -> !mo.tensor<[1, 32, 1024, 128], f32, gpu:0>
  %30 = mo.transpose(%29, %5) : (!mo.tensor<[1, 32, 1024, 128], f32, gpu:0>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 1024, 32, 128], f32, gpu:0>
  mo.output %30 : !mo.tensor<[1, 1024, 32, 128], f32, gpu:0>
}


// CHECK-LABEL: @multihead_flash_attention_with_unsafe_bcasts_reshapes
// We should have some runtime checks before and after the flash attention kernel
// due to extraneous reshapes/bcasts

// CHECK: mo.composite.masked_flash_attention_gpu
// CHECK: kgen.param.declare MHA_ROOT_TRANSPOSE_BCAST_OR_RESHAPE_CHECK
// CHECK: kgen.param.declare MHA_ROOT_TRANSPOSE_BCAST_OR_RESHAPE_CHECK
// CHECK: kgen.param.declare MHA_ROOT_TRANSPOSE_BCAST_OR_RESHAPE_CHECK
// CHECK: kgen.param.declare MHA_ROOT_TRANSPOSE_BCAST_OR_RESHAPE_CHECK
// CHECK: mo.output

// `%dummy` is just meant to populate the parameter A and make it valid
mo.graph @multihead_flash_attention_with_unsafe_bcasts_reshapes<A, B>(%dummy: !mo.tensor<[A, B], f32, gpu:0>) -> !mo.tensor<[1, 1024, 32, 128], f32, gpu:0> {
  %0 = mo.constant <#M.dense_array<2.000000e+00> : tensor<1xf32>> : !mo.tensor<[], f32, gpu:0>
  %1 = mo.constant <#M.dense_array<3.000000e+00> : tensor<1xf32>> : !mo.tensor<[], f32, gpu:0>
  %2 = mo.constant <#M.dense_array<4.000000e+00> : tensor<1xf32>> : !mo.tensor<[], f32, gpu:0>
  %3 = mo.constant <#M.dense_array<2.000000e+00> : tensor<1xf32>> : !mo.tensor<[], f32, gpu:0>
  %4 = mo.constant <#M.dense_array<8.000000e+00> : tensor<1xf32>> : !mo.tensor<[], f32, gpu:0>
  %5 = mo.constant <#M.dense_array<0, 2, 1, 3> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %6 = mo.constant <#M.dense_array<0, 2, 3, 1> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %7 = mo.constant <#M.dense_array<0, 0, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %8 = mo.constant <#M.dense_array<1, 1024, 1024> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %9 = mo.constant <#M.dense_array<1, 1, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %10 = mo.static.broadcast_to(%0) : !mo.tensor<[], f32, gpu:0> -> !mo.tensor<[1, 1024, 32, 128], f32, gpu:0>
  %11 = mo.static.broadcast_to(%1) : !mo.tensor<[], f32, gpu:0> -> !mo.tensor<[1, 1024, 32, 128], f32, gpu:0>
  %12 = mo.static.broadcast_to(%2) : !mo.tensor<[], f32, gpu:0> -> !mo.tensor<[1, 1024, 32, 128], f32, gpu:0>

  // Before: Extraneous reshapes/bcasts we assume do nothing and should have runtime checks to verify this
  %13 = mo.static.reshape(%12) : !mo.tensor<[1, 1024, 32, 128], f32, gpu:0> -> !mo.tensor<[1, A, 32, 128], f32, gpu:0>
  %14 = mo.static.broadcast_to(%13) : !mo.tensor<[1, A, 32, 128], f32, gpu:0> -> !mo.tensor<[B, A, 32, 128], f32, gpu:0>
  %15 = mo.static.reshape(%14) : !mo.tensor<[B, A, 32, 128], f32, gpu:0> -> !mo.tensor<[1, 1024, 32, 128], f32, gpu:0>
  %16 = mo.transpose(%10, %5) : (!mo.tensor<[1, 1024, 32, 128], f32, gpu:0>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 32, 1024, 128], f32, gpu:0>
  %17 = mo.transpose(%11, %6) : (!mo.tensor<[1, 1024, 32, 128], f32, gpu:0>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 32, 128, 1024], f32, gpu:0>
  %18 = mo.transpose(%15, %5) : (!mo.tensor<[1, 1024, 32, 128], f32, gpu:0>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 32, 1024, 128], f32, gpu:0>
  %19 = mo.batch_matmul(%16, %17) : (!mo.tensor<[1, 32, 1024, 128], f32, gpu:0>, !mo.tensor<[1, 32, 128, 1024], f32, gpu:0>) -> !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>
  %20 = mo.static.broadcast_to(%4) : !mo.tensor<[], f32, gpu:0> -> !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>
  %21 = mo.div(%19, %20) : !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>
  %22 = mo.static.broadcast_to(%3) : !mo.tensor<[], f32, gpu:0> -> !mo.tensor<[1, 2048, 2048], f32, gpu:0>
  %23 = mo.slice(%22, %7, %8, %9) : (!mo.tensor<[1, 2048, 2048], f32, gpu:0>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[1, 1024, 1024], f32, gpu:0>
  %24 = mo.static.broadcast_to(%23) : !mo.tensor<[1, 1024, 1024], f32, gpu:0> -> !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>
  %25 = mo.add(%21, %24) : !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>
  %26 = mo.reduce.softmax(%25) [axis = 3] : (!mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>) -> !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>
  %27 = mo.static.broadcast_to(%26) : !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0> -> !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>
  %28 = mo.batch_matmul(%27, %18) : (!mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>, !mo.tensor<[1, 32, 1024, 128], f32, gpu:0>) -> !mo.tensor<[1, 32, 1024, 128], f32, gpu:0>

  // After: Extraneous reshapes/bcasts we assume do nothing and should have runtime checks to verify this
  %29 = mo.static.broadcast_to(%28) : !mo.tensor<[1, 32, 1024, 128], f32, gpu:0> -> !mo.tensor<[1, 32, A, 128], f32, gpu:0>
  %30 = mo.static.reshape(%29) : !mo.tensor<[1, 32, A, 128], f32, gpu:0> -> !mo.tensor<[B, 32, A, 128], f32, gpu:0>
  %31 = mo.static.reshape(%30) : !mo.tensor<[B, 32, A, 128], f32, gpu:0> -> !mo.tensor<[1, 32, 1024, 128], f32, gpu:0>
  %32 = mo.transpose(%31, %5) : (!mo.tensor<[1, 32, 1024, 128], f32, gpu:0>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 1024, 32, 128], f32, gpu:0>
  mo.output %32 : !mo.tensor<[1, 1024, 32, 128], f32, gpu:0>
}

// CHECK-LABEL: @multihead_flash_attention_non_matching_D_dim
// CHECK-NOT: mo.custom
mo.graph @multihead_flash_attention_non_matching_D_dim() -> !mo.tensor<[1, 1024, 32, 256], f32, gpu:0> {
  %0 = mo.constant <#M.dense_array<2.000000e+00> : tensor<1xf32>> : !mo.tensor<[], f32, gpu:0>
  %1 = mo.constant <#M.dense_array<3.000000e+00> : tensor<1xf32>> : !mo.tensor<[], f32, gpu:0>
  %2 = mo.constant <#M.dense_array<4.000000e+00> : tensor<1xf32>> : !mo.tensor<[], f32, gpu:0>
  %3 = mo.constant <#M.dense_array<8.000000e+00> : tensor<1xf32>> : !mo.tensor<[], f32, gpu:0>
  %4 = mo.constant <#M.dense_array<0, 2, 1, 3> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %5 = mo.constant <#M.dense_array<0, 2, 3, 1> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %6 = mo.constant <#M.dense_array<0, 0, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %7 = mo.constant <#M.dense_array<1, 1024, 1024> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %8 = mo.constant <#M.dense_array<1, 1, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %9 = mo.static.broadcast_to(%0) : !mo.tensor<[], f32, gpu:0> -> !mo.tensor<[1, 1024, 32, 256], f32, gpu:0>
  %10 = mo.static.broadcast_to(%1) : !mo.tensor<[], f32, gpu:0> -> !mo.tensor<[1, 1024, 32, 256], f32, gpu:0>
  %11 = mo.static.broadcast_to(%2) : !mo.tensor<[], f32, gpu:0> -> !mo.tensor<[1, 1024, 32, 256], f32, gpu:0>
  %12 = mo.transpose(%9, %4) : (!mo.tensor<[1, 1024, 32, 256], f32, gpu:0>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 32, 1024, 256], f32, gpu:0>
  %13 = mo.transpose(%10, %5) : (!mo.tensor<[1, 1024, 32, 256], f32, gpu:0>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 32, 256, 1024], f32, gpu:0>
  %14 = mo.transpose(%11, %4) : (!mo.tensor<[1, 1024, 32, 256], f32, gpu:0>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 32, 1024, 256], f32, gpu:0>
  %15 = mo.batch_matmul(%12, %13) : (!mo.tensor<[1, 32, 1024, 256], f32, gpu:0>, !mo.tensor<[1, 32, 256, 1024], f32, gpu:0>) -> !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>
  %16 = mo.static.broadcast_to(%3) : !mo.tensor<[], f32, gpu:0> -> !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>
  %17 = mo.div(%15, %16) : !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>
  %18 = mo.static.broadcast_to(%0) : !mo.tensor<[], f32, gpu:0> -> !mo.tensor<[1, 2048, 2048], f32, gpu:0>
  %19 = mo.slice(%18, %6, %7, %8) : (!mo.tensor<[1, 2048, 2048], f32, gpu:0>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[1, 1024, 1024], f32, gpu:0>
  %20 = mo.static.broadcast_to(%19) : !mo.tensor<[1, 1024, 1024], f32, gpu:0> -> !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>
  %21 = mo.add(%17, %20) : !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>
  %22 = mo.reduce.softmax(%21) [axis = 3] : (!mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>) -> !mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>
  %23 = mo.batch_matmul(%22, %14) : (!mo.tensor<[1, 32, 1024, 1024], f32, gpu:0>, !mo.tensor<[1, 32, 1024, 256], f32, gpu:0>) -> !mo.tensor<[1, 32, 1024, 256], f32, gpu:0>
  %24 = mo.transpose(%23, %4) : (!mo.tensor<[1, 32, 1024, 256], f32, gpu:0>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 1024, 32, 256], f32, gpu:0>
  mo.output %24 : !mo.tensor<[1, 1024, 32, 256], f32, gpu:0>
}

// CHECK-LABEL: @multihead_flash_attention_dynamic_H_dim
// CHECK-NOT: mo.custom
mo.graph @multihead_flash_attention_dynamic_H_dim<H0>() -> !mo.tensor<[1, 1024, H0, 128], f32, gpu:0> {
  %0 = mo.constant <#M.dense_array<2.000000e+00> : tensor<1xf32>> : !mo.tensor<[], f32, gpu:0>
  %1 = mo.constant <#M.dense_array<3.000000e+00> : tensor<1xf32>> : !mo.tensor<[], f32, gpu:0>
  %2 = mo.constant <#M.dense_array<4.000000e+00> : tensor<1xf32>> : !mo.tensor<[], f32, gpu:0>
  %3 = mo.constant <#M.dense_array<2.000000e+00> : tensor<1xf32>> : !mo.tensor<[], f32, gpu:0>
  %4 = mo.constant <#M.dense_array<8.000000e+00> : tensor<1xf32>> : !mo.tensor<[], f32, gpu:0>
  %5 = mo.constant <#M.dense_array<0, 2, 1, 3> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %6 = mo.constant <#M.dense_array<0, 2, 3, 1> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %7 = mo.constant <#M.dense_array<0, 0, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %8 = mo.constant <#M.dense_array<1, 1024, 1024> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %9 = mo.constant <#M.dense_array<1, 1, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %10 = mo.static.broadcast_to(%0) : !mo.tensor<[], f32, gpu:0> -> !mo.tensor<[1, 1024, H0, 128], f32, gpu:0>
  %11 = mo.static.broadcast_to(%1) : !mo.tensor<[], f32, gpu:0> -> !mo.tensor<[1, 1024, H0, 128], f32, gpu:0>
  %12 = mo.static.broadcast_to(%2) : !mo.tensor<[], f32, gpu:0> -> !mo.tensor<[1, 1024, H0, 128], f32, gpu:0>
  %13 = mo.transpose(%10, %5) : (!mo.tensor<[1, 1024, H0, 128], f32, gpu:0>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, H0, 1024, 128], f32, gpu:0>
  %14 = mo.transpose(%11, %6) : (!mo.tensor<[1, 1024, H0, 128], f32, gpu:0>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, H0, 128, 1024], f32, gpu:0>
  %15 = mo.transpose(%12, %5) : (!mo.tensor<[1, 1024, H0, 128], f32, gpu:0>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, H0, 1024, 128], f32, gpu:0>
  %16 = mo.batch_matmul(%13, %14) : (!mo.tensor<[1, H0, 1024, 128], f32, gpu:0>, !mo.tensor<[1, H0, 128, 1024], f32, gpu:0>) -> !mo.tensor<[1, H0, 1024, 1024], f32, gpu:0>
  %17 = mo.static.broadcast_to(%4) : !mo.tensor<[], f32, gpu:0> -> !mo.tensor<[1, H0, 1024, 1024], f32, gpu:0>
  %18 = mo.div(%16, %17) : !mo.tensor<[1, H0, 1024, 1024], f32, gpu:0>
  %19 = mo.static.broadcast_to(%3) : !mo.tensor<[], f32, gpu:0> -> !mo.tensor<[1, 2048, 2048], f32, gpu:0>
  %20 = mo.slice(%19, %7, %8, %9) : (!mo.tensor<[1, 2048, 2048], f32, gpu:0>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[1, 1024, 1024], f32, gpu:0>
  %21 = mo.static.broadcast_to(%20) : !mo.tensor<[1, 1024, 1024], f32, gpu:0> -> !mo.tensor<[1, H0, 1024, 1024], f32, gpu:0>
  %22 = mo.add(%18, %21) : !mo.tensor<[1, H0, 1024, 1024], f32, gpu:0>
  %23 = mo.reduce.softmax(%22) [axis = 3] : (!mo.tensor<[1, H0, 1024, 1024], f32, gpu:0>) -> !mo.tensor<[1, H0, 1024, 1024], f32, gpu:0>
  %24 = mo.static.broadcast_to(%23) : !mo.tensor<[1, H0, 1024, 1024], f32, gpu:0> -> !mo.tensor<[1, H0, 1024, 1024], f32, gpu:0>
  %25 = mo.batch_matmul(%24, %15) : (!mo.tensor<[1, H0, 1024, 1024], f32, gpu:0>, !mo.tensor<[1, H0, 1024, 128], f32, gpu:0>) -> !mo.tensor<[1, H0, 1024, 128], f32, gpu:0>
  %26 = mo.transpose(%25, %5) : (!mo.tensor<[1, H0, 1024, 128], f32, gpu:0>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 1024, H0, 128], f32, gpu:0>
  mo.output %26 : !mo.tensor<[1, 1024, H0, 128], f32, gpu:0>
}
