// RUN: mo-opt --mo-legalize-rmo-operators --canonicalize %s | FileCheck %s --enable-var-scope

// CHECK-LABEL: mo.graph @conv
// CHECK: (%[[INPUT:.*]]: !mo.tensor<[10, 5, 5, 32], f32>,
// CHECK:  %[[FILTER:.*]]: !mo.tensor<[2, 2, 32, 64], f32>)
// CHECK: -> !mo.tensor<[10, 4, 4, 64], f32>

mo.graph @conv(
  %input: !mo.tensor<[10, 5, 5, 32], f32>,
  %filter: !mo.tensor<[2, 2, 32, 64], f32>
) -> !mo.tensor<[10, 4, 4, 64], f32> {

  // CHECK-DAG: %[[GROUPS:.*]] = mo.constant <#M.dense_array<1> : !M.array<1xsi64>> : !mo.tensor<[], si64>
  // CHECK-DAG: %[[C1:.*]] = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[C0:.*]] = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>

  // CHECK: %[[CONV:.*]] = mo.conv(%arg0, %arg1) [strides = %[[C1]], dilations = %[[C1]], paddings = %[[C0]], num_groups = %[[GROUPS]], filter_layout = ""] : (!mo.tensor<[10, 5, 5, 32], f32>, !mo.tensor<[2, 2, 32, 64], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[10, 4, 4, 64], f32>

  %out = rmo.conv(%input, %filter) [strides = #mosh<ape[1, 1]> : !mosh.ape, dilations = #mosh<ape[1, 1]> : !mosh.ape, paddings = #mosh<ape[0, 0, 0, 0]> : !mosh.ape, num_groups = 1]: (!mo.tensor<[10, 5, 5, 32], f32>, !mo.tensor<[2, 2, 32, 64], f32>) -> !mo.tensor<[10, 4, 4, 64], f32>

  // CHECK: mo.output %[[CONV]] : !mo.tensor<[10, 4, 4, 64], f32>
  mo.output %out : !mo.tensor<[10, 4, 4, 64], f32>
}

// -----

// CHECK-LABEL: mo.graph @conv_gpu
// CHECK: (%[[INPUT:.*]]: !mo.tensor<[10, 5, 5, 32], f32, gpu:0>,
// CHECK:  %[[FILTER:.*]]: !mo.tensor<[2, 2, 32, 64], f32, gpu:0>)
// CHECK: -> !mo.tensor<[10, 4, 4, 64], f32, gpu:0>

mo.graph @conv_gpu(
  %input: !mo.tensor<[10, 5, 5, 32], f32, gpu:0>,
  %filter: !mo.tensor<[2, 2, 32, 64], f32, gpu:0>
) -> !mo.tensor<[10, 4, 4, 64], f32, gpu:0> {

  // CHECK-DAG: %[[GROUPS:.*]] = mo.constant <#M.dense_array<1> : !M.array<1xsi64>> : !mo.tensor<[], si64>
  // CHECK-DAG: %[[C1:.*]] = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[C0:.*]] = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>

  // CHECK: %[[CONV:.*]] = mo.conv(%arg0, %arg1) [strides = %[[C1]], dilations = %[[C1]], paddings = %[[C0]], num_groups = %[[GROUPS]], filter_layout = ""] : (!mo.tensor<[10, 5, 5, 32], f32, gpu:0>, !mo.tensor<[2, 2, 32, 64], f32, gpu:0>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[10, 4, 4, 64], f32, gpu:0>
  %out = rmo.conv(%input, %filter) [strides = #mosh<ape[1, 1]> : !mosh.ape, dilations = #mosh<ape[1, 1]> : !mosh.ape, paddings = #mosh<ape[0, 0, 0, 0]> : !mosh.ape, num_groups = 1]: (!mo.tensor<[10, 5, 5, 32], f32, gpu:0>, !mo.tensor<[2, 2, 32, 64], f32, gpu:0>) -> !mo.tensor<[10, 4, 4, 64], f32, gpu:0>

  // CHECK: mo.output %[[CONV]] : !mo.tensor<[10, 4, 4, 64], f32, gpu:0>
  mo.output %out : !mo.tensor<[10, 4, 4, 64], f32, gpu:0>
}

// -----

// CHECK-LABEL: mo.graph @symbolic_conv<height, width, channels>
// CHECK: (%[[INPUT:.*]]: !mo.tensor<[1, height, width, channels], bf16>,
// CHECK:  %[[FILTER:.*]]: !mo.tensor<[16, 16, 3, 1024], bf16>)
// CHECK: -> !mo.tensor<[1, div(height, 16), div(width, 16), 1024], bf16>
mo.graph @symbolic_conv<height, width, channels>(%arg0: !mo.tensor<[1, height, width, channels], bf16>, %arg1: !mo.tensor<[16, 16, 3, 1024], bf16>) -> !mo.tensor<[1, div(height, 16), div(width, 16), 1024], bf16> {

  // CHECK-DAG: %[[GROUPS:.*]] = mo.constant <#M.dense_array<1> : !M.array<1xsi64>> : !mo.tensor<[], si64>
  // CHECK-DAG: %[[STRIDES:.*]] = mo.constant <#M.dense_array<16, 16> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[PADDINGS:.*]] = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  // CHECK-DAG: %[[DILATIONS:.*]] = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>

  // CHECK: %[[CONV:.*]] = mo.conv(%arg0, %arg1) [strides = %[[STRIDES]], dilations = %[[DILATIONS]], paddings = %[[PADDINGS]], num_groups = %[[GROUPS]], filter_layout = ""] : (!mo.tensor<[1, height, width, channels], bf16>, !mo.tensor<[16, 16, 3, 1024], bf16>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[1, div(height, 16), div(width, 16), 1024], bf16>
  %1 = rmo.conv(%arg0, %arg1) [strides = #mosh<ape[16, 16]> : !mosh.ape, dilations = #mosh<ape[1, 1]> : !mosh.ape, paddings = #mosh<ape[0, 0, 0, 0]> : !mosh.ape, num_groups = 1] : (!mo.tensor<[1, height, width, channels], bf16>, !mo.tensor<[16, 16, 3, 1024], bf16>) -> !mo.tensor<[1, div(height, 16), div(width, 16), 1024], bf16>

  // CHECK: mo.output %[[CONV]] : !mo.tensor<[1, div(height, 16), div(width, 16), 1024], bf16>
  mo.output %1 : !mo.tensor<[1, div(height, 16), div(width, 16), 1024], bf16>
}
