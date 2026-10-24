// RUN: mo-opt %s --gc-pipeline=constant-fold=false --allow-unregistered-dialect -o %t.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

mo.graph @resize_nearest_half_pixel_down(%input : !mo.tensor<[1, 1, 2, 2], f32>) -> !mo.tensor<[1, 1, 7, 8], f32> {
    %size = mo.constant <#M.dense_array<1, 1, 7, 8> : tensor<4xsi64>> : !mo.tensor<[4], si64>
    %input_vals = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0> : tensor<4xf32>> : !mo.tensor<[1, 1, 2, 2], f32>
    %input2 = mo.mul(%input, %input_vals) : !mo.tensor<[1, 1, 2, 2], f32>
    // roundMode = HalfDown
    %output = mo.resize.nearest(%input2, %size) [coordinate_transform_mode = #mo.coord_trans_mode<half_pixel>]: (!mo.tensor<[1, 1, 2, 2], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 1, 7, 8], f32>
    mo.output %output : !mo.tensor<[1, 1, 7, 8], f32>
}
// CHECK-LABEL: --- Running 'resize_nearest_half_pixel_down':
// CHECK-NEXT: 'resize_nearest_half_pixel_down' returned tensor<1x1x7x8xf32> [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00]

mo.graph @resize_nearest_half_pixel_ceil(%input : !mo.tensor<[1, 1, 2, 2], f32>) -> !mo.tensor<[1, 1, 7, 8], f32> {
    %size = mo.constant <#M.dense_array<1, 1, 7, 8> : tensor<4xsi64>> : !mo.tensor<[4], si64>
    %input_vals = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0> : tensor<4xf32>> : !mo.tensor<[1, 1, 2, 2], f32>
    %input2 = mo.mul(%input, %input_vals) : !mo.tensor<[1, 1, 2, 2], f32>
    // roundMode = Ceil
    %output = mo.resize.nearest(%input2, %size) [coordinate_transform_mode = #mo.coord_trans_mode<half_pixel>, round_mode = 3]: (!mo.tensor<[1, 1, 2, 2], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 1, 7, 8], f32>
    mo.output %output : !mo.tensor<[1, 1, 7, 8], f32>
}

// CHECK-LABEL: --- Running 'resize_nearest_half_pixel_ceil':
// CHECK-NEXT: 'resize_nearest_half_pixel_ceil' returned tensor<1x1x7x8xf32> [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 3.000000e+00, 3.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 3.000000e+00, 3.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 3.000000e+00, 3.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 3.000000e+00, 3.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 3.000000e+00, 3.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00]

mo.graph @resize_linear_align_corners(%input : !mo.tensor<[1, 1, 2, 2], f32>) -> !mo.tensor<[1, 1, 4, 4], f32> {
    %size = mo.constant <#M.dense_array<1, 1, 4, 4> : tensor<4xsi64>> : !mo.tensor<[4], si64>
    %input_vals = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0> : tensor<4xf32>> : !mo.tensor<[1, 1, 2, 2], f32>
    %input2 = mo.mul(%input, %input_vals) : !mo.tensor<[1, 1, 2, 2], f32>
    %output = mo.resize.linear(%input2, %size) [coordinate_transform_mode = #mo.coord_trans_mode<align_corners>, antialias = false]: (!mo.tensor<[1, 1, 2, 2], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 1, 4, 4], f32>
    mo.output %output : !mo.tensor<[1, 1, 4, 4], f32>
}
// CHECK-LABEL: --- Running 'resize_linear_align_corners':
// CHECK-NEXT: 'resize_linear_align_corners' returned tensor<1x1x4x4xf32> [1.000000e+00, 1.333333e+00, 1.666667e+00, 2.000000e+00, 1.666667e+00, 2.000000e+00, 2.333333e+00, 2.666667e+00, 2.333333e+00, 2.666667e+00, 3.000000e+00, 3.333333e+00, 3.000000e+00, 3.333333e+00, 3.666667e+00, 4.000000e+00]

mo.graph @resize_linear_half_pixel_1D(%input : !mo.tensor<[1, 1, 2, 2], f32>) -> !mo.tensor<[1, 1, 4, 4], f32> {
    %size = mo.constant <#M.dense_array<1, 1, 4, 4> : tensor<4xsi64>> : !mo.tensor<[4], si64>
    %input_vals = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0> : tensor<4xf32>> : !mo.tensor<[1, 1, 2, 2], f32>
    %input2 = mo.mul(%input, %input_vals) : !mo.tensor<[1, 1, 2, 2], f32>
    %output = mo.resize.linear(%input2, %size) [coordinate_transform_mode = #mo.coord_trans_mode<half_pixel_1D>, antialias = false]: (!mo.tensor<[1, 1, 2, 2], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 1, 4, 4], f32>
    mo.output %output : !mo.tensor<[1, 1, 4, 4], f32>
}
// CHECK-LABEL: --- Running 'resize_linear_half_pixel_1D':
// CHECK-NEXT: 'resize_linear_half_pixel_1D' returned tensor<1x1x4x4xf32> [1.000000e+00, 1.250000e+00, 1.750000e+00, 2.000000e+00, 1.500000e+00, 1.750000e+00, 2.250000e+00, 2.500000e+00, 2.500000e+00, 2.750000e+00, 3.250000e+00, 3.500000e+00, 3.000000e+00, 3.250000e+00, 3.750000e+00, 4.000000e+00]

mo.graph @resize_fallback_shape_func<() -> D0, D1, D2, D3>(%one : !mo.tensor<[], si64>, %input : !mo.tensor<[1, 1, 2, 2], f32>, %size : !mo.tensor<[4], si64>) -> !mo.tensor<[D0, D1, D2, D3], si64> {
    mo.invoke_shape_func["mo.resize.linear"]<() -> D4, D5, D6, D7>(%input, %size) [dataDeptTensors = [1 : i32]] : (!mo.tensor<[1, 1, 2, 2], f32>, !mo.tensor<[4], si64>)
    %val = mo.static.broadcast_to(%one) : !mo.tensor<[], si64> -> !mo.tensor<[D4, D5, D6, D7], si64>
    mo.output<D4, D5, D6, D7> %val : !mo.tensor<[D4, D5, D6, D7], si64>
}
// CHECK-LABEL: --- Running 'resize_fallback_shape_func':
// CHECK-NEXT: 'resize_fallback_shape_func' returned tensor<1x1x1x1xsi64> [1]
