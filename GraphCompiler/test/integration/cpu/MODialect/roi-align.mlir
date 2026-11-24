// RUN: mo-opt %s --gc-pipeline --allow-unregistered-dialect -o %t.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

mo.graph @roi_align(%inp_ones: !mo.tensor<[1, 5, 5, 1], f32>) -> !mo.tensor<[1, 2, 2, 1], f32> {
    %output_height = mo.constant <#M.dense_array<2> : tensor<1xsi64>> : !mo.tensor<[], si64>
    %spatial_scale = mo.constant <#M.dense_array<1.0> : tensor<1xf32>> : !mo.tensor<[], f32>
    %sampling_ratio = mo.constant <#M.dense_array<2.0> : tensor<1xf32>> : !mo.tensor<[], f32>
    %inp_vals = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0> : tensor<25xf32>> : !mo.tensor<[1, 5, 5, 1], f32>
    %inp = mo.mul(%inp_ones, %inp_vals) : !mo.tensor<[1, 5, 5, 1], f32>
    // Change this
    %rois = mo.constant <#M.dense_array<0.0, 0.0, 0.0, 2.0, 2.0> : tensor<5xf32>> : !mo.tensor<[1, 5], f32>

    %output = mo.roi_align(%inp, %rois, %output_height, %output_height, %spatial_scale, %sampling_ratio) [aligned = false, mode = "AVG"] : (!mo.tensor<[1, 5, 5, 1], f32>, !mo.tensor<[1, 5], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[1, 2, 2, 1], f32>
    mo.output %output : !mo.tensor<[1, 2, 2, 1], f32>
}

// CHECK-LABEL: --- Running 'roi_align':
// CHECK-NEXT:  'roi_align' returned tensor<1x2x2x1xf32> [4.000000e+00, 5.000000e+00, 9.000000e+00, 1.000000e+01]

mo.graph @roi_align_fallback_shape_func<() -> D0, D1, D2, D3>(%one: !mo.tensor<[], si64>, %inp_ones: !mo.tensor<[1, 5, 5, 1], f32>, %roi_ones : !mo.tensor<[1, 5], f32>) -> !mo.tensor<[D0, D1, D2, D3], si64> {
    %output_height = mo.constant <#M.dense_array<2> : tensor<1xsi64>> : !mo.tensor<[], si64>
    %spatial_scale = mo.constant <#M.dense_array<1.0> : tensor<1xf32>> : !mo.tensor<[], f32>
    %sampling_ratio = mo.constant <#M.dense_array<2.0> : tensor<1xf32>> : !mo.tensor<[], f32>
    mo.invoke_shape_func["mo.roi_align"]<() -> D4, D5, D6, D7>(%inp_ones, %roi_ones, %output_height, %output_height, %spatial_scale, %sampling_ratio) [dataDeptTensors = [2 : i32, 3 : i32]] : (!mo.tensor<[1, 5, 5, 1], f32>, !mo.tensor<[1, 5], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>)
    %val = mo.static.broadcast_to(%one) : !mo.tensor<[], si64> -> !mo.tensor<[D4, D5, D6, D7], si64>
    mo.output<D4, D5, D6, D7> %val : !mo.tensor<[D4, D5, D6, D7], si64>
}

// CHECK-LABEL: --- Running 'roi_align_fallback_shape_func':
// CHECK-NEXT: 'roi_align_fallback_shape_func' returned tensor<1x2x2x1xsi64>
