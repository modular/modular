// RUN: mo-opt %s --gc-pipeline --allow-unregistered-dialect -o %t.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

// GEX-4020 regression: output_height/output_width are block arguments here
// (not mo.constant), the shape a runtime-operand eager graph produces. The
// fallback previously truncated to the first four operands, dropping
// spatial_scale/sampling_ratio, which the registered shape function still
// declares -- so the emitted invoke_shape_func must carry all six.
mo.graph @roi_align_dynamic_output_size<() -> D0, D1, D2, D3>(
  %inp_ones: !mo.tensor<[1, 5, 5, 1], f32>,
  %output_height: !mo.tensor<[], si64>,
  %output_width: !mo.tensor<[], si64>
) -> !mo.tensor<[D0, D1, D2, D3], f32> {
    %spatial_scale = mo.constant <#M.dense_array<1.0> : tensor<1xf32>> : !mo.tensor<[], f32>
    %sampling_ratio = mo.constant <#M.dense_array<2.0> : tensor<1xf32>> : !mo.tensor<[], f32>
    %inp_vals = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0> : tensor<25xf32>> : !mo.tensor<[1, 5, 5, 1], f32>
    %inp = mo.mul(%inp_ones, %inp_vals) : !mo.tensor<[1, 5, 5, 1], f32>
    %rois = mo.constant <#M.dense_array<0.0, 0.0, 0.0, 2.0, 2.0> : tensor<5xf32>> : !mo.tensor<[1, 5], f32>

    %output = mo.roi_align<() -> D4, D5, D6, D7>(%inp, %rois, %output_height, %output_width, %spatial_scale, %sampling_ratio) [aligned = false, mode = "AVG"] : (!mo.tensor<[1, 5, 5, 1], f32>, !mo.tensor<[1, 5], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[D4, D5, D6, D7], f32>
    mo.output<D4, D5, D6, D7> %output : !mo.tensor<[D4, D5, D6, D7], f32>
}

// CHECK{LITERAL}: 'roi_align_dynamic_output_size' returned tensor<1x1x1x1xf32> [7.000000e+00]
